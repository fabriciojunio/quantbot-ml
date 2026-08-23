"""
Carteira PAPER — o bot opera sozinho com dinheiro SIMULADO.

Ideia: durante os 6 meses, o bot toma as MESMAS decisões que tomaria com
dinheiro real (comprar os FIIs aprovados no topo, vender os que saíram da
lista), mas com capital fictício. Assim ele cria um TRACK RECORD honesto —
acertos e erros — que alimenta o aprendizado (income/feedback.py).

Não é backtest com dados passados: é "forward test", decidindo no presente
e medindo o resultado conforme o tempo passa. É o jeito certo de validar
uma estratégia antes de arriscar dinheiro de verdade.

Persistência: data/paper_book.json (estado) — fora do controle de versão.

Honestidade: 6 meses é pouco para uma estratégia de dividendos (que é de
anos). O sinal de aprendizado será fraco e ruidoso no começo. O mecanismo,
porém, é real e correto, e fica mais útil quanto mais tempo roda.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("quantbot.income.paper_book")


def _hoje() -> str:
    return date.today().isoformat()


def _dias(d1: str, d2: str) -> int:
    try:
        a = datetime.fromisoformat(d1).date()
        b = datetime.fromisoformat(d2).date()
        return (b - a).days
    except ValueError:
        return 0


# Custo de operar FII na B3 para pessoa física. A maioria das corretoras zerou
# a corretagem, mas emolumentos e liquidação continuam existindo, e em fundo
# pouco líquido o spread do book é o que realmente pesa.
EMOLUMENTOS_PCT = 0.0325 / 100   # B3, por ponta
SPREAD_MEIO_PCT = 0.15 / 100     # metade do spread típico, por ponta
CUSTO_PONTA = EMOLUMENTOS_PCT + SPREAD_MEIO_PCT


@dataclass
class Posicao:
    ticker: str
    cotas: float
    preco_entrada: float
    data_entrada: str
    score_entrada: float
    dy_entrada: float
    sentimento_entrada: str = "neutro"
    ultima_avaliacao: str = ""       # última foto de desempenho (aprendizado)
    ultimo_provento: str = ""        # último accrual de provento — campo separado
    dividendos: float = 0.0          # proventos acumulados (vão pro caixa)
    custo_entrada: float = 0.0       # emolumentos + spread pagos na compra


class PaperBook:
    """Carteira simulada que segue os sinais do screener."""

    def __init__(
        self,
        arquivo: str = "data/paper_book.json",
        capital_inicial: float = 10_000.0,
        max_posicoes: int = 12,
    ):
        self.arquivo = Path(arquivo)
        self.max_posicoes = max_posicoes
        self.capital_inicial = capital_inicial
        self.estado = self._carregar()

    # ── persistência ────────────────────────────────────────────────────

    def _carregar(self) -> dict:
        if self.arquivo.exists():
            try:
                return json.loads(self.arquivo.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("paper_book.json corrompido; recriando.")
        return {
            "capital_inicial": self.capital_inicial,
            "caixa": self.capital_inicial,
            "posicoes": {},      # ticker -> dict(Posicao)
            "fechadas": [],      # operações encerradas
            "serie_valor": [],   # [{data, valor_total}]
            "log": [],           # decisões
        }

    def salvar(self) -> None:
        self.arquivo.parent.mkdir(parents=True, exist_ok=True)
        self.arquivo.write_text(
            json.dumps(self.estado, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── helpers ─────────────────────────────────────────────────────────

    @property
    def caixa(self) -> float:
        return self.estado["caixa"]

    def _posicoes(self) -> Dict[str, dict]:
        return self.estado["posicoes"]

    def valor_total(self, precos: Dict[str, float]) -> float:
        total = self.caixa
        for tk, p in self._posicoes().items():
            preco = precos.get(tk, p["preco_entrada"])
            total += p["cotas"] * preco
        return round(total, 2)

    # ── ciclo principal ─────────────────────────────────────────────────

    def marcar_mercado(self, precos: Dict[str, float], cdi_anual: float = 14.4) -> None:
        """Atualiza valor, acumula proventos (DY) no caixa e registra a série.

        O accrual usa `ultimo_provento`, um campo só dele, e o atualiza sempre.
        Até 23/08/2026 esta função lia `ultima_avaliacao` — campo que só a
        `avaliar()` atualiza, e apenas de 30 em 30 dias. O provento de todos os
        dias desde a entrada era relançado a cada marcação, então em n dias
        entravam n(n+1)/2 dias de rendimento em vez de n. Em 61 dias isso
        multiplicou o provento por 14, e o track record inteiro saiu errado.
        """
        hoje = _hoje()
        for tk, p in self._posicoes().items():
            ref = p.get("ultimo_provento") or p["data_entrada"]
            d = _dias(ref, hoje)
            if d <= 0 or p["dy_entrada"] <= 0:
                continue
            prov = p["cotas"] * p["preco_entrada"] * (p["dy_entrada"] / 100) * (d / 365)
            p["dividendos"] = round(p["dividendos"] + prov, 2)
            self.estado["caixa"] = round(self.caixa + prov, 2)
            p["ultimo_provento"] = hoje  # sem isto, o accrual vira cascata
        self.estado["serie_valor"].append({"data": hoje, "valor_total": self.valor_total(precos)})

    def decidir(
        self,
        aprovados: list,
        precos: Dict[str, float],
        sentimento_por_ticker: Optional[Dict[str, tuple]] = None,
    ) -> List[dict]:
        """
        Decide compras e vendas com base no screening.

        aprovados: lista de objetos com .ticker, .score, .dy_12m, .pvp
                   (já ordenados por score desc — ex.: FIIScreener.top_n()).
        precos: {ticker: preço atual}.
        Retorna a lista de decisões tomadas nesta rodada.
        """
        hoje = _hoje()
        sentimento_por_ticker = sentimento_por_ticker or {}
        aprovados_set = {a.ticker for a in aprovados}
        decisoes: List[dict] = []
        pos = self._posicoes()

        # 1) VENDER o que saiu da lista de aprovados (sinal de saída)
        for tk in list(pos.keys()):
            if tk not in aprovados_set:
                preco = precos.get(tk, pos[tk]["preco_entrada"])
                dec = self._vender(tk, preco, motivo="saiu da lista de aprovados")
                decisoes.append(dec)

        # 2) COMPRAR os melhores que ainda não temos (até o máximo de posições)
        alvo = self.valor_total(precos) / self.max_posicoes
        for a in aprovados:
            if len(pos) >= self.max_posicoes:
                break
            if a.ticker in pos:
                continue
            preco = precos.get(a.ticker, 0)
            if preco <= 0:
                continue
            investir = min(self.caixa, alvo)
            cotas = investir // preco  # cotas inteiras
            if cotas < 1:
                continue
            custo = cotas * preco
            taxa = custo * CUSTO_PONTA          # emolumentos + spread na compra
            self.estado["caixa"] = round(self.caixa - custo - taxa, 2)
            sent = sentimento_por_ticker.get(a.ticker, ("neutro", 0.5))[0]
            pos[a.ticker] = asdict(Posicao(
                ticker=a.ticker, cotas=cotas, preco_entrada=preco,
                data_entrada=hoje, score_entrada=round(a.score, 1),
                dy_entrada=round(a.dy_12m, 2), sentimento_entrada=sent,
                ultima_avaliacao=hoje, ultimo_provento=hoje,
                custo_entrada=round(taxa, 2),
            ))
            dec = {"data": hoje, "acao": "COMPRAR", "ticker": a.ticker,
                   "cotas": cotas, "preco": preco, "score": round(a.score, 1),
                   "dy": round(a.dy_12m, 2)}
            decisoes.append(dec)

        self.estado["log"].extend(decisoes)
        return decisoes

    def _vender(self, ticker: str, preco: float, motivo: str) -> dict:
        p = self._posicoes().pop(ticker)
        valor = p["cotas"] * preco
        taxa_saida = valor * CUSTO_PONTA
        self.estado["caixa"] = round(self.caixa + valor - taxa_saida, 2)

        investido = p["cotas"] * p["preco_entrada"]
        retorno_pct = (preco - p["preco_entrada"]) / p["preco_entrada"] * 100
        div_pct = (p["dividendos"] / investido * 100) if investido else 0.0
        custo_total = p.get("custo_entrada", 0.0) + taxa_saida
        custo_pct = (custo_total / investido * 100) if investido else 0.0
        total_pct = retorno_pct + div_pct - custo_pct

        fechada = {
            "ticker": ticker, "data_entrada": p["data_entrada"], "data_saida": _hoje(),
            "preco_entrada": p["preco_entrada"], "preco_saida": preco,
            "dias": _dias(p["data_entrada"], _hoje()),
            "retorno_preco_pct": round(retorno_pct, 2),
            "dividendos": round(p["dividendos"], 2),
            "dividendos_pct": round(div_pct, 2),
            "custo_total": round(custo_total, 2),
            "custo_pct": round(custo_pct, 3),
            "retorno_total_pct": round(total_pct, 2),
            "score_entrada": p["score_entrada"], "dy_entrada": p["dy_entrada"],
            "sentimento_entrada": p["sentimento_entrada"], "motivo": motivo,
        }
        self.estado["fechadas"].append(fechada)
        return {"data": _hoje(), "acao": "VENDER", "ticker": ticker,
                "preco": preco, "retorno_total_pct": round(total_pct, 2), "motivo": motivo}

    def avaliar(self, precos: Dict[str, float], cdi_anual: float = 14.4,
                dias_min: int = 30) -> List[dict]:
        """
        Tira uma "foto" de desempenho das posições antigas (>= dias_min desde
        a última avaliação) SEM vender — gera os sinais de acerto/erro que o
        feedback usa para aprender. Compara o retorno anualizado vs CDI.
        """
        hoje = _hoje()
        resultados: List[dict] = []
        for tk, p in self._posicoes().items():
            d = _dias(p["ultima_avaliacao"] or p["data_entrada"], hoje)
            if d < dias_min:
                continue
            preco = precos.get(tk, p["preco_entrada"])
            investido = p["cotas"] * p["preco_entrada"]
            ret_preco = (preco - p["preco_entrada"]) / p["preco_entrada"] * 100
            ret_div = (p["dividendos"] / investido * 100) if investido else 0.0
            # Desconta o custo das duas pontas: quem avalia posição sem contar
            # a saída está medindo um resultado que não dá para realizar.
            custo = ((p.get("custo_entrada", 0.0) + preco * p["cotas"] * CUSTO_PONTA)
                     / investido * 100) if investido else 0.0
            total = ret_preco + ret_div - custo
            dias_tot = max(1, _dias(p["data_entrada"], hoje))
            anualizado = total * (365 / dias_tot)
            excedente = anualizado - cdi_anual  # bateu a renda fixa?
            resultados.append({
                "ticker": tk, "score_entrada": p["score_entrada"],
                "dy_entrada": p["dy_entrada"], "sentimento_entrada": p["sentimento_entrada"],
                "retorno_total_pct": round(total, 2),
                "anualizado_pct": round(anualizado, 2),
                "excedente_cdi_pct": round(excedente, 2),
                "acerto": excedente > 0,
            })
            p["ultima_avaliacao"] = hoje
        return resultados

    # ── resumo ──────────────────────────────────────────────────────────

    def resumo(self, precos: Dict[str, float]) -> dict:
        vt = self.valor_total(precos)
        ret = (vt - self.capital_inicial) / self.capital_inicial * 100
        fechadas = self.estado["fechadas"]
        acertos = sum(1 for f in fechadas if f["retorno_total_pct"] > 0)
        return {
            "valor_total": vt,
            "capital_inicial": self.capital_inicial,
            "retorno_pct": round(ret, 2),
            "caixa": round(self.caixa, 2),
            "n_posicoes": len(self._posicoes()),
            "n_fechadas": len(fechadas),
            "acertos_fechadas": acertos,
        }
