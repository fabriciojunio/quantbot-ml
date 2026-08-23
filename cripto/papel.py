"""
Book de paper trading.

O que o painel antigo chamava de "bot rodando" era um saldo de US$ 10.000 no
localStorage do navegador, fechando posição em +US$ 0,10 sem pagar taxa. Aqui
o paper trading serve ao propósito real dele: medir, com preço de verdade e
custo de verdade, se a estratégia sobrevive antes de qualquer centavo entrar.

Diferenças que fazem o resultado ser comparável ao real:

* preço vem do book ao vivo, não do último preço da tela;
* entrada paga o ask, saída paga o bid, mais taxa e slippage;
* toda ordem passa pelo `RiskManager` do QuantBot antes de existir;
* toda decisão vai para o `DecisionLogger`, com o score que a motivou;
* o estado fica em disco (`data/cripto/paper_book.json`), não no navegador.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from core.decision_logger import DecisionLogger, DecisionOutcome, DecisionType
from risk.kill_switch import KillSwitchActivated, RiskLimits, RiskManager, RiskViolation

from .custos import ModeloCusto
from .exchanges import Roteador, cliente_padrao

logger = logging.getLogger("quantbot.cripto.papel")

ARQUIVO = Path("data/cripto/paper_book.json")


@dataclass
class Posicao:
    simbolo: str
    lado: int
    quantidade: float
    preco_entrada: float
    notional_usd: float
    taxa_entrada_usd: float
    stop: float
    alvo: float
    aberta_em: str
    score_entrada: float
    id_decisao: str | None = None

    def pnl_usd(self, preco_atual: float) -> float:
        return self.quantidade * (preco_atual - self.preco_entrada) * self.lado

    def pnl_pct(self, preco_atual: float) -> float:
        if self.notional_usd <= 0:
            return 0.0
        return self.pnl_usd(preco_atual) / self.notional_usd * 100


@dataclass
class Fechada:
    simbolo: str
    lado: int
    preco_entrada: float
    preco_saida: float
    notional_usd: float
    custo_total_usd: float
    pnl_liquido_usd: float
    retorno_pct: float
    motivo: str
    aberta_em: str
    fechada_em: str
    score_entrada: float


@dataclass
class Book:
    capital_inicial: float = 1_000.0
    caixa: float = 1_000.0
    posicoes: dict[str, Posicao] = field(default_factory=dict)
    fechadas: list[Fechada] = field(default_factory=list)
    serie_valor: list[dict] = field(default_factory=list)
    criado_em: str = ""

    def valor_total(self, precos: dict[str, float]) -> float:
        aberto = sum(
            p.notional_usd + p.pnl_usd(precos.get(s, p.preco_entrada))
            for s, p in self.posicoes.items()
        )
        return self.caixa + aberto

    def como_dict(self) -> dict:
        return {
            "capital_inicial": self.capital_inicial,
            "caixa": round(self.caixa, 4),
            "posicoes": {s: asdict(p) for s, p in self.posicoes.items()},
            "fechadas": [asdict(f) for f in self.fechadas],
            "serie_valor": self.serie_valor,
            "criado_em": self.criado_em,
        }


def carregar(caminho: Path = ARQUIVO, capital_inicial: float = 1_000.0) -> Book:
    if not caminho.exists():
        return Book(
            capital_inicial=capital_inicial,
            caixa=capital_inicial,
            criado_em=datetime.now(timezone.utc).isoformat(),
        )
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    return Book(
        capital_inicial=dados.get("capital_inicial", capital_inicial),
        caixa=dados.get("caixa", capital_inicial),
        posicoes={s: Posicao(**p) for s, p in dados.get("posicoes", {}).items()},
        fechadas=[Fechada(**f) for f in dados.get("fechadas", [])],
        serie_valor=dados.get("serie_valor", []),
        criado_em=dados.get("criado_em", datetime.now(timezone.utc).isoformat()),
    )


def salvar(book: Book, caminho: Path = ARQUIVO) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(book.como_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


class OperadorPapel:
    """Executa o ciclo de paper trading de um conjunto de símbolos."""

    def __init__(
        self,
        book: Book | None = None,
        cliente: Roteador | None = None,
        modelo_custo: ModeloCusto | None = None,
        limites: RiskLimits | None = None,
        fracao_por_operacao: float = 0.25,
        max_posicoes: int = 3,
    ):
        self.book = book or carregar()
        self.cliente = cliente or cliente_padrao()
        self.custo = modelo_custo or ModeloCusto()
        self.fracao = fracao_por_operacao
        self.max_posicoes = max_posicoes

        self.risco = RiskManager(
            limites or RiskLimits(max_trades_per_day=6, max_position_size_pct=0.30),
            state_file="data/cripto/risk_state.json",
        )
        self.decisoes = DecisionLogger(log_dir="data/cripto/decisoes")

    # ── Preço ────────────────────────────────────────────────────────────────

    def _cotacao(self, simbolo: str) -> tuple[float, float, float]:
        """Devolve (bid, ask, meio) do book ao vivo."""
        book = self.cliente.book(simbolo, profundidade=20)
        bid, ask = book.melhor_bid, book.melhor_ask
        if bid <= 0 or ask <= 0:
            raise ValueError(f"book inválido para {simbolo}")
        return bid, ask, (bid + ask) / 2

    # ── Abertura ─────────────────────────────────────────────────────────────

    def abrir(
        self, simbolo: str, lado: int, stop: float, alvo: float, score: float
    ) -> str | None:
        if simbolo in self.book.posicoes:
            logger.info("Já existe posição aberta em %s.", simbolo)
            return None
        if len(self.book.posicoes) >= self.max_posicoes:
            logger.info("Limite de %d posições simultâneas atingido.", self.max_posicoes)
            return None

        bid, ask, meio = self._cotacao(simbolo)
        precos = {s: meio for s in self.book.posicoes}
        precos[simbolo] = meio
        valor = self.book.valor_total(precos)
        notional = min(valor * self.fracao, self.book.caixa)

        if notional < 10:
            logger.info("Caixa de US$ %.2f não cobre a ordem mínima.", self.book.caixa)
            return None

        self.risco.initialize(current_capital=valor)
        try:
            self.risco.check_order(capital=valor, order_value=notional)
        except (RiskViolation, KillSwitchActivated) as exc:
            self.decisoes.log(
                tipo=DecisionType.RISK_VIOLATION,
                ticker=simbolo,
                contexto={"motivo": str(exc), "notional": notional, "score": score},
                notas="Ordem barrada pela camada de risco antes de existir.",
                tags=["cripto", "risco"],
            )
            logger.warning("Risco barrou a ordem em %s: %s", simbolo, exc)
            return None

        # Comprando você paga o ask; vendendo, recebe o bid. Mais slippage.
        base = ask if lado == 1 else bid
        preco = self.custo.preco_efetivo(base, notional, comprando=(lado == 1))
        taxa = notional * self.custo.taxa_taker
        quantidade = notional / preco

        id_decisao = self.decisoes.log(
            tipo=DecisionType.ORDER_FILLED,
            ticker=simbolo,
            valor_envolvido=notional,
            confianca=min(1.0, abs(score - 50) / 50),
            contexto={
                "lado": "COMPRA" if lado == 1 else "VENDA",
                "preco_executado": round(preco, 6),
                "melhor_bid": bid,
                "melhor_ask": ask,
                "taxa_usd": round(taxa, 4),
                "stop": round(stop, 6),
                "alvo": round(alvo, 6),
                "score_confluencia": round(score, 1),
                "fonte_preco": self.cliente.ultima_fonte,
            },
            tags=["cripto", "paper"],
        )

        self.book.posicoes[simbolo] = Posicao(
            simbolo=simbolo,
            lado=lado,
            quantidade=quantidade,
            preco_entrada=preco,
            notional_usd=notional,
            taxa_entrada_usd=taxa,
            stop=stop,
            alvo=alvo,
            aberta_em=datetime.now(timezone.utc).isoformat(),
            score_entrada=score,
            id_decisao=id_decisao,
        )
        self.book.caixa -= notional + taxa
        salvar(self.book)

        logger.info(
            "ABERTA %s %s: US$ %.2f a %.6f (taxa US$ %.4f)",
            "COMPRA" if lado == 1 else "VENDA",
            simbolo,
            notional,
            preco,
            taxa,
        )
        return id_decisao

    # ── Fechamento ───────────────────────────────────────────────────────────

    def fechar(self, simbolo: str, motivo: str) -> Fechada | None:
        pos = self.book.posicoes.get(simbolo)
        if pos is None:
            return None

        bid, ask, _ = self._cotacao(simbolo)
        base = bid if pos.lado == 1 else ask
        preco = self.custo.preco_efetivo(base, pos.notional_usd, comprando=(pos.lado == -1))
        taxa_saida = pos.quantidade * preco * self.custo.taxa_taker

        pnl = pos.quantidade * (preco - pos.preco_entrada) * pos.lado - taxa_saida
        custo_total = pos.taxa_entrada_usd + taxa_saida

        fechada = Fechada(
            simbolo=simbolo,
            lado=pos.lado,
            preco_entrada=pos.preco_entrada,
            preco_saida=preco,
            notional_usd=pos.notional_usd,
            custo_total_usd=round(custo_total, 4),
            pnl_liquido_usd=round(pnl, 4),
            retorno_pct=round(pnl / pos.notional_usd * 100, 4) if pos.notional_usd else 0.0,
            motivo=motivo,
            aberta_em=pos.aberta_em,
            fechada_em=datetime.now(timezone.utc).isoformat(),
            score_entrada=pos.score_entrada,
        )

        self.book.caixa += pos.notional_usd + pnl
        self.book.fechadas.append(fechada)
        del self.book.posicoes[simbolo]

        if pos.id_decisao:
            self.decisoes.update_outcome(
                pos.id_decisao,
                DecisionOutcome.SUCCESS if pnl > 0 else DecisionOutcome.FAILURE,
                {
                    "pnl_liquido_usd": fechada.pnl_liquido_usd,
                    "retorno_pct": fechada.retorno_pct,
                    "custo_total_usd": fechada.custo_total_usd,
                    "motivo": motivo,
                },
            )

        self.risco.record_trade_result(pnl=pnl, capital=self.book.caixa)
        salvar(self.book)

        logger.info(
            "FECHADA %s (%s): PnL líquido US$ %.4f | custo US$ %.4f",
            simbolo,
            motivo,
            pnl,
            custo_total,
        )
        return fechada

    # ── Manutenção ───────────────────────────────────────────────────────────

    def revisar_posicoes(self) -> list[Fechada]:
        """Checa stop e alvo de cada posição contra o preço vivo."""
        fechadas = []
        for simbolo in list(self.book.posicoes):
            pos = self.book.posicoes[simbolo]
            try:
                bid, ask, meio = self._cotacao(simbolo)
            except Exception as exc:
                logger.warning("Sem cotação de %s agora: %s", simbolo, exc)
                continue

            if pos.lado == 1:
                if meio <= pos.stop:
                    fechadas.append(self.fechar(simbolo, "stop"))
                elif meio >= pos.alvo:
                    fechadas.append(self.fechar(simbolo, "alvo"))
            else:
                if meio >= pos.stop:
                    fechadas.append(self.fechar(simbolo, "stop"))
                elif meio <= pos.alvo:
                    fechadas.append(self.fechar(simbolo, "alvo"))

        return [f for f in fechadas if f is not None]

    def marcar_valor(self) -> float:
        """Registra o valor da carteira no momento, para a curva de capital."""
        precos = {}
        for simbolo in self.book.posicoes:
            try:
                _, _, meio = self._cotacao(simbolo)
                precos[simbolo] = meio
            except Exception:
                precos[simbolo] = self.book.posicoes[simbolo].preco_entrada

        valor = self.book.valor_total(precos)
        self.book.serie_valor.append(
            {
                "quando": datetime.now(timezone.utc).isoformat(),
                "valor_usd": round(valor, 4),
                "caixa": round(self.book.caixa, 4),
                "posicoes_abertas": len(self.book.posicoes),
            }
        )
        salvar(self.book)
        return valor

    def resumo(self) -> dict:
        fechadas = self.book.fechadas
        ganhos = [f.pnl_liquido_usd for f in fechadas if f.pnl_liquido_usd > 0]
        perdas = [f.pnl_liquido_usd for f in fechadas if f.pnl_liquido_usd <= 0]
        valor = self.book.serie_valor[-1]["valor_usd"] if self.book.serie_valor else self.book.caixa

        return {
            "capital_inicial_usd": self.book.capital_inicial,
            "valor_atual_usd": round(valor, 2),
            "retorno_pct": round((valor / self.book.capital_inicial - 1) * 100, 2),
            "caixa_usd": round(self.book.caixa, 2),
            "posicoes_abertas": len(self.book.posicoes),
            "operacoes_fechadas": len(fechadas),
            "acertos": len(ganhos),
            "taxa_acerto_pct": round(len(ganhos) / len(fechadas) * 100, 1) if fechadas else 0.0,
            "custo_total_pago_usd": round(sum(f.custo_total_usd for f in fechadas), 2),
            "pnl_liquido_usd": round(sum(f.pnl_liquido_usd for f in fechadas), 2),
            "ganho_medio_usd": round(sum(ganhos) / len(ganhos), 4) if ganhos else 0.0,
            "perda_media_usd": round(sum(perdas) / len(perdas), 4) if perdas else 0.0,
            "risco": self.risco.get_status(),
        }
