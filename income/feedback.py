"""
Aprendizado do QuantBot — fecha o ciclo decisão → resultado → ajuste.

Aprende de TRÊS fontes ao mesmo tempo (o foco é aprender o máximo):

1. DECISÕES (acertos/erros): para cada posição paper avaliada, guarda em
   qual "faixa" ela estava (score, DY, sentimento na entrada) e se bateu o
   CDI. Com o tempo, descobre quais faixas costumam dar certo e ajusta o
   score do screener para cima/baixo — bem de leve e com limite.

2. NOTÍCIAS / SENTIMENTO: cruza o índice de sentimento das notícias (de
   cada dia) com o retorno do portfólio nos dias seguintes. Assim aprende,
   na prática, SE o sentimento tem poder preditivo (ex.: "comprar quando o
   mercado está com medo costuma pagar?").

3. DADOS (macro/preços): entram via o retorno realizado vs CDI, que já
   reflete juros e mercado.

Honestidade: em 6 meses o sinal é fraco e ruidoso. Por isso todo ajuste é
LIMITADO (±3 pontos no score, ±viés pequeno) e só vale com amostra mínima.
O valor real aparece no longo prazo — mas o mecanismo é correto desde já.

Persistência: data/feedback.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("quantbot.income.feedback")

_MIN_AMOSTRA = 5          # nº mínimo de observações para confiar numa faixa
_AJUSTE_MAX = 3.0         # nudge máximo no score (pontos)
_VIES_MAX = 2.0           # viés máximo de mercado (pontos)


def _faixa_score(s: float) -> str:
    if s >= 80:
        return "80+"
    if s >= 75:
        return "75-80"
    return "<75"


def _faixa_dy(dy: float) -> str:
    if dy >= 12:
        return "12+"
    if dy >= 10:
        return "10-12"
    return "<10"


def _faixa_sent(idx: float) -> str:
    if idx <= -0.3:
        return "medo"
    if idx >= 0.3:
        return "otimismo"
    return "neutro"


class Feedback:
    """Memória de aprendizado do bot."""

    def __init__(self, arquivo: str = "data/feedback.json"):
        self.arquivo = Path(arquivo)
        self.estado = self._carregar()

    def _carregar(self) -> dict:
        if self.arquivo.exists():
            try:
                return json.loads(self.arquivo.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {
            "faixas": {},          # "score|dy|sent" -> {n, soma_excedente, acertos}
            "sentimento": {},      # "medo|neutro|otimismo" -> {n, soma_retorno_fwd}
            "atualizado_em": None,
        }

    def salvar(self) -> None:
        from datetime import date
        self.estado["atualizado_em"] = date.today().isoformat()
        self.arquivo.parent.mkdir(parents=True, exist_ok=True)
        self.arquivo.write_text(
            json.dumps(self.estado, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── 1. aprende com as decisões (acertos/erros) ──────────────────────

    def atualizar_decisoes(self, resultados: List[dict]) -> None:
        faixas = self.estado["faixas"]
        for r in resultados:
            chave = (f"{_faixa_score(r['score_entrada'])}|"
                     f"{_faixa_dy(r['dy_entrada'])}|{r['sentimento_entrada']}")
            b = faixas.setdefault(chave, {"n": 0, "soma_excedente": 0.0, "acertos": 0})
            b["n"] += 1
            b["soma_excedente"] += r["excedente_cdi_pct"]
            b["acertos"] += 1 if r["acerto"] else 0

    def ajuste_score(self, score: float, dy: float, sentimento: str) -> float:
        """Nudge aprendido para o score de um candidato (limitado)."""
        chave = f"{_faixa_score(score)}|{_faixa_dy(dy)}|{sentimento}"
        b = self.estado["faixas"].get(chave)
        if not b or b["n"] < _MIN_AMOSTRA:
            return 0.0
        media_excedente = b["soma_excedente"] / b["n"]
        # cada 1% de excedente médio sobre o CDI vira ~0.2 ponto, com teto
        nudge = max(-_AJUSTE_MAX, min(_AJUSTE_MAX, media_excedente * 0.2))
        return round(nudge, 2)

    # ── 2. aprende com as notícias (sentimento → retorno futuro) ────────

    def aprender_sentimento(
        self,
        serie_sentimento: List[dict],   # [{data, indice_sentimento}]
        serie_valor: List[dict],        # [{data, valor_total}]
        horizonte_dias: int = 7,
    ) -> None:
        """Cruza o sentimento de cada dia com o retorno do portfólio à frente."""
        valor_por_data = {v["data"]: v["valor_total"] for v in serie_valor if v.get("valor_total")}
        datas_valor = sorted(valor_por_data)
        if len(datas_valor) < 2:
            return

        from datetime import date as _d
        sent = {"medo": {"n": 0, "soma_retorno_fwd": 0.0},
                "neutro": {"n": 0, "soma_retorno_fwd": 0.0},
                "otimismo": {"n": 0, "soma_retorno_fwd": 0.0}}

        for s in serie_sentimento:
            d0 = s.get("data")
            idx = s.get("indice_sentimento")
            if d0 not in valor_por_data or idx is None:
                continue
            # acha o primeiro valor com data >= d0 + horizonte
            alvo = None
            for dv in datas_valor:
                if (_d.fromisoformat(dv) - _d.fromisoformat(d0)).days >= horizonte_dias:
                    alvo = dv
                    break
            if alvo is None:
                continue
            v0, v1 = valor_por_data[d0], valor_por_data[alvo]
            if v0 <= 0:
                continue
            ret_fwd = (v1 - v0) / v0 * 100
            faixa = _faixa_sent(idx)
            sent[faixa]["n"] += 1
            sent[faixa]["soma_retorno_fwd"] += ret_fwd

        self.estado["sentimento"] = sent

    def vies_de_mercado(self, indice_hoje: float) -> float:
        """
        A partir do que aprendeu, devolve um viés (pontos) para o sentimento
        ATUAL. Ex.: se historicamente 'medo' precedeu retornos positivos, o
        viés para um dia de medo é levemente positivo. Limitado.
        """
        faixa = _faixa_sent(indice_hoje)
        b = self.estado["sentimento"].get(faixa)
        if not b or b["n"] < _MIN_AMOSTRA:
            return 0.0
        media_fwd = b["soma_retorno_fwd"] / b["n"]
        return round(max(-_VIES_MAX, min(_VIES_MAX, media_fwd * 0.5)), 2)

    # ── resumo do que aprendeu (para o relatório) ───────────────────────

    def resumo_texto(self) -> str:
        faixas = self.estado["faixas"]
        sent = self.estado["sentimento"]
        linhas = ["### O que o bot aprendeu até aqui"]

        confiaveis = {k: v for k, v in faixas.items() if v["n"] >= _MIN_AMOSTRA}
        if confiaveis:
            melhor = max(confiaveis.items(), key=lambda kv: kv[1]["soma_excedente"] / kv[1]["n"])
            k, v = melhor
            taxa = v["acertos"] / v["n"] * 100
            linhas.append(
                f"- Melhor perfil de pick: **{k}** "
                f"({v['n']} obs, {taxa:.0f}% bateram o CDI, "
                f"+{v['soma_excedente']/v['n']:.1f}% médio sobre o CDI a.a.)."
            )
        else:
            linhas.append("- Picks: ainda coletando (precisa de mais avaliações).")

        aprendeu_sent = any(b.get("n", 0) >= _MIN_AMOSTRA for b in sent.values())
        if aprendeu_sent:
            partes = []
            for faixa, b in sent.items():
                if b["n"] >= _MIN_AMOSTRA:
                    partes.append(f"{faixa}: {b['soma_retorno_fwd']/b['n']:+.2f}% (n={b['n']})")
            linhas.append("- Sentimento → retorno ~7 dias depois: " + " · ".join(partes))
        else:
            linhas.append("- Sentimento: ainda sem amostra suficiente para concluir.")
        return "\n".join(linhas)
