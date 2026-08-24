"""
Motor de backtest event-driven.

Regras que este motor respeita e que a maioria dos backtests caseiros quebra:

* **Sem olhar o futuro.** O sinal da barra `t` é executado na ABERTURA da
  barra `t+1`. Nenhuma decisão usa o fechamento da barra em que foi tomada.
* **Custo em toda ponta.** Entrada e saída pagam taxa, meio spread e
  slippage, via `custos.ModeloCusto`.
* **Stop e alvo checados dentro da barra.** Usa máxima e mínima, não só o
  fechamento. Quando os dois são tocados na mesma barra, assume o pior caso
  (stop primeiro) — o otimista aqui é o mesmo erro que faz backtest mentir.
* **Sem posição fracionada infinita.** O tamanho respeita o capital e o
  limite por operação.

O resultado é um `Resultado` com a curva de capital, a lista de operações e
as métricas que importam depois do custo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .custos import ModeloCusto
from .sinais import ConfigSinal, gerar_sinais

logger = logging.getLogger("quantbot.cripto.backtest")

# Quantas barras existem em um ano, por intervalo — usado para anualizar.
BARRAS_ANO = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}


@dataclass
class Operacao:
    entrada_em: pd.Timestamp
    saida_em: pd.Timestamp | None
    lado: int                       # 1 comprado, -1 vendido
    preco_entrada: float            # já com spread/slippage
    preco_saida: float | None
    notional_usd: float
    custo_usd: float
    pnl_bruto_usd: float
    pnl_liquido_usd: float
    retorno_pct: float
    motivo_saida: str
    barras: int
    score_entrada: float

    def como_dict(self) -> dict:
        return {
            "entrada_em": str(self.entrada_em),
            "saida_em": str(self.saida_em) if self.saida_em is not None else None,
            "lado": "COMPRA" if self.lado == 1 else "VENDA",
            "preco_entrada": round(self.preco_entrada, 6),
            "preco_saida": round(self.preco_saida, 6) if self.preco_saida else None,
            "notional_usd": round(self.notional_usd, 2),
            "custo_usd": round(self.custo_usd, 4),
            "pnl_bruto_usd": round(self.pnl_bruto_usd, 4),
            "pnl_liquido_usd": round(self.pnl_liquido_usd, 4),
            "retorno_pct": round(self.retorno_pct * 100, 4),
            "motivo_saida": self.motivo_saida,
            "barras": self.barras,
            "score_entrada": round(self.score_entrada, 1),
        }


@dataclass
class ConfigBacktest:
    capital_inicial: float = 1_000.0
    fracao_por_operacao: float = 0.25   # 25% do capital por posição
    intervalo: str = "1h"
    modelo_custo: ModeloCusto = field(default_factory=ModeloCusto)
    config_sinal: ConfigSinal = field(default_factory=ConfigSinal)
    permitir_alavancagem: bool = False


@dataclass
class Resultado:
    capital: pd.Series
    operacoes: list[Operacao]
    config: ConfigBacktest
    simbolo: str

    # ── Métricas ─────────────────────────────────────────────────────────────

    @property
    def retorno_total_pct(self) -> float:
        if self.capital.empty:
            return 0.0
        return float(self.capital.iloc[-1] / self.capital.iloc[0] - 1) * 100

    @property
    def retornos(self) -> pd.Series:
        return self.capital.pct_change().dropna()

    @property
    def sharpe(self) -> float:
        r = self.retornos
        if len(r) < 2 or r.std(ddof=1) == 0:
            return 0.0
        barras_ano = BARRAS_ANO.get(self.config.intervalo, 8_760)
        return float(r.mean() / r.std(ddof=1) * np.sqrt(barras_ano))

    @property
    def max_drawdown_pct(self) -> float:
        if self.capital.empty:
            return 0.0
        pico = self.capital.cummax()
        return float((self.capital / pico - 1).min()) * 100

    @property
    def taxa_acerto(self) -> float:
        fechadas = [o for o in self.operacoes if o.saida_em is not None]
        if not fechadas:
            return 0.0
        return sum(1 for o in fechadas if o.pnl_liquido_usd > 0) / len(fechadas)

    @property
    def custo_total_usd(self) -> float:
        return sum(o.custo_usd for o in self.operacoes)

    @property
    def pnl_bruto_usd(self) -> float:
        return sum(o.pnl_bruto_usd for o in self.operacoes)

    @property
    def pnl_liquido_usd(self) -> float:
        return sum(o.pnl_liquido_usd for o in self.operacoes)

    @property
    def payoff(self) -> float:
        ganhos = [o.pnl_liquido_usd for o in self.operacoes if o.pnl_liquido_usd > 0]
        perdas = [-o.pnl_liquido_usd for o in self.operacoes if o.pnl_liquido_usd < 0]
        if not ganhos or not perdas:
            return 0.0
        return float(np.mean(ganhos) / np.mean(perdas))

    @property
    def expectancy_pct(self) -> float:
        fechadas = [o for o in self.operacoes if o.saida_em is not None]
        if not fechadas:
            return 0.0
        return float(np.mean([o.retorno_pct for o in fechadas])) * 100

    def buy_and_hold_pct(self, precos: pd.Series) -> float:
        """Comparação obrigatória: a estratégia bateu simplesmente comprar e segurar?"""
        if precos.empty:
            return 0.0
        return float(precos.iloc[-1] / precos.iloc[0] - 1) * 100

    def resumo(self, precos: pd.Series | None = None) -> dict:
        fechadas = [o for o in self.operacoes if o.saida_em is not None]
        d = {
            "simbolo": self.simbolo,
            "intervalo": self.config.intervalo,
            "capital_inicial": round(self.config.capital_inicial, 2),
            "capital_final": round(float(self.capital.iloc[-1]), 2) if not self.capital.empty else 0.0,
            "retorno_total_pct": round(self.retorno_total_pct, 2),
            "sharpe_anualizado": round(self.sharpe, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "operacoes": len(fechadas),
            "taxa_acerto_pct": round(self.taxa_acerto * 100, 1),
            "payoff": round(self.payoff, 2),
            "expectancy_por_operacao_pct": round(self.expectancy_pct, 4),
            "pnl_bruto_usd": round(self.pnl_bruto_usd, 2),
            "custo_total_usd": round(self.custo_total_usd, 2),
            "pnl_liquido_usd": round(self.pnl_liquido_usd, 2),
        }
        if precos is not None:
            bh = self.buy_and_hold_pct(precos)
            d["buy_and_hold_pct"] = round(bh, 2)
            d["alpha_vs_buy_and_hold_pp"] = round(self.retorno_total_pct - bh, 2)
        return d


def executar(
    df: pd.DataFrame,
    simbolo: str = "BTCUSDT",
    config: ConfigBacktest | None = None,
) -> Resultado:
    """Roda o backtest sobre um DataFrame OHLCV.

    O `df` pode vir cru: os indicadores e o score são calculados aqui dentro.
    """
    config = config or ConfigBacktest()
    dados = gerar_sinais(df, config.config_sinal)

    n = len(dados)
    if n < 250:
        raise ValueError(
            f"série curta demais ({n} barras). Indicadores como a EMA 200 "
            "precisam de aquecimento — use pelo menos 250 barras."
        )

    aberturas = dados["abertura"].to_numpy(dtype=float)
    maximas = dados["maxima"].to_numpy(dtype=float)
    minimas = dados["minima"].to_numpy(dtype=float)
    fechamentos = dados["fechamento"].to_numpy(dtype=float)
    alvo = dados["posicao_alvo"].to_numpy(dtype=float)
    scores = dados["score"].to_numpy(dtype=float)
    atrs = dados["atr_14"].to_numpy(dtype=float)
    indice = dados.index

    capital = config.capital_inicial
    curva = np.full(n, np.nan)
    operacoes: list[Operacao] = []

    # Estado da posição aberta. Guardamos o preço "bruto" (meio do book) e o
    # efetivo (com spread e slippage) para separar resultado de custo.
    lado = 0
    preco_entrada = 0.0
    preco_entrada_bruto = 0.0
    notional = 0.0
    quantidade = 0.0
    taxa_entrada = 0.0
    stop = 0.0
    take = 0.0
    barra_entrada = 0
    score_entrada = 0.0

    cs = config.config_sinal
    mc = config.modelo_custo

    for i in range(n):
        preco_marcacao = fechamentos[i]

        if lado != 0:
            # ── Gestão da posição aberta ─────────────────────────────────────
            motivo = ""
            preco_saida_bruto = 0.0

            if lado == 1:
                # Pior caso: se stop e alvo caem na mesma barra, o stop vale.
                if minimas[i] <= stop:
                    preco_saida_bruto, motivo = stop, "stop"
                elif maximas[i] >= take:
                    preco_saida_bruto, motivo = take, "alvo"
            else:
                if maximas[i] >= stop:
                    preco_saida_bruto, motivo = stop, "stop"
                elif minimas[i] <= take:
                    preco_saida_bruto, motivo = take, "alvo"

            if not motivo and (i - barra_entrada) >= cs.max_barras_posicao:
                preco_saida_bruto, motivo = fechamentos[i], "tempo"

            if not motivo and alvo[i] != lado and not np.isnan(alvo[i]):
                # O sinal virou: sai no fechamento desta barra.
                preco_saida_bruto, motivo = fechamentos[i], "sinal"

            if motivo:
                preco_saida = mc.preco_efetivo(
                    preco_saida_bruto, notional, comprando=(lado == -1)
                )
                taxa_saida = quantidade * preco_saida * mc.taxa_taker

                # Bruto = o que a operação renderia no meio do book, sem atrito.
                pnl_bruto = quantidade * (preco_saida_bruto - preco_entrada_bruto) * lado
                # Líquido = preço realmente executado, menos as duas taxas.
                pnl_liquido = (
                    quantidade * (preco_saida - preco_entrada) * lado - taxa_entrada - taxa_saida
                )
                custo_total = pnl_bruto - pnl_liquido

                capital += pnl_liquido + taxa_entrada  # a taxa de entrada já saiu no início
                operacoes.append(
                    Operacao(
                        entrada_em=indice[barra_entrada],
                        saida_em=indice[i],
                        lado=lado,
                        preco_entrada=preco_entrada,
                        preco_saida=preco_saida,
                        notional_usd=notional,
                        custo_usd=custo_total,
                        pnl_bruto_usd=pnl_bruto,
                        pnl_liquido_usd=pnl_liquido,
                        retorno_pct=pnl_liquido / notional if notional else 0.0,
                        motivo_saida=motivo,
                        barras=i - barra_entrada,
                        score_entrada=score_entrada,
                    )
                )
                lado = 0
                quantidade = 0.0
                notional = 0.0

        # ── Marcação a mercado ───────────────────────────────────────────────
        if lado != 0:
            curva[i] = capital + quantidade * (preco_marcacao - preco_entrada) * lado
        else:
            curva[i] = capital

        # ── Abertura de posição: decide agora, executa na abertura seguinte ──
        if lado == 0 and i + 1 < n:
            desejo = alvo[i]
            if desejo != 0 and not np.isnan(desejo) and not np.isnan(atrs[i]):
                notional_desejado = capital * config.fracao_por_operacao
                if not config.permitir_alavancagem:
                    notional_desejado = min(notional_desejado, capital)

                if notional_desejado >= 10.0:  # ordem mínima típica de exchange
                    lado = int(desejo)
                    preco_entrada_bruto = aberturas[i + 1]
                    preco_entrada = mc.preco_efetivo(
                        preco_entrada_bruto, notional_desejado, comprando=(lado == 1)
                    )
                    quantidade = notional_desejado / preco_entrada
                    notional = notional_desejado
                    taxa_entrada = notional * mc.taxa_taker
                    capital -= taxa_entrada

                    dist = cs.atr_stop * atrs[i]
                    ganho = cs.atr_alvo * atrs[i]
                    stop = preco_entrada - dist * lado
                    take = preco_entrada + ganho * lado

                    barra_entrada = i + 1
                    score_entrada = scores[i] if not np.isnan(scores[i]) else 50.0

    # Fecha o que ficou aberto na última barra, para não inflar o resultado.
    if lado != 0:
        preco_saida = mc.preco_efetivo(fechamentos[-1], notional, comprando=(lado == -1))
        taxa_saida = quantidade * preco_saida * mc.taxa_taker
        pnl_bruto = quantidade * (fechamentos[-1] - preco_entrada_bruto) * lado
        pnl_liquido = (
            quantidade * (preco_saida - preco_entrada) * lado - taxa_entrada - taxa_saida
        )
        capital += pnl_liquido + taxa_entrada
        operacoes.append(
            Operacao(
                entrada_em=indice[barra_entrada],
                saida_em=indice[-1],
                lado=lado,
                preco_entrada=preco_entrada,
                preco_saida=preco_saida,
                notional_usd=notional,
                custo_usd=pnl_bruto - pnl_liquido,
                pnl_bruto_usd=pnl_bruto,
                pnl_liquido_usd=pnl_liquido,
                retorno_pct=pnl_liquido / notional if notional else 0.0,
                motivo_saida="fim_da_serie",
                barras=len(dados) - 1 - barra_entrada,
                score_entrada=score_entrada,
            )
        )
        curva[-1] = capital

    serie_capital = pd.Series(curva, index=indice, name="capital").ffill()
    return Resultado(
        capital=serie_capital, operacoes=operacoes, config=config, simbolo=simbolo
    )
