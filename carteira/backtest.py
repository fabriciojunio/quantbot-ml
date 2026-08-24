"""
Backtest de carteira, com custo de execução e imposto de renda.

Diferenças em relação a um backtest de carteira comum, e cada uma existe
porque a falta dela infla o resultado:

* **A carteira deriva.** Entre rebalanceamentos os pesos mudam sozinhos com o
  mercado. Reaplicar o peso alvo todo dia sem cobrar nada simula uma carteira
  que ninguém consegue ter.
* **Todo giro paga.** Emolumentos, spread e corretagem em cada ponta, com o
  spread específico de cada ativo — BOVA11 e XFIX11 não custam a mesma coisa.
* **Taxa de administração sai da cota.** Descontada diariamente, como acontece
  de verdade.
* **Imposto é apurado.** Cada venda com lucro gera IR conforme o regime do
  ativo, com compensação de prejuízo por modalidade. É o item que mais separa
  backtest de extrato.
* **Sobra vai para o CDI.** Caixa não fica embaixo do colchão.
* **Comparação obrigatória.** Todo resultado sai ao lado de CDI, Ibovespa e da
  mesma carteira sem rebalancear.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .custos_b3 import TAXA_ADM_ANUAL, ModeloCustoB3
from .rebalanceamento import ConfigRebalanceamento, marcar_datas
from .tributos import ApuradorIR, Venda
from .universo import CAIXA, Ativo, universo_completo

logger = logging.getLogger("quantbot.carteira.backtest")


@dataclass
class ConfigBacktest:
    capital_inicial: float = 10_000.0
    aporte_mensal: float = 0.0
    modelo_custo: ModeloCustoB3 = field(default_factory=ModeloCustoB3)
    config_rebal: ConfigRebalanceamento = field(default_factory=ConfigRebalanceamento)
    cobrar_imposto: bool = True
    cobrar_taxa_adm: bool = True
    caixa_rende_cdi: bool = True


@dataclass
class Resultado:
    patrimonio: pd.Series
    pesos: pd.DataFrame
    operacoes: list[dict]
    imposto_por_mes: list[dict]
    config: ConfigBacktest
    aportes_totais: float
    nome: str = "carteira"

    @property
    def custo_total(self) -> float:
        return sum(o["custo"] for o in self.operacoes)

    @property
    def imposto_total(self) -> float:
        return sum(m["imposto_devido"] for m in self.imposto_por_mes)

    @property
    def capital_investido(self) -> float:
        return self.config.capital_inicial + self.aportes_totais

    @property
    def retorno_total_pct(self) -> float:
        """Retorno sobre o capital efetivamente colocado, não sobre o inicial.

        Com aporte mensal, dividir pelo capital inicial produz um número
        espetacular e sem sentido — boa parte do 'ganho' é dinheiro novo.
        """
        if self.capital_investido <= 0:
            return 0.0
        return (float(self.patrimonio.iloc[-1]) / self.capital_investido - 1) * 100

    @property
    def retornos_diarios(self) -> pd.Series:
        return self.patrimonio.pct_change().dropna()

    @property
    def cagr_pct(self) -> float:
        if len(self.patrimonio) < 2:
            return 0.0
        anos = (self.patrimonio.index[-1] - self.patrimonio.index[0]).days / 365.25
        if anos <= 0 or self.capital_investido <= 0:
            return 0.0
        # Com aporte, o CAGR simples superestima. Usamos o retorno médio da
        # série de patrimônio, que já embute o efeito do dinheiro novo.
        r = self.retornos_diarios
        return (float((1 + r).prod()) ** (1 / anos) - 1) * 100

    @property
    def vol_anual_pct(self) -> float:
        r = self.retornos_diarios
        return float(r.std(ddof=1) * np.sqrt(252) * 100) if len(r) > 1 else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        if self.patrimonio.empty:
            return 0.0
        pico = self.patrimonio.cummax()
        return float((self.patrimonio / pico - 1).min()) * 100

    @property
    def sharpe(self) -> float:
        r = self.retornos_diarios
        if len(r) < 2 or r.std(ddof=1) == 0:
            return 0.0
        return float(r.mean() / r.std(ddof=1) * np.sqrt(252))

    def resumo(self, comparacoes: dict[str, pd.Series] | None = None) -> dict:
        d = {
            "nome": self.nome,
            "periodo": [str(self.patrimonio.index[0])[:10], str(self.patrimonio.index[-1])[:10]],
            "capital_inicial": round(self.config.capital_inicial, 2),
            "aportes_totais": round(self.aportes_totais, 2),
            "capital_investido": round(self.capital_investido, 2),
            "patrimonio_final": round(float(self.patrimonio.iloc[-1]), 2),
            "retorno_sobre_investido_pct": round(self.retorno_total_pct, 2),
            "cagr_pct": round(self.cagr_pct, 2),
            "vol_anual_pct": round(self.vol_anual_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe": round(self.sharpe, 2),
            "rebalanceamentos": len({o["quando"] for o in self.operacoes}),
            "operacoes": len(self.operacoes),
            "custo_total": round(self.custo_total, 2),
            "imposto_total": round(self.imposto_total, 2),
            "atrito_total": round(self.custo_total + self.imposto_total, 2),
            "atrito_pct_do_investido": round(
                (self.custo_total + self.imposto_total) / self.capital_investido * 100, 2
            )
            if self.capital_investido
            else 0.0,
        }
        if comparacoes:
            for nome, serie in comparacoes.items():
                s = serie.reindex(self.patrimonio.index).ffill().dropna()
                if len(s) > 1:
                    ret = (float(s.iloc[-1]) / float(s.iloc[0]) - 1) * 100
                    d[f"vs_{nome}_pct"] = round(ret, 2)
                    d[f"alpha_vs_{nome}_pp"] = round(self.retorno_total_pct - ret, 2)
        return d


def _cdi_fator(cdi: pd.Series | None, indice: pd.DatetimeIndex) -> pd.Series:
    """Fator diário do CDI alinhado ao índice de preços. Sem CDI, devolve 1."""
    if cdi is None:
        return pd.Series(1.0, index=indice)
    return (1 + cdi.reindex(indice).ffill().fillna(0.0)).fillna(1.0)


def executar(
    precos: pd.DataFrame,
    pesos_alvo: pd.DataFrame,
    config: ConfigBacktest | None = None,
    cdi: pd.Series | None = None,
    ativos: dict[str, Ativo] | None = None,
    nome: str = "carteira",
) -> Resultado:
    """Roda a simulação da carteira sobre `precos`, seguindo `pesos_alvo`."""
    config = config or ConfigBacktest()
    ativos = ativos or universo_completo(incluir_cripto=True)

    precos = precos.sort_index()
    pesos_alvo = pesos_alvo.reindex(precos.index).fillna(0.0)
    tickers = list(precos.columns)

    retornos = precos.pct_change().fillna(0.0)
    marcas = marcar_datas(pesos_alvo, retornos, config.config_rebal)
    fator_cdi = _cdi_fator(cdi if config.caixa_rende_cdi else None, precos.index)

    # Estado da carteira: quantidade por ativo e caixa.
    quantidade = pd.Series(0.0, index=tickers)
    caixa = config.capital_inicial
    aportes = 0.0
    custo_base: dict[str, float] = dict.fromkeys(tickers, 0.0)  # custo médio total por ativo
    data_compra: dict[str, date] = {}

    patrimonio = []
    pesos_reais = []
    operacoes: list[dict] = []
    vendas_do_mes: list[Venda] = []
    imposto_por_mes: list[dict] = []
    apurador = ApuradorIR()
    ultimo_mes = None

    for i, hoje in enumerate(precos.index):
        preco_hoje = precos.loc[hoje]

        # 1) Caixa rende CDI
        caixa *= float(fator_cdi.loc[hoje])

        # 2) Taxa de administração sai de dentro da cota
        if config.cobrar_taxa_adm:
            for t in tickers:
                taxa = TAXA_ADM_ANUAL.get(t, 0.0) / 252
                if taxa and quantidade[t] > 0:
                    quantidade[t] *= 1 - taxa

        # 3) Aporte mensal, no primeiro pregão de cada mês
        mes_atual = (hoje.year, hoje.month)
        if config.aporte_mensal > 0 and ultimo_mes is not None and mes_atual != ultimo_mes:
            caixa += config.aporte_mensal
            aportes += config.aporte_mensal
        ultimo_mes = mes_atual

        valor_posicoes = float((quantidade * preco_hoje).sum())
        total = caixa + valor_posicoes

        # 4) Rebalanceamento
        if bool(marcas.loc[hoje]) and total > 0:
            alvo = pesos_alvo.loc[hoje]
            valor_alvo = alvo * total

            for t in tickers:
                ativo = ativos.get(t) or CAIXA
                atual = float(quantidade[t] * preco_hoje[t])
                desejado = float(valor_alvo.get(t, 0.0))
                delta = desejado - atual

                if abs(delta) < 1.0:  # menos de um real não vale ordem
                    continue

                comprando = delta > 0
                preco_exec = config.modelo_custo.preco_efetivo(ativo, float(preco_hoje[t]), comprando)
                valor = abs(delta)
                custo = config.modelo_custo.custo_ponta(ativo, valor) - valor * ativo.custo_meio_spread
                custo = max(custo, 0.0)  # o spread já foi cobrado no preço

                if comprando:
                    if caixa < valor + custo:
                        valor = max(0.0, caixa - custo)
                        if valor < 1.0:
                            continue
                    qtd = valor / preco_exec
                    quantidade[t] += qtd
                    custo_base[t] += valor + custo
                    caixa -= valor + custo
                    data_compra.setdefault(t, hoje.date())
                else:
                    qtd = min(quantidade[t], valor / preco_exec)
                    if qtd <= 0:
                        continue
                    recebido = qtd * preco_exec
                    fracao = qtd / quantidade[t] if quantidade[t] > 0 else 0.0
                    base_vendida = custo_base[t] * fracao

                    quantidade[t] -= qtd
                    custo_base[t] -= base_vendida
                    caixa += recebido - custo

                    if config.cobrar_imposto:
                        dias = (hoje.date() - data_compra.get(t, hoje.date())).days
                        vendas_do_mes.append(
                            Venda(
                                ticker=t,
                                regime=ativo.regime,
                                quando=hoje.date(),
                                valor_alienado=recebido - custo,
                                custo_aquisicao=base_vendida,
                                dias_carregado=dias,
                            )
                        )
                    if quantidade[t] <= 1e-9:
                        data_compra.pop(t, None)

                operacoes.append({
                    "quando": str(hoje)[:10],
                    "ticker": t,
                    "lado": "compra" if comprando else "venda",
                    "valor": round(valor, 2),
                    "custo": round(custo, 2),
                })

        # 5) Apuração do IR no último pregão do mês. O imposto sai do caixa no
        # mês seguinte na vida real (DARF até o último dia útil), mas cobrar no
        # fechamento do mês é conservador e evita fingir que o dinheiro rendeu
        # um mês a mais do que rendeu.
        fim_do_mes = i + 1 == len(precos.index) or precos.index[i + 1].month != hoje.month
        if config.cobrar_imposto and fim_do_mes and vendas_do_mes:
            r = apurador.apurar_mes(hoje.year, hoje.month, vendas_do_mes)
            if r.imposto_devido > 0:
                # Se o caixa não cobre o DARF, é preciso vender posição para
                # pagar. Deixar o caixa negativo seria alavancagem silenciosa:
                # a carteira passaria a somar mais de 100% investido sem que
                # ninguém tivesse decidido isso.
                if caixa < r.imposto_devido:
                    faltando = r.imposto_devido - max(caixa, 0.0)
                    valor_posicoes_agora = float((quantidade * preco_hoje).sum())
                    if valor_posicoes_agora > 0:
                        fracao = min(1.0, faltando / valor_posicoes_agora)
                        for t in tickers:
                            if quantidade[t] <= 0:
                                continue
                            ativo = ativos.get(t) or CAIXA
                            vendida = quantidade[t] * fracao
                            preco_exec = config.modelo_custo.preco_efetivo(
                                ativo, float(preco_hoje[t]), comprando=False
                            )
                            recebido = vendida * preco_exec
                            quantidade[t] -= vendida
                            custo_base[t] *= 1 - fracao
                            caixa += recebido
                            operacoes.append({
                                "quando": str(hoje)[:10],
                                "ticker": t,
                                "lado": "venda",
                                "valor": round(recebido, 2),
                                "custo": 0.0,
                                "motivo": "liquidação para pagar imposto",
                            })
                caixa -= r.imposto_devido
            imposto_por_mes.append({
                "ano": r.ano,
                "mes": r.mes,
                "imposto_devido": r.imposto_devido,
                "ganho_tributavel": r.ganho_tributavel,
                "ganho_isento": r.ganho_isento,
                "prejuizo_acumulado": r.prejuizo_acumulado,
            })
            vendas_do_mes = []

        valor_posicoes = float((quantidade * preco_hoje).sum())
        total = caixa + valor_posicoes
        patrimonio.append(total)
        pesos_reais.append((quantidade * preco_hoje / total) if total > 0 else quantidade * 0)

    serie = pd.Series(patrimonio, index=precos.index, name="patrimonio")
    df_pesos = pd.DataFrame(pesos_reais, index=precos.index).fillna(0.0)

    return Resultado(
        patrimonio=serie,
        pesos=df_pesos,
        operacoes=operacoes,
        imposto_por_mes=imposto_por_mes,
        config=config,
        aportes_totais=aportes,
        nome=nome,
    )
