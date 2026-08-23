"""
Motor de imposto de renda para pessoa física.

Este é o módulo mais importante do pacote, e a razão é simples: **imposto é a
única vantagem que não depende de acertar o mercado**. Prever preço é uma
aposta; a alíquota é lei. Duas carteiras com o mesmo retorno bruto e
tributação diferente entregam resultados diferentes, todo ano, sem exceção e
sem risco.

Regras implementadas, vigentes em agosto de 2026:

**FII** — o provento distribuído é isento de IR para pessoa física, desde que
o fundo tenha ao menos 100 cotistas, seja negociado em bolsa e o investidor
detenha menos de 10% das cotas. A MP 1.303/2025 propôs tributar em 5% ou 17,5%
e caducou sem virar lei. O ganho de capital na venda da cota é tributado em
20%, sem isenção por valor.

**Ações** — 15% sobre o ganho em operação comum, com **isenção quando o total
vendido no mês fica até R$ 20 mil**. Day trade é 20% e não tem isenção.
Prejuízo compensa ganho futuro da mesma modalidade, sem prazo.

**ETF de renda variável** — 15% sobre o ganho. **Não tem a isenção de
R$ 20 mil**: essa é a pegadinha que faz muita gente achar que BOVA11 e uma
carteira de ações têm a mesma tributação. Não têm.

**ETF de renda fixa** — tabela regressiva por prazo médio da carteira do fundo
(25%, 20% ou 15%), retida na fonte na venda.

**Tesouro Direto e CDB** — tabela regressiva por prazo do investimento:
22,5% até 180 dias, 20% até 360, 17,5% até 720, 15% acima.

**Cripto** — 15% a 22,5% conforme o ganho, com **isenção quando o total
alienado no mês fica até R$ 35 mil**.

**LCI, LCA, poupança e debênture incentivada** — isentos.

Duas consequências de projeto que saem daqui e valem mais que qualquer sinal:

1. Girar carteira custa imposto além de custo de corretagem. Uma posição que
   fica dois anos parada paga 15%; a mesma posição girada a cada trimestre paga
   22,5% e paga oito vezes o custo de corretagem.
2. Provento de FII isento vale mais que juro tributado de mesmo percentual. Um
   FII pagando 9,5% líquido equivale a um CDB pagando cerca de 11,6% bruto para
   quem fica menos de 180 dias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .universo import Regime

# ── Alíquotas ────────────────────────────────────────────────────────────────

ALIQUOTA_ACAO_COMUM = 0.15
ALIQUOTA_ACAO_DAYTRADE = 0.20
ALIQUOTA_ETF_RV = 0.15
ALIQUOTA_FII_GANHO = 0.20
ALIQUOTA_CRIPTO_BASE = 0.15

# Isenções mensais por modalidade, em reais.
ISENCAO_MENSAL_ACOES = 20_000.0
ISENCAO_MENSAL_CRIPTO = 35_000.0

# Tabela regressiva de renda fixa: (dias corridos até, alíquota).
TABELA_REGRESSIVA = ((180, 0.225), (360, 0.20), (720, 0.175), (10**9, 0.15))

# Faixas de cripto acima da isenção, por ganho no mês.
FAIXAS_CRIPTO = ((5_000_000.0, 0.15), (10_000_000.0, 0.175), (30_000_000.0, 0.20), (float("inf"), 0.225))


def aliquota_regressiva(dias: int) -> float:
    """Alíquota da tabela regressiva de renda fixa para um prazo em dias."""
    if dias < 0:
        raise ValueError("prazo não pode ser negativo")
    for limite, aliquota in TABELA_REGRESSIVA:
        if dias <= limite:
            return aliquota
    return 0.15


def aliquota_cripto(ganho_no_mes: float) -> float:
    for limite, aliquota in FAIXAS_CRIPTO:
        if ganho_no_mes <= limite:
            return aliquota
    return 0.225


@dataclass
class Venda:
    """Uma alienação, com o que o cálculo de imposto precisa saber."""

    ticker: str
    regime: Regime
    quando: date
    valor_alienado: float      # quanto entrou na venda
    custo_aquisicao: float     # quanto custou comprar o que foi vendido
    dias_carregado: int = 0
    day_trade: bool = False

    @property
    def ganho(self) -> float:
        return self.valor_alienado - self.custo_aquisicao


@dataclass
class ResultadoMensal:
    ano: int
    mes: int
    imposto_devido: float
    ganho_tributavel: float
    ganho_isento: float
    prejuizo_acumulado: float
    detalhe: list[dict] = field(default_factory=list)


class ApuradorIR:
    """Apura o IR mês a mês, com compensação de prejuízo por modalidade.

    A compensação é separada por modalidade porque a lei separa: prejuízo em
    FII só compensa ganho em FII, prejuízo em day trade só compensa day trade.
    Misturar tudo num caixa só é o erro que faz o backtest pagar menos imposto
    do que a declaração vai cobrar.
    """

    def __init__(self) -> None:
        # modalidade -> prejuízo acumulado disponível para compensar
        self.prejuizo: dict[str, float] = {}

    @staticmethod
    def modalidade(venda: Venda) -> str:
        if venda.day_trade:
            return "day_trade"
        if venda.regime == Regime.FII:
            return "fii"
        if venda.regime == Regime.CRIPTO:
            return "cripto"
        if venda.regime in (Regime.ETF_RENDA_FIXA, Regime.TESOURO):
            return "renda_fixa"
        return "renda_variavel"

    def _aliquota(self, venda: Venda, ganho_modalidade_no_mes: float) -> float:
        if venda.day_trade:
            return ALIQUOTA_ACAO_DAYTRADE
        if venda.regime == Regime.FII:
            return ALIQUOTA_FII_GANHO
        if venda.regime == Regime.CRIPTO:
            return aliquota_cripto(ganho_modalidade_no_mes)
        if venda.regime in (Regime.ETF_RENDA_FIXA, Regime.TESOURO):
            return aliquota_regressiva(venda.dias_carregado)
        if venda.regime == Regime.ISENTO:
            return 0.0
        if venda.regime == Regime.ACAO_INDIVIDUAL:
            return ALIQUOTA_ACAO_COMUM
        return ALIQUOTA_ETF_RV

    @staticmethod
    def _tem_isencao(vendas: list[Venda]) -> bool:
        """A isenção mensal vale para ação individual e para cripto, e é sobre o
        TOTAL VENDIDO no mês, não sobre o lucro. ETF de renda variável não tem."""
        if not vendas:
            return False
        regime = vendas[0].regime
        total = sum(v.valor_alienado for v in vendas)

        if regime == Regime.ACAO_INDIVIDUAL and not any(v.day_trade for v in vendas):
            return total <= ISENCAO_MENSAL_ACOES
        if regime == Regime.CRIPTO:
            return total <= ISENCAO_MENSAL_CRIPTO
        return False

    def apurar_mes(self, ano: int, mes: int, vendas: list[Venda]) -> ResultadoMensal:
        """Apura o imposto de um mês. Vendas de meses diferentes não se somam."""
        do_mes = [v for v in vendas if v.quando.year == ano and v.quando.month == mes]

        imposto_total = 0.0
        tributavel_total = 0.0
        isento_total = 0.0
        detalhe: list[dict] = []

        # Agrupa por modalidade e, dentro dela, por regime, porque a isenção
        # mensal é por regime (ação individual, cripto).
        por_modalidade: dict[str, list[Venda]] = {}
        for v in do_mes:
            por_modalidade.setdefault(self.modalidade(v), []).append(v)

        for modalidade, lista in por_modalidade.items():
            por_regime: dict[Regime, list[Venda]] = {}
            for v in lista:
                por_regime.setdefault(v.regime, []).append(v)

            ganho_modalidade = sum(v.ganho for v in lista)

            for regime, vendas_regime in por_regime.items():
                ganho_regime = sum(v.ganho for v in vendas_regime)

                if self._tem_isencao(vendas_regime):
                    if ganho_regime > 0:
                        isento_total += ganho_regime
                        detalhe.append({
                            "modalidade": modalidade,
                            "regime": regime.value,
                            "ganho": round(ganho_regime, 2),
                            "imposto": 0.0,
                            "motivo": "isenção mensal por valor alienado",
                        })
                        continue
                    # Prejuízo dentro da isenção não gera crédito compensável.
                    detalhe.append({
                        "modalidade": modalidade,
                        "regime": regime.value,
                        "ganho": round(ganho_regime, 2),
                        "imposto": 0.0,
                        "motivo": "prejuízo em operação isenta, sem crédito",
                    })
                    continue

                if ganho_regime <= 0:
                    self.prejuizo[modalidade] = self.prejuizo.get(modalidade, 0.0) + abs(ganho_regime)
                    detalhe.append({
                        "modalidade": modalidade,
                        "regime": regime.value,
                        "ganho": round(ganho_regime, 2),
                        "imposto": 0.0,
                        "motivo": "prejuízo acumulado para compensação futura",
                    })
                    continue

                # Compensa prejuízo anterior da mesma modalidade.
                credito = self.prejuizo.get(modalidade, 0.0)
                compensado = min(credito, ganho_regime)
                base = ganho_regime - compensado
                self.prejuizo[modalidade] = credito - compensado

                aliquota = self._aliquota(vendas_regime[0], ganho_modalidade)
                imposto = base * aliquota

                imposto_total += imposto
                tributavel_total += base
                detalhe.append({
                    "modalidade": modalidade,
                    "regime": regime.value,
                    "ganho": round(ganho_regime, 2),
                    "prejuizo_compensado": round(compensado, 2),
                    "base": round(base, 2),
                    "aliquota": aliquota,
                    "imposto": round(imposto, 2),
                    "motivo": "tributado",
                })

        return ResultadoMensal(
            ano=ano,
            mes=mes,
            imposto_devido=round(imposto_total, 2),
            ganho_tributavel=round(tributavel_total, 2),
            ganho_isento=round(isento_total, 2),
            prejuizo_acumulado=round(sum(self.prejuizo.values()), 2),
            detalhe=detalhe,
        )


def imposto_sobre_provento(regime: Regime, valor: float) -> float:
    """Imposto sobre rendimento distribuído.

    Provento de FII é isento para pessoa física. ETF de FII **não é**: o ETF é
    um fundo de índice e o rendimento dele não herda a isenção do FII que ele
    carrega. É a confusão mais cara do mercado brasileiro de varejo.
    """
    if valor <= 0:
        return 0.0
    if regime == Regime.FII:
        return 0.0
    if regime == Regime.ISENTO:
        return 0.0
    return valor * ALIQUOTA_ETF_RV


def equivalente_bruto(rendimento_liquido_pct: float, dias: int = 365) -> float:
    """Quanto um investimento TRIBUTADO precisa render para empatar com um isento.

    Serve para comparar FII com CDB na mesma régua: 9,5% isento de FII exige
    cerca de 11,2% bruto de um CDB de um ano.
    """
    aliquota = aliquota_regressiva(dias)
    return rendimento_liquido_pct / (1 - aliquota)


def custo_fiscal_do_giro(
    ganho_pct: float,
    valor_carteira: float,
    giros_por_ano: int,
    regime: Regime = Regime.ETF_RENDA_VARIAVEL,
) -> dict:
    """Quanto o giro custa em imposto, comparado a ficar parado.

    Cada realização de lucro antecipa imposto que ficaria diferido. Em renda
    fixa há um agravante: a alíquota regressiva sobe quando o prazo encurta.
    """
    if giros_por_ano < 0:
        raise ValueError("giros por ano não pode ser negativo")

    ganho_anual = valor_carteira * (ganho_pct / 100)

    if regime in (Regime.ETF_RENDA_FIXA, Regime.TESOURO):
        # Girar N vezes por ano significa prazo médio de 365/N dias por posição,
        # e a tabela regressiva pune prazo curto.
        dias_por_giro = 365 // max(1, giros_por_ano)
        aliquota_girando = aliquota_regressiva(dias_por_giro)
        aliquota_parado = aliquota_regressiva(365 * 3)  # 15%, o piso da tabela
    else:
        aliquota_girando = aliquota_parado = (
            ALIQUOTA_FII_GANHO if regime == Regime.FII else ALIQUOTA_ETF_RV
        )

    # Sem giro não há realização, e sem realização não há imposto no ano. Esse
    # é o diferimento: o imposto continua existindo, mas só acontece no resgate,
    # e até lá o dinheiro dele rende junto com o resto da carteira.
    if giros_por_ano == 0:
        return {
            "ganho_anual": round(ganho_anual, 2),
            "giros_por_ano": 0,
            "aliquota_girando": 0.0,
            "aliquota_se_ficasse_parado": aliquota_parado,
            "imposto_girando": 0.0,
            "imposto_diferido_para_o_resgate": round(ganho_anual * aliquota_parado, 2),
            "custo_extra_do_giro": 0.0,
            "custo_extra_pct_da_carteira": 0.0,
        }

    imposto_girando = ganho_anual * aliquota_girando
    imposto_se_girasse_uma_vez = ganho_anual * aliquota_parado

    return {
        "ganho_anual": round(ganho_anual, 2),
        "giros_por_ano": giros_por_ano,
        "aliquota_girando": aliquota_girando,
        "aliquota_se_ficasse_parado": aliquota_parado,
        "imposto_girando": round(imposto_girando, 2),
        "imposto_diferido_para_o_resgate": round(imposto_se_girasse_uma_vez, 2),
        "custo_extra_do_giro": round(imposto_girando - imposto_se_girasse_uma_vez, 2),
        "custo_extra_pct_da_carteira": round(
            (imposto_girando - imposto_se_girasse_uma_vez) / valor_carteira * 100, 3
        )
        if valor_carteira
        else 0.0,
    }
