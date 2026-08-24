"""
Universo de ativos acessíveis a pessoa física no Brasil.

Cada ativo traz a informação que o motor de imposto e o de custo precisam:
classe, regime tributário, liquidez típica e spread observado. Um ETF de renda
fixa e um FII têm tributação completamente diferente, e ignorar isso é o tipo
de erro que faz um backtest render vários pontos a mais do que a realidade.

Os spreads são estimativas conservadoras de book em horário normal. Em leilão
de abertura, fechamento ou dia de estresse eles pioram, e o modelo de custo
tem folga para isso.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Classe(str, Enum):
    ACOES_BR = "acoes_br"
    ACOES_GLOBAL = "acoes_global"
    SMALL_CAPS = "small_caps"
    RENDA_FIXA_IPCA = "renda_fixa_ipca"
    RENDA_FIXA_PRE = "renda_fixa_pre"
    RENDA_FIXA_POS = "renda_fixa_pos"
    IMOBILIARIO = "imobiliario"
    OURO = "ouro"
    CRIPTO = "cripto"
    CAIXA = "caixa"


class Regime(str, Enum):
    """Regime de imposto de renda aplicável à pessoa física."""

    ETF_RENDA_VARIAVEL = "etf_rv"        # 15% sobre ganho, sem isenção mensal
    ETF_RENDA_FIXA = "etf_rf"            # tabela regressiva 15% a 25%
    ACAO_INDIVIDUAL = "acao"             # 15%, com isenção até R$ 20 mil/mês
    FII = "fii"                          # provento isento; ganho de capital 20%
    CRIPTO = "cripto"                    # 15% a 22,5%, isenção até R$ 35 mil/mês
    TESOURO = "tesouro"                  # tabela regressiva
    ISENTO = "isento"                    # LCI, LCA, poupança


@dataclass(frozen=True)
class Ativo:
    ticker: str
    nome: str
    classe: Classe
    regime: Regime
    spread_pct: float          # spread relativo típico do book, em fração
    volume_medio_brl: float    # volume diário médio, para checar se cabe a ordem
    dy_anual_pct: float = 0.0  # rendimento distribuído, quando houver
    observacao: str = ""

    @property
    def custo_meio_spread(self) -> float:
        return self.spread_pct / 2


# ETFs e fundos listados na B3 com liquidez suficiente para ordem de pessoa
# física. Volume é ordem de grandeza observada em 2026, não valor exato.
UNIVERSO_B3: dict[str, Ativo] = {
    "BOVA11.SA": Ativo(
        "BOVA11.SA", "Ibovespa", Classe.ACOES_BR, Regime.ETF_RENDA_VARIAVEL,
        spread_pct=0.0004, volume_medio_brl=400_000_000,
        observacao="O ETF mais líquido da B3. Taxa de administração 0,30% a.a.",
    ),
    "IVVB11.SA": Ativo(
        "IVVB11.SA", "S&P 500 em reais", Classe.ACOES_GLOBAL, Regime.ETF_RENDA_VARIAVEL,
        spread_pct=0.0006, volume_medio_brl=120_000_000,
        observacao="Exposição a bolsa americana e ao dólar ao mesmo tempo.",
    ),
    "SMAL11.SA": Ativo(
        "SMAL11.SA", "Small caps Brasil", Classe.SMALL_CAPS, Regime.ETF_RENDA_VARIAVEL,
        spread_pct=0.0012, volume_medio_brl=40_000_000,
        observacao="Mais volátil e com spread pior que o BOVA11.",
    ),
    "DIVO11.SA": Ativo(
        "DIVO11.SA", "Ações de dividendo", Classe.ACOES_BR, Regime.ETF_RENDA_VARIAVEL,
        spread_pct=0.0015, volume_medio_brl=12_000_000,
        dy_anual_pct=7.0,
    ),
    "IMAB11.SA": Ativo(
        "IMAB11.SA", "Tesouro IPCA+ (IMA-B)", Classe.RENDA_FIXA_IPCA, Regime.ETF_RENDA_FIXA,
        spread_pct=0.0008, volume_medio_brl=25_000_000,
        observacao="Marcado a mercado: sobe e desce com a curva de juros real.",
    ),
    "B5P211.SA": Ativo(
        "B5P211.SA", "Tesouro IPCA+ curto (IMA-B 5)", Classe.RENDA_FIXA_IPCA, Regime.ETF_RENDA_FIXA,
        spread_pct=0.0010, volume_medio_brl=10_000_000,
        observacao="Duration curta: oscila bem menos que o IMAB11.",
    ),
    "FIXA11.SA": Ativo(
        "FIXA11.SA", "Prefixado (IRF-M)", Classe.RENDA_FIXA_PRE, Regime.ETF_RENDA_FIXA,
        spread_pct=0.0015, volume_medio_brl=5_000_000,
    ),
    "XFIX11.SA": Ativo(
        "XFIX11.SA", "IFIX (fundos imobiliários)", Classe.IMOBILIARIO, Regime.ETF_RENDA_VARIAVEL,
        spread_pct=0.0018, volume_medio_brl=6_000_000,
        dy_anual_pct=9.5,
        observacao="ETF de FII NÃO herda a isenção do provento: aqui o rendimento é tributado.",
    ),
    "GOLD11.SA": Ativo(
        "GOLD11.SA", "Ouro", Classe.OURO, Regime.ETF_RENDA_VARIAVEL,
        spread_pct=0.0020, volume_medio_brl=8_000_000,
        observacao="Proteção contra crise e contra dólar, não fonte de renda.",
    ),
}

# Cripto entra como fatia pequena, com o próprio regime tributário.
UNIVERSO_CRIPTO: dict[str, Ativo] = {
    "BTC-USD": Ativo(
        "BTC-USD", "Bitcoin", Classe.CRIPTO, Regime.CRIPTO,
        spread_pct=0.0002, volume_medio_brl=100_000_000_000,
        observacao="Isento de IR enquanto a venda total do mês ficar abaixo de R$ 35 mil.",
    ),
    "ETH-USD": Ativo(
        "ETH-USD", "Ethereum", Classe.CRIPTO, Regime.CRIPTO,
        spread_pct=0.0003, volume_medio_brl=40_000_000_000,
    ),
}

CAIXA = Ativo(
    "CDI", "Caixa remunerado ao CDI", Classe.CAIXA, Regime.TESOURO,
    spread_pct=0.0, volume_medio_brl=float("inf"),
    observacao="Tesouro Selic ou CDB de liquidez diária a 100% do CDI.",
)


def universo_completo(incluir_cripto: bool = False) -> dict[str, Ativo]:
    u = dict(UNIVERSO_B3)
    if incluir_cripto:
        u.update(UNIVERSO_CRIPTO)
    return u


def por_classe(classe: Classe, incluir_cripto: bool = True) -> list[Ativo]:
    return [a for a in universo_completo(incluir_cripto).values() if a.classe == classe]


def cabe_a_ordem(ativo: Ativo, valor_brl: float, fatia_maxima: float = 0.01) -> bool:
    """A ordem cabe no volume diário sem mover o preço?

    Para uma carteira de pessoa física a resposta é quase sempre sim, e é bom
    que o código diga isso em vez de deixar implícito: o problema de quem tem
    R$ 5 mil nunca é liquidez, é custo e imposto.
    """
    return valor_brl <= ativo.volume_medio_brl * fatia_maxima


# Carteiras de referência. Servem de baseline honesto: qualquer estratégia
# precisa bater estas antes de ser considerada.
CARTEIRAS_REFERENCIA: dict[str, dict[str, float]] = {
    "so_cdi": {"CDI": 1.0},
    "so_bova": {"BOVA11.SA": 1.0},
    "60_40": {"BOVA11.SA": 0.60, "IMAB11.SA": 0.40},
    "tres_pilares": {"BOVA11.SA": 0.34, "IVVB11.SA": 0.33, "IMAB11.SA": 0.33},
    "diversificada": {
        "BOVA11.SA": 0.20,
        "IVVB11.SA": 0.20,
        "SMAL11.SA": 0.10,
        "IMAB11.SA": 0.20,
        "B5P211.SA": 0.10,
        "XFIX11.SA": 0.10,
        "GOLD11.SA": 0.10,
    },
}
