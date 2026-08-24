"""
Reconstrução de histórico longo a partir de índices.

O problema: a carteira usa ETFs que são novos. O GOLD11 nasceu em dezembro de
2020, o B5P211 em novembro de 2020, o XFIX11 em novembro de 2020. Com o
universo completo só existem cinco anos de história — e cinco anos que foram,
por acaso, um período específico (pandemia, juro a 2%, juro a 15%). Concluir
"a carteira rende X" a partir de uma janela dessas é o erro que este projeto
inteiro existe para não cometer.

A saída é reconstruir cada classe pelo índice que ela replica, que tem história
muito mais longa, e **cobrar** dela o que o ETF cobraria:

    BOVA11  <-  ^BVSP                   desde 1993, menos 0,30% a.a.
    IVVB11  <-  ^SP500TR x PTAX         desde 1993, menos 0,23% a.a.
    GOLD11  <-  GC=F x PTAX             desde 2000, menos 0,50% a.a.

O câmbio vem da **PTAX do Banco Central**, não do Yahoo. Essa escolha não é
detalhe: `USDBRL=X` não tem um único dia de 2005 e tem um buraco de 575 dias
terminando em março de 2006. Depois do alinhamento entre ativos, o buraco
sumia e o dia seguinte a ele virava um retorno de 66% em um pregão — número
que contamina volatilidade, queda máxima e contagem de janela sem dar nenhum
sinal de erro. A PTAX é a taxa oficial, publicada desde 1984, e é contínua.

Com a PTAX, a reconstrução completa vai de **2000-08 até hoje** (limite do
contrato de ouro), cerca de 26 anos — o que inclui a crise de 2008 inteira e
dá janelas móveis de 5 e de 10 anos suficientes para falar em distribuição de
resultado em vez de um número só.

O que esta reconstrução **não** cobre, e por que:

* **SMAL11 e XFIX11** — não existe índice gratuito com história longa para
  small caps e para o IFIX. Ficam de fora. Não invento proxy.
* **IMAB11 e B5P211** — o IMA-B é da ANBIMA e não está na API aberta do Banco
  Central. A faixa de renda fixa da carteira longa usa **CDI**, que é dado
  real. A distorção é conhecida e vai declarada no resultado: o IMA-B rendeu
  um pouco acima do CDI no período e oscilou bastante mais, então a carteira
  longa **subestima** tanto o retorno quanto a volatilidade da faixa.

Nenhuma série aqui é sintética. Todas vêm de cotação real; o que o código faz
é multiplicar séries reais entre si e descontar a taxa que o ETF cobraria.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from .perfil import Faixa
from .universo import Ativo, Classe, Regime

logger = logging.getLogger("quantbot.carteira.serie_longa")

# Prefixo que marca um fator vindo do Banco Central em vez do Yahoo.
PREFIXO_BCB = "bcb:"

# Antes disto o câmbio era administrado e o real acabara de nascer; qualquer
# série que cruze esse ponto mistura dois regimes cambiais diferentes.
INICIO_DO_CAMBIO_FLUTUANTE = pd.Timestamp("1999-02-01")


@dataclass(frozen=True)
class Receita:
    """Como reconstruir um ativo a partir de índices reais.

    `fatores` são multiplicados entre si. Dois fatores significam índice em
    dólar convertido para reais.
    """

    alvo: str
    fatores: tuple[str, ...]
    taxa_adm_anual: float
    desde: str
    fonte: str
    aviso: str = ""

    def __post_init__(self) -> None:
        if not self.fatores:
            raise ValueError(f"{self.alvo}: receita sem nenhum fator")
        if not 0 <= self.taxa_adm_anual < 0.10:
            raise ValueError(f"{self.alvo}: taxa de administração implausível")


RECEITAS: dict[str, Receita] = {
    "BOVA11.SA": Receita(
        alvo="BOVA11.SA",
        fatores=("^BVSP",),
        taxa_adm_anual=0.0030,
        desde="1993-04",
        fonte="Ibovespa",
        aviso="O Ibovespa já é índice de retorno total: reinveste o provento das ações.",
    ),
    "IVVB11.SA": Receita(
        alvo="IVVB11.SA",
        fatores=("^SP500TR", PREFIXO_BCB + "USD_BRL"),
        taxa_adm_anual=0.0023,
        desde="1993-04",
        fonte="S&P 500 Total Return convertido pela PTAX venda do Banco Central",
        aviso="Usa a versão total return do S&P: a de preço puro esconde ~1,8% a.a. de dividendo.",
    ),
    "GOLD11.SA": Receita(
        alvo="GOLD11.SA",
        fatores=("GC=F", PREFIXO_BCB + "USD_BRL"),
        taxa_adm_anual=0.0050,
        desde="2000-08",
        fonte="Ouro futuro em dólar convertido pela PTAX venda do Banco Central",
        aviso="Ouro não distribui rendimento, então preço puro é a medida certa aqui.",
    ),
}

# Ativos que a reconstrução longa não cobre, com o motivo. Fica no código para
# que a ausência seja uma decisão registrada e não um esquecimento.
SEM_HISTORICO_LONGO: dict[str, str] = {
    "SMAL11.SA": "não há índice de small caps brasileiro gratuito com história longa",
    "XFIX11.SA": "o IFIX não está disponível em série gratuita longa",
    "IMAB11.SA": "o IMA-B é da ANBIMA e não está na API aberta do Banco Central",
    "B5P211.SA": "mesmo motivo do IMA-B",
    "DIVO11.SA": "índice de dividendos sem série longa gratuita",
    "FIXA11.SA": "o IRF-M é da ANBIMA e não está na API aberta",
}


# Composição das faixas quando se usa a série longa. Cada faixa é representada
# só pelo que existe de verdade no período — e a renda fixa cai para o CDI.
COMPOSICAO_LONGA: dict[Faixa, dict[str, float]] = {
    Faixa.CAIXA: {"CDI": 1.0},
    Faixa.RENDA_FIXA: {"CDI": 1.0},
    Faixa.DOLAR: {"IVVB11.SA": 1.0},
    Faixa.BOLSA: {"BOVA11.SA": 1.0},
    Faixa.ALTERNATIVO: {"GOLD11.SA": 1.0},
    Faixa.SATELITE: {},
}

AVISOS_DA_COMPOSICAO_LONGA: tuple[str, ...] = (
    "A faixa de renda fixa usa CDI no lugar do IMA-B: subestima retorno e volatilidade.",
    "A faixa de bolsa usa só o Ibovespa: sem a fatia de small caps, que é mais volátil.",
    "A faixa alternativa usa só ouro: sem a fatia de fundos imobiliários.",
    "Os preços são índices com a taxa de administração do ETF descontada dia a dia, "
    "mas sem o erro de rastreamento que um ETF real tem.",
)


# Ativos sintetizados precisam existir no universo para o modelo de custo saber
# o spread deles. Reaproveitamos a definição do universo real quando existe.
def _ativo_reconstruido(ticker: str, base: dict[str, Ativo]) -> Ativo:
    if ticker in base:
        return base[ticker]
    return Ativo(
        ticker, ticker, Classe.ACOES_BR, Regime.ETF_RENDA_VARIAVEL,
        spread_pct=0.0010, volume_medio_brl=1e9,
    )


def _juntar_fatores(
    fatores: set[str],
    anos: int,
    baixador,
    cambista,
) -> pd.DataFrame:
    """Traz cada fator da sua fonte e alinha no calendário comum.

    Fatores com o prefixo ``bcb:`` vêm do Banco Central; o resto vem do Yahoo.
    O câmbio é reindexado para o calendário do mercado com preenchimento para
    frente, o que aqui é correto e não é o `ffill` proibido em outros pontos:
    a PTAX não é publicada em feriado bancário, e a taxa que vale nesse dia é
    de fato a do último fechamento.
    """
    de_mercado = sorted(f for f in fatores if not f.startswith(PREFIXO_BCB))
    do_bcb = sorted(f for f in fatores if f.startswith(PREFIXO_BCB))

    if not de_mercado:
        raise ValueError("série longa: nenhuma cotação de mercado nas receitas")

    df = baixador(de_mercado, anos=anos).sort_index()
    df = df.dropna(how="any")
    if df.empty:
        raise ValueError("série longa: nenhum dia com cotação de todos os fatores de mercado")

    for f in do_bcb:
        serie = cambista(anos=anos, moeda=f[len(PREFIXO_BCB):])
        df[f] = serie.reindex(df.index, method="ffill")

    return df.dropna(how="any")


def montar(
    anos: int = 26,
    receitas: dict[str, Receita] | None = None,
    baixador=None,
    cambista=None,
    cobrar_taxa_adm: bool = True,
    max_dias_sem_pregao: int = 12,
) -> pd.DataFrame:
    """Reconstrói os preços longos, já com taxa de administração descontada.

    Devolve um DataFrame com uma coluna por ativo alvo, base 100 no primeiro
    dia em que **todos** têm cotação. Base 100 é adequado porque o simulador
    de aportes compra frações de cota; o nível absoluto do preço não muda
    nenhum resultado, e ancorar em 100 evita comparar reais de 2003 com reais
    de hoje sem querer.
    """
    receitas = receitas or RECEITAS
    from . import dados

    baixador = baixador or dados.precos
    cambista = cambista or dados.cambio_diario

    fatores = {f for r in receitas.values() for f in r.fatores}
    bruto = _juntar_fatores(fatores, anos, baixador, cambista)

    faltando = fatores - set(bruto.columns)
    if faltando:
        raise ValueError(f"série longa: faltaram os fatores {sorted(faltando)}")

    bruto = bruto.loc[bruto.index >= INICIO_DO_CAMBIO_FLUTUANTE]
    if bruto.empty:
        raise ValueError("série longa: nenhum dia com cotação de todos os fatores")

    # Um buraco de calendário sobrevive calado ao alinhamento e vira um retorno
    # gigante de um dia só. É a falha que a fonte anterior de câmbio tinha.
    dados.valida_continuidade(bruto, "fatores da série longa", max_dias_sem_pregao)

    colunas: dict[str, pd.Series] = {}
    for alvo, r in receitas.items():
        serie = bruto[list(r.fatores)].prod(axis=1)
        if cobrar_taxa_adm:
            # A taxa é cobrada dentro da cota, todo dia útil, e por isso é
            # invisível no extrato. Aqui ela aparece.
            dias = pd.Series(range(len(serie)), index=serie.index)
            serie = serie * (1 - r.taxa_adm_anual / 252) ** dias
        colunas[alvo] = serie / float(serie.iloc[0]) * 100.0

    df = pd.DataFrame(colunas).sort_index()
    logger.info(
        "Série longa: %d pregões, de %s a %s, ativos %s",
        len(df), df.index.min().date(), df.index.max().date(), list(df.columns),
    )
    return df


def procedencia(receitas: dict[str, Receita] | None = None) -> list[dict]:
    """De onde veio cada série. Vai junto de todo relatório que usa série longa."""
    receitas = receitas or RECEITAS
    linhas = [
        {
            "ativo": r.alvo,
            "reconstruido_de": " x ".join(r.fatores),
            "fonte": r.fonte,
            "desde": r.desde,
            "taxa_adm_descontada_pct": round(r.taxa_adm_anual * 100, 2),
            "aviso": r.aviso,
        }
        for r in receitas.values()
    ]
    linhas += [
        {"ativo": t, "reconstruido_de": "—", "fonte": "sem série longa",
         "desde": "—", "taxa_adm_descontada_pct": 0.0, "aviso": motivo}
        for t, motivo in SEM_HISTORICO_LONGO.items()
    ]
    return linhas


@dataclass(frozen=True)
class Cobertura:
    """Quanto de história cada configuração alcança."""

    inicio: pd.Timestamp
    fim: pd.Timestamp
    pregoes: int
    ativos: tuple[str, ...]
    avisos: tuple[str, ...] = field(default_factory=tuple)

    @property
    def anos(self) -> float:
        return (self.fim - self.inicio).days / 365.25

    def janelas_de(self, anos: int) -> int:
        """Quantas janelas móveis de N anos cabem, com passo de um mês."""
        meses = int(self.anos * 12) - anos * 12
        return max(0, meses + 1)


def cobertura(precos: pd.DataFrame, avisos: tuple[str, ...] = ()) -> Cobertura:
    return Cobertura(
        inicio=precos.index.min(),
        fim=precos.index.max(),
        pregoes=len(precos),
        ativos=tuple(precos.columns),
        avisos=avisos,
    )
