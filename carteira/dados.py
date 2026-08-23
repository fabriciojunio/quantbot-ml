"""
Coleta de preços e indicadores macro, com cache em disco.

Preço de ETF e ação vem do Yahoo Finance com `auto_adjust=True`, que já
incorpora provento e desdobramento no histórico — sem isso, todo backtest de
ativo que distribui rendimento sai subestimado. CDI, Selic e IPCA vêm do
Banco Central, pelo cliente que o QuantBot já tinha.

Como no pacote `cripto/`, a regra é: se a fonte não responde e não há cache,
a função levanta erro. Nenhuma série sintética entra para "não quebrar o
gráfico".
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.bcb_client import BCBClient

logger = logging.getLogger("quantbot.carteira.dados")

CACHE_DIR = Path("data/carteira/precos")
CACHE_MACRO = Path("data/carteira/macro")


class ErroDados(RuntimeError):
    """A fonte não respondeu e não havia cache utilizável."""


def _yf():
    """Importa o yfinance só quando precisa e cala o ruído dele nos logs."""
    warnings.filterwarnings("ignore", module="yfinance")
    import yfinance as yf

    return yf


def _achatar(df: pd.DataFrame) -> pd.DataFrame:
    """O yfinance devolve colunas em MultiIndex quando baixa um ticker só.
    Achatar aqui evita o `float(Series)` que quebra silenciosamente adiante."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    return df


def _caminho(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('.', '_').replace('-', '_')}.csv"


def _ler_cache(ticker: str) -> pd.DataFrame | None:
    caminho = _caminho(ticker)
    if not caminho.exists():
        return None
    try:
        df = pd.read_csv(caminho, index_col="data", parse_dates=["data"])
        return df.sort_index()
    except Exception as exc:
        logger.warning("Cache ilegível de %s (%s). Vou rebaixar.", ticker, exc)
        return None


def _gravar_cache(ticker: str, df: pd.DataFrame) -> None:
    caminho = _caminho(ticker)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index_label="data")


def precos(
    tickers: list[str] | str,
    anos: int = 5,
    usar_cache: bool = True,
    validade_horas: int = 12,
) -> pd.DataFrame:
    """Baixa o fechamento ajustado de um ou mais tickers.

    Devolve DataFrame com uma coluna por ticker, indexado por data, sem
    preencher para frente entre feriados diferentes — quem alinha o calendário
    é quem usa, de forma explícita.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    series: dict[str, pd.Series] = {}
    faltando: list[str] = []

    for t in tickers:
        cache = _ler_cache(t) if usar_cache else None
        if cache is not None and not cache.empty:
            idade = datetime.now() - cache.index.max().to_pydatetime()
            cobre = cache.index.min().date() <= date.today() - timedelta(days=anos * 365 - 10)
            if idade < timedelta(hours=validade_horas) and cobre:
                series[t] = cache["fechamento"]
                continue
        faltando.append(t)

    if faltando:
        yf = _yf()
        for t in faltando:
            try:
                bruto = yf.download(
                    t, period=f"{max(anos, 1)}y", interval="1d",
                    progress=False, auto_adjust=True, threads=False,
                )
                bruto = _achatar(bruto)
                if bruto.empty or "Close" not in bruto:
                    raise ErroDados(f"resposta vazia para {t}")
                s = bruto["Close"].astype(float).dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                s.name = "fechamento"
                series[t] = s
                if usar_cache:
                    _gravar_cache(t, s.to_frame())
                logger.info("%s: %d pregões (%s a %s)", t, len(s),
                            s.index.min().date(), s.index.max().date())
            except Exception as exc:
                cache = _ler_cache(t)
                if cache is not None and not cache.empty:
                    logger.warning("Falhou %s (%s). Usando cache antigo.", t, exc)
                    series[t] = cache["fechamento"]
                else:
                    logger.error("Sem dado nenhum para %s: %s", t, exc)

    if not series:
        raise ErroDados(f"nenhuma série obtida para {tickers}")

    df = pd.DataFrame(series).sort_index()
    return df.loc[df.index >= pd.Timestamp(date.today() - timedelta(days=anos * 365))]


def cdi_diario(anos: int = 5) -> pd.Series:
    """Série do CDI diário (fator, não percentual) vinda do Banco Central.

    Devolve o retorno de cada dia útil já em fração: 0,0005 para 0,05% ao dia.

    A API do BCB recusa com HTTP 406 qualquer janela maior que 10 anos, então
    a busca é fatiada em blocos de 8 anos e emendada. Sem isso, um pedido de
    12 anos volta vazio — e o backtest passa a comparar contra CDI zero, o que
    faz qualquer carteira parecer vencedora nas janelas antigas.
    """
    CACHE_MACRO.mkdir(parents=True, exist_ok=True)
    caminho = CACHE_MACRO / "cdi_diario.csv"

    if caminho.exists():
        idade = datetime.now() - datetime.fromtimestamp(caminho.stat().st_mtime)
        cache = pd.read_csv(caminho, index_col="data", parse_dates=["data"])["cdi"].sort_index()
        cobre = cache.index.min().date() <= date.today() - timedelta(days=anos * 365 - 30)
        if idade < timedelta(hours=24) and cobre:
            return cache

    cliente = BCBClient()
    pedacos: list[pd.Series] = []
    fim = date.today()
    restante = max(1, anos)

    while restante > 0:
        bloco = min(8, restante)
        inicio = fim - timedelta(days=bloco * 365)
        try:
            obs = cliente.historico("CDI_DIARIO", data_inicial=inicio, data_final=fim)
            if obs:
                pedacos.append(
                    pd.Series(
                        [o.valor / 100 for o in obs],
                        index=pd.to_datetime([o.data for o in obs]),
                    )
                )
        except Exception as exc:
            logger.warning("Bloco de CDI %s a %s falhou: %s", inicio, fim, exc)
        fim = inicio - timedelta(days=1)
        restante -= bloco

    if pedacos:
        s = pd.concat(pedacos).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s.name = "cdi"
        s.to_frame().to_csv(caminho, index_label="data")
        logger.info("CDI: %d dias, de %s a %s", len(s), s.index.min().date(), s.index.max().date())
        return s

    if caminho.exists():
        logger.warning("BCB não respondeu. Usando cache de CDI.")
        return pd.read_csv(caminho, index_col="data", parse_dates=["data"])["cdi"].sort_index()
    raise ErroDados("sem CDI: BCB não respondeu e não há cache")


def macro() -> dict:
    """Selic, CDI anual, IPCA 12m e dólar, agora."""
    try:
        return BCBClient().snapshot()
    except Exception as exc:
        raise ErroDados(f"snapshot macro indisponível: {exc}") from exc


def alinhar(df: pd.DataFrame, minimo_de_ativos: int = 2) -> pd.DataFrame:
    """Alinha o calendário entre ativos, mantendo só os dias com preço para todos.

    Preencher para frente aqui seria conveniente e errado: um feriado da B3 que
    não é feriado nos Estados Unidos criaria retorno zero para o ativo local no
    dia em que o global anda, e o backtest passaria a medir uma correlação que
    não existe.
    """
    if df.shape[1] < minimo_de_ativos:
        return df.dropna()
    return df.dropna(how="any")


def retornos(df: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Retornos diários. `log=True` para retorno logarítmico."""
    if log:
        return np.log(df / df.shift(1)).dropna(how="all")
    return df.pct_change().dropna(how="all")


def valida_precos(df: pd.DataFrame, nome: str = "série de preços") -> None:
    """Falha alto em preço zerado, negativo, buraco grande ou índice fora de ordem."""
    if df.empty:
        raise ValueError(f"{nome}: sem dados")
    if (df <= 0).any().any():
        colunas = df.columns[(df <= 0).any()].tolist()
        raise ValueError(f"{nome}: preço zerado ou negativo em {colunas}")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{nome}: índice fora de ordem")
    if df.index.duplicated().any():
        raise ValueError(f"{nome}: há datas duplicadas")

    # Um salto acima de 60% em um dia é quase sempre desdobramento não ajustado
    # ou erro da fonte — não movimento de mercado de um ETF.
    variacao = df.pct_change().abs()
    suspeitos = variacao[variacao > 0.60].dropna(how="all")
    if not suspeitos.empty:
        raise ValueError(
            f"{nome}: {len(suspeitos)} dia(s) com variação acima de 60%. "
            "Provável desdobramento não ajustado ou erro da fonte."
        )
