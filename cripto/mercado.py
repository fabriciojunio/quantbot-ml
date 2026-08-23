"""
Download, cache e normalização de candles.

O ponto principal deste módulo é o `historico()`: as exchanges devolvem no
máximo 1000 candles por chamada, então para ter dois anos de 1h é preciso
paginar. O cache em CSV evita rebaixar a mesma janela toda vez que um
backtest roda, o que também mantém o consumo de limite de taxa civilizado.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .exchanges import Candle, ErroExchange, Roteador, cliente_padrao

logger = logging.getLogger("quantbot.cripto.mercado")

CACHE_DIR = Path("data/cripto/candles")

# Duração de cada intervalo em milissegundos, para calcular a paginação.
DURACAO_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

COLUNAS = ["abertura", "maxima", "minima", "fechamento", "volume"]


def para_dataframe(candles: list[Candle]) -> pd.DataFrame:
    """Converte candles em DataFrame indexado por datetime UTC."""
    if not candles:
        return pd.DataFrame(columns=COLUNAS, index=pd.DatetimeIndex([], tz="UTC", name="data"))

    df = pd.DataFrame(
        {
            "data": [datetime.fromtimestamp(c.abertura_ms / 1000, tz=timezone.utc) for c in candles],
            "abertura": [c.abertura for c in candles],
            "maxima": [c.maxima for c in candles],
            "minima": [c.minima for c in candles],
            "fechamento": [c.fechamento for c in candles],
            "volume": [c.volume for c in candles],
        }
    )
    df = df.set_index("data").sort_index()
    return df[~df.index.duplicated(keep="last")]


def _caminho_cache(simbolo: str, intervalo: str) -> Path:
    return CACHE_DIR / f"{simbolo.upper()}_{intervalo}.csv"


def _ler_cache(simbolo: str, intervalo: str) -> pd.DataFrame | None:
    caminho = _caminho_cache(simbolo, intervalo)
    if not caminho.exists():
        return None
    try:
        df = pd.read_csv(caminho, index_col="data", parse_dates=["data"])
        if df.index.tz is None:
            df.index = df.index.tz_localize("utc")
        return df.sort_index()
    except Exception as exc:
        logger.warning("Cache ilegível em %s (%s). Vou rebaixar.", caminho, exc)
        return None


def _gravar_cache(simbolo: str, intervalo: str, df: pd.DataFrame) -> None:
    caminho = _caminho_cache(simbolo, intervalo)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index_label="data")


def historico(
    simbolo: str,
    intervalo: str = "1h",
    dias: int = 365,
    cliente: Roteador | None = None,
    usar_cache: bool = True,
    pausa_s: float = 0.25,
) -> pd.DataFrame:
    """Baixa `dias` de candles, paginando e reaproveitando o cache.

    Levanta `ErroExchange` se nenhuma fonte responder e não houver cache —
    nunca devolve série sintética para "não quebrar o gráfico".
    """
    if intervalo not in DURACAO_MS:
        raise ValueError(f"intervalo {intervalo!r} não suportado")

    simbolo = simbolo.upper()
    cliente = cliente or cliente_padrao()
    agora = datetime.now(timezone.utc)
    inicio_alvo = agora - timedelta(days=dias)

    cache = _ler_cache(simbolo, intervalo) if usar_cache else None
    if cache is not None and not cache.empty:
        # Só busca o que falta na ponta nova; a ponta velha já está em disco.
        if cache.index.min() <= inicio_alvo:
            faltando_desde = cache.index.max()
            if (agora - faltando_desde) < timedelta(milliseconds=DURACAO_MS[intervalo] * 2):
                logger.info("Cache de %s %s já cobre a janela pedida.", simbolo, intervalo)
                return cache.loc[cache.index >= inicio_alvo].copy()
            inicio_download = faltando_desde
        else:
            inicio_download = inicio_alvo
    else:
        inicio_download = inicio_alvo

    passo_ms = DURACAO_MS[intervalo]
    inicio_ms = int(inicio_download.timestamp() * 1000)
    fim_ms = int(agora.timestamp() * 1000)

    pedacos: list[pd.DataFrame] = []
    cursor = inicio_ms
    while cursor < fim_ms:
        candles = cliente.candles(
            simbolo, intervalo, limite=1000, inicio_ms=cursor, fim_ms=fim_ms
        )
        if not candles:
            break
        pedaco = para_dataframe(candles)
        pedacos.append(pedaco)

        ultimo = candles[-1].abertura_ms
        if ultimo <= cursor:  # a fonte não avançou; evita laço infinito
            break
        cursor = ultimo + passo_ms
        time.sleep(pausa_s)

    if not pedacos:
        if cache is not None and not cache.empty:
            logger.warning("Download vazio para %s. Usando só o cache.", simbolo)
            return cache.loc[cache.index >= inicio_alvo].copy()
        raise ErroExchange(f"nenhum candle obtido para {simbolo} {intervalo}")

    novo = pd.concat(pedacos)
    if cache is not None and not cache.empty:
        novo = pd.concat([cache, novo])

    novo = novo[~novo.index.duplicated(keep="last")].sort_index()
    novo = novo[COLUNAS]

    if usar_cache:
        _gravar_cache(simbolo, intervalo, novo)

    logger.info(
        "%s %s: %d candles de %s até %s (fonte: %s)",
        simbolo,
        intervalo,
        len(novo),
        novo.index.min().date(),
        novo.index.max().date(),
        cliente.ultima_fonte,
    )
    return novo.loc[novo.index >= inicio_alvo].copy()


def valida_serie(df: pd.DataFrame, nome: str = "série") -> None:
    """Falha alto se a série tiver buraco grave, preço zerado ou OHLC impossível.

    Um backtest sobre dado corrompido produz um número bonito e mentiroso.
    Melhor quebrar aqui.
    """
    if df.empty:
        raise ValueError(f"{nome}: sem candles")

    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise ValueError(f"{nome}: faltam colunas {faltando}")

    if (df[["abertura", "maxima", "minima", "fechamento"]] <= 0).any().any():
        raise ValueError(f"{nome}: há preço zerado ou negativo")

    invalidos = (df["maxima"] < df["minima"]) | (df["maxima"] < df["fechamento"]) | (df["minima"] > df["fechamento"])
    if invalidos.any():
        raise ValueError(f"{nome}: {int(invalidos.sum())} candles com OHLC inconsistente")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{nome}: índice fora de ordem")
