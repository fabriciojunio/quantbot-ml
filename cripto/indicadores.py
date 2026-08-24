"""
Indicadores técnicos calculados de OHLCV real.

Contraste deliberado com o sistema antigo: lá os oito "indicadores" saíam de
`isBullish = change24h >= 0` e de strings fixas ("MACD +1.42"). Aqui cada
função recebe a série de preço e devolve o número que a definição manda, sem
atalho e sem valor decorativo.

Todas as funções devolvem `pd.Series` alinhada ao índice de entrada, com NaN
no período de aquecimento. Nada é preenchido para trás — preencher para trás
é vazamento de informação do futuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _serie(x: pd.Series | pd.DataFrame, coluna: str = "fechamento") -> pd.Series:
    if isinstance(x, pd.DataFrame):
        return x[coluna].astype(float)
    return x.astype(float)


def sma(preco: pd.Series, periodo: int) -> pd.Series:
    return _serie(preco).rolling(periodo, min_periods=periodo).mean()


def ema(preco: pd.Series, periodo: int) -> pd.Series:
    return _serie(preco).ewm(span=periodo, adjust=False, min_periods=periodo).mean()


def rsi(preco: pd.Series, periodo: int = 14) -> pd.Series:
    """RSI de Wilder (suavização exponencial com alpha = 1/periodo)."""
    p = _serie(preco)
    delta = p.diff()
    ganho = delta.clip(lower=0.0)
    perda = -delta.clip(upper=0.0)

    media_ganho = ganho.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    media_perda = perda.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()

    rs = media_ganho / media_perda.replace(0.0, np.nan)
    resultado = 100 - (100 / (1 + rs))
    # Perda média zero significa alta ininterrupta: RSI 100 por definição.
    return resultado.where(media_perda != 0, 100.0).where(media_ganho.notna())


def macd(
    preco: pd.Series, rapida: int = 12, lenta: int = 26, sinal: int = 9
) -> pd.DataFrame:
    p = _serie(preco)
    linha = ema(p, rapida) - ema(p, lenta)
    linha_sinal = linha.ewm(span=sinal, adjust=False, min_periods=sinal).mean()
    return pd.DataFrame(
        {"macd": linha, "sinal": linha_sinal, "histograma": linha - linha_sinal}
    )


def bollinger(preco: pd.Series, periodo: int = 20, desvios: float = 2.0) -> pd.DataFrame:
    p = _serie(preco)
    media = p.rolling(periodo, min_periods=periodo).mean()
    desvio = p.rolling(periodo, min_periods=periodo).std(ddof=0)
    superior = media + desvios * desvio
    inferior = media - desvios * desvio
    largura = (superior - inferior) / media.replace(0.0, np.nan)
    # %B: 0 na banda inferior, 1 na superior.
    pct_b = (p - inferior) / (superior - inferior).replace(0.0, np.nan)
    return pd.DataFrame(
        {"media": media, "superior": superior, "inferior": inferior, "largura": largura, "pct_b": pct_b}
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    alta, baixa, fecha = df["maxima"], df["minima"], df["fechamento"]
    fecha_ant = fecha.shift(1)
    return pd.concat(
        [alta - baixa, (alta - fecha_ant).abs(), (baixa - fecha_ant).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    """ATR de Wilder. É a régua de volatilidade que dimensiona stop e alvo."""
    return true_range(df).ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()


def adx(df: pd.DataFrame, periodo: int = 14) -> pd.DataFrame:
    """ADX com +DI e -DI (Wilder). ADX alto = tendência; baixo = lateralidade."""
    alta, baixa = df["maxima"], df["minima"]
    up = alta.diff()
    down = -baixa.diff()

    dm_mais = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    dm_menos = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr_suave = true_range(df).ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    di_mais = 100 * dm_mais.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean() / tr_suave.replace(0.0, np.nan)
    di_menos = 100 * dm_menos.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean() / tr_suave.replace(0.0, np.nan)

    dx = 100 * (di_mais - di_menos).abs() / (di_mais + di_menos).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "adx": dx.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean(),
            "di_mais": di_mais,
            "di_menos": di_menos,
        }
    )


def estocastico(df: pd.DataFrame, periodo: int = 14, suave_k: int = 3, suave_d: int = 3) -> pd.DataFrame:
    baixa_min = df["minima"].rolling(periodo, min_periods=periodo).min()
    alta_max = df["maxima"].rolling(periodo, min_periods=periodo).max()
    faixa = (alta_max - baixa_min).replace(0.0, np.nan)
    k_bruto = 100 * (df["fechamento"] - baixa_min) / faixa
    k = k_bruto.rolling(suave_k, min_periods=suave_k).mean()
    d = k.rolling(suave_d, min_periods=suave_d).mean()
    return pd.DataFrame({"k": k, "d": d})


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: volume acumulado com o sinal da variação do fechamento."""
    direcao = np.sign(df["fechamento"].diff()).fillna(0.0)
    return (direcao * df["volume"]).cumsum()


def vwap_sessao(df: pd.DataFrame) -> pd.Series:
    """VWAP reiniciado a cada dia UTC — referência de preço justo intradiário."""
    tipico = (df["maxima"] + df["minima"] + df["fechamento"]) / 3
    dia = df.index.tz_convert("UTC").date if df.index.tz is not None else df.index.date
    grupos = pd.Series(dia, index=df.index)
    pv = (tipico * df["volume"]).groupby(grupos).cumsum()
    vol = df["volume"].groupby(grupos).cumsum().replace(0.0, np.nan)
    return pv / vol


def supertrend(df: pd.DataFrame, periodo: int = 10, multiplicador: float = 3.0) -> pd.DataFrame:
    """SuperTrend clássico sobre ATR, com as bandas travadas de forma recursiva."""
    faixa = atr(df, periodo)
    meio = (df["maxima"] + df["minima"]) / 2
    banda_sup = meio + multiplicador * faixa
    banda_inf = meio - multiplicador * faixa

    fecha = df["fechamento"].to_numpy(dtype=float)
    sup = banda_sup.to_numpy(dtype=float).copy()
    inf = banda_inf.to_numpy(dtype=float).copy()
    tendencia = np.ones(len(df), dtype=int)

    for i in range(1, len(df)):
        if np.isnan(sup[i]) or np.isnan(sup[i - 1]):
            continue
        # A banda só aperta a favor da tendência; nunca afrouxa contra.
        if fecha[i - 1] <= sup[i - 1]:
            sup[i] = min(sup[i], sup[i - 1])
        if fecha[i - 1] >= inf[i - 1]:
            inf[i] = max(inf[i], inf[i - 1])

        if fecha[i] > sup[i - 1]:
            tendencia[i] = 1
        elif fecha[i] < inf[i - 1]:
            tendencia[i] = -1
        else:
            tendencia[i] = tendencia[i - 1]

    linha = np.where(tendencia == 1, inf, sup)
    return pd.DataFrame(
        {"supertrend": linha, "tendencia": tendencia, "banda_sup": sup, "banda_inf": inf},
        index=df.index,
    )


def retorno_log(preco: pd.Series, periodos: int = 1) -> pd.Series:
    p = _serie(preco)
    return np.log(p / p.shift(periodos))


def volatilidade_realizada(preco: pd.Series, janela: int = 24, periodos_ano: int = 24 * 365) -> pd.Series:
    """Desvio-padrão dos retornos log, anualizado. Em 1h, `periodos_ano` = 8760."""
    return retorno_log(preco).rolling(janela, min_periods=janela).std(ddof=1) * np.sqrt(periodos_ano)


def tabela_completa(df: pd.DataFrame) -> pd.DataFrame:
    """Monta de uma vez o conjunto de indicadores usado pelas estratégias."""
    out = df.copy()
    fecha = out["fechamento"]

    out["ema_9"] = ema(fecha, 9)
    out["ema_21"] = ema(fecha, 21)
    out["ema_50"] = ema(fecha, 50)
    out["ema_200"] = ema(fecha, 200)
    out["rsi_14"] = rsi(fecha, 14)

    m = macd(fecha)
    out["macd"] = m["macd"]
    out["macd_sinal"] = m["sinal"]
    out["macd_hist"] = m["histograma"]

    b = bollinger(fecha)
    out["bb_superior"] = b["superior"]
    out["bb_inferior"] = b["inferior"]
    out["bb_largura"] = b["largura"]
    out["bb_pct_b"] = b["pct_b"]

    out["atr_14"] = atr(out, 14)
    out["atr_pct"] = out["atr_14"] / fecha

    a = adx(out, 14)
    out["adx"] = a["adx"]
    out["di_mais"] = a["di_mais"]
    out["di_menos"] = a["di_menos"]

    e = estocastico(out)
    out["stoch_k"] = e["k"]
    out["stoch_d"] = e["d"]

    out["obv"] = obv(out)
    out["obv_ema_21"] = ema(out["obv"], 21)

    st = supertrend(out)
    out["supertrend"] = st["supertrend"]
    out["supertrend_tendencia"] = st["tendencia"]

    out["retorno_log"] = retorno_log(fecha)
    out["vol_realizada"] = volatilidade_realizada(fecha)

    return out
