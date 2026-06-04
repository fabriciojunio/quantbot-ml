"""
CUSUM Filter + Diferenciação Fracionária.

CUSUM (Cumulative Sum) filtra eventos significativos,
eliminando ruído e reduzindo falsos sinais em 30-40%.

Diferenciação Fracionária torna séries estacionárias
sem perder toda a memória (López de Prado Cap. 5).

Referências:
    - López de Prado (2018) — Cap. 2.5 CUSUM Filter, Cap. 5 Frac Diff
    - Hudson & Thames — mlfinlab CUSUM implementation
    - Springer (2025) — CUSUM + Triple Barrier em crypto
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger("quantbot.data.cusum_filter")


def cusum_filter(
    close: pd.Series,
    threshold: Optional[float] = None,
    vol_lookback: int = 20,
    vol_multiplier: float = 2.5,
) -> pd.Series:
    """
    Filtro CUSUM simétrico de López de Prado.

    Detecta mudanças significativas na tendência, filtrando
    ruído do mercado. Só gera evento quando a variação
    acumulada excede um threshold baseado na volatilidade.

    Args:
        close: Série de preços de fechamento
        threshold: Limite fixo (se None, usa volatilidade dinâmica)
        vol_lookback: Janela para calcular volatilidade
        vol_multiplier: Multiplicador da volatilidade

    Returns:
        Series com eventos: +1 (breakout alta), -1 (breakout baixa), 0 (nada)
    """
    returns = close.pct_change().fillna(0)
    events = pd.Series(0, index=close.index, dtype=int)

    s_pos = 0.0
    s_neg = 0.0

    for i in range(1, len(returns)):
        if threshold is None:
            start = max(0, i - vol_lookback)
            vol = returns.iloc[start:i].std()
            h = vol * vol_multiplier if vol > 0 else 0.02
        else:
            h = threshold

        r = returns.iloc[i]
        s_pos = max(0, s_pos + r)
        s_neg = min(0, s_neg + r)

        if s_pos > h:
            events.iloc[i] = 1
            s_pos = 0

        if s_neg < -h:
            events.iloc[i] = -1
            s_neg = 0

    n_up = (events == 1).sum()
    n_down = (events == -1).sum()
    total = len(events)
    logger.info(
        f"CUSUM: {n_up} eventos de alta, {n_down} de baixa "
        f"({(n_up + n_down) / total * 100:.1f}% do total)"
    )
    return events


def cusum_event_timestamps(
    close: pd.Series,
    threshold: Optional[float] = None,
    vol_lookback: int = 20,
    vol_multiplier: float = 2.5,
) -> pd.DatetimeIndex:
    """Retorna timestamps dos eventos CUSUM (para uso com Triple Barrier)."""
    events = cusum_filter(close, threshold, vol_lookback, vol_multiplier)
    return events[events != 0].index


class FractionalDifferentiation:
    """
    Diferenciação Fracionária — López de Prado Cap. 5.

    Torna séries temporais estacionárias SEM perder toda a memória.
    d=0 → série original (não estacionária)
    d=0.3-0.5 → preserva memória parcial (ideal para ML)
    d=1 → diferenciação inteira (perde memória)
    """

    @staticmethod
    def get_weights(d: float, size: int) -> np.ndarray:
        """Calcula pesos para diferenciação fracionária."""
        w = [1.0]
        for k in range(1, size):
            w.append(-w[-1] * (d - k + 1) / k)
        return np.array(w[::-1]).reshape(-1, 1)

    @staticmethod
    def frac_diff(
        series: pd.Series, d: float = 0.4, threshold: float = 1e-5
    ) -> pd.Series:
        """Aplica diferenciação fracionária em uma série."""
        weights = FractionalDifferentiation.get_weights(d, len(series))
        weights_abs = np.abs(weights)
        cutoff = len(weights) - np.argmax(weights_abs[::-1] > threshold)

        result = pd.Series(index=series.index, dtype=float)
        for i in range(cutoff, len(series)):
            window = series.iloc[i - cutoff + 1 : i + 1].values
            w = weights[-(len(window)) :].flatten()
            if len(window) == len(w):
                result.iloc[i] = np.dot(w, window)
        return result

    @staticmethod
    def add_frac_diff_features(df: pd.DataFrame, d: float = 0.4) -> pd.DataFrame:
        """Adiciona versões fracionariamente diferenciadas ao DataFrame."""
        df = df.copy()
        if "Close" in df.columns:
            df["close_frac_diff"] = FractionalDifferentiation.frac_diff(df["Close"], d=d)
        if "Volume" in df.columns:
            df["volume_frac_diff"] = FractionalDifferentiation.frac_diff(
                df["Volume"].astype(float), d=d
            )
        return df
