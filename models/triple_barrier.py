"""Triple Barrier Labeling + Meta-Labeling (López de Prado, AFML cap. 3).

Rotula cada observação por qual barreira é tocada primeiro dentro de um
horizonte: take-profit (+1), stop-loss (-1) ou expiração de tempo (0). As
barreiras são dinâmicas, escaladas pela volatilidade diária estimada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_daily_volatility(close: pd.Series, span: int = 20) -> pd.Series:
    """Volatilidade diária via EWMA dos retornos (sempre >= 0)."""
    returns = close.pct_change()
    vol = returns.ewm(span=span).std()
    return vol.abs()


class TripleBarrierLabeler:
    """Aplica o método das três barreiras a um DataFrame OHLCV."""

    def __init__(
        self,
        tp_mult: float = 2.0,
        sl_mult: float = 1.0,
        max_holding: int = 10,
        vol_span: int = 20,
    ):
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.max_holding = max_holding
        self.vol_span = vol_span

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        close = (df["Close"] if "Close" in df.columns else df["close"]).astype(float)
        vol = get_daily_volatility(close, self.vol_span).fillna(0.0)
        prices = close.values
        vols = vol.values
        n = len(prices)
        labels = np.zeros(n, dtype=int)

        for i in range(n):
            v = vols[i]
            if v <= 0:
                continue
            entry = prices[i]
            tp = entry * (1 + self.tp_mult * v)
            sl = entry * (1 - self.sl_mult * v)
            end = min(i + self.max_holding, n - 1)
            label = 0
            for j in range(i + 1, end + 1):
                if prices[j] >= tp:
                    label = 1
                    break
                if prices[j] <= sl:
                    label = -1
                    break
            labels[i] = label

        result = df.copy()
        result["tb_label"] = labels
        result["tb_volatility"] = vol.values
        return result


def meta_labels(primary: pd.Series, tb_label: pd.Series) -> pd.Series:
    """Meta-label: 1 se o sinal primário acertou a direção da barreira."""
    primary = pd.Series(primary)
    tb_label = pd.Series(tb_label).reindex(primary.index)
    agree = (np.sign(primary) == np.sign(tb_label)) & (primary != 0)
    return agree.astype(int)
