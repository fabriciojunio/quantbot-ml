"""Detecção de regime de mercado e sizing adaptativo.

Classifica cada barra em um regime (combinação de tendência e volatilidade)
e define quanto da posição-base é permitido operar. Em regimes de baixa
(bear) com alta volatilidade, o sistema zera a exposição automaticamente.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    NEUTRAL = "neutral"


# Fração da posição-base permitida por regime.
_REGIME_SCALE = {
    MarketRegime.BULL_LOW_VOL.value: 1.0,
    MarketRegime.BULL_HIGH_VOL.value: 0.5,
    MarketRegime.NEUTRAL.value: 0.3,
    MarketRegime.BEAR_LOW_VOL.value: 0.0,
    MarketRegime.BEAR_HIGH_VOL.value: 0.0,
}


class RegimeDetector:
    """Detecta regime via tendência (preço vs média) e volatilidade."""

    def __init__(self, trend_window: int = 50, vol_window: int = 20):
        self.trend_window = trend_window
        self.vol_window = vol_window

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        close = (df["Close"] if "Close" in df.columns else df["close"]).astype(float)

        sma = close.rolling(self.trend_window, min_periods=1).mean()
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window, min_periods=1).std().fillna(0.0)
        vol_threshold = vol.median()

        is_bull = close >= sma
        is_high_vol = vol > vol_threshold

        regimes = []
        scales = []
        for bull, high_vol in zip(is_bull, is_high_vol):
            if bull and not high_vol:
                reg = MarketRegime.BULL_LOW_VOL
            elif bull and high_vol:
                reg = MarketRegime.BULL_HIGH_VOL
            elif not bull and not high_vol:
                reg = MarketRegime.BEAR_LOW_VOL
            else:
                reg = MarketRegime.BEAR_HIGH_VOL
            regimes.append(reg.value)
            scales.append(_REGIME_SCALE[reg.value])

        result = df.copy()
        result["regime"] = regimes
        result["position_scale"] = scales
        result["can_trade"] = (pd.Series(scales, index=df.index) > 0).astype(bool)
        # Score contínuo: distância normalizada do preço à média.
        result["regime_score"] = ((close - sma) / sma.replace(0, np.nan)).fillna(0.0)
        return result


class AdaptivePositionSizer:
    """Converte regime em percentual de posição a partir de uma base."""

    def __init__(self, base_position_pct: float = 0.10):
        self.base_position_pct = base_position_pct

    def get_position_pct(self, regime_value: str) -> float:
        scale = _REGIME_SCALE.get(regime_value, 0.0)
        return self.base_position_pct * scale
