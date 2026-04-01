"""
Estratégia: Índice de Força Relativa (RSI).

COMPRA quando RSI < 30 (sobrevendido). VENDA quando RSI > 70.
Sinais fortes nos extremos (<20 / >80).
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from config.settings import Signal


class RSIStrategy(BaseStrategy):
    name = "RSI"
    description = "Compra em sobrevendido (RSI<30), vende em sobrecomprado (RSI>70)."

    def __init__(self, oversold=30, overbought=70, extreme_low=20, extreme_high=80):
        self.oversold = oversold
        self.overbought = overbought
        self.extreme_low = extreme_low
        self.extreme_high = extreme_high

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "RSI" not in df.columns:
            df["signal"] = Signal.HOLD
            return df

        df["signal"] = Signal.HOLD
        df.loc[df["RSI"] < self.extreme_low, "signal"] = Signal.STRONG_BUY
        df.loc[(df["RSI"] >= self.extreme_low) & (df["RSI"] < self.oversold), "signal"] = Signal.BUY
        df.loc[df["RSI"] > self.extreme_high, "signal"] = Signal.STRONG_SELL
        df.loc[(df["RSI"] <= self.extreme_high) & (df["RSI"] > self.overbought), "signal"] = Signal.SELL

        return df

    def explain(self, row: pd.Series) -> str:
        sig = row.get("signal", Signal.HOLD)
        rsi = row.get("RSI", 50)
        if sig in (Signal.BUY, Signal.STRONG_BUY):
            return f"[RSI] COMPRA — RSI={rsi:.1f} (sobrevendido)"
        elif sig in (Signal.SELL, Signal.STRONG_SELL):
            return f"[RSI] VENDA — RSI={rsi:.1f} (sobrecomprado)"
        return f"[RSI] MANTER — RSI={rsi:.1f} na zona neutra"
