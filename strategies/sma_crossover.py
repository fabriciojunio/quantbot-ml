"""Estratégia de cruzamento de médias móveis (SMA curta x SMA longa)."""

from __future__ import annotations

import pandas as pd

from config.settings import Signal
from strategies.base import Strategy


class SMACrossoverStrategy(Strategy):
    name = "SMA Crossover"

    def __init__(self, short_col: str = "SMA_short", long_col: str = "SMA_long"):
        # Aceita tanto SMA_short/SMA_long quanto SMA_20/SMA_50.
        self.short_col = short_col
        self.long_col = long_col

    def _resolve_columns(self, df: pd.DataFrame) -> tuple[str, str] | None:
        if self.short_col in df.columns and self.long_col in df.columns:
            return self.short_col, self.long_col
        if "SMA_20" in df.columns and "SMA_50" in df.columns:
            return "SMA_20", "SMA_50"
        return None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._resolve_columns(df)
        if cols is None:
            return self._hold_frame(df)

        short_col, long_col = cols
        result = df.copy()
        short = result[short_col]
        long = result[long_col]
        diff = short - long
        prev_diff = diff.shift(1)

        signals = []
        for d, pd_ in zip(diff, prev_diff):
            if pd.isna(d) or pd.isna(pd_):
                signals.append(Signal.HOLD)
            elif pd_ <= 0 < d:          # cruzamento de alta
                signals.append(Signal.STRONG_BUY)
            elif pd_ >= 0 > d:          # cruzamento de baixa
                signals.append(Signal.STRONG_SELL)
            elif d > 0:
                signals.append(Signal.BUY)
            elif d < 0:
                signals.append(Signal.SELL)
            else:
                signals.append(Signal.HOLD)

        result["signal"] = signals
        return result

    def explain(self, row: pd.Series) -> str:
        sig = row.get("signal", Signal.HOLD)
        return f"SMA Crossover: sinal {sig} pelo cruzamento das médias curta e longa."
