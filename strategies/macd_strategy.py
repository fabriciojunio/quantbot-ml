"""Estratégia de MACD (cruzamento da linha MACD com a linha de sinal)."""

from __future__ import annotations

import pandas as pd

from config.settings import Signal
from strategies.base import Strategy


class MACDStrategy(Strategy):
    name = "MACD"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "MACD" not in df.columns or "MACD_signal" not in df.columns:
            return self._hold_frame(df)

        result = df.copy()
        macd = result["MACD"]
        macd_signal = result["MACD_signal"]
        diff = macd - macd_signal
        prev_diff = diff.shift(1)

        signals = []
        for d, pd_ in zip(diff, prev_diff):
            if pd.isna(d) or pd.isna(pd_):
                signals.append(Signal.HOLD)
            elif pd_ <= 0 < d:
                signals.append(Signal.STRONG_BUY)
            elif pd_ >= 0 > d:
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
        return f"MACD: sinal {sig} pelo cruzamento da linha MACD com a linha de sinal."
