"""Estratégia de reversão por RSI (sobrevendido compra, sobrecomprado vende)."""

from __future__ import annotations

import pandas as pd

from config.settings import Signal
from strategies.base import Strategy


class RSIStrategy(Strategy):
    name = "RSI"

    def __init__(self, oversold: float = 30.0, overbought: float = 70.0):
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "RSI" not in df.columns:
            return self._hold_frame(df)

        result = df.copy()
        signals = []
        # Margem extra para sinal "forte" sem violar o limiar base.
        strong_margin = 10.0
        for rsi in result["RSI"]:
            if pd.isna(rsi):
                signals.append(Signal.HOLD)
            elif rsi < self.oversold:
                if rsi < self.oversold - strong_margin:
                    signals.append(Signal.STRONG_BUY)
                else:
                    signals.append(Signal.BUY)
            elif rsi > self.overbought:
                if rsi > self.overbought + strong_margin:
                    signals.append(Signal.STRONG_SELL)
                else:
                    signals.append(Signal.SELL)
            else:
                signals.append(Signal.HOLD)

        result["signal"] = signals
        return result

    def explain(self, row: pd.Series) -> str:
        rsi = row.get("RSI", float("nan"))
        sig = row.get("signal", Signal.HOLD)
        return f"RSI={rsi:.1f}: sinal {sig} (oversold<{self.oversold}, overbought>{self.overbought})."
