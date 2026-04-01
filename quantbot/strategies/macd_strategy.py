"""
Estratégia: MACD (Moving Average Convergence Divergence).

COMPRA no cruzamento MACD acima da linha de sinal.
VENDA no cruzamento MACD abaixo da linha de sinal.
Sinais fortes quando histograma > 1 desvio padrão.
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from config.settings import Signal


class MACDStrategy(BaseStrategy):
    name = "MACD"
    description = "Compra/vende no cruzamento MACD com linha de sinal."

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "MACD" not in df.columns or "MACD_signal" not in df.columns:
            df["signal"] = Signal.HOLD
            return df

        diff = df["MACD"] - df["MACD_signal"]
        diff_prev = diff.shift(1)

        cross_up = (diff > 0) & (diff_prev <= 0)
        cross_down = (diff < 0) & (diff_prev >= 0)

        hist_col = "MACD_hist" if "MACD_hist" in df.columns else None
        hist_std = df[hist_col].std() if hist_col else 0.5
        strong = df[hist_col].abs() > hist_std if hist_col else pd.Series(False, index=df.index)

        df["signal"] = Signal.HOLD
        df.loc[cross_up & strong, "signal"] = Signal.STRONG_BUY
        df.loc[cross_up & ~strong, "signal"] = Signal.BUY
        df.loc[cross_down & strong, "signal"] = Signal.STRONG_SELL
        df.loc[cross_down & ~strong, "signal"] = Signal.SELL

        return df

    def explain(self, row: pd.Series) -> str:
        sig = row.get("signal", Signal.HOLD)
        macd = row.get("MACD", 0)
        macd_sig = row.get("MACD_signal", 0)
        hist = row.get("MACD_hist", 0)
        if sig in (Signal.BUY, Signal.STRONG_BUY):
            return f"[MACD] COMPRA — MACD ({macd:.4f}) cruzou acima do sinal ({macd_sig:.4f})"
        elif sig in (Signal.SELL, Signal.STRONG_SELL):
            return f"[MACD] VENDA — MACD ({macd:.4f}) cruzou abaixo do sinal ({macd_sig:.4f})"
        return f"[MACD] MANTER — MACD={macd:.4f}, Sinal={macd_sig:.4f}"
