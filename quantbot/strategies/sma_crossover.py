"""
Estratégia: Cruzamento de Médias Móveis Simples (SMA Crossover).

COMPRA quando SMA curta (20d) cruza acima da SMA longa (50d).
VENDA quando SMA curta cruza abaixo da SMA longa.
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from config.settings import Signal


class SMACrossoverStrategy(BaseStrategy):
    name = "SMA Crossover"
    description = "Compra/vende no cruzamento das médias móveis 20/50."

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        sma_s = df.get("SMA_20", df.get("SMA_short"))
        sma_l = df.get("SMA_50", df.get("SMA_long"))

        if sma_s is None or sma_l is None:
            df["signal"] = Signal.HOLD
            return df

        position = np.where(sma_s > sma_l, 1, 0)
        cross = pd.Series(position, index=df.index).diff()

        df["signal"] = Signal.HOLD
        df.loc[cross == 1, "signal"] = Signal.BUY
        df.loc[cross == -1, "signal"] = Signal.SELL

        return df

    def explain(self, row: pd.Series) -> str:
        sig = row.get("signal", Signal.HOLD)
        sma_s = row.get("SMA_20", row.get("SMA_short", 0))
        sma_l = row.get("SMA_50", row.get("SMA_long", 0))

        if sig == Signal.BUY:
            return f"[SMA] COMPRA — Curta ({sma_s:.2f}) cruzou acima da longa ({sma_l:.2f})"
        elif sig == Signal.SELL:
            return f"[SMA] VENDA — Curta ({sma_s:.2f}) cruzou abaixo da longa ({sma_l:.2f})"
        return f"[SMA] MANTER — Curta={sma_s:.2f}, Longa={sma_l:.2f}"
