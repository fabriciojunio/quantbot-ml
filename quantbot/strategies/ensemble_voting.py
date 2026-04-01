"""
Estratégia Ensemble — Votação Ponderada de SMA + RSI + MACD.

Cada estratégia vota com um peso. O sinal final é a média ponderada.
Opcionalmente inclui o modelo ML do quantbot com peso maior.

Referências:
    - Krauss et al. (2017) — ensemble methods em finanças
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, SIGNAL_NUMERIC, numeric_to_signal
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy
from config.settings import Signal


class EnsembleVotingStrategy(BaseStrategy):
    name = "Ensemble (Votação)"
    description = "Combina SMA, RSI, MACD por votação ponderada."

    def __init__(self, use_ml: bool = False):
        self.strategies = [
            (SMACrossoverStrategy(), 1.0),
            (RSIStrategy(), 1.0),
            (MACDStrategy(), 1.2),
        ]
        self.use_ml = use_ml
        self.individual_signals = {}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = 0.0

        for strategy, weight in self.strategies:
            result = strategy.generate_signals(df.copy())
            # Converte Signal enum para numérico
            sig_numeric = result["signal"].map(
                lambda s: SIGNAL_NUMERIC.get(s, 0)
            ).astype(float)
            self.individual_signals[strategy.name] = result["signal"]
            weighted_sum += sig_numeric * weight
            total_weight += weight

        avg_signal = weighted_sum / total_weight

        df["signal"] = avg_signal.map(numeric_to_signal)
        df["_ensemble_score"] = avg_signal

        return df

    def explain(self, row: pd.Series) -> str:
        sig = row.get("signal", Signal.HOLD)
        score = row.get("_ensemble_score", 0)

        parts = []
        for strategy, _ in self.strategies:
            if strategy.name in self.individual_signals:
                idx = row.name if hasattr(row, "name") else None
                if idx is not None and idx in self.individual_signals[strategy.name].index:
                    s = self.individual_signals[strategy.name].loc[idx]
                    val = s.value if hasattr(s, "value") else str(s)
                    parts.append(f"  {strategy.name}: {val}")

        votes = "\n".join(parts)

        if sig in (Signal.BUY, Signal.STRONG_BUY):
            return f"[Ensemble] COMPRA — Score={score:.2f}\n{votes}"
        elif sig in (Signal.SELL, Signal.STRONG_SELL):
            return f"[Ensemble] VENDA — Score={score:.2f}\n{votes}"
        return f"[Ensemble] MANTER — Score={score:.2f}\n{votes}"
