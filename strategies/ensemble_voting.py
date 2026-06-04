"""Ensemble por votação: agrega SMA, RSI e MACD num único sinal.

O score é a média das intensidades numéricas das estratégias-membro; o
sinal final vem da conversão desse score de volta para a escala discreta.
"""

from __future__ import annotations

import pandas as pd

from config.settings import Signal
from strategies.base import SIGNAL_NUMERIC, Strategy, numeric_to_signal
from strategies.macd_strategy import MACDStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.sma_crossover import SMACrossoverStrategy


class EnsembleVotingStrategy(Strategy):
    name = "Ensemble Voting"

    def __init__(self, weights: dict | None = None):
        self.members = {
            "SMA": SMACrossoverStrategy(),
            "RSI": RSIStrategy(),
            "MACD": MACDStrategy(),
        }
        self.weights = weights or {k: 1.0 for k in self.members}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        member_scores = pd.DataFrame(index=df.index)
        total_weight = 0.0
        for key, strat in self.members.items():
            signals = strat.generate_signals(df)["signal"]
            member_scores[key] = signals.map(SIGNAL_NUMERIC).fillna(0.0)
            member_scores[key] *= self.weights[key]
            total_weight += self.weights[key]

        score = member_scores.sum(axis=1) / max(total_weight, 1e-9)
        result["_ensemble_score"] = score.astype(float)
        result["signal"] = [numeric_to_signal(s) for s in score]
        return result

    def explain(self, row: pd.Series) -> str:
        score = row.get("_ensemble_score", 0.0)
        sig = row.get("signal", Signal.HOLD)
        return (
            f"Ensemble Voting: score médio {score:.2f} -> sinal {sig} "
            f"(votos combinados de SMA, RSI e MACD)."
        )
