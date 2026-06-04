"""Estratégias técnicas modulares e ensemble por votação."""

from strategies.base import (
    Strategy,
    SIGNAL_NUMERIC,
    numeric_to_signal,
)
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.ensemble_voting import EnsembleVotingStrategy

__all__ = [
    "Strategy",
    "SIGNAL_NUMERIC",
    "numeric_to_signal",
    "SMACrossoverStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "EnsembleVotingStrategy",
]
