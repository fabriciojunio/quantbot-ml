"""Estratégias de trading modulares."""
from strategies.base import BaseStrategy
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.ensemble_voting import EnsembleVotingStrategy
