"""Pacote de configuração do QuantBot.

Centraliza enums, o universo de ativos e os parâmetros de dados, ML,
backtest e risco. Tudo que muda o comportamento do sistema deve ser
parametrizado aqui (e, em produção, sobrescrito por variáveis de ambiente).
"""

from config.settings import (
    Signal,
    Market,
    RiskProfile,
    AssetConfig,
    UNIVERSE,
    DATA_CONFIG,
    ML_CONFIG,
    BACKTEST_CONFIG,
    FEATURE_COLUMNS,
)

__all__ = [
    "Signal",
    "Market",
    "RiskProfile",
    "AssetConfig",
    "UNIVERSE",
    "DATA_CONFIG",
    "ML_CONFIG",
    "BACKTEST_CONFIG",
    "FEATURE_COLUMNS",
]
