"""
Classe base para estratégias de trading técnico.

Cada estratégia herda de BaseStrategy e implementa generate_signals().
Os sinais usam o enum Signal do config/settings.py do quantbot.

Referências:
    - Strategy Pattern (GoF)
    - López de Prado (2018) — Advances in Financial ML
"""

from abc import ABC, abstractmethod
import pandas as pd
from config.settings import Signal


# Mapeamento numérico para facilitar cálculos de votação
SIGNAL_NUMERIC = {
    Signal.STRONG_SELL: -2,
    Signal.SELL: -1,
    Signal.HOLD: 0,
    Signal.BUY: 1,
    Signal.STRONG_BUY: 2,
}

NUMERIC_TO_SIGNAL = {v: k for k, v in SIGNAL_NUMERIC.items()}


def numeric_to_signal(value: float) -> Signal:
    """Converte valor numérico para Signal mais próximo."""
    if value >= 1.5:
        return Signal.STRONG_BUY
    elif value >= 0.5:
        return Signal.BUY
    elif value <= -1.5:
        return Signal.STRONG_SELL
    elif value <= -0.5:
        return Signal.SELL
    return Signal.HOLD


class BaseStrategy(ABC):
    """
    Interface-base para estratégias de trading.

    Cada estratégia recebe um DataFrame com preços e indicadores
    e retorna o DataFrame com coluna 'signal' contendo valores Signal.
    """

    name: str = "BaseStrategy"
    description: str = ""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera sinais. Retorna df com coluna 'signal' (Signal enum values)."""
        pass

    def explain(self, row: pd.Series) -> str:
        """Explicação textual do sinal gerado para uma linha."""
        signal = row.get("signal", Signal.HOLD)
        if isinstance(signal, str):
            return f"[{self.name}] Sinal: {signal}"
        return f"[{self.name}] Sinal: {signal.value if hasattr(signal, 'value') else signal}"
