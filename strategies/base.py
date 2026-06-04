"""Base das estratégias técnicas e mapeamento sinal <-> número.

Cada estratégia recebe um DataFrame com indicadores e devolve uma cópia com
a coluna `signal` (enum Signal). O mapeamento numérico permite agregar
estratégias por votação (ensemble).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from config.settings import Signal

# Intensidade direcional de cada sinal.
SIGNAL_NUMERIC = {
    Signal.STRONG_BUY: 2,
    Signal.BUY: 1,
    Signal.HOLD: 0,
    Signal.SELL: -1,
    Signal.STRONG_SELL: -2,
}


def numeric_to_signal(value: float) -> Signal:
    """Converte um score contínuo de volta para um sinal discreto."""
    if value >= 1.5:
        return Signal.STRONG_BUY
    if value >= 0.5:
        return Signal.BUY
    if value <= -1.5:
        return Signal.STRONG_SELL
    if value <= -0.5:
        return Signal.SELL
    return Signal.HOLD


class Strategy(ABC):
    """Contrato comum a todas as estratégias."""

    name: str = "Strategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recebe indicadores e devolve cópia do df com coluna `signal`."""

    @abstractmethod
    def explain(self, row: pd.Series) -> str:
        """Explica em texto a decisão para uma linha do resultado."""

    @staticmethod
    def _hold_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Resultado neutro (tudo HOLD) quando faltam colunas necessárias."""
        result = df.copy()
        result["signal"] = Signal.HOLD
        return result
