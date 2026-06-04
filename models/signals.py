"""Sinal de ML emitido pelos modelos para a camada de risco/execução."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import Signal


@dataclass
class MLSignal:
    """Sinal direcional com score e confiança do modelo.

    Attributes:
        symbol: ativo ao qual o sinal se refere.
        score: pontuação 0-100 (quanto maior, mais comprador).
        signal: sinal discreto (BUY/SELL/HOLD/...).
        confidence: confiança 0-100 do modelo na predição.
    """
    symbol: str
    score: float
    signal: Signal
    confidence: float

    @property
    def is_actionable(self) -> bool:
        """True se o sinal não for HOLD."""
        return self.signal is not Signal.HOLD
