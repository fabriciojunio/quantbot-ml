"""Gerenciador de risco: position sizing, stop-loss e take-profit.

Parametrizado pelo perfil de risco (config.RISK_PROFILES). O sizing é
proporcional à confiança do sinal e inversamente proporcional à
volatilidade do ativo, sempre limitado pela exposição máxima do perfil.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import RiskProfile, RISK_PROFILES
from models.signals import MLSignal


@dataclass
class Position:
    """Posição aberta em carteira."""
    symbol: str
    name: str
    market: str
    sector: str
    quantity: float
    entry_price: float
    current_price: float

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price


class RiskManager:
    """Aplica as regras de risco do perfil escolhido."""

    # Volatilidade-alvo usada como referência no dimensionamento.
    TARGET_VOLATILITY = 0.20

    def __init__(self, profile: RiskProfile = RiskProfile.MODERATE):
        self.profile = profile
        self.config = RISK_PROFILES[profile]

    def calculate_position_size(
        self,
        capital: float,
        signal: MLSignal,
        volatility: float,
    ) -> float:
        """Tamanho (em moeda) da posição, limitado pela exposição máxima.

        Escala pela confiança do sinal e pelo inverso da volatilidade,
        nunca ultrapassando capital * max_position_pct.
        """
        max_alloc = capital * self.config.max_position_pct
        confidence_factor = max(0.0, min(1.0, signal.confidence / 100.0))
        vol = max(volatility, 1e-6)
        vol_factor = min(1.0, self.TARGET_VOLATILITY / vol)
        size = max_alloc * confidence_factor * vol_factor
        # Piso mínimo para um sinal acionável, teto na exposição máxima.
        size = min(size, max_alloc)
        return max(size, 0.0)

    def should_stop_loss(self, position: Position) -> bool:
        return position.pnl_pct <= -self.config.stop_loss_pct

    def should_take_profit(self, position: Position) -> bool:
        return position.pnl_pct >= self.config.take_profit_pct
