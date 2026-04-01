"""
Gerenciador de risco do portfólio.

Controla position sizing, stop-loss, take-profit e limites
de alocação baseados no perfil de risco.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from config.settings import RiskProfile, RISK_PROFILES
from models.signals import MLSignal
from utils.logger import get_logger

logger = get_logger("quantbot.risk.manager")


@dataclass
class Position:
    """Representa uma posição no portfólio."""
    symbol: str
    name: str
    market: str
    sector: str
    quantity: float
    avg_price: float
    current_price: float = 0.0

    @property
    def value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost(self) -> float:
        return self.quantity * self.avg_price

    @property
    def pnl_value(self) -> float:
        return self.value - self.cost

    @property
    def pnl_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return ((self.current_price - self.avg_price) / self.avg_price) * 100


class RiskManager:
    """
    Gerencia risco e position sizing.

    Implementa:
    - Position sizing baseado em volatilidade e confiança
    - Stop-loss e take-profit dinâmicos
    - Limites de concentração
    - Verificação de drawdown
    """

    def __init__(self, profile: RiskProfile = RiskProfile.MODERATE):
        self.profile = profile
        self.limits = RISK_PROFILES[profile]
        logger.info(f"Risk Manager inicializado: perfil {profile.value}")

    def calculate_position_size(
        self,
        portfolio_value: float,
        signal: MLSignal,
        volatility: float,
    ) -> float:
        """
        Calcula tamanho ideal da posição.

        Fórmula: Size = MaxPos * ConfidenceFactor * ScoreFactor * VolAdjust

        Args:
            portfolio_value: Valor total do portfólio
            signal: Sinal ML com score e confiança
            volatility: Volatilidade anualizada do ativo

        Returns:
            Valor em $ para a posição
        """
        max_position = portfolio_value * self.limits["max_position_pct"]

        # Fator de confiança (0-1)
        confidence_factor = signal.confidence / 100

        # Fator de score (distância do neutro)
        score_factor = abs(signal.score - 50) / 50

        # Ajuste por volatilidade (reduz posição se vol alta)
        max_vol = self.limits["max_annual_volatility"]
        vol_adjustment = min(1.0, max_vol / max(volatility, 0.01))

        size = max_position * confidence_factor * score_factor * vol_adjustment

        return min(size, max_position)

    def should_stop_loss(self, position: Position) -> bool:
        """Verifica se deve acionar stop-loss."""
        threshold = -self.limits["stop_loss_pct"] * 100
        triggered = position.pnl_pct <= threshold

        if triggered:
            logger.warning(
                f"⛔ STOP-LOSS: {position.symbol} "
                f"P&L={position.pnl_pct:.2f}% < {threshold:.1f}%"
            )

        return triggered

    def should_take_profit(self, position: Position) -> bool:
        """Verifica se deve acionar take-profit."""
        threshold = self.limits["take_profit_pct"] * 100
        triggered = position.pnl_pct >= threshold

        if triggered:
            logger.info(
                f"🎯 TAKE-PROFIT: {position.symbol} "
                f"P&L={position.pnl_pct:.2f}% >= {threshold:.1f}%"
            )

        return triggered

    def check_concentration(
        self, positions: list, new_symbol: str, new_value: float
    ) -> bool:
        """
        Verifica se adicionar posição viola limites de concentração.

        Returns:
            True se a posição é permitida
        """
        total_value = sum(p.value for p in positions) + new_value
        if total_value == 0:
            return True

        new_weight = new_value / total_value

        if new_weight > self.limits["max_position_pct"]:
            logger.warning(
                f"Concentração excedida: {new_symbol} "
                f"seria {new_weight:.1%} > {self.limits['max_position_pct']:.1%}"
            )
            return False

        return True
