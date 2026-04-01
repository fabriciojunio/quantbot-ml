"""
Geração e interpretação de sinais de Machine Learning.

Converte probabilidades do ensemble em sinais acionáveis
(STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL) com score e confiança.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import pandas as pd

from config.settings import Signal, ML_CONFIG, FEATURE_COLUMNS
from models.ensemble import EnsembleModel
from utils.logger import get_logger

logger = get_logger("quantbot.models.signals")


@dataclass
class MLSignal:
    """Sinal de ML com metadados para análise."""
    symbol: str
    score: float  # 0-100 (probabilidade de BUY)
    signal: Signal
    confidence: float  # 0-100 (concordância dos modelos)
    model_votes: Dict[str, str] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Serializa para dicionário."""
        return {
            "symbol": self.symbol,
            "score": self.score,
            "signal": self.signal.value,
            "confidence": self.confidence,
            "model_votes": self.model_votes,
            "top_features": dict(list(self.feature_importance.items())[:5]),
            "timestamp": self.timestamp.isoformat(),
        }


class SignalGenerator:
    """Gera sinais de ML a partir de modelos treinados."""

    def __init__(self, config=None):
        self.config = config or ML_CONFIG

    def generate(
        self,
        model: EnsembleModel,
        features: pd.DataFrame,
        symbol: str,
    ) -> MLSignal:
        """
        Gera sinal de ML para o ativo.

        Processo:
        1. Extrai features mais recentes
        2. Obtém probabilidades do ensemble
        3. Calcula score (0-100) e sinal
        4. Determina confiança pela concordância dos modelos
        5. Extrai feature importance

        Args:
            model: Modelo ensemble treinado
            features: DataFrame de features
            symbol: Símbolo do ativo

        Returns:
            MLSignal com score, sinal e metadados
        """
        if not model.is_fitted:
            logger.warning(f"{symbol}: modelo não treinado")
            return MLSignal(
                symbol=symbol, score=50.0,
                signal=Signal.HOLD, confidence=0.0,
            )

        # Seleciona features disponíveis
        available = [c for c in FEATURE_COLUMNS if c in features.columns]
        latest = features[available].iloc[[-1]].copy()

        # Trata NaN
        if latest.isna().any(axis=1).iloc[0]:
            latest = latest.ffill().fillna(0)

        # Probabilidades do ensemble
        probs = model.predict_proba(latest)
        buy_prob = probs[0, 1]  # Probabilidade de BUY
        score = round(buy_prob * 100, 1)

        # Determina sinal baseado nos thresholds
        signal = self._score_to_signal(score)

        # Confiança: concordância entre modelos
        model_votes = model.get_model_predictions(latest)
        confidence = self._calculate_confidence(model_votes)

        # Feature importance
        feat_imp = model.get_feature_importance()

        return MLSignal(
            symbol=symbol,
            score=score,
            signal=signal,
            confidence=round(confidence, 1),
            model_votes=model_votes,
            feature_importance=feat_imp,
        )

    def _score_to_signal(self, score: float) -> Signal:
        """Converte score numérico em sinal categórico."""
        if score >= self.config.strong_buy_threshold:
            return Signal.STRONG_BUY
        elif score >= self.config.buy_threshold:
            return Signal.BUY
        elif score >= self.config.hold_upper_threshold:
            return Signal.HOLD
        elif score >= self.config.sell_threshold:
            return Signal.SELL
        else:
            return Signal.STRONG_SELL

    @staticmethod
    def _calculate_confidence(votes: Dict[str, str]) -> float:
        """
        Calcula confiança baseada na concordância dos modelos.

        100% = todos concordam
        50% = metade discorda
        """
        if not votes:
            return 0.0

        total = len(votes)
        buy_votes = sum(1 for v in votes.values() if v == "BUY")
        majority = max(buy_votes, total - buy_votes)

        return (majority / total) * 100
