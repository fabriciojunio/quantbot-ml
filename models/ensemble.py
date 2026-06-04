"""Ensemble de classificadores para previsão direcional.

Combina Random Forest + Gradient Boosting (+ XGBoost quando disponível)
por média das probabilidades (soft voting). A escolha por ensemble reduz
variância e é menos suscetível a overfitting que um único modelo profundo.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from config.settings import ML_CONFIG
from utils.logger import get_logger

logger = get_logger("quantbot.models.ensemble")

try:  # XGBoost é opcional — degrada graciosamente se ausente.
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # pragma: no cover - depende do ambiente
    _HAS_XGB = False


class EnsembleModel:
    """Ensemble soft-voting para classificação binária (sobe / não sobe)."""

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state or ML_CONFIG.random_state
        self.is_fitted = False
        self._feature_names: list[str] = []
        self._build_estimators()

    def _build_estimators(self) -> None:
        self.estimators: Dict[str, object] = {
            "rf": RandomForestClassifier(
                n_estimators=ML_CONFIG.n_estimators,
                max_depth=ML_CONFIG.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=ML_CONFIG.n_estimators,
                max_depth=ML_CONFIG.max_depth,
                random_state=self.random_state,
            ),
        }
        if _HAS_XGB:
            self.estimators["xgb"] = XGBClassifier(
                n_estimators=ML_CONFIG.n_estimators,
                max_depth=ML_CONFIG.max_depth,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )

    @staticmethod
    def _clean(values) -> np.ndarray:
        """Remove inf/nan dos dados antes de alimentar os estimadores."""
        arr = np.asarray(values, dtype=float)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        if len(X) == 0:
            raise ValueError("X vazio: não há dados para treinar")
        self._feature_names = list(X.columns)
        X_clean = self._clean(X.values)
        y_clean = np.asarray(y).ravel()
        for name, est in self.estimators.items():
            est.fit(X_clean, y_clean)
            logger.debug("Estimador %s treinado", name)
        self.is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado: chame fit() antes de prever")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_clean = self._clean(np.asarray(X))
        probas = [est.predict_proba(X_clean) for est in self.estimators.values()]
        return np.mean(probas, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self) -> Dict[str, float]:
        """Importância média entre os estimadores, normalizada para somar 1."""
        self._check_fitted()
        total = np.zeros(len(self._feature_names))
        count = 0
        for est in self.estimators.values():
            imp = getattr(est, "feature_importances_", None)
            if imp is not None:
                total += np.asarray(imp, dtype=float)
                count += 1
        if count == 0 or total.sum() == 0:
            # fallback uniforme
            uniform = 1.0 / len(self._feature_names)
            return {f: uniform for f in self._feature_names}
        total /= total.sum()
        return dict(zip(self._feature_names, total))
