"""
Ensemble de modelos de Machine Learning.

Implementa um sistema de votação entre Random Forest, XGBoost
e Gradient Boosting para geração de sinais mais robustos.

Referências:
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Chen, T. & Guestrin, C. (2016). XGBoost. KDD '16.
- Krauss et al. (2017). Statistical arbitrage on the S&P 500.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from config.settings import ML_CONFIG
from utils.logger import get_logger

logger = get_logger("quantbot.models")

# Importação condicional de XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.info("XGBoost não disponível. Usando GradientBoosting como substituto.")


class EnsembleModel:
    """
    Ensemble de modelos para classificação binária (BUY/SELL).

    Combina múltiplos modelos através de votação ponderada
    pela probabilidade de cada um, gerando um score final
    de 0 a 100.
    """

    def __init__(self, config=None):
        self.config = config or ML_CONFIG
        self.models: Dict[str, object] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list = []
        self.is_fitted = False

    def _create_models(self) -> Dict[str, object]:
        """Instancia os modelos do ensemble."""
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                class_weight="balanced",
                random_state=self.config.random_state,
                n_jobs=-1,
            ),
        }

        if HAS_XGBOOST:
            models["xgboost"] = XGBClassifier(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                eval_metric="logloss",
                random_state=self.config.random_state,
                verbosity=0,
            )
        else:
            models["gradient_boosting"] = GradientBoostingClassifier(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.xgb_subsample,
                random_state=self.config.random_state,
            )

        return models

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        """
        Treina todos os modelos do ensemble.

        Args:
            X: Features (já filtradas por NaN)
            y: Target binário (0/1)

        Returns:
            self (para encadeamento)
        """
        if len(X) < self.config.min_training_samples:
            raise ValueError(
                f"Amostras insuficientes: {len(X)} < {self.config.min_training_samples}"
            )

        self.feature_names = list(X.columns)

        # Scaling
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns,
        )

        # Treina cada modelo
        self.models = self._create_models()
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            logger.debug(f"  Modelo treinado: {name}")

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidade média do ensemble.

        Args:
            X: Features para predição

        Returns:
            Array com probabilidades [P(SELL), P(BUY)]
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
        )

        all_probs = []
        for model in self.models.values():
            probs = model.predict_proba(X_scaled)
            all_probs.append(probs)

        # Média das probabilidades
        return np.mean(all_probs, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna predições binárias (0/1)."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retorna importância agregada das features.

        Combina feature_importances_ de todos os modelos.
        """
        if not self.is_fitted:
            return {}

        aggregated = {}
        count = 0

        for model in self.models.values():
            if hasattr(model, "feature_importances_"):
                for fname, imp in zip(self.feature_names, model.feature_importances_):
                    aggregated[fname] = aggregated.get(fname, 0) + imp
                count += 1

        if count == 0:
            return {}

        # Normaliza
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}

        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))

    def get_model_predictions(self, X: pd.DataFrame) -> Dict[str, str]:
        """Retorna predição individual de cada modelo."""
        if not self.is_fitted:
            return {}

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
        )

        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            predictions[name] = "BUY" if pred == 1 else "SELL"

        return predictions
