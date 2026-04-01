"""
Pipeline de treinamento com validação temporal.

Utiliza TimeSeriesSplit para evitar data leakage, que é um
dos erros mais comuns em ML aplicado a finanças.

Referência: López de Prado (2018), Cap. 7 - Cross-Validation in Finance.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config.settings import ML_CONFIG, FEATURE_COLUMNS
from models.ensemble import EnsembleModel
from utils.logger import get_logger

logger = get_logger("quantbot.models.trainer")


class ModelTrainer:
    """
    Gerencia o treinamento e avaliação de modelos por ativo.

    Mantém um modelo treinado para cada ativo, com métricas
    de validação cruzada temporal.
    """

    def __init__(self, config=None):
        self.config = config or ML_CONFIG
        self.trained_models: Dict[str, EnsembleModel] = {}
        self.cv_results: Dict[str, Dict] = {}

    def _prepare_data(
        self, features: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara dados para treino: filtra features, remove NaN.

        Returns:
            Tupla (X, y) filtrados e alinhados
        """
        # Seleciona apenas features configuradas
        available = [c for c in FEATURE_COLUMNS if c in features.columns]

        if len(available) < 10:
            raise ValueError(
                f"Features insuficientes: {len(available)} disponíveis, "
                f"mínimo: 10"
            )

        X = features[available].copy()
        y = target.copy()

        # Remove linhas com NaN em X ou y
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        return X, y

    def train_and_evaluate(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        symbol: str,
    ) -> Dict:
        """
        Treina modelo com cross-validation temporal e avalia.

        Processo:
        1. Prepara dados (filtra NaN, seleciona features)
        2. Executa TimeSeriesSplit com N folds
        3. Calcula métricas em cada fold
        4. Treina modelo final em todos os dados
        5. Armazena modelo e métricas

        Args:
            features: DataFrame com features calculadas
            target: Series com target binário
            symbol: Símbolo do ativo

        Returns:
            Dicionário com métricas de CV
        """
        X, y = self._prepare_data(features, target)

        if len(X) < self.config.min_training_samples:
            logger.warning(
                f"{symbol}: dados insuficientes ({len(X)} < "
                f"{self.config.min_training_samples})"
            )
            return {}

        logger.info(
            f"  Treinando {symbol}: {len(X)} amostras, "
            f"{len(X.columns)} features"
        )

        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)

        fold_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Treina modelo temporário para este fold
            fold_model = EnsembleModel(self.config)
            fold_model.fit(X_train, y_train)

            # Avalia
            y_pred = fold_model.predict(X_val)

            fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
            fold_metrics["precision"].append(
                precision_score(y_val, y_pred, zero_division=0)
            )
            fold_metrics["recall"].append(
                recall_score(y_val, y_pred, zero_division=0)
            )
            fold_metrics["f1"].append(
                f1_score(y_val, y_pred, zero_division=0)
            )

        # Métricas médias
        cv_results = {
            metric: {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            }
            for metric, values in fold_metrics.items()
        }

        # Treina modelo final em todos os dados
        final_model = EnsembleModel(self.config)
        final_model.fit(X, y)

        # Armazena
        self.trained_models[symbol] = final_model
        self.cv_results[symbol] = cv_results

        logger.info(
            f"  {symbol}: accuracy={cv_results['accuracy']['mean']:.3f} "
            f"(±{cv_results['accuracy']['std']:.3f}), "
            f"f1={cv_results['f1']['mean']:.3f}"
        )

        return cv_results

    def get_model(self, symbol: str) -> EnsembleModel:
        """Retorna modelo treinado para um ativo."""
        if symbol not in self.trained_models:
            raise KeyError(
                f"Modelo não encontrado para {symbol}. "
                f"Treinados: {list(self.trained_models.keys())}"
            )
        return self.trained_models[symbol]

    def get_cv_summary(self) -> pd.DataFrame:
        """Retorna resumo de CV de todos os ativos."""
        rows = []
        for symbol, results in self.cv_results.items():
            row = {"symbol": symbol}
            for metric, data in results.items():
                row[f"{metric}_mean"] = data["mean"]
                row[f"{metric}_std"] = data["std"]
            rows.append(row)

        return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()
