"""
Walk-Forward Optimization — Validação mais robusta que TimeSeriesSplit.

O TimeSeriesSplit treina uma vez e testa uma vez. Walk-Forward
retreina o modelo periodicamente com dados mais recentes,
simulando o que aconteceria em produção real.

Isso é o que diferencia um backtest acadêmico de um backtest
que realmente prevê performance futura.

Como funciona:
    1. Treina no período [0, T]
    2. Testa no período [T, T+step]
    3. Move a janela: treina em [step, T+step]
    4. Testa em [T+step, T+2*step]
    5. Repete até o final dos dados

Por que é melhor:
    - Detecta overfitting (modelo que só funciona em um período)
    - Simula retreinamento real em produção
    - Gera múltiplas amostras out-of-sample
    - Fundos como Two Sigma e DE Shaw usam isso

Referências:
    - López de Prado (2018) — Cap. 7, Backtesting pitfalls
    - Bailey et al. (2014) — Probability of Backtest Overfitting
    - QuantifiedStrategies (2026) — Walk-forward testing
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("quantbot.models.walk_forward")


@dataclass
class WalkForwardFold:
    """Resultado de um fold do walk-forward."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    sharpe: float = 0.0
    return_pct: float = 0.0
    n_trades: int = 0


@dataclass
class WalkForwardResult:
    """Resultado agregado do walk-forward."""
    folds: List[WalkForwardFold] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def avg_accuracy(self) -> float:
        if not self.folds:
            return 0
        return np.mean([f.accuracy for f in self.folds])

    @property
    def std_accuracy(self) -> float:
        if not self.folds:
            return 0
        return np.std([f.accuracy for f in self.folds])

    @property
    def avg_sharpe(self) -> float:
        if not self.folds:
            return 0
        return np.mean([f.sharpe for f in self.folds])

    @property
    def avg_return(self) -> float:
        if not self.folds:
            return 0
        return np.mean([f.return_pct for f in self.folds])

    @property
    def consistency(self) -> float:
        """% de folds com retorno positivo."""
        if not self.folds:
            return 0
        positive = sum(1 for f in self.folds if f.return_pct > 0)
        return positive / len(self.folds)

    @property
    def is_overfit(self) -> bool:
        """
        Heurística de overfitting:
        - Se accuracy varia muito entre folds (std > 10%)
        - Se menos de 50% dos folds são positivos
        """
        return self.std_accuracy > 0.10 or self.consistency < 0.5

    def summary(self) -> str:
        overfit = "SIM" if self.is_overfit else "NAO"
        return (
            f"Walk-Forward: {self.n_folds} folds\n"
            f"  Accuracy: {self.avg_accuracy:.1%} (+/- {self.std_accuracy:.1%})\n"
            f"  Sharpe medio: {self.avg_sharpe:.2f}\n"
            f"  Retorno medio: {self.avg_return:+.2f}%\n"
            f"  Consistencia: {self.consistency:.0%} dos folds positivos\n"
            f"  Overfitting detectado: {overfit}"
        )


class WalkForwardValidator:
    """
    Walk-Forward Optimization para validação de modelos.

    Divide os dados em múltiplas janelas de treino+teste
    que avançam no tempo, retreinando o modelo em cada janela.
    """

    def __init__(
        self,
        train_size: int = 252,    # 1 ano de treino
        test_size: int = 63,      # 3 meses de teste
        step_size: int = 21,      # Avança 1 mês por fold
        min_train_size: int = 126, # Mínimo 6 meses de treino
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.min_train_size = min_train_size

    def generate_splits(
        self, n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Gera índices de treino/teste para cada fold.

        Returns:
            Lista de (train_indices, test_indices)
        """
        splits = []
        start = 0

        while start + self.train_size + self.test_size <= n_samples:
            train_idx = np.arange(start, start + self.train_size)
            test_idx = np.arange(
                start + self.train_size,
                min(start + self.train_size + self.test_size, n_samples)
            )

            if len(train_idx) >= min(self.min_train_size, self.train_size) and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

            start += self.step_size

        logger.info(f"Walk-Forward: {len(splits)} folds gerados")
        return splits

    def validate(
        self,
        df: pd.DataFrame,
        model_fn,
        feature_cols: List[str],
        target_col: str = "_target",
    ) -> WalkForwardResult:
        """
        Executa walk-forward validation completa.

        Args:
            df: DataFrame com features e target
            model_fn: Função que retorna um modelo sklearn-like
                      (deve ter .fit() e .predict_proba())
            feature_cols: Colunas de features
            target_col: Coluna do target

        Returns:
            WalkForwardResult com métricas por fold
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.preprocessing import StandardScaler

        # Limpa dados
        valid_cols = [c for c in feature_cols if c in df.columns]
        clean = df.dropna(subset=valid_cols + [target_col])

        if len(clean) < self.train_size + self.test_size:
            logger.warning("Dados insuficientes para walk-forward")
            return WalkForwardResult()

        splits = self.generate_splits(len(clean))
        result = WalkForwardResult()

        for fold_id, (train_idx, test_idx) in enumerate(splits):
            train = clean.iloc[train_idx]
            test = clean.iloc[test_idx]

            X_train = train[valid_cols]
            y_train = train[target_col]
            X_test = test[valid_cols]
            y_test = test[target_col]

            # Normaliza
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            # Treina
            model = model_fn()
            model.fit(X_train_sc, y_train)

            # Prediz
            y_pred = model.predict(X_test_sc)

            # Métricas
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=str(train.index[0]),
                train_end=str(train.index[-1]),
                test_start=str(test.index[0]),
                test_end=str(test.index[-1]),
                train_size=len(train),
                test_size=len(test),
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                f1=f1_score(y_test, y_pred, zero_division=0),
            )

            # Retorno simulado do fold
            if "Close" in test.columns:
                returns = test["Close"].pct_change().fillna(0)
                signals = pd.Series(y_pred, index=test.index)
                strategy_returns = returns * signals.shift(1).fillna(0)
                fold.return_pct = strategy_returns.sum() * 100
                fold.n_trades = (signals.diff().abs() > 0).sum()

                if strategy_returns.std() > 0:
                    fold.sharpe = (
                        strategy_returns.mean() / strategy_returns.std()
                        * np.sqrt(252)
                    )

            result.folds.append(fold)

            logger.debug(
                f"  Fold {fold_id}: acc={fold.accuracy:.1%}, "
                f"ret={fold.return_pct:+.2f}%, sharpe={fold.sharpe:.2f}"
            )

        logger.info(result.summary())
        return result
