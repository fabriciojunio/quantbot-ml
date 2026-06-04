"""Walk-Forward Optimization.

Gera janelas deslizantes de treino/teste em ordem cronológica (sem
embaralhar) e agrega os resultados por fold. A consistência entre folds é
um sinal mais honesto de robustez do que um único backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    accuracy: float = 0.0
    return_pct: float = 0.0


@dataclass
class WalkForwardResult:
    folds: List[WalkForwardFold] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def avg_accuracy(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.accuracy for f in self.folds]))

    @property
    def avg_return(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.return_pct for f in self.folds]))

    @property
    def consistency(self) -> float:
        """Fração de folds com retorno positivo (estabilidade do edge)."""
        if not self.folds:
            return 0.0
        positive = sum(1 for f in self.folds if f.return_pct > 0)
        return positive / len(self.folds)


class WalkForwardValidator:
    """Produz índices de treino/teste em janelas deslizantes."""

    def __init__(self, train_size: int, test_size: int, step_size: int | None = None):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size

    def generate_splits(
        self, n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        start = 0
        while start + self.train_size + self.test_size <= n_samples:
            train_idx = np.arange(start, start + self.train_size)
            test_idx = np.arange(
                start + self.train_size,
                start + self.train_size + self.test_size,
            )
            splits.append((train_idx, test_idx))
            start += self.step_size
        return splits
