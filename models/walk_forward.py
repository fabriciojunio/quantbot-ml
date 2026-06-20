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
    """Produz índices de treino/teste em janelas deslizantes.

    Purging + embargo (López de Prado, AFML cap. 7)
    -----------------------------------------------
    Quando os rótulos olham para o futuro (ex.: triple barrier com
    ``max_holding`` dias), as últimas observações do treino "enxergam" preços
    que caem dentro do teste — isso é vazamento (leakage) e infla a acurácia.

    Para eliminar:
      - ``purge``: nº de barras descartadas no FIM do treino (= horizonte do
        rótulo). Remove os rótulos de treino que se sobrepõem ao teste.
      - ``embargo``: nº de barras de folga ENTRE treino e teste, além do purge,
        para neutralizar a autocorrelação de curto prazo.

    IMPORTANTE: ao usar rótulos com horizonte H (ex.: triple barrier
    ``max_holding=H``), passe ``purge=H`` para o resultado ser honesto.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        purge: int = 0,
        embargo: int = 0,
    ):
        if purge < 0 or embargo < 0:
            raise ValueError("purge e embargo não podem ser negativos")
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.purge = purge
        self.embargo = embargo

    def generate_splits(
        self, n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        gap = self.purge + self.embargo  # folga entre fim do treino e início do teste
        start = 0
        while start + self.train_size + gap + self.test_size <= n_samples:
            # Treino com as últimas `purge` barras removidas (purging).
            train_end = start + self.train_size - self.purge
            train_idx = np.arange(start, train_end)
            # Teste começa após o purge + embargo (sem sobreposição de rótulos).
            test_start = start + self.train_size + self.embargo
            test_idx = np.arange(test_start, test_start + self.test_size)
            splits.append((train_idx, test_idx))
            start += self.step_size
        return splits
