"""
Purged & Embargoed Cross-Validation + Combinatorial Purged CV (CPCV).

Por que isso existe
-------------------
O k-fold padrão do scikit-learn QUEBRA uma regra fundamental de finanças:
ele permite que o modelo "veja o futuro". Em séries temporais financeiras,
um label em t depende de preços em [t, t+h] (horizonte de holding). Se um
ponto de treino cai dentro dessa janela de um ponto de teste, há vazamento
(leakage): o modelo aprende indiretamente a resposta.

A solução de López de Prado (Advances in Financial ML, cap. 7):
    1. PURGING  — remove do treino qualquer amostra cujo label se sobreponha
                  no tempo ao intervalo de teste.
    2. EMBARGO  — bloqueia uma fração de amostras logo APÓS o teste, porque
                  a autocorrelação serial ainda pode vazar informação.

CPCV (Combinatorial Purged Cross-Validation) vai além: em vez de gerar um
único caminho histórico (como o walk-forward), ele combina grupos de teste
para gerar MÚLTIPLOS caminhos de backtest out-of-sample. Isso permite
estimar a distribuição de performance — e não confiar em um único número
que pode ser sorte.

Referências:
    - López de Prado, M. (2018). Advances in Financial Machine Learning,
      Cap. 7 ("Cross-Validation in Finance") e Cap. 12 (CPCV).
    - Wikipedia: Purged cross-validation (verbete técnico, 2017).
"""

from itertools import combinations
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


def get_train_times(
    samples_info_sets: pd.Series,
    test_times: pd.Series,
) -> pd.Series:
    """
    Remove (purga) do conjunto de treino as observações cujos labels se
    sobrepõem no tempo aos intervalos de teste.

    Args:
        samples_info_sets: Série indexada pelo início do evento, com valor =
                           timestamp do FIM do label (quando a barreira fecha).
        test_times: Mesma estrutura, mas só com os eventos de teste.

    Returns:
        Série de treino purgada (sem overlap temporal com o teste).
    """
    train = samples_info_sets.copy(deep=True)
    for start_test, end_test in test_times.items():
        # Caso 1: treino começa dentro do intervalo de teste
        df0 = train[(start_test <= train.index) & (train.index <= end_test)].index
        # Caso 2: treino termina dentro do intervalo de teste
        df1 = train[(start_test <= train) & (train <= end_test)].index
        # Caso 3: treino envolve completamente o intervalo de teste
        df2 = train[(train.index <= start_test) & (end_test <= train)].index
        train = train.drop(df0.union(df1).union(df2))
    return train


class PurgedKFold:
    """
    K-Fold com purging e embargo para séries temporais financeiras.

    Compatível com a interface de splitters do scikit-learn (tem .split()),
    mas respeita a estrutura temporal dos labels.
    """

    def __init__(
        self,
        n_splits: int = 5,
        samples_info_sets: Optional[pd.Series] = None,
        embargo_pct: float = 0.01,
    ):
        """
        Args:
            n_splits: Número de folds.
            samples_info_sets: Série (início -> fim do label). Se None, assume
                               labels pontuais (sem horizonte) — ainda aplica
                               embargo, mas o purging fica trivial.
            embargo_pct: Fração do total de amostras a embargar após cada teste.
                         López de Prado sugere ~1% (0.01) como ponto de partida.
        """
        if n_splits < 2:
            raise ValueError("n_splits deve ser >= 2")
        if not 0 <= embargo_pct < 1:
            raise ValueError("embargo_pct deve estar em [0, 1)")
        self.n_splits = n_splits
        self.samples_info_sets = samples_info_sets
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Gera índices (train, test) purgados e com embargo."""
        if self.samples_info_sets is not None:
            if not X.index.equals(self.samples_info_sets.index):
                raise ValueError(
                    "X e samples_info_sets devem ter o mesmo índice."
                )

        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.embargo_pct)

        test_starts = [
            (i[0], i[-1] + 1)
            for i in np.array_split(indices, self.n_splits)
        ]

        for start, end in test_starts:
            test_indices = indices[start:end]

            if self.samples_info_sets is None:
                # Sem horizonte de label: purge simplificado por posição
                train_indices = np.concatenate((
                    indices[:max(0, start - embargo)],
                    indices[min(X.shape[0], end + embargo):],
                ))
                yield train_indices, test_indices
                continue

            # Com horizonte: purging temporal completo
            t1 = self.samples_info_sets
            test_t1 = t1.iloc[test_indices]
            test_start_time = test_t1.index.min()
            test_end_time = test_t1.max()

            # Máximo timestamp do label de teste (para embargo)
            max_test_idx = test_indices.max()
            embargo_end_idx = min(X.shape[0], max_test_idx + embargo + 1)

            train_mask = np.ones(X.shape[0], dtype=bool)
            train_mask[test_indices] = False

            # Purge: remove treino que se sobrepõe ao período de teste
            for i in indices[train_mask]:
                evt_start = t1.index[i]
                evt_end = t1.iloc[i]
                overlaps = (
                    (evt_start <= test_end_time)
                    and (evt_end >= test_start_time)
                )
                if overlaps:
                    train_mask[i] = False

            # Embargo: bloqueia amostras logo após o teste
            if embargo > 0:
                train_mask[max_test_idx + 1:embargo_end_idx] = False

            train_indices = indices[train_mask]
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Em vez de 1 caminho de backtest (walk-forward), gera C(N, k) combinações
    de grupos de teste, produzindo MÚLTIPLOS caminhos out-of-sample. Cada
    caminho é purgado e embargado. O resultado é uma DISTRIBUIÇÃO de Sharpe
    ratios — base para calcular a Probability of Backtest Overfitting.

    Args:
        n_groups: Número total de grupos (N). Típico: 6 a 10.
        n_test_groups: Grupos por teste (k). Típico: 2.
        embargo_pct: Fração de embargo.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
    ):
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups deve ser < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    @property
    def n_paths(self) -> int:
        """Número de caminhos de backtest gerados."""
        from math import comb
        n_splits = comb(self.n_groups, self.n_test_groups)
        # Cada split contribui n_test_groups grupos; cada grupo participa de
        # C(N-1, k-1) splits. O número de caminhos completos é:
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def split(
        self, X: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Gera todos os splits combinatórios purgados."""
        indices = np.arange(X.shape[0])
        groups = np.array_split(indices, self.n_groups)
        embargo = int(X.shape[0] * self.embargo_pct)

        for test_group_ids in combinations(range(self.n_groups), self.n_test_groups):
            test_indices = np.concatenate([groups[g] for g in test_group_ids])
            test_indices = np.sort(test_indices)

            train_mask = np.ones(X.shape[0], dtype=bool)
            train_mask[test_indices] = False

            # Purge + embargo nas bordas de cada bloco de teste contíguo
            for g in test_group_ids:
                g_start, g_end = groups[g][0], groups[g][-1]
                lo = max(0, g_start - embargo)
                hi = min(X.shape[0], g_end + embargo + 1)
                train_mask[lo:hi] = train_mask[lo:hi] & False
            # Restaura grupos de treino que não tocam o embargo
            train_mask[test_indices] = False

            train_indices = indices[train_mask]
            yield train_indices, test_indices
