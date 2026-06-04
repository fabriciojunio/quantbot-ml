"""
Probability of Backtest Overfitting (PBO) via CSCV.

A pergunta que o PBO responde
-----------------------------
"Se eu escolho a estratégia que parece melhor no backtest (in-sample),
qual a probabilidade de ela ficar ABAIXO da mediana no período real
(out-of-sample)?"

Se essa probabilidade for alta (> 0.5), significa que escolher pela melhor
performance histórica não tem valor preditivo — você está fazendo overfitting.
PBO baixo (< 0.2) é o que se busca.

Método: Combinatorially Symmetric Cross-Validation (CSCV)
    1. Monte uma matriz M (T observações x N estratégias) de retornos.
    2. Divida as T linhas em S blocos.
    3. Para cada combinação de S/2 blocos como in-sample (IS) e o resto
       como out-of-sample (OOS):
        a. Escolha a estratégia n* com melhor performance IS.
        b. Veja o rank de n* no OOS.
        c. Calcule o "logit" do rank relativo.
    4. PBO = fração de combinações em que n* fica abaixo da mediana OOS.

Referências:
    - Bailey, D., Borwein, J., López de Prado, M., Zhu, Q. (2016).
      "The Probability of Backtest Overfitting." Journal of Computational
      Finance, 20(4).
"""

from itertools import combinations
from typing import Dict

import numpy as np


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = 10,
    metric: str = "sharpe",
) -> Dict[str, float]:
    """
    Calcula o PBO via CSCV.

    Args:
        returns_matrix: Matriz (T x N). Cada coluna é a série de retornos de
                        uma estratégia/configuração testada. T = observações.
        n_splits: Número de blocos S (deve ser par). Típico: 10 a 16.
        metric: "sharpe" ou "mean".

    Returns:
        Dicionário com:
            - pbo: Probability of Backtest Overfitting em [0, 1].
            - n_combinations: número de combinações avaliadas.
            - performance_degradation: mediana da diferença IS - OOS da melhor.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("returns_matrix deve ser 2D (T x N).")
    T, N = M.shape
    if N < 2:
        raise ValueError("São necessárias pelo menos 2 estratégias.")
    if n_splits % 2 != 0:
        raise ValueError("n_splits deve ser par.")
    if n_splits > T:
        raise ValueError("n_splits não pode exceder o número de observações.")

    def _perf(block: np.ndarray) -> np.ndarray:
        if metric == "sharpe":
            mu = block.mean(axis=0)
            sd = block.std(axis=0, ddof=1)
            sd = np.where(sd == 0, np.nan, sd)
            return mu / sd
        return block.mean(axis=0)

    # Divide as linhas em S blocos
    row_blocks = np.array_split(np.arange(T), n_splits)
    block_ids = list(range(n_splits))
    half = n_splits // 2

    logits = []
    degradations = []
    n_below_median = 0
    n_total = 0

    for is_blocks in combinations(block_ids, half):
        oos_blocks = [b for b in block_ids if b not in is_blocks]

        is_rows = np.concatenate([row_blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([row_blocks[b] for b in oos_blocks])

        is_perf = _perf(M[is_rows])
        oos_perf = _perf(M[oos_rows])

        if np.all(np.isnan(is_perf)):
            continue

        # Melhor estratégia in-sample
        n_star = int(np.nanargmax(is_perf))

        # Rank relativo dela no out-of-sample
        valid = ~np.isnan(oos_perf)
        if valid.sum() < 2:
            continue
        oos_valid = oos_perf[valid]
        # rank de n* entre as válidas (1 = pior)
        rank = np.sum(oos_valid <= oos_perf[n_star])
        relative_rank = rank / (len(oos_valid) + 1)
        relative_rank = min(max(relative_rank, 1e-6), 1 - 1e-6)

        logit = np.log(relative_rank / (1 - relative_rank))
        logits.append(logit)

        if relative_rank < 0.5:
            n_below_median += 1
        n_total += 1

        degradations.append(
            float(np.nanmax(is_perf) - oos_perf[n_star])
        )

    if n_total == 0:
        return {"pbo": float("nan"), "n_combinations": 0,
                "performance_degradation": float("nan")}

    pbo = n_below_median / n_total

    return {
        "pbo": round(pbo, 4),
        "n_combinations": n_total,
        "performance_degradation": round(float(np.median(degradations)), 4),
        "interpretation": _interpret_pbo(pbo),
    }


def _interpret_pbo(pbo: float) -> str:
    if pbo < 0.20:
        return "BAIXO RISCO de overfitting (PBO < 0.20)"
    if pbo < 0.50:
        return "RISCO MODERADO de overfitting (0.20 <= PBO < 0.50)"
    return "ALTO RISCO de overfitting — a selecao pelo backtest nao tem valor (PBO >= 0.50)"
