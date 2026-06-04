"""
Suíte de validação estatística anti-overfitting.

Reúne as ferramentas que separam um backtest acadêmico (que quase sempre
mente) de uma avaliação honesta de skill:

    - PurgedKFold / CombinatorialPurgedCV: cross-validation sem vazamento
      temporal (López de Prado, cap. 7 e 12).
    - Deflated Sharpe Ratio: corrige o Sharpe por nº de tentativas e
      não-normalidade (Bailey & López de Prado, 2014).
    - Probability of Backtest Overfitting: mede se escolher pela melhor
      performance histórica tem valor preditivo (Bailey et al., 2016).
"""

from .purged_cv import PurgedKFold, CombinatorialPurgedCV, get_train_times
from .deflated_sharpe import (
    sharpe_ratio,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    evaluate_strategy,
)
from .pbo import probability_of_backtest_overfitting
from .validated_trainer import ValidatedTrainer

__all__ = [
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "get_train_times",
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "evaluate_strategy",
    "probability_of_backtest_overfitting",
    "ValidatedTrainer",
]
