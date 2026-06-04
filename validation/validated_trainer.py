"""
ValidatedTrainer — treino com validação estatística honesta.

Substitui o TimeSeriesSplit "cru" do trainer original por:
    1. PurgedKFold (cross-validation sem vazamento temporal)
    2. Track record out-of-sample concatenado dos folds
    3. Deflated Sharpe Ratio sobre esse track record (detecta sorte)
    4. Probability of Backtest Overfitting sobre um sweep de configurações

A diferença prática: o trainer antigo responde "o modelo acerta X% das vezes".
Este responde "existe evidência estatística de que esse edge é real, ou é
provável que seja sorte/overfitting?" — que é a pergunta que importa antes de
arriscar capital.

Projetado para ser agnóstico ao modelo: aceita qualquer objeto com .fit(X, y)
e .predict(X) (a EnsembleModel do projeto se encaixa).
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .purged_cv import PurgedKFold
from .deflated_sharpe import evaluate_strategy
from .pbo import probability_of_backtest_overfitting


def _signal_to_returns(
    predictions: np.ndarray,
    forward_returns: np.ndarray,
) -> np.ndarray:
    """
    Converte sinais do modelo em retornos da estratégia.

    Sinal 1 (comprar) -> captura o retorno futuro do ativo.
    Sinal 0 (não operar) -> retorno zero (fica de fora).

    Esta é a ponte entre "o modelo classifica bem" e "a estratégia ganha
    dinheiro" — que NÃO são a mesma coisa.
    """
    predictions = np.asarray(predictions, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)
    return predictions * forward_returns


class ValidatedTrainer:
    """
    Treina e valida um modelo com rigor estatístico anti-overfitting.

    Args:
        model_factory: função que retorna uma nova instância do modelo
                       (cada fold treina um modelo limpo).
        n_splits: número de folds para o PurgedKFold.
        embargo_pct: fração de embargo após cada teste.
    """

    def __init__(
        self,
        model_factory: Callable[[], object],
        n_splits: int = 6,
        embargo_pct: float = 0.01,
    ):
        self.model_factory = model_factory
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        forward_returns: pd.Series,
        samples_info_sets: Optional[pd.Series] = None,
        n_trials: int = 1,
        periods_per_year: int = 252,
    ) -> Dict:
        """
        Cross-validation purgada + avaliação de skill (DSR).

        Args:
            X: features alinhadas pelo índice temporal.
            y: target binário (1 = oportunidade de compra).
            forward_returns: retorno futuro do ativo em cada ponto (alinhado a X).
            samples_info_sets: série (início -> fim do label) para purging
                               temporal. Se None, usa purge por posição.
            n_trials: nº de configurações que VOCÊ testou no total (para o DSR
                      penalizar data snooping). Seja honesto aqui.
            periods_per_year: 252 para dados diários.

        Returns:
            Relatório com métricas de classificação, track record OOS e DSR.
        """
        if not (len(X) == len(y) == len(forward_returns)):
            raise ValueError("X, y e forward_returns devem ter o mesmo tamanho.")

        cv = PurgedKFold(
            n_splits=self.n_splits,
            samples_info_sets=samples_info_sets,
            embargo_pct=self.embargo_pct,
        )

        oos_returns: List[float] = []
        fold_accuracy: List[float] = []
        fold_returns_by_fold: List[np.ndarray] = []

        for train_idx, test_idx in cv.split(X):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            fwd_test = forward_returns.iloc[test_idx].to_numpy()

            model = self.model_factory()
            model.fit(X_train, y_train)
            preds = np.asarray(model.predict(X_test))

            fold_accuracy.append(float(np.mean(preds == y_test.to_numpy())))

            strat_ret = _signal_to_returns(preds, fwd_test)
            oos_returns.extend(strat_ret.tolist())
            fold_returns_by_fold.append(strat_ret)

        oos_returns_arr = np.array(oos_returns)

        # Avaliação honesta do track record out-of-sample
        skill = evaluate_strategy(
            oos_returns_arr,
            n_trials=n_trials,
            sr_variance=self._estimate_sr_variance(fold_returns_by_fold),
            periods_per_year=periods_per_year,
        )

        return {
            "n_samples": int(len(X)),
            "n_folds": len(fold_accuracy),
            "mean_accuracy": round(float(np.mean(fold_accuracy)), 4) if fold_accuracy else 0.0,
            "std_accuracy": round(float(np.std(fold_accuracy)), 4) if fold_accuracy else 0.0,
            "oos_track_record": skill,
        }

    @staticmethod
    def _estimate_sr_variance(fold_returns: List[np.ndarray]) -> float:
        """Estima a variância dos Sharpe ratios entre folds (para o DSR)."""
        srs = []
        for r in fold_returns:
            r = r[~np.isnan(r)]
            if r.size >= 2 and r.std(ddof=1) > 1e-10:
                srs.append(r.mean() / r.std(ddof=1))
        if len(srs) < 2:
            return 0.5  # default conservador
        return float(np.var(srs, ddof=1))

    def evaluate_config_sweep(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        forward_returns: pd.Series,
        config_factories: List[Callable[[], object]],
        n_splits_pbo: int = 10,
    ) -> Dict:
        """
        Avalia um conjunto de configurações e calcula o PBO.

        Treina cada configuração, gera o track record de cada uma, monta a
        matriz (tempo x configurações) e calcula a Probability of Backtest
        Overfitting. Use isto quando você comparar várias variações do modelo.

        Args:
            config_factories: lista de funções, cada uma retornando um modelo
                              de uma configuração diferente.
            n_splits_pbo: blocos para o CSCV (par).

        Returns:
            Relatório com PBO e degradação de performance.
        """
        if len(config_factories) < 2:
            raise ValueError("PBO requer ao menos 2 configurações para comparar.")

        # Split temporal simples (treino na 1a metade, retorno OOS na 2a)
        split = len(X) // 2
        X_train, X_oos = X.iloc[:split], X.iloc[split:]
        y_train = y.iloc[:split]
        fwd_oos = forward_returns.iloc[split:].to_numpy()

        returns_columns = []
        for factory in config_factories:
            model = factory()
            model.fit(X_train, y_train)
            preds = np.asarray(model.predict(X_oos))
            returns_columns.append(_signal_to_returns(preds, fwd_oos))

        returns_matrix = np.column_stack(returns_columns)
        pbo_result = probability_of_backtest_overfitting(
            returns_matrix, n_splits=n_splits_pbo
        )
        return {
            "n_configs": len(config_factories),
            "pbo": pbo_result,
        }
