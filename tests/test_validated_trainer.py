"""
Testes do ValidatedTrainer.

Estratégia de teste: criar dois mundos sintéticos —
  (A) um onde as features REALMENTE preveem o retorno futuro (sinal real)
  (B) um onde tudo é ruído (sem sinal)
e verificar que o trainer atribui skill ao primeiro e nega ao segundo.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from validation import ValidatedTrainer


def _make_signal_dataset(n=1500, seed=0):
    """Dataset onde feature f0 prevê o retorno futuro (edge real)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    # Retorno futuro depende de f0 (com ruído) -> existe edge aprendível
    noise = rng.normal(0, 0.01, n)
    forward = 0.004 * np.tanh(f0) + noise
    X = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2}, index=idx)
    y = pd.Series((forward > 0).astype(int), index=idx)
    fwd = pd.Series(forward, index=idx)
    return X, y, fwd


def _make_noise_dataset(n=1500, seed=1):
    """Dataset puramente aleatório: features não preveem nada."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    X = pd.DataFrame(rng.normal(0, 1, (n, 3)),
                     columns=["f0", "f1", "f2"], index=idx)
    forward = rng.normal(0, 0.01, n)  # retorno futuro independente das features
    y = pd.Series((forward > 0).astype(int), index=idx)
    fwd = pd.Series(forward, index=idx)
    return X, y, fwd


def _factory():
    return DecisionTreeClassifier(max_depth=4, random_state=0)


class TestValidatedTrainerSignal:
    def test_runs_and_returns_report(self):
        X, y, fwd = _make_signal_dataset()
        trainer = ValidatedTrainer(_factory, n_splits=6, embargo_pct=0.01)
        report = trainer.cross_validate(X, y, fwd, n_trials=5)
        assert "oos_track_record" in report
        assert report["n_folds"] >= 5
        assert "dsr" in report["oos_track_record"]

    def test_signal_dataset_beats_noise_dataset(self):
        """
        TESTE-CHAVE: o dataset COM sinal deve ter Sharpe OOS maior que o
        dataset de ruído. É a prova de que a ponte sinal->retorno funciona.
        """
        Xs, ys, fwds = _make_signal_dataset()
        Xn, yn, fwdn = _make_noise_dataset()
        trainer = ValidatedTrainer(_factory, n_splits=6)

        rep_signal = trainer.cross_validate(Xs, ys, fwds, n_trials=5)
        rep_noise = trainer.cross_validate(Xn, yn, fwdn, n_trials=5)

        sharpe_signal = rep_signal["oos_track_record"]["sharpe_annualized"]
        sharpe_noise = rep_noise["oos_track_record"]["sharpe_annualized"]
        assert sharpe_signal > sharpe_noise

    def test_noise_dataset_denied_skill(self):
        """Ruído puro NÃO pode receber selo de skill (DSR alto)."""
        Xn, yn, fwdn = _make_noise_dataset()
        trainer = ValidatedTrainer(_factory, n_splits=6)
        rep = trainer.cross_validate(Xn, yn, fwdn, n_trials=50)
        assert rep["oos_track_record"]["has_skill"] is False


class TestConfigSweepPBO:
    def test_pbo_on_config_sweep(self):
        X, y, fwd = _make_signal_dataset()
        trainer = ValidatedTrainer(_factory)
        configs = [
            lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
            lambda: DecisionTreeClassifier(max_depth=5, random_state=0),
            lambda: DecisionTreeClassifier(max_depth=7, random_state=0),
            lambda: DecisionTreeClassifier(max_depth=10, random_state=0),
        ]
        result = trainer.evaluate_config_sweep(X, y, fwd, configs, n_splits_pbo=10)
        assert "pbo" in result
        assert 0.0 <= result["pbo"]["pbo"] <= 1.0

    def test_sweep_requires_two_configs(self):
        X, y, fwd = _make_signal_dataset()
        trainer = ValidatedTrainer(_factory)
        try:
            trainer.evaluate_config_sweep(X, y, fwd, [_factory], n_splits_pbo=10)
            assert False, "deveria ter levantado ValueError"
        except ValueError:
            pass
