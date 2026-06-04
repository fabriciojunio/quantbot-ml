"""
Testes da suíte de validação estatística.

Estes testes verificam PROPRIEDADES estatísticas conhecidas — não apenas
que o código executa. Por exemplo: uma estratégia de puro ruído DEVE ter
DSR baixo; uma matriz de ruído puro DEVE ter PBO ~0.5.
"""

import numpy as np
import pandas as pd
import pytest

from validation import (
    PurgedKFold,
    CombinatorialPurgedCV,
    sharpe_ratio,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    evaluate_strategy,
    probability_of_backtest_overfitting,
)


# ═══════════════════════════════════════════════════════════════
# PurgedKFold
# ═══════════════════════════════════════════════════════════════

class TestPurgedKFold:
    def test_no_overlap_between_train_and_test(self):
        """Treino e teste nunca podem compartilhar índices."""
        X = pd.DataFrame(np.random.randn(200, 3))
        cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        for train_idx, test_idx in cv.split(X):
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_all_test_indices_covered(self):
        """A união dos testes deve cobrir todas as amostras."""
        X = pd.DataFrame(np.random.randn(200, 3))
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        all_test = np.concatenate([test for _, test in cv.split(X)])
        assert np.array_equal(np.sort(all_test), np.arange(200))

    def test_embargo_removes_samples(self):
        """Com embargo > 0, o treino deve ser menor do que sem embargo."""
        X = pd.DataFrame(np.random.randn(300, 3))
        cv_no = PurgedKFold(n_splits=5, embargo_pct=0.0)
        cv_emb = PurgedKFold(n_splits=5, embargo_pct=0.05)
        train_no = sum(len(tr) for tr, _ in cv_no.split(X))
        train_emb = sum(len(tr) for tr, _ in cv_emb.split(X))
        assert train_emb < train_no

    def test_purging_with_label_horizon(self):
        """Com horizonte de label, o purging deve remover overlaps temporais."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        X = pd.DataFrame(np.random.randn(n, 2), index=idx)
        # Cada label fecha 5 dias depois
        info = pd.Series(idx.shift(5, freq="D"), index=idx)
        cv = PurgedKFold(n_splits=4, samples_info_sets=info, embargo_pct=0.0)
        for train_idx, test_idx in cv.split(X):
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            assert len(train_idx) > 0

    def test_invalid_n_splits(self):
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)

    def test_invalid_embargo(self):
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=5, embargo_pct=1.5)


# ═══════════════════════════════════════════════════════════════
# CombinatorialPurgedCV
# ═══════════════════════════════════════════════════════════════

class TestCombinatorialPurgedCV:
    def test_generates_multiple_paths(self):
        """CPCV deve gerar C(N, k) splits."""
        from math import comb
        X = pd.DataFrame(np.random.randn(600, 3))
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.0)
        splits = list(cv.split(X))
        assert len(splits) == comb(6, 2)  # 15 splits

    def test_no_overlap(self):
        X = pd.DataFrame(np.random.randn(600, 3))
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        for train_idx, test_idx in cv.split(X):
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            CombinatorialPurgedCV(n_groups=2, n_test_groups=2)


# ═══════════════════════════════════════════════════════════════
# Sharpe / PSR / DSR
# ═══════════════════════════════════════════════════════════════

class TestSharpe:
    def test_sharpe_of_constant_positive_returns(self):
        """Retornos constantes positivos: std=0 -> Sharpe definido como 0."""
        assert sharpe_ratio(np.ones(100) * 0.01) == 0.0

    def test_sharpe_positive_for_good_returns(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 1000)  # média positiva
        assert sharpe_ratio(returns) > 0

    def test_psr_bounds(self):
        psr = probabilistic_sharpe_ratio(0.1, 0.0, 500, 0.0, 3.0)
        assert 0.0 <= psr <= 1.0

    def test_psr_higher_with_more_data(self):
        """Mais observações -> mais confiança (PSR maior) para mesmo SR."""
        psr_small = probabilistic_sharpe_ratio(0.1, 0.0, 50, 0.0, 3.0)
        psr_large = probabilistic_sharpe_ratio(0.1, 0.0, 5000, 0.0, 3.0)
        assert psr_large > psr_small

    def test_expected_max_sharpe_grows_with_trials(self):
        """Quanto mais tentativas, maior o Sharpe esperado por sorte."""
        sr_var = 0.25
        em_10 = expected_max_sharpe(10, sr_var)
        em_1000 = expected_max_sharpe(1000, sr_var)
        assert em_1000 > em_10 > 0

    def test_dsr_penalizes_many_trials(self):
        """
        Para o MESMO Sharpe observado, testar mais estratégias deve REDUZIR
        o DSR (mais chance de ter sido sorte).
        """
        dsr_few = deflated_sharpe_ratio(0.15, 0.25, n_trials=5,
                                        n_observations=1000, skewness=0, kurtosis=3)
        dsr_many = deflated_sharpe_ratio(0.15, 0.25, n_trials=500,
                                         n_observations=1000, skewness=0, kurtosis=3)
        assert dsr_few > dsr_many


class TestEvaluateStrategy:
    def test_pure_noise_has_low_dsr(self):
        """
        TESTE-CHAVE: uma estratégia de puro ruído (sem skill) deve ter
        DSR baixo e ser marcada como SEM skill.
        """
        rng = np.random.default_rng(7)
        noise_returns = rng.normal(0.0, 0.01, 500)  # média zero = sem edge
        result = evaluate_strategy(noise_returns, n_trials=100, sr_variance=0.5)
        assert result["has_skill"] is False
        assert result["dsr"] < 0.95

    def test_insufficient_data(self):
        result = evaluate_strategy(np.array([0.01]), n_trials=10, sr_variance=0.1)
        assert result["verdict"] == "DADOS INSUFICIENTES"

    def test_result_has_required_keys(self):
        rng = np.random.default_rng(1)
        result = evaluate_strategy(rng.normal(0, 0.01, 300), n_trials=10,
                                   sr_variance=0.2)
        for key in ["sharpe_annualized", "psr", "dsr", "verdict", "has_skill"]:
            assert key in result


# ═══════════════════════════════════════════════════════════════
# PBO
# ═══════════════════════════════════════════════════════════════

class TestPBO:
    def test_pure_noise_pbo_near_half(self):
        """
        TESTE-CHAVE: com N estratégias de puro ruído (nenhuma tem skill),
        escolher a "melhor" in-sample não ajuda no out-of-sample.

        A PROPRIEDADE estatística é que o PBO ESPERADO ~ 0.5. Em uma única
        amostra há grande variância, então verificamos a MÉDIA em várias
        seeds (que é o que a teoria garante).
        """
        pbos = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, 0.01, (1000, 20))
            result = probability_of_backtest_overfitting(noise, n_splits=10)
            pbos.append(result["pbo"])
        assert 0.4 < np.mean(pbos) < 0.6  # média ~0.5 conforme a teoria

    def test_genuine_skill_lowers_pbo(self):
        """
        Se UMA estratégia tem skill real consistente, o PBO deve cair
        (escolher pela melhor IS passa a ter valor preditivo).
        """
        rng = np.random.default_rng(99)
        T, N = 1000, 20
        M = rng.normal(0, 0.01, (T, N))
        # Estratégia 0 tem edge real e persistente
        M[:, 0] += rng.normal(0.004, 0.002, T)
        result = probability_of_backtest_overfitting(M, n_splits=10)
        assert result["pbo"] < 0.4

    def test_requires_even_splits(self):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.random.randn(100, 5), n_splits=7)

    def test_requires_multiple_strategies(self):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.random.randn(100, 1))
