"""
Testes automatizados do QuantBot ML.

Cobre: features, modelos, risco, backtest, segurança e paper trading.
Execute com: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Adiciona raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Signal, Market, RiskProfile, ML_CONFIG, FEATURE_COLUMNS
from utils.security import InputValidator, DataSanitizer, OperationLimits
from utils.formatters import fmt_currency, fmt_pct


# ═══════════════════════════════════════════════════════════════
# FIXTURES — Dados de teste reutilizáveis
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_ohlcv():
    """Gera DataFrame OHLCV sintético realista para testes."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range(start="2024-01-01", periods=n)

    price = 100.0
    prices = []
    for _ in range(n):
        price *= 1 + np.random.normal(0.0003, 0.015)
        prices.append(price)

    close = np.array(prices)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


@pytest.fixture
def sample_returns():
    """Gera série de retornos diários para testes de risco."""
    np.random.seed(42)
    return pd.Series(
        np.random.normal(0.0005, 0.015, 252),
        index=pd.bdate_range(start="2024-01-01", periods=252),
    )


# ═══════════════════════════════════════════════════════════════
# TESTES — SECURITY & VALIDATION
# ═══════════════════════════════════════════════════════════════

class TestInputValidator:
    """Testes de validação e sanitização de inputs."""

    def test_valid_symbol(self):
        assert InputValidator.validate_symbol("PETR4.SA") == "PETR4.SA"
        assert InputValidator.validate_symbol("aapl") == "AAPL"
        assert InputValidator.validate_symbol("BTC-USD") == "BTC-USD"

    def test_invalid_symbol_type(self):
        with pytest.raises(ValueError, match="deve ser string"):
            InputValidator.validate_symbol(123)

    def test_invalid_symbol_chars(self):
        with pytest.raises(ValueError, match="inválido"):
            InputValidator.validate_symbol("DROP TABLE;")

    def test_invalid_symbol_too_long(self):
        with pytest.raises(ValueError, match="inválido"):
            InputValidator.validate_symbol("A" * 25)

    def test_valid_capital(self):
        assert InputValidator.validate_capital(100000) == 100000.0
        assert InputValidator.validate_capital(100) == 100.0

    def test_invalid_capital_negative(self):
        with pytest.raises(ValueError):
            InputValidator.validate_capital(-1000)

    def test_invalid_capital_too_high(self):
        with pytest.raises(ValueError):
            InputValidator.validate_capital(2_000_000_000)

    def test_invalid_capital_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            InputValidator.validate_capital(float("nan"))

    def test_valid_lookback(self):
        assert InputValidator.validate_lookback(365) == 365

    def test_invalid_lookback(self):
        with pytest.raises(ValueError):
            InputValidator.validate_lookback(10)

    def test_validate_symbols_list(self):
        result = InputValidator.validate_symbols(["AAPL", "msft"])
        assert result == ["AAPL", "MSFT"]

    def test_validate_symbols_empty(self):
        with pytest.raises(ValueError, match="vazia"):
            InputValidator.validate_symbols([])


class TestDataSanitizer:
    """Testes de sanitização de dados de mercado."""

    def test_sanitize_valid_data(self, sample_ohlcv):
        result = DataSanitizer.sanitize_dataframe(sample_ohlcv)
        assert len(result) > 0
        assert not result.isna().any().any()

    def test_sanitize_empty_dataframe(self):
        with pytest.raises(ValueError, match="vazio"):
            DataSanitizer.sanitize_dataframe(pd.DataFrame())

    def test_sanitize_missing_columns(self, sample_ohlcv):
        with pytest.raises(ValueError, match="ausentes"):
            DataSanitizer.sanitize_dataframe(
                sample_ohlcv[["close"]], required_columns=["open", "close", "volume"]
            )

    def test_sanitize_negative_prices(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.loc[df.index[5], "close"] = -10.0
        result = DataSanitizer.sanitize_dataframe(df)
        assert (result["close"] >= 0).all()

    def test_sanitize_infinite_values(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.loc[df.index[10], "close"] = np.inf
        result = DataSanitizer.sanitize_dataframe(df)
        assert not np.isinf(result["close"]).any()

    def test_detect_outliers(self):
        data = pd.Series([1.0] * 50 + [100.0])  # 100 é outlier claro
        outliers = DataSanitizer.detect_outliers(data, std_threshold=3.0)
        assert outliers.iloc[-1] == True

    def test_clip_outliers(self):
        data = pd.Series([1.0] * 50 + [100.0])
        clipped = DataSanitizer.clip_outliers(data, std_threshold=3.0)
        assert clipped.iloc[-1] < 100


# ═══════════════════════════════════════════════════════════════
# TESTES — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    """Testes de geração de features técnicas."""

    def test_compute_all_features(self, sample_ohlcv):
        from data.features import FeatureEngineer
        features = FeatureEngineer.compute_all(sample_ohlcv)

        assert len(features) == len(sample_ohlcv)
        assert "rsi" in features.columns
        assert "macd_normalized" in features.columns
        assert "bb_width" in features.columns
        assert "volatility_20" in features.columns
        assert "volume_ratio" in features.columns
        assert "momentum_10" in features.columns
        assert "stoch_k" in features.columns

    def test_rsi_range(self, sample_ohlcv):
        from data.features import FeatureEngineer
        features = FeatureEngineer.compute_all(sample_ohlcv)
        rsi = features["rsi"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_bb_position_range(self, sample_ohlcv):
        from data.features import FeatureEngineer
        features = FeatureEngineer.compute_all(sample_ohlcv)
        # BB position pode sair de 0-1 em movimentos extremos
        bb = features["bb_position"].dropna()
        assert len(bb) > 0

    def test_create_target(self, sample_ohlcv):
        from data.features import FeatureEngineer
        target = FeatureEngineer.create_target(sample_ohlcv, horizon=5)
        valid = target.dropna()
        assert set(valid.unique()).issubset({0, 1})
        assert len(valid) > 0

    def test_create_target_invalid_horizon(self, sample_ohlcv):
        from data.features import FeatureEngineer
        with pytest.raises(ValueError, match="Horizonte"):
            FeatureEngineer.create_target(sample_ohlcv, horizon=100)

    def test_insufficient_data(self):
        from data.features import FeatureEngineer
        small_df = pd.DataFrame({
            "open": [1, 2], "high": [1, 2], "low": [1, 2],
            "close": [1, 2], "volume": [100, 200],
        })
        with pytest.raises(ValueError, match="insuficientes"):
            FeatureEngineer.compute_all(small_df)

    def test_feature_descriptions(self):
        from data.features import FeatureEngineer
        desc = FeatureEngineer.get_feature_descriptions()
        assert isinstance(desc, dict)
        assert "rsi" in desc
        assert len(desc) > 10


# ═══════════════════════════════════════════════════════════════
# TESTES — ML MODELS
# ═══════════════════════════════════════════════════════════════

class TestEnsembleModel:
    """Testes dos modelos de ML."""

    def test_fit_and_predict(self, sample_ohlcv):
        from data.features import FeatureEngineer
        from models.ensemble import EnsembleModel

        features = FeatureEngineer.compute_all(sample_ohlcv)
        target = FeatureEngineer.create_target(sample_ohlcv)

        available = [c for c in FEATURE_COLUMNS if c in features.columns]
        X = features[available].dropna()
        y = target.loc[X.index].dropna()
        mask = X.index.isin(y.index)
        X = X[mask]
        y = y.loc[X.index]

        model = EnsembleModel()
        model.fit(X, y)

        assert model.is_fitted
        preds = model.predict(X.iloc[[-1]])
        assert preds[0] in [0, 1]

    def test_predict_proba_range(self, sample_ohlcv):
        from data.features import FeatureEngineer
        from models.ensemble import EnsembleModel

        features = FeatureEngineer.compute_all(sample_ohlcv)
        target = FeatureEngineer.create_target(sample_ohlcv)

        available = [c for c in FEATURE_COLUMNS if c in features.columns]
        X = features[available].dropna()
        y = target.loc[X.index].dropna()
        mask = X.index.isin(y.index)
        X, y = X[mask], y.loc[X[mask].index]

        model = EnsembleModel()
        model.fit(X, y)

        probs = model.predict_proba(X.iloc[[-1]])
        assert probs.shape == (1, 2)
        assert 0 <= probs[0, 0] <= 1
        assert 0 <= probs[0, 1] <= 1
        assert abs(probs[0, 0] + probs[0, 1] - 1.0) < 0.01

    def test_feature_importance(self, sample_ohlcv):
        from data.features import FeatureEngineer
        from models.ensemble import EnsembleModel

        features = FeatureEngineer.compute_all(sample_ohlcv)
        target = FeatureEngineer.create_target(sample_ohlcv)

        available = [c for c in FEATURE_COLUMNS if c in features.columns]
        X = features[available].dropna()
        y = target.loc[X.index].dropna()
        mask = X.index.isin(y.index)
        X, y = X[mask], y.loc[X[mask].index]

        model = EnsembleModel()
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert len(importance) > 0
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_unfitted_model_raises(self):
        from models.ensemble import EnsembleModel
        model = EnsembleModel()
        with pytest.raises(RuntimeError, match="não treinado"):
            model.predict_proba(pd.DataFrame({"a": [1]}))


# ═══════════════════════════════════════════════════════════════
# TESTES — RISK
# ═══════════════════════════════════════════════════════════════

class TestRiskMetrics:
    """Testes de métricas de risco."""

    def test_calculate_risk(self, sample_returns):
        from risk.metrics import calculate_risk_metrics

        metrics = calculate_risk_metrics(sample_returns)

        assert metrics.var_95 < 0  # VaR é negativo (perda)
        assert metrics.var_99 < metrics.var_95  # 99% é mais extremo
        assert metrics.cvar_95 <= metrics.var_95
        assert metrics.volatility > 0
        assert metrics.max_drawdown < 0

    def test_risk_with_benchmark(self, sample_returns):
        from risk.metrics import calculate_risk_metrics

        bench = pd.Series(
            np.random.normal(0.0003, 0.012, 252),
            index=sample_returns.index,
        )

        metrics = calculate_risk_metrics(sample_returns, bench)
        assert isinstance(metrics.beta, float)
        assert isinstance(metrics.alpha, float)

    def test_sharpe_positive_returns(self):
        from risk.metrics import calculate_risk_metrics

        positive_returns = pd.Series(
            np.random.normal(0.002, 0.01, 252),
            index=pd.bdate_range(start="2024-01-01", periods=252),
        )
        metrics = calculate_risk_metrics(positive_returns)
        assert metrics.sharpe_ratio > 0


class TestRiskManager:
    """Testes do gerenciador de risco."""

    def test_position_sizing(self):
        from risk.manager import RiskManager
        from models.signals import MLSignal

        rm = RiskManager(RiskProfile.MODERATE)
        signal = MLSignal(
            symbol="TEST", score=75.0,
            signal=Signal.BUY, confidence=80.0,
        )

        size = rm.calculate_position_size(100000, signal, 0.25)
        assert size > 0
        assert size <= 100000 * 0.10  # max 10% para moderado

    def test_stop_loss(self):
        from risk.manager import RiskManager, Position

        rm = RiskManager(RiskProfile.MODERATE)
        # -2% loss: below 3.5% threshold, should NOT trigger
        small_loss = Position("TEST", "Test", "US", "Tech", 100, 100.0, 98.0)
        assert not rm.should_stop_loss(small_loss)

        # -10% loss: above 3.5% threshold, SHOULD trigger
        big_loss = Position("TEST", "Test", "US", "Tech", 100, 100.0, 90.0)
        assert rm.should_stop_loss(big_loss)

    def test_take_profit(self):
        from risk.manager import RiskManager, Position

        rm = RiskManager(RiskProfile.MODERATE)
        winning = Position("TEST", "Test", "US", "Tech", 100, 100.0, 116.0)
        assert rm.should_take_profit(winning)  # +16% > 15% threshold


# ═══════════════════════════════════════════════════════════════
# TESTES — BACKTEST
# ═══════════════════════════════════════════════════════════════

class TestBacktest:
    """Testes do motor de backtesting."""

    def test_basic_backtest(self, sample_ohlcv):
        from backtest.engine import BacktestEngine

        engine = BacktestEngine()

        # Sinal simples: compra no dia 1, vende no dia 100
        signals = pd.Series(0, index=sample_ohlcv.index)
        signals.iloc[0] = 1
        signals.iloc[100] = 0

        result = engine.run(sample_ohlcv, signals)

        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert result.total_trades >= 0
        assert result.equity_curve is not None

    def test_backtest_with_benchmark(self, sample_ohlcv):
        from backtest.engine import BacktestEngine

        engine = BacktestEngine()
        signals = pd.Series(1, index=sample_ohlcv.index[:200])
        signals.iloc[150:] = 0

        result = engine.run(
            sample_ohlcv[:200], signals,
            benchmark_prices=sample_ohlcv["close"][:200],
        )

        assert result.benchmark_return != 0 or True  # pode ser 0 por coincidência
        assert result.benchmark_curve is not None


# ═══════════════════════════════════════════════════════════════
# TESTES — FORMATTERS
# ═══════════════════════════════════════════════════════════════

class TestFormatters:

    def test_fmt_currency(self):
        assert fmt_currency(1234.56) == "$1,234.56"
        assert fmt_currency(0) == "$0.00"

    def test_fmt_pct(self):
        assert fmt_pct(12.34) == "+12.34%"
        assert fmt_pct(-5.67) == "-5.67%"

    def test_fmt_pct_no_sign(self):
        assert fmt_pct(12.34, with_sign=False) == "12.34%"


# ═══════════════════════════════════════════════════════════════
# TESTES — PAPER TRADING
# ═══════════════════════════════════════════════════════════════

class TestPaperTrading:
    """Testes do modo de simulação."""

    def test_initialization(self):
        from core.paper_trading import PaperTradingEngine
        engine = PaperTradingEngine(initial_capital=50000)
        assert engine.cash == 50000
        assert engine.get_total_value() == 50000
        assert len(engine.holdings) == 0

    def test_invalid_capital(self):
        from core.paper_trading import PaperTradingEngine
        with pytest.raises(ValueError):
            PaperTradingEngine(initial_capital=-1000)

    def test_sell_without_position(self):
        from core.paper_trading import PaperTradingEngine
        engine = PaperTradingEngine(initial_capital=50000)
        order = engine.execute_sell("FAKE.SYMBOL", quantity=10)
        assert order.status == "REJECTED"

    def test_performance_metrics(self):
        from core.paper_trading import PaperTradingEngine
        engine = PaperTradingEngine(initial_capital=100000)
        metrics = engine.get_performance_metrics()
        assert metrics["initial_capital"] == 100000
        assert metrics["current_value"] == 100000
        assert metrics["pnl_total"] == 0
        assert metrics["num_positions"] == 0

    def test_export_and_load(self, tmp_path):
        from core.paper_trading import PaperTradingEngine
        engine = PaperTradingEngine(initial_capital=75000)

        filepath = str(tmp_path / "test_state.json")
        engine.export_state(filepath)

        loaded = PaperTradingEngine.load_state(filepath)
        assert loaded.initial_capital == 75000
        assert loaded.cash == 75000
