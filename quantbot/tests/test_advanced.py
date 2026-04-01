"""
Testes dos módulos institucionais avançados.

Cobertura:
    - Triple Barrier + Meta-Labeling
    - Regime Detection
    - Walk-Forward Optimization
    - Dynamic Trailing Stop
    - CUSUM Filter + Fractional Differentiation
    - Macro Data (BCB + FED + Fundamentus)
    - Correlação / Monte Carlo / Análise Estatística
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def price_series():
    """Série de preços sintética com tendência e ruído."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    trend = np.linspace(100, 120, n)
    noise = np.cumsum(np.random.randn(n) * 1.5)
    close = trend + noise
    close = np.maximum(close, 10)
    return pd.Series(close, index=dates, name="Close")


@pytest.fixture
def ohlcv_df(price_series):
    """DataFrame OHLCV completo."""
    close = price_series.values
    return pd.DataFrame({
        "Open": close - np.random.uniform(0, 1, len(close)),
        "High": close + np.random.uniform(0, 2, len(close)),
        "Low": close - np.random.uniform(0, 2, len(close)),
        "Close": close,
        "Volume": np.random.randint(500_000, 5_000_000, len(close)),
    }, index=price_series.index)


# ═══════════════════════════════════════════════════════════════
# CUSUM FILTER
# ═══════════════════════════════════════════════════════════════

class TestCUSUMFilter:
    def test_returns_correct_shape(self, price_series):
        from data.cusum_filter import cusum_filter
        events = cusum_filter(price_series)
        assert len(events) == len(price_series)

    def test_values_are_valid(self, price_series):
        from data.cusum_filter import cusum_filter
        events = cusum_filter(price_series)
        assert set(events.unique()).issubset({-1, 0, 1})

    def test_generates_events(self, price_series):
        from data.cusum_filter import cusum_filter
        events = cusum_filter(price_series)
        assert (events != 0).sum() > 0

    def test_fixed_threshold(self, price_series):
        from data.cusum_filter import cusum_filter
        events = cusum_filter(price_series, threshold=0.05)
        assert len(events) == len(price_series)

    def test_timestamps(self, price_series):
        from data.cusum_filter import cusum_event_timestamps
        ts = cusum_event_timestamps(price_series)
        assert len(ts) > 0
        assert all(t in price_series.index for t in ts)

    def test_higher_threshold_fewer_events(self, price_series):
        from data.cusum_filter import cusum_filter
        events_low = cusum_filter(price_series, threshold=0.01)
        events_high = cusum_filter(price_series, threshold=0.10)
        assert (events_low != 0).sum() >= (events_high != 0).sum()


class TestFractionalDifferentiation:
    def test_frac_diff_output_shape(self, price_series):
        from data.cusum_filter import FractionalDifferentiation
        result = FractionalDifferentiation.frac_diff(price_series, d=0.4)
        assert len(result) == len(price_series)

    def test_add_features(self, ohlcv_df):
        from data.cusum_filter import FractionalDifferentiation
        result = FractionalDifferentiation.add_frac_diff_features(ohlcv_df)
        assert "close_frac_diff" in result.columns
        assert "volume_frac_diff" in result.columns


# ═══════════════════════════════════════════════════════════════
# TRIPLE BARRIER
# ═══════════════════════════════════════════════════════════════

class TestTripleBarrier:
    def test_labels_shape(self, ohlcv_df):
        from models.triple_barrier import TripleBarrierLabeler
        labeler = TripleBarrierLabeler(tp_mult=2.0, sl_mult=1.0)
        result = labeler.fit_transform(ohlcv_df)
        assert "tb_label" in result.columns
        assert "tb_volatility" in result.columns
        assert len(result) == len(ohlcv_df)

    def test_label_values_valid(self, ohlcv_df):
        from models.triple_barrier import TripleBarrierLabeler
        labeler = TripleBarrierLabeler()
        result = labeler.fit_transform(ohlcv_df)
        assert set(result["tb_label"].unique()).issubset({-1, 0, 1})

    def test_meta_labels(self, ohlcv_df):
        from models.triple_barrier import TripleBarrierLabeler, meta_labels
        labeler = TripleBarrierLabeler()
        result = labeler.fit_transform(ohlcv_df)
        primary = pd.Series(np.random.choice([-1, 0, 1], len(ohlcv_df)), index=ohlcv_df.index)
        ml = meta_labels(primary, result["tb_label"])
        assert set(ml.unique()).issubset({0, 1})

    def test_volatility_positive(self, ohlcv_df):
        from models.triple_barrier import get_daily_volatility
        vol = get_daily_volatility(ohlcv_df["Close"])
        valid = vol.dropna()
        assert (valid >= 0).all()


# ═══════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════

class TestRegimeDetection:
    def test_detect_returns_columns(self, ohlcv_df):
        from models.regime import RegimeDetector
        det = RegimeDetector()
        result = det.detect(ohlcv_df)
        assert "regime" in result.columns
        assert "can_trade" in result.columns
        assert "position_scale" in result.columns
        assert "regime_score" in result.columns

    def test_regime_values_valid(self, ohlcv_df):
        from models.regime import RegimeDetector, MarketRegime
        det = RegimeDetector()
        result = det.detect(ohlcv_df)
        valid_regimes = {r.value for r in MarketRegime}
        for val in result["regime"].unique():
            assert val in valid_regimes

    def test_position_scale_range(self, ohlcv_df):
        from models.regime import RegimeDetector
        det = RegimeDetector()
        result = det.detect(ohlcv_df)
        assert (result["position_scale"] >= 0).all()
        assert (result["position_scale"] <= 1).all()

    def test_can_trade_is_bool(self, ohlcv_df):
        from models.regime import RegimeDetector
        det = RegimeDetector()
        result = det.detect(ohlcv_df)
        assert result["can_trade"].dtype == bool

    def test_adaptive_sizer(self):
        from models.regime import AdaptivePositionSizer, MarketRegime
        sizer = AdaptivePositionSizer(base_position_pct=0.10)
        assert sizer.get_position_pct(MarketRegime.BULL_LOW_VOL.value) == 0.10
        assert sizer.get_position_pct(MarketRegime.BULL_HIGH_VOL.value) == 0.05
        assert sizer.get_position_pct(MarketRegime.BEAR_HIGH_VOL.value) == 0.0


# ═══════════════════════════════════════════════════════════════
# DYNAMIC TRAILING STOP
# ═══════════════════════════════════════════════════════════════

class TestDynamicStop:
    def test_open_and_update(self):
        from risk.dynamic_stop import DynamicStopManager
        dsm = DynamicStopManager()
        dsm.open_position("AAPL", 150.0, "2024-01-01", atr=3.0)
        assert "AAPL" in dsm.positions
        assert dsm.positions["AAPL"].current_stop < 150.0

    def test_trailing_stop_rises(self):
        from risk.dynamic_stop import DynamicStopManager
        dsm = DynamicStopManager()
        dsm.open_position("AAPL", 150.0, "2024-01-01", atr=3.0)
        initial_stop = dsm.positions["AAPL"].current_stop
        dsm.update("AAPL", 160.0, 2.8)
        assert dsm.positions["AAPL"].current_stop >= initial_stop

    def test_stop_never_decreases(self):
        from risk.dynamic_stop import DynamicStopManager
        dsm = DynamicStopManager()
        dsm.open_position("AAPL", 150.0, "2024-01-01", atr=3.0)
        dsm.update("AAPL", 160.0, 2.8)
        high_stop = dsm.positions["AAPL"].current_stop
        dsm.update("AAPL", 155.0, 3.0)  # Price drops but stop shouldn't
        assert dsm.positions["AAPL"].current_stop >= high_stop

    def test_stop_triggered(self):
        from risk.dynamic_stop import DynamicStopManager
        dsm = DynamicStopManager()
        dsm.open_position("AAPL", 150.0, "2024-01-01", atr=3.0)
        result = dsm.update("AAPL", 130.0, 3.0)
        assert result == "stop_triggered"

    def test_time_exit(self):
        from risk.dynamic_stop import DynamicStopManager
        dsm = DynamicStopManager(max_holding_days=5)
        dsm.open_position("AAPL", 150.0, "2024-01-01", atr=3.0)
        for _ in range(4):
            dsm.update("AAPL", 152.0, 3.0)
        result = dsm.update("AAPL", 152.0, 3.0)
        assert result == "time_exit"

    def test_breakeven(self):
        from risk.dynamic_stop import DynamicStopManager
        dsm = DynamicStopManager(breakeven_trigger=0.02)
        dsm.open_position("AAPL", 100.0, "2024-01-01", atr=2.0)
        result = dsm.update("AAPL", 103.0, 2.0)  # +3% > 2% trigger
        assert result == "breakeven_set"
        assert dsm.positions["AAPL"].current_stop >= 100.0

    def test_close_position(self):
        from risk.dynamic_stop import DynamicStopManager
        dsm = DynamicStopManager()
        dsm.open_position("AAPL", 150.0, "2024-01-01", atr=3.0)
        pos = dsm.close_position("AAPL")
        assert pos is not None
        assert "AAPL" not in dsm.positions

    def test_get_all_stops(self):
        from risk.dynamic_stop import DynamicStopManager
        dsm = DynamicStopManager()
        dsm.open_position("AAPL", 150.0, "2024-01-01", atr=3.0)
        dsm.open_position("TSLA", 200.0, "2024-01-01", atr=5.0)
        stops = dsm.get_all_stops()
        assert "AAPL" in stops
        assert "TSLA" in stops


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD
# ═══════════════════════════════════════════════════════════════

class TestWalkForward:
    def test_generate_splits(self):
        from models.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator(train_size=100, test_size=30, step_size=30)
        splits = wfv.generate_splits(300)
        assert len(splits) > 0
        for train, test in splits:
            assert len(train) >= 100
            assert len(test) > 0
            assert train[-1] < test[0]  # No overlap

    def test_no_splits_insufficient_data(self):
        from models.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator(train_size=200, test_size=50)
        splits = wfv.generate_splits(100)
        assert len(splits) == 0

    def test_result_properties(self):
        from models.walk_forward import WalkForwardResult, WalkForwardFold
        result = WalkForwardResult(folds=[
            WalkForwardFold(0, "", "", "", "", 100, 30, accuracy=0.6, return_pct=5.0),
            WalkForwardFold(1, "", "", "", "", 100, 30, accuracy=0.55, return_pct=-2.0),
            WalkForwardFold(2, "", "", "", "", 100, 30, accuracy=0.65, return_pct=3.0),
        ])
        assert result.n_folds == 3
        assert 0.5 < result.avg_accuracy < 0.7
        assert result.consistency == 2 / 3


# ═══════════════════════════════════════════════════════════════
# MACRO DATA
# ═══════════════════════════════════════════════════════════════

class TestMacroData:
    def test_bcb_series_codes(self):
        from data.macro_data import BancoCentralAPI
        assert BancoCentralAPI.SERIES["selic_meta"] == 432
        assert BancoCentralAPI.SERIES["cdi"] == 12
        assert BancoCentralAPI.SERIES["ipca"] == 433

    def test_correlation_analyzer(self):
        from data.macro_data import CorrelationAnalyzer
        np.random.seed(42)
        a = pd.Series(np.random.randn(100).cumsum() + 100)
        b = pd.Series(np.random.randn(100).cumsum() + 50)
        corr = CorrelationAnalyzer.rolling_correlation(a, b, window=30)
        valid = corr.dropna()
        assert len(valid) > 0
        assert -1 <= valid.iloc[-1] <= 1

    def test_correlation_matrix(self):
        from data.macro_data import CorrelationAnalyzer
        df = pd.DataFrame({
            "A": np.random.randn(100).cumsum(),
            "B": np.random.randn(100).cumsum(),
        })
        corr = CorrelationAnalyzer.correlation_matrix(df)
        assert corr.shape == (2, 2)
        assert corr.loc["A", "A"] == pytest.approx(1.0)

    def test_monte_carlo(self):
        from data.macro_data import MonteCarloPortfolio
        np.random.seed(42)
        returns = pd.DataFrame({
            "A": np.random.randn(252) * 0.02,
            "B": np.random.randn(252) * 0.015,
        })
        result = MonteCarloPortfolio.simulate(returns, n_portfolios=100)
        assert len(result) == 100
        assert "return" in result.columns
        assert "sharpe" in result.columns
        assert result["is_optimal"].sum() == 1

    def test_optimal_weights_sum_to_one(self):
        from data.macro_data import MonteCarloPortfolio
        np.random.seed(42)
        returns = pd.DataFrame({
            "X": np.random.randn(252) * 0.02,
            "Y": np.random.randn(252) * 0.015,
        })
        result = MonteCarloPortfolio.simulate(returns, n_portfolios=100)
        weights = MonteCarloPortfolio.get_optimal_weights(result)
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_extreme_event_analysis(self):
        from data.macro_data import ExtremeEventAnalysis
        np.random.seed(42)
        returns = pd.Series(np.random.randn(2000) * 0.015)
        analysis = ExtremeEventAnalysis.analyze_tail_risk(returns, threshold=-0.03)
        assert "prob_normal" in analysis
        assert "prob_t_student" in analysis
        assert analysis["prob_normal"] > 0

    def test_impact_missing_days(self):
        from data.macro_data import ExtremeEventAnalysis
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        impact = ExtremeEventAnalysis.impact_of_missing_days(returns, n_days=5)
        assert "total_return" in impact
        assert "without_5_best_days" in impact
        assert "without_5_worst_days" in impact

    def test_cota_calculator(self):
        from data.macro_data import CotaCalculator
        values = pd.Series(np.linspace(100000, 115000, 50))
        flows = pd.Series(0.0, index=range(50))
        flows.iloc[20] = 10000
        result = CotaCalculator.calculate(values, flows)
        assert "vl_cota" in result.columns
        assert "qtd_cotas" in result.columns
        assert result["vl_cota"].iloc[-1] > 0
