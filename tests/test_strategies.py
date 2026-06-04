"""
Testes das estratégias modulares (SMA, RSI, MACD, Ensemble Votação).
"""

import pytest
import numpy as np
import pandas as pd
from config.settings import Signal


@pytest.fixture
def df_with_indicators():
    """DataFrame com indicadores técnicos para teste."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    close = np.maximum(close, 1)

    df = pd.DataFrame({
        "Open": close - 0.5, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": np.random.randint(500_000, 5_000_000, n),
    }, index=dates)

    # Indicadores manuais
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_short"] = df["SMA_20"]
    df["SMA_long"] = df["SMA_50"]

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20

    return df


class TestSMACrossover:
    def test_generates_signals(self, df_with_indicators):
        from strategies.sma_crossover import SMACrossoverStrategy
        s = SMACrossoverStrategy()
        result = s.generate_signals(df_with_indicators)
        assert "signal" in result.columns

    def test_signals_are_valid_enum(self, df_with_indicators):
        from strategies.sma_crossover import SMACrossoverStrategy
        result = SMACrossoverStrategy().generate_signals(df_with_indicators)
        for val in result["signal"].unique():
            assert val in (Signal.BUY, Signal.SELL, Signal.HOLD,
                           Signal.STRONG_BUY, Signal.STRONG_SELL)

    def test_generates_some_signals(self, df_with_indicators):
        from strategies.sma_crossover import SMACrossoverStrategy
        result = SMACrossoverStrategy().generate_signals(df_with_indicators)
        non_hold = result[result["signal"] != Signal.HOLD]
        assert len(non_hold) > 0

    def test_missing_columns_returns_hold(self):
        from strategies.sma_crossover import SMACrossoverStrategy
        df = pd.DataFrame({"Close": [100, 101, 102]})
        result = SMACrossoverStrategy().generate_signals(df)
        assert (result["signal"] == Signal.HOLD).all()

    def test_explain(self, df_with_indicators):
        from strategies.sma_crossover import SMACrossoverStrategy
        s = SMACrossoverStrategy()
        result = s.generate_signals(df_with_indicators)
        explanation = s.explain(result.iloc[-1])
        assert isinstance(explanation, str)
        assert "SMA" in explanation


class TestRSIStrategy:
    def test_generates_signals(self, df_with_indicators):
        from strategies.rsi_strategy import RSIStrategy
        result = RSIStrategy().generate_signals(df_with_indicators)
        assert "signal" in result.columns

    def test_buys_when_oversold(self, df_with_indicators):
        from strategies.rsi_strategy import RSIStrategy
        result = RSIStrategy().generate_signals(df_with_indicators)
        buys = result[result["signal"].isin([Signal.BUY, Signal.STRONG_BUY])]
        if len(buys) > 0:
            assert (df_with_indicators.loc[buys.index, "RSI"] < 30).all()

    def test_sells_when_overbought(self, df_with_indicators):
        from strategies.rsi_strategy import RSIStrategy
        result = RSIStrategy().generate_signals(df_with_indicators)
        sells = result[result["signal"].isin([Signal.SELL, Signal.STRONG_SELL])]
        if len(sells) > 0:
            assert (df_with_indicators.loc[sells.index, "RSI"] > 70).all()

    def test_custom_thresholds(self, df_with_indicators):
        from strategies.rsi_strategy import RSIStrategy
        s = RSIStrategy(oversold=40, overbought=60)
        result = s.generate_signals(df_with_indicators)
        non_hold = result[result["signal"] != Signal.HOLD]
        assert len(non_hold) > 0

    def test_no_rsi_returns_hold(self):
        from strategies.rsi_strategy import RSIStrategy
        df = pd.DataFrame({"Close": [100, 101]})
        result = RSIStrategy().generate_signals(df)
        assert (result["signal"] == Signal.HOLD).all()


class TestMACDStrategy:
    def test_generates_signals(self, df_with_indicators):
        from strategies.macd_strategy import MACDStrategy
        result = MACDStrategy().generate_signals(df_with_indicators)
        assert "signal" in result.columns

    def test_signals_are_valid(self, df_with_indicators):
        from strategies.macd_strategy import MACDStrategy
        result = MACDStrategy().generate_signals(df_with_indicators)
        valid_signals = {Signal.BUY, Signal.SELL, Signal.HOLD,
                         Signal.STRONG_BUY, Signal.STRONG_SELL}
        for val in result["signal"].unique():
            assert val in valid_signals

    def test_no_macd_returns_hold(self):
        from strategies.macd_strategy import MACDStrategy
        df = pd.DataFrame({"Close": [100, 101]})
        result = MACDStrategy().generate_signals(df)
        assert (result["signal"] == Signal.HOLD).all()

    def test_explain(self, df_with_indicators):
        from strategies.macd_strategy import MACDStrategy
        s = MACDStrategy()
        result = s.generate_signals(df_with_indicators)
        text = s.explain(result.iloc[-1])
        assert "MACD" in text


class TestEnsembleVoting:
    def test_generates_signals(self, df_with_indicators):
        from strategies.ensemble_voting import EnsembleVotingStrategy
        result = EnsembleVotingStrategy().generate_signals(df_with_indicators)
        assert "signal" in result.columns
        assert "_ensemble_score" in result.columns

    def test_score_is_numeric(self, df_with_indicators):
        from strategies.ensemble_voting import EnsembleVotingStrategy
        result = EnsembleVotingStrategy().generate_signals(df_with_indicators)
        assert result["_ensemble_score"].dtype in [np.float64, np.float32, float]

    def test_explain_has_individual_votes(self, df_with_indicators):
        from strategies.ensemble_voting import EnsembleVotingStrategy
        s = EnsembleVotingStrategy()
        result = s.generate_signals(df_with_indicators)
        text = s.explain(result.iloc[-1])
        assert "Ensemble" in text

    def test_signals_valid_enum(self, df_with_indicators):
        from strategies.ensemble_voting import EnsembleVotingStrategy
        result = EnsembleVotingStrategy().generate_signals(df_with_indicators)
        valid = {Signal.BUY, Signal.SELL, Signal.HOLD,
                 Signal.STRONG_BUY, Signal.STRONG_SELL}
        for v in result["signal"].unique():
            assert v in valid


class TestBaseStrategy:
    def test_signal_numeric_mapping(self):
        from strategies.base import SIGNAL_NUMERIC, numeric_to_signal
        assert SIGNAL_NUMERIC[Signal.STRONG_BUY] == 2
        assert SIGNAL_NUMERIC[Signal.STRONG_SELL] == -2
        assert SIGNAL_NUMERIC[Signal.HOLD] == 0

    def test_numeric_to_signal(self):
        from strategies.base import numeric_to_signal
        assert numeric_to_signal(2.0) == Signal.STRONG_BUY
        assert numeric_to_signal(0.7) == Signal.BUY
        assert numeric_to_signal(0.0) == Signal.HOLD
        assert numeric_to_signal(-0.7) == Signal.SELL
        assert numeric_to_signal(-2.0) == Signal.STRONG_SELL


class TestVisualization:
    def test_plot_signals(self, df_with_indicators):
        from strategies.ensemble_voting import EnsembleVotingStrategy
        from visualization import plot_backtest_signals
        df = EnsembleVotingStrategy().generate_signals(df_with_indicators)
        path = plot_backtest_signals(df, "TEST", "Ensemble")
        assert path.exists()

    def test_plot_equity(self):
        from visualization import plot_equity_curve
        eq = pd.Series([100000, 101000, 99500, 102000],
                       index=pd.date_range("2024-01-01", periods=4))
        path = plot_equity_curve(eq, "TEST", "Test")
        assert path.exists()

    def test_plot_comparison(self):
        from visualization import plot_strategy_comparison
        data = [{"strategy_name": "A", "total_trades": 5, "win_rate": 0.6,
                  "total_return_pct": 3.5, "sharpe_ratio": 1.2, "max_drawdown_pct": 5.0}]
        path = plot_strategy_comparison(data, "TEST")
        assert path.exists()
