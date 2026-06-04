"""Testes para o rastreador de performance por período."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from core.performance import PerformanceTracker, PeriodMetrics


@pytest.fixture
def equity_252():
    """Equity curve de 252 dias (1 ano)."""
    np.random.seed(42)
    days = 252
    daily_ret = 0.20 / 252  # ~20% a.a.
    vol = 0.015
    rets = np.random.normal(daily_ret, vol, days)
    prices = 100000 * np.cumprod(1 + rets)
    return pd.Series(prices, index=pd.bdate_range(start="2025-04-01", periods=days))


@pytest.fixture
def sample_trades():
    dates = pd.bdate_range(start="2025-04-01", periods=252)
    trades = []
    for i in range(0, 240, 20):
        trades.append({"date": str(dates[i]), "type": "BUY", "price": 100, "shares": 10})
        pnl = np.random.choice([2.5, -1.5, 3.0, -0.8, 1.2])
        trades.append({"date": str(dates[i + 10]), "type": "SELL", "price": 100, "shares": 10, "pnl": pnl})
    return trades


class TestPerformanceTracker:

    def test_init(self, equity_252):
        tracker = PerformanceTracker(equity_252)
        assert tracker is not None

    def test_init_empty_raises(self):
        with pytest.raises(ValueError, match="pelo menos 2"):
            PerformanceTracker(pd.Series([]))

    def test_weekly(self, equity_252, sample_trades):
        tracker = PerformanceTracker(equity_252, sample_trades)
        w = tracker.get_weekly()
        assert w.period_type == "weekly"
        assert w.trading_days == 5
        assert isinstance(w.return_pct, float)
        assert isinstance(w.cdi_return, float)
        assert w.cdi_return > 0

    def test_monthly(self, equity_252, sample_trades):
        tracker = PerformanceTracker(equity_252, sample_trades)
        m = tracker.get_monthly()
        assert m.period_type == "monthly"
        assert m.trading_days == 21
        assert m.cdi_return > m.sp500_return  # CDI ~13% > SP500 ~10% em período curto

    def test_annual(self, equity_252, sample_trades):
        tracker = PerformanceTracker(equity_252, sample_trades)
        a = tracker.get_annual()
        assert a.period_type == "annual"
        assert a.trading_days >= 250  # returns tem N-1 pontos
        assert a.return_annualized != 0
        assert a.volatility > 0
        assert a.cdi_return > 10  # CDI ~12.5% a.a.

    def test_annual_benchmarks(self, equity_252):
        tracker = PerformanceTracker(equity_252)
        a = tracker.get_annual()
        # CDI anual deve ser ~12.5%
        assert 10 < a.cdi_return < 15
        # Ibov ~12%
        assert 9 < a.ibov_return < 15
        # S&P ~10%
        assert 7 < a.sp500_return < 13

    def test_alpha_calculation(self, equity_252):
        tracker = PerformanceTracker(equity_252)
        a = tracker.get_annual()
        # Alpha = retorno do bot - retorno do benchmark
        assert abs(a.alpha_vs_cdi - (a.return_pct - a.cdi_return)) < 0.01

    def test_full_summary(self, equity_252, sample_trades):
        tracker = PerformanceTracker(equity_252, sample_trades)
        summary = tracker.get_full_summary()
        assert "weekly" in summary
        assert "monthly" in summary
        assert "annual" in summary

    def test_all_months(self, equity_252):
        tracker = PerformanceTracker(equity_252)
        months = tracker.get_all_months()
        assert len(months) >= 10  # 252 dias ~ 12 meses

    def test_metrics_to_dict(self, equity_252):
        tracker = PerformanceTracker(equity_252)
        m = tracker.get_weekly()
        d = m.to_dict()
        assert "return_pct" in d
        assert "cdi_return" in d
        assert "alpha_vs_cdi" in d

    def test_short_equity(self):
        """Equity com poucos dias deve funcionar sem crash."""
        eq = pd.Series([100000, 101000, 102000],
                       index=pd.bdate_range(start="2025-01-01", periods=3))
        tracker = PerformanceTracker(eq)
        w = tracker.get_weekly()
        assert w.trading_days >= 2  # returns = N-1
        assert w.return_pct > 0

    def test_negative_return_period(self):
        """Período com retorno negativo."""
        np.random.seed(99)
        rets = np.random.normal(-0.003, 0.02, 21)
        prices = 100000 * np.cumprod(1 + rets)
        eq = pd.Series(prices, index=pd.bdate_range(start="2025-01-01", periods=21))
        tracker = PerformanceTracker(eq)
        m = tracker.get_monthly()
        assert m.return_pct < 0
        assert m.max_drawdown < 0
        assert m.alpha_vs_cdi < 0  # Perdeu pro CDI
