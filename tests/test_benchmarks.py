"""Testes para o comparador de benchmarks."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from backtest.benchmarks import BenchmarkComparator, BenchmarkResult, ComparisonReport


class TestBenchmarkComparator:

    def _make_equity(self, annual_return=0.20, days=252):
        """Gera equity curve simulada."""
        np.random.seed(42)
        daily = annual_return / 252
        vol = 0.015
        rets = np.random.normal(daily, vol, days)
        prices = 100000 * np.cumprod(1 + rets)
        return pd.Series(prices, index=pd.bdate_range(start="2025-04-01", periods=days))

    def test_compare_basic(self):
        eq = self._make_equity(annual_return=0.20, days=252)
        comp = BenchmarkComparator(initial_capital=100000)
        report = comp.compare(eq, cdi_rate=13.0)

        assert isinstance(report, ComparisonReport)
        assert report.period_days == 252
        assert "CDI" in report.benchmarks
        assert report.bot_return != 0

    def test_cdi_benchmark(self):
        eq = self._make_equity(days=252)
        comp = BenchmarkComparator()
        report = comp.compare(eq, cdi_rate=13.0)

        cdi = report.benchmarks["CDI"]
        assert cdi.annual_return == 13.0
        assert cdi.total_return > 0
        assert cdi.final_value > 100000

    def test_alpha_calculation(self):
        eq = self._make_equity(annual_return=0.25, days=252)
        comp = BenchmarkComparator()
        report = comp.compare(eq, cdi_rate=10.0)

        # Bot com ~25% deve ter alpha positivo vs CDI 10%
        assert report.alpha_vs["CDI"] > 0

    def test_empty_equity_raises(self):
        comp = BenchmarkComparator()
        with pytest.raises(ValueError, match="vazia"):
            comp.compare(pd.Series([]))

    def test_estimated_benchmarks(self):
        comp = BenchmarkComparator()
        data = comp._generate_estimated(252)
        assert "Ibovespa" in data
        assert "S&P 500" in data
        assert "Bitcoin" in data
        assert len(data["Ibovespa"]) == 252

    def test_report_to_dict(self):
        eq = self._make_equity(days=100)
        comp = BenchmarkComparator()
        report = comp.compare(eq, cdi_rate=12.0)
        d = report.to_dict()
        assert "bot" in d
        assert "benchmarks" in d
        assert "alpha" in d
        assert "CDI" in d["benchmarks"]

    def test_final_values_reasonable(self):
        eq = self._make_equity(annual_return=0.15, days=252)
        comp = BenchmarkComparator(initial_capital=100000)
        report = comp.compare(eq, cdi_rate=13.0)

        # Valor final do CDI deve ser ~113k
        cdi = report.benchmarks["CDI"]
        assert 110000 < cdi.final_value < 120000
