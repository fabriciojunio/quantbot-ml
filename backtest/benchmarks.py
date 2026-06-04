"""Comparação da equity do bot contra benchmarks (CDI, Ibovespa, S&P, BTC).

Responde à pergunta que importa: o bot bateu o custo de oportunidade?
O CDI é o benchmark central no Brasil — se o sistema não supera o CDI com
folga, ele não justifica o risco de renda variável.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


@dataclass
class BenchmarkResult:
    name: str
    annual_return: float    # taxa anualizada em % (ex.: 13.0)
    total_return: float     # retorno no período em %
    final_value: float      # valor final do capital inicial


@dataclass
class ComparisonReport:
    period_days: int
    bot_return: float                 # retorno do bot no período (%)
    bot_final_value: float
    initial_capital: float
    benchmarks: Dict[str, BenchmarkResult] = field(default_factory=dict)
    alpha_vs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "period_days": self.period_days,
            "bot": {
                "return_pct": self.bot_return,
                "final_value": self.bot_final_value,
            },
            "benchmarks": {
                name: {
                    "annual_return": b.annual_return,
                    "total_return": b.total_return,
                    "final_value": b.final_value,
                }
                for name, b in self.benchmarks.items()
            },
            "alpha": dict(self.alpha_vs),
        }


class BenchmarkComparator:
    """Compara a curva de capital do bot com benchmarks de referência."""

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital

    def compare(self, equity: pd.Series, cdi_rate: float = 13.0) -> ComparisonReport:
        equity = pd.Series(equity).dropna()
        if len(equity) == 0:
            raise ValueError("Equity curve vazia")

        period_days = len(equity)
        years = period_days / _TRADING_DAYS

        bot_final = float(equity.iloc[-1])
        initial = float(equity.iloc[0])
        bot_return = (bot_final / initial - 1.0) * 100.0

        benchmarks: Dict[str, BenchmarkResult] = {}

        # CDI: taxa anual capitalizada pelo tempo do período.
        cdi_total = ((1 + cdi_rate / 100.0) ** years - 1.0) * 100.0
        benchmarks["CDI"] = BenchmarkResult(
            name="CDI",
            annual_return=cdi_rate,
            total_return=cdi_total,
            final_value=self.initial_capital * (1 + cdi_total / 100.0),
        )

        # Benchmarks de renda variável estimados.
        estimated = self._generate_estimated(period_days)
        for name, series in estimated.items():
            total = (series[-1] / series[0] - 1.0) * 100.0
            annual = ((1 + total / 100.0) ** (1 / years) - 1.0) * 100.0 if years > 0 else total
            benchmarks[name] = BenchmarkResult(
                name=name,
                annual_return=annual,
                total_return=total,
                final_value=self.initial_capital * (1 + total / 100.0),
            )

        alpha_vs = {
            name: bot_return - b.total_return for name, b in benchmarks.items()
        }

        return ComparisonReport(
            period_days=period_days,
            bot_return=bot_return,
            bot_final_value=self.initial_capital * (1 + bot_return / 100.0),
            initial_capital=self.initial_capital,
            benchmarks=benchmarks,
            alpha_vs=alpha_vs,
        )

    @staticmethod
    def _generate_estimated(n: int) -> Dict[str, np.ndarray]:
        """Séries sintéticas plausíveis para comparação visual/relativa.

        Não são dados reais — servem de referência ilustrativa quando não há
        séries históricas carregadas. Substitua por dados reais em produção.
        """
        rng = np.random.default_rng(7)
        params = {
            "Ibovespa": (0.12, 0.22),   # (retorno anual, vol anual)
            "S&P 500": (0.10, 0.16),
            "Bitcoin": (0.30, 0.65),
        }
        out: Dict[str, np.ndarray] = {}
        for name, (mu, sigma) in params.items():
            daily_mu = mu / _TRADING_DAYS
            daily_sigma = sigma / np.sqrt(_TRADING_DAYS)
            rets = rng.normal(daily_mu, daily_sigma, n)
            out[name] = 100_000 * np.cumprod(1 + rets)
        return out
