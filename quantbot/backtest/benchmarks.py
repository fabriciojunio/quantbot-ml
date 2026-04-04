"""
Benchmarks de Comparação — CDI, Ibovespa, S&P 500.

Compara o retorno do bot com os principais benchmarks
do mercado brasileiro e americano, permitindo ao investidor
(e ao leitor do TCC) entender se o bot agrega valor.

Benchmarks:
- CDI: taxa livre de risco no Brasil (~13% a.a. em 2025/2026)
- Ibovespa (BOVA11): principal índice de ações do Brasil
- S&P 500 (SPY/^GSPC): principal índice dos EUA
- Bitcoin: benchmark para o mercado crypto

Uso:
    bench = BenchmarkComparator()
    bench.fetch_benchmarks()
    report = bench.compare(bot_equity_curve)
    report.print_summary()
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("quantbot.backtest.benchmarks")

# Tenta importar yfinance
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ═══════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Resultado de comparação com um benchmark."""
    name: str
    annual_return: float       # retorno anualizado %
    total_return: float        # retorno total no período %
    volatility: float          # volatilidade anualizada %
    sharpe: float              # sharpe ratio
    max_drawdown: float        # max drawdown %
    final_value: float         # valor final de $100k investidos


@dataclass
class ComparisonReport:
    """Relatório completo de comparação."""
    bot_return: float          # retorno anualizado do bot %
    bot_total_return: float    # retorno total do bot %
    bot_sharpe: float
    bot_max_dd: float
    bot_final_value: float
    period_days: int
    initial_capital: float
    benchmarks: Dict[str, BenchmarkResult] = field(default_factory=dict)
    alpha_vs: Dict[str, float] = field(default_factory=dict)  # alpha vs cada benchmark

    def print_summary(self):
        """Imprime relatório formatado no terminal."""
        print("\n" + "═" * 80)
        print("  📊 COMPARAÇÃO COM BENCHMARKS — RETORNO ANUALIZADO")
        print("═" * 80)
        print(f"\n  Período: {self.period_days} dias úteis (~{self.period_days/252:.1f} ano)")
        print(f"  Capital inicial: ${self.initial_capital:,.0f}\n")

        # Header
        print(f"  {'':20} {'Retorno Anual':>14} {'Retorno Total':>14} "
              f"{'Sharpe':>8} {'Max DD':>8} {'Valor Final':>14}")
        print("  " + "─" * 78)

        # Bot
        print(f"  {'🤖 QuantBot ML':<20} {self.bot_return:>+13.1f}% "
              f"{self.bot_total_return:>+13.1f}% {self.bot_sharpe:>8.2f} "
              f"{self.bot_max_dd:>7.1f}% ${self.bot_final_value:>13,.0f}")

        # Benchmarks
        for name, b in self.benchmarks.items():
            alpha = self.alpha_vs.get(name, 0)
            print(f"  {name:<20} {b.annual_return:>+13.1f}% "
                  f"{b.total_return:>+13.1f}% {b.sharpe:>8.2f} "
                  f"{b.max_drawdown:>7.1f}% ${b.final_value:>13,.0f}")

        print("  " + "─" * 78)

        # Alpha
        print("\n  ALPHA (retorno acima do benchmark):")
        for name, alpha in self.alpha_vs.items():
            emoji = "✅" if alpha > 0 else "❌"
            print(f"  {emoji} vs {name:<16} {alpha:>+.1f}% ao ano")

        print("\n" + "═" * 80)

    def to_dict(self) -> dict:
        return {
            "bot": {
                "annual_return": self.bot_return,
                "total_return": self.bot_total_return,
                "sharpe": self.bot_sharpe,
                "max_drawdown": self.bot_max_dd,
                "final_value": self.bot_final_value,
            },
            "benchmarks": {
                name: {
                    "annual_return": b.annual_return,
                    "total_return": b.total_return,
                    "sharpe": b.sharpe,
                    "max_drawdown": b.max_drawdown,
                    "final_value": b.final_value,
                }
                for name, b in self.benchmarks.items()
            },
            "alpha": self.alpha_vs,
            "period_days": self.period_days,
        }


# ═══════════════════════════════════════════════════════════════
# BENCHMARK COMPARATOR
# ═══════════════════════════════════════════════════════════════

# CDI rates por ano (aproximado, fonte BCB)
CDI_ANNUAL_RATES = {
    2020: 2.75, 2021: 4.42, 2022: 12.39, 2023: 13.04,
    2024: 10.93, 2025: 13.25, 2026: 12.50,  # estimativa
}


class BenchmarkComparator:
    """
    Compara performance do bot com benchmarks de mercado.

    Busca dados reais do Ibovespa e S&P 500 via yfinance
    e calcula CDI com taxa fixa anualizada.
    """

    BENCHMARK_SYMBOLS = {
        "Ibovespa": "^BVSP",
        "S&P 500": "^GSPC",
        "Bitcoin": "BTC-USD",
    }

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self._data: Dict[str, pd.Series] = {}

    def fetch_benchmarks(self, period_days: int = 252) -> Dict[str, pd.Series]:
        """
        Busca dados históricos dos benchmarks.

        Args:
            period_days: Período em dias úteis

        Returns:
            Dicionário {nome: Series de preços}
        """
        if not HAS_YF:
            logger.warning("yfinance não disponível. Usando dados estimados.")
            return self._generate_estimated(period_days)

        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=int(period_days * 1.5))

        for name, symbol in self.BENCHMARK_SYMBOLS.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start, end=end)
                if not hist.empty:
                    self._data[name] = hist["Close"].tail(period_days)
                    logger.info(f"  ✓ {name}: {len(self._data[name])} pontos")
            except Exception as e:
                logger.warning(f"  ✗ {name}: {e}")

        return self._data

    def _generate_estimated(self, period_days: int) -> Dict[str, pd.Series]:
        """Gera dados estimados quando yfinance não está disponível."""
        np.random.seed(42)
        end_date = pd.Timestamp.today().normalize()
        # Request extra dates then slice to guarantee exactly period_days entries
        dates = pd.bdate_range(end=end_date, periods=period_days + 10)[-period_days:]

        estimates = {
            "Ibovespa": {"annual_return": 0.12, "annual_vol": 0.22},
            "S&P 500": {"annual_return": 0.10, "annual_vol": 0.16},
            "Bitcoin": {"annual_return": 0.40, "annual_vol": 0.65},
        }

        for name, params in estimates.items():
            daily_ret = params["annual_return"] / 252
            daily_vol = params["annual_vol"] / np.sqrt(252)
            returns = np.random.normal(daily_ret, daily_vol, period_days)
            prices = 100 * np.cumprod(1 + returns)
            self._data[name] = pd.Series(prices, index=dates)

        return self._data

    def compare(
        self,
        bot_equity: pd.Series,
        cdi_rate: float = None,
    ) -> ComparisonReport:
        """
        Compara performance do bot com todos os benchmarks.

        Args:
            bot_equity: Série com valor do portfólio do bot ao longo do tempo
            cdi_rate: Taxa CDI anual (%) — se None, usa a mais recente

        Returns:
            ComparisonReport completo
        """
        if bot_equity is None or len(bot_equity) < 2:
            raise ValueError("Equity curve do bot vazia ou insuficiente")

        period_days = len(bot_equity)
        years = period_days / 252

        # ── Bot metrics ──
        bot_returns = bot_equity.pct_change().dropna()
        bot_total_ret = (bot_equity.iloc[-1] / bot_equity.iloc[0] - 1) * 100
        bot_annual_ret = ((1 + bot_total_ret / 100) ** (1 / years) - 1) * 100
        bot_vol = bot_returns.std() * np.sqrt(252)
        bot_sharpe = (bot_returns.mean() * 252) / (bot_vol) if bot_vol > 0 else 0
        cum = (1 + bot_returns).cumprod()
        bot_max_dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min() * 100
        bot_final = self.initial_capital * (1 + bot_total_ret / 100)

        # ── Benchmarks ──
        benchmarks = {}

        # CDI
        if cdi_rate is None:
            cdi_rate = CDI_ANNUAL_RATES.get(2026, 12.5)

        cdi_total = ((1 + cdi_rate / 100) ** years - 1) * 100
        benchmarks["CDI"] = BenchmarkResult(
            name="CDI",
            annual_return=cdi_rate,
            total_return=cdi_total,
            volatility=0.1,  # CDI tem vol ~0
            sharpe=0.0,  # RF é o próprio benchmark do Sharpe
            max_drawdown=0.0,
            final_value=self.initial_capital * (1 + cdi_total / 100),
        )

        # Market benchmarks
        if not self._data:
            self.fetch_benchmarks(period_days)

        for name, prices in self._data.items():
            if len(prices) < 2:
                continue

            rets = prices.pct_change().dropna()
            total_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
            bench_years = len(prices) / 252
            annual_ret = ((1 + total_ret / 100) ** (1 / bench_years) - 1) * 100
            vol = rets.std() * np.sqrt(252) * 100
            sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
            cum_b = (1 + rets).cumprod()
            max_dd = ((cum_b - cum_b.expanding().max()) / cum_b.expanding().max()).min() * 100

            benchmarks[name] = BenchmarkResult(
                name=name,
                annual_return=annual_ret,
                total_return=total_ret,
                volatility=vol,
                sharpe=sharpe,
                max_drawdown=max_dd,
                final_value=self.initial_capital * (1 + total_ret / 100),
            )

        # Alpha vs cada benchmark
        alpha_vs = {
            name: round(bot_annual_ret - b.annual_return, 1)
            for name, b in benchmarks.items()
        }

        return ComparisonReport(
            bot_return=round(bot_annual_ret, 1),
            bot_total_return=round(bot_total_ret, 1),
            bot_sharpe=round(bot_sharpe, 2),
            bot_max_dd=round(bot_max_dd, 1),
            bot_final_value=round(bot_final, 0),
            period_days=period_days,
            initial_capital=self.initial_capital,
            benchmarks=benchmarks,
            alpha_vs=alpha_vs,
        )
