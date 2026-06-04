"""
Rastreador de Performance por Período — Semanal, Mensal e Anual.

Calcula e exibe métricas financeiras detalhadas para cada
período de tempo, permitindo acompanhar a evolução do bot
em diferentes horizontes.

Métricas por período:
- Retorno (%) e valor absoluto ($)
- Retorno anualizado
- Volatilidade
- Sharpe Ratio
- Max Drawdown
- Melhor e pior dia
- Número de trades
- Win Rate
- Comparação com benchmarks (CDI, Ibovespa, S&P 500)

Uso:
    tracker = PerformanceTracker(equity_curve, trades)
    weekly = tracker.get_weekly()
    monthly = tracker.get_monthly()
    annual = tracker.get_annual()
    tracker.print_full_report()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.formatters import fmt_currency, fmt_pct

logger = get_logger("quantbot.core.performance")


# CDI diário aproximado por ano
CDI_DAILY = {
    2024: 10.93 / 100 / 252,
    2025: 13.25 / 100 / 252,
    2026: 12.50 / 100 / 252,
}
DEFAULT_CDI_DAILY = 12.50 / 100 / 252

# Ibovespa retorno médio diário histórico (~12% a.a.)
IBOV_DAILY = 12.0 / 100 / 252

# S&P 500 retorno médio diário histórico (~10% a.a.)
SP500_DAILY = 10.0 / 100 / 252


@dataclass
class PeriodMetrics:
    """Métricas de um período específico."""
    period_label: str          # "Semana Atual", "Março 2026", "2026"
    period_type: str           # "weekly", "monthly", "annual"
    start_date: str
    end_date: str
    trading_days: int

    # Retornos
    return_pct: float          # retorno do período %
    return_value: float        # retorno em $
    return_annualized: float   # retorno anualizado %
    start_value: float         # valor inicial $
    end_value: float           # valor final $

    # Risco
    volatility: float          # volatilidade anualizada %
    sharpe_ratio: float
    max_drawdown: float        # max drawdown no período %
    best_day: float            # melhor retorno diário %
    worst_day: float           # pior retorno diário %

    # Trading
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float            # %
    profit_factor: float

    # Benchmarks
    cdi_return: float          # CDI no período %
    cdi_value: float           # CDI em $
    ibov_return: float         # Ibovespa estimado %
    sp500_return: float        # S&P 500 estimado %

    # Alpha
    alpha_vs_cdi: float        # retorno - CDI %
    alpha_vs_ibov: float       # retorno - Ibovespa %
    alpha_vs_sp500: float      # retorno - S&P 500 %

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


class PerformanceTracker:
    """
    Calcula métricas de performance para semana, mês e ano.

    Aceita uma equity curve (série de valores do portfólio)
    e opcionalmente uma lista de trades para métricas de trading.
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trades: List[dict] = None,
        initial_capital: float = 100_000.0,
    ):
        """
        Args:
            equity_curve: Series indexada por data com valor do portfólio
            trades: Lista de dicts com {date, type, pnl}
            initial_capital: Capital inicial
        """
        if equity_curve is None or len(equity_curve) < 2:
            raise ValueError("Equity curve precisa de pelo menos 2 pontos")

        self.equity = equity_curve.copy()
        self.returns = equity_curve.pct_change().dropna()
        self.trades = trades or []
        self.initial_capital = initial_capital

    def _calc_period(
        self,
        label: str,
        period_type: str,
        eq_slice: pd.Series,
        ret_slice: pd.Series,
        trades_slice: List[dict],
    ) -> PeriodMetrics:
        """Calcula métricas para um período arbitrário."""

        if len(eq_slice) < 2:
            # Retorna métricas zeradas
            return PeriodMetrics(
                period_label=label, period_type=period_type,
                start_date="", end_date="", trading_days=0,
                return_pct=0, return_value=0, return_annualized=0,
                start_value=0, end_value=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, best_day=0, worst_day=0,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, profit_factor=0,
                cdi_return=0, cdi_value=0, ibov_return=0, sp500_return=0,
                alpha_vs_cdi=0, alpha_vs_ibov=0, alpha_vs_sp500=0,
            )

        days = len(ret_slice)
        years = days / 252 if days > 0 else 1

        # Datas
        start_date = str(eq_slice.index[0])[:10]
        end_date = str(eq_slice.index[-1])[:10]

        # Retornos
        start_val = eq_slice.iloc[0]
        end_val = eq_slice.iloc[-1]
        ret_pct = (end_val / start_val - 1) * 100
        ret_val = end_val - start_val
        ret_annual = ((1 + ret_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Risco
        if len(ret_slice) > 1:
            vol = ret_slice.std() * np.sqrt(252) * 100
            daily_mean = ret_slice.mean()
            daily_std = ret_slice.std()
            sharpe = (daily_mean * 252) / (daily_std * np.sqrt(252)) if daily_std > 0 else 0

            cum = (1 + ret_slice).cumprod()
            rolling_max = cum.expanding().max()
            dd = ((cum - rolling_max) / rolling_max)
            max_dd = dd.min() * 100

            best_day = ret_slice.max() * 100
            worst_day = ret_slice.min() * 100
        else:
            vol = sharpe = max_dd = best_day = worst_day = 0

        # Trading
        wins = [t for t in trades_slice if t.get("pnl", 0) > 0]
        losses = [t for t in trades_slice if t.get("pnl", 0) <= 0 and t.get("type") == "SELL"]
        total_t = len(trades_slice)
        sell_t = [t for t in trades_slice if t.get("type") == "SELL"]
        win_count = len(wins)
        lose_count = len(losses)
        win_rate = win_count / len(sell_t) * 100 if sell_t else 0

        gross_profit = sum(t.get("pnl", 0) for t in wins)
        gross_loss = abs(sum(t.get("pnl", 0) for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        # Benchmarks
        cdi_daily = DEFAULT_CDI_DAILY
        cdi_ret = ((1 + cdi_daily) ** days - 1) * 100
        cdi_val = start_val * (cdi_ret / 100)

        ibov_ret = ((1 + IBOV_DAILY) ** days - 1) * 100
        sp500_ret = ((1 + SP500_DAILY) ** days - 1) * 100

        return PeriodMetrics(
            period_label=label,
            period_type=period_type,
            start_date=start_date,
            end_date=end_date,
            trading_days=days,
            return_pct=round(ret_pct, 2),
            return_value=round(ret_val, 2),
            return_annualized=round(ret_annual, 2),
            start_value=round(start_val, 2),
            end_value=round(end_val, 2),
            volatility=round(vol, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown=round(max_dd, 2),
            best_day=round(best_day, 2),
            worst_day=round(worst_day, 2),
            total_trades=total_t,
            winning_trades=win_count,
            losing_trades=lose_count,
            win_rate=round(win_rate, 1),
            profit_factor=round(pf, 2),
            cdi_return=round(cdi_ret, 2),
            cdi_value=round(cdi_val, 2),
            ibov_return=round(ibov_ret, 2),
            sp500_return=round(sp500_ret, 2),
            alpha_vs_cdi=round(ret_pct - cdi_ret, 2),
            alpha_vs_ibov=round(ret_pct - ibov_ret, 2),
            alpha_vs_sp500=round(ret_pct - sp500_ret, 2),
        )

    def _filter_trades(self, start: datetime, end: datetime) -> List[dict]:
        """Filtra trades por período."""
        filtered = []
        for t in self.trades:
            try:
                t_date = pd.Timestamp(t.get("date", ""))
                if start <= t_date <= end:
                    filtered.append(t)
            except (ValueError, TypeError):
                continue
        return filtered

    def get_weekly(self) -> PeriodMetrics:
        """Métricas da última semana (5 dias úteis)."""
        n = min(5, len(self.equity))
        eq = self.equity.iloc[-n:]
        ret = self.returns.iloc[-n:]
        start = eq.index[0]
        end = eq.index[-1]
        trades = self._filter_trades(start, end)
        return self._calc_period("Semana Atual", "weekly", eq, ret, trades)

    def get_monthly(self) -> PeriodMetrics:
        """Métricas do último mês (~21 dias úteis)."""
        n = min(21, len(self.equity))
        eq = self.equity.iloc[-n:]
        ret = self.returns.iloc[-n:]
        start = eq.index[0]
        end = eq.index[-1]
        trades = self._filter_trades(start, end)

        # Label com nome do mês
        month_names = ["", "Janeiro", "Fevereiro", "Março", "Abril", "Maio",
                       "Junho", "Julho", "Agosto", "Setembro", "Outubro",
                       "Novembro", "Dezembro"]
        end_dt = pd.Timestamp(end)
        label = f"{month_names[end_dt.month]} {end_dt.year}"

        return self._calc_period(label, "monthly", eq, ret, trades)

    def get_annual(self) -> PeriodMetrics:
        """Métricas do último ano (~252 dias úteis)."""
        n = min(252, len(self.equity))
        eq = self.equity.iloc[-n:]
        ret = self.returns.iloc[-n:]
        start = eq.index[0]
        end = eq.index[-1]
        trades = self._filter_trades(start, end)

        end_dt = pd.Timestamp(end)
        label = f"Ano {end_dt.year}"

        return self._calc_period(label, "annual", eq, ret, trades)

    def get_all_months(self) -> List[PeriodMetrics]:
        """Retorna métricas mês a mês."""
        months = []
        eq_df = self.equity.to_frame("value")
        eq_df["month"] = eq_df.index.to_period("M")

        for period, group in eq_df.groupby("month"):
            eq = group["value"]
            ret = eq.pct_change().dropna()
            start = eq.index[0]
            end = eq.index[-1]
            trades = self._filter_trades(start, end)

            month_names = ["", "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                           "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
            label = f"{month_names[period.month]} {period.year}"

            months.append(self._calc_period(label, "monthly", eq, ret, trades))

        return months

    def get_full_summary(self) -> Dict[str, PeriodMetrics]:
        """Retorna resumo completo: semana, mês e ano."""
        return {
            "weekly": self.get_weekly(),
            "monthly": self.get_monthly(),
            "annual": self.get_annual(),
        }

    def print_full_report(self):
        """Imprime relatório completo no terminal."""
        summary = self.get_full_summary()

        print("\n" + "═" * 90)
        print("  📊 PERFORMANCE — SEMANAL | MENSAL | ANUAL")
        print("═" * 90)

        # Header
        print(f"\n  {'Métrica':<28} {'Semana':>16} {'Mês':>16} {'Ano':>16}")
        print("  " + "─" * 86)

        w, m, a = summary["weekly"], summary["monthly"], summary["annual"]

        rows = [
            ("Período", w.period_label, m.period_label, a.period_label),
            ("Dias úteis", str(w.trading_days), str(m.trading_days), str(a.trading_days)),
            ("", "", "", ""),
            ("── RETORNOS ──", "", "", ""),
            ("Retorno", fmt_pct(w.return_pct), fmt_pct(m.return_pct), fmt_pct(a.return_pct)),
            ("Retorno ($)", fmt_currency(w.return_value), fmt_currency(m.return_value), fmt_currency(a.return_value)),
            ("Retorno Anualizado", fmt_pct(w.return_annualized), fmt_pct(m.return_annualized), fmt_pct(a.return_annualized)),
            ("Valor Inicial", fmt_currency(w.start_value), fmt_currency(m.start_value), fmt_currency(a.start_value)),
            ("Valor Final", fmt_currency(w.end_value), fmt_currency(m.end_value), fmt_currency(a.end_value)),
            ("", "", "", ""),
            ("── RISCO ──", "", "", ""),
            ("Volatilidade (anual.)", f"{w.volatility:.1f}%", f"{m.volatility:.1f}%", f"{a.volatility:.1f}%"),
            ("Sharpe Ratio", f"{w.sharpe_ratio:.2f}", f"{m.sharpe_ratio:.2f}", f"{a.sharpe_ratio:.2f}"),
            ("Max Drawdown", fmt_pct(w.max_drawdown), fmt_pct(m.max_drawdown), fmt_pct(a.max_drawdown)),
            ("Melhor Dia", fmt_pct(w.best_day), fmt_pct(m.best_day), fmt_pct(a.best_day)),
            ("Pior Dia", fmt_pct(w.worst_day), fmt_pct(m.worst_day), fmt_pct(a.worst_day)),
            ("", "", "", ""),
            ("── TRADING ──", "", "", ""),
            ("Total Trades", str(w.total_trades), str(m.total_trades), str(a.total_trades)),
            ("Win Rate", f"{w.win_rate:.1f}%", f"{m.win_rate:.1f}%", f"{a.win_rate:.1f}%"),
            ("Profit Factor", f"{w.profit_factor:.2f}", f"{m.profit_factor:.2f}", f"{a.profit_factor:.2f}"),
            ("", "", "", ""),
            ("── BENCHMARKS ──", "", "", ""),
            ("CDI", fmt_pct(w.cdi_return), fmt_pct(m.cdi_return), fmt_pct(a.cdi_return)),
            ("Ibovespa (est.)", fmt_pct(w.ibov_return), fmt_pct(m.ibov_return), fmt_pct(a.ibov_return)),
            ("S&P 500 (est.)", fmt_pct(w.sp500_return), fmt_pct(m.sp500_return), fmt_pct(a.sp500_return)),
            ("", "", "", ""),
            ("── ALPHA ──", "", "", ""),
            ("vs CDI", fmt_pct(w.alpha_vs_cdi), fmt_pct(m.alpha_vs_cdi), fmt_pct(a.alpha_vs_cdi)),
            ("vs Ibovespa", fmt_pct(w.alpha_vs_ibov), fmt_pct(m.alpha_vs_ibov), fmt_pct(a.alpha_vs_ibov)),
            ("vs S&P 500", fmt_pct(w.alpha_vs_sp500), fmt_pct(m.alpha_vs_sp500), fmt_pct(a.alpha_vs_sp500)),
        ]

        for row in rows:
            if row[1] == "" and row[2] == "" and row[3] == "":
                if "──" in row[0]:
                    print(f"\n  {row[0]}")
                else:
                    print()
            else:
                print(f"  {row[0]:<28} {row[1]:>16} {row[2]:>16} {row[3]:>16}")

        print("\n" + "═" * 90)

        # Resumo mês a mês se tiver dados suficientes
        if len(self.equity) > 42:
            months = self.get_all_months()
            if len(months) > 1:
                print("\n  📅 RETORNO MÊS A MÊS")
                print("  " + "─" * 60)
                for m in months:
                    bar_len = max(0, int(abs(m.return_pct) * 2))
                    bar_char = "█" if m.return_pct >= 0 else "░"
                    bar_color = "+" if m.return_pct >= 0 else "-"
                    bar = bar_char * min(bar_len, 30)
                    print(f"  {m.period_label:<12} {m.return_pct:>+7.2f}% "
                          f"({fmt_currency(m.return_value):>12}) [{bar}]")
                print("  " + "─" * 60)
