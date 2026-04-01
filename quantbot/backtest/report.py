"""
Gerador de relatórios visuais para backtesting.

Gera gráficos profissionais com equity curve, distribuição
de retornos, drawdown e métricas consolidadas.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from models.signals import MLSignal
from risk.metrics import RiskMetrics
from backtest.engine import BacktestResult
from utils.logger import get_logger
from utils.formatters import fmt_currency, fmt_pct

logger = get_logger("quantbot.backtest.report")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class ReportGenerator:
    """Gera relatórios visuais e textuais."""

    DARK_BG = "#0d1117"
    CARD_BG = "#111622"
    TEXT_COLOR = "#e6edf3"
    GRID_COLOR = "#1a1f2e"
    GREEN = "#00ff87"
    RED = "#f87171"
    PURPLE = "#6366f1"
    YELLOW = "#fbbf24"

    @staticmethod
    def print_backtest(result: BacktestResult):
        """Imprime resultados do backtest no terminal."""
        print("\n" + "═" * 70)
        print("  📊 RESULTADOS DO BACKTEST")
        print("═" * 70)

        metrics = [
            ("Retorno Total", fmt_pct(result.total_return)),
            ("Retorno Benchmark", fmt_pct(result.benchmark_return)),
            ("Alpha", fmt_pct(result.alpha)),
            ("Sharpe Ratio", f"{result.sharpe_ratio:.3f}"),
            ("Sortino Ratio", f"{result.sortino_ratio:.3f}"),
            ("Max Drawdown", fmt_pct(result.max_drawdown)),
            ("Win Rate", f"{result.win_rate:.1f}%"),
            ("Profit Factor", f"{result.profit_factor:.2f}"),
            ("Calmar Ratio", f"{result.calmar_ratio:.2f}"),
            ("Total Trades", f"{result.total_trades}"),
        ]

        for label, value in metrics:
            print(f"  {label:<25} {value:>15}")

        print("═" * 70)

    @staticmethod
    def print_risk(risk: RiskMetrics):
        """Imprime métricas de risco no terminal."""
        print("\n" + "═" * 70)
        print("  🛡️  MÉTRICAS DE RISCO")
        print("═" * 70)

        metrics = [
            ("VaR 95% (diário)", f"{risk.var_95*100:.2f}%"),
            ("VaR 99% (diário)", f"{risk.var_99*100:.2f}%"),
            ("CVaR 95%", f"{risk.cvar_95*100:.2f}%"),
            ("Beta", f"{risk.beta:.3f}"),
            ("Alpha (anual)", f"{risk.alpha*100:.2f}%"),
            ("Volatilidade (anual)", f"{risk.volatility*100:.2f}%"),
            ("Sharpe Ratio", f"{risk.sharpe_ratio:.3f}"),
            ("Sortino Ratio", f"{risk.sortino_ratio:.3f}"),
            ("Max Drawdown", f"{risk.max_drawdown*100:.2f}%"),
            ("Drawdown Atual", f"{risk.current_drawdown*100:.2f}%"),
        ]

        for label, value in metrics:
            print(f"  {label:<25} {value:>15}")

        print("═" * 70)

    @staticmethod
    def print_signals(signals: Dict[str, MLSignal]):
        """Imprime sinais de ML."""
        print("\n" + "═" * 70)
        print("  🤖 SINAIS DE MACHINE LEARNING")
        print("═" * 70)

        sorted_sigs = sorted(signals.values(), key=lambda s: s.score, reverse=True)

        for sig in sorted_sigs:
            bar_len = int(sig.score / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)

            print(f"\n  {sig.symbol:<12} Score: {sig.score:>5.1f} [{bar}]")
            print(f"  {'':12} Sinal: {sig.signal.value:<14} Confiança: {sig.confidence:.0f}%")

            votes = " | ".join(f"{k}: {v}" for k, v in sig.model_votes.items())
            print(f"  {'':12} Modelos: {votes}")

            feats = list(sig.feature_importance.items())[:5]
            feat_str = " | ".join(f"{k}: {v:.1%}" for k, v in feats)
            if feat_str:
                print(f"  {'':12} Top Features: {feat_str}")

        print("\n" + "═" * 70)

    @classmethod
    def generate_charts(
        cls,
        backtest: BacktestResult,
        signals: Dict[str, MLSignal] = None,
        risk: RiskMetrics = None,
        output_path: str = "quantbot_report.png",
    ):
        """Gera relatório visual completo em PNG."""
        if not HAS_MPL:
            logger.warning("Matplotlib não disponível.")
            return

        fig = plt.figure(figsize=(18, 12), facecolor=cls.DARK_BG)
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor(cls.DARK_BG)

        if backtest.equity_curve is not None:
            ax1.plot(backtest.equity_curve.index, backtest.equity_curve.values,
                     color=cls.GREEN, linewidth=1.5, label="Bot ML")
            ax1.fill_between(backtest.equity_curve.index, backtest.equity_curve.values,
                             backtest.equity_curve.values.min(), alpha=0.1, color=cls.GREEN)

        if backtest.benchmark_curve is not None:
            ax1.plot(backtest.benchmark_curve.index, backtest.benchmark_curve.values,
                     color="#4a5568", linewidth=1, linestyle="--", label="Benchmark")

        ax1.set_title("Equity Curve — Bot ML vs Benchmark",
                       color=cls.TEXT_COLOR, fontsize=12, fontweight="bold")
        ax1.legend(facecolor=cls.CARD_BG, edgecolor=cls.GRID_COLOR, labelcolor=cls.TEXT_COLOR)
        ax1.tick_params(colors=cls.TEXT_COLOR)
        ax1.grid(True, alpha=0.1)
        for spine in ax1.spines.values():
            spine.set_color(cls.GRID_COLOR)

        # 2. ML Scores
        if signals:
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.set_facecolor(cls.DARK_BG)
            sorted_sigs = sorted(signals.values(), key=lambda s: s.score, reverse=True)[:10]
            symbols = [s.symbol.replace(".SA", "").replace("-USD", "") for s in sorted_sigs]
            scores = [s.score for s in sorted_sigs]
            colors = [cls.GREEN if s > 60 else cls.YELLOW if s > 40 else cls.RED for s in scores]
            ax2.barh(symbols, scores, color=colors, height=0.6)
            ax2.set_xlim(0, 100)
            ax2.set_title("ML Scores", color=cls.TEXT_COLOR, fontsize=11, fontweight="bold")
            ax2.tick_params(colors=cls.TEXT_COLOR)
            ax2.invert_yaxis()
            for spine in ax2.spines.values():
                spine.set_color(cls.GRID_COLOR)

        # 3. Returns Distribution
        if backtest.equity_curve is not None:
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.set_facecolor(cls.DARK_BG)
            returns = backtest.equity_curve.pct_change().dropna()
            ax3.hist(returns * 100, bins=50, color=cls.PURPLE, alpha=0.7, edgecolor=cls.GRID_COLOR)
            ax3.axvline(0, color=cls.TEXT_COLOR, linewidth=0.5, linestyle="--")
            ax3.set_title("Distribuição de Retornos Diários (%)",
                          color=cls.TEXT_COLOR, fontsize=11, fontweight="bold")
            ax3.tick_params(colors=cls.TEXT_COLOR)
            for spine in ax3.spines.values():
                spine.set_color(cls.GRID_COLOR)

        # 4. Risk Metrics
        if risk:
            ax4 = fig.add_subplot(gs[1, 2])
            ax4.set_facecolor(cls.DARK_BG)
            labels = ["VaR 95%", "VaR 99%", "CVaR", "Max DD"]
            values = [abs(risk.var_95)*100, abs(risk.var_99)*100,
                      abs(risk.cvar_95)*100, abs(risk.max_drawdown)*100]
            colors = [cls.YELLOW, cls.RED, "#ef4444", cls.RED]
            ax4.bar(labels, values, color=colors, width=0.5)
            ax4.set_title("Métricas de Risco (%)",
                          color=cls.TEXT_COLOR, fontsize=11, fontweight="bold")
            ax4.tick_params(colors=cls.TEXT_COLOR)
            for spine in ax4.spines.values():
                spine.set_color(cls.GRID_COLOR)

        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=cls.DARK_BG, edgecolor="none")
        plt.close()
        logger.info(f"📊 Relatório salvo: {output_path}")
