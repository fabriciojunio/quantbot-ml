"""
Visualização Profissional — Gráficos de Backtest e Análise.

4 tipos de gráficos:
    1. Preço + sinais + indicadores (4 subplots)
    2. Curva de equity
    3. Comparativo de estratégias
    4. Feature importance do ML
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger("quantbot.visualization")

plt.rcParams.update({
    "figure.facecolor": "#0a0a0a",       # Preto profundo
    "axes.facecolor": "#111111",          # Preto suave
    "axes.edgecolor": "#2a2a2a",          # Borda sutil
    "axes.labelcolor": "#D4A843",         # Dourado nos labels
    "text.color": "#e8e0d0",             # Texto off-white quente
    "xtick.color": "#8a7d6b",            # Ticks dourado fosco
    "ytick.color": "#8a7d6b",
    "grid.color": "#1a1a1a",             # Grid quase invisível
    "grid.alpha": 0.5,
    "font.size": 10,
    "figure.dpi": 120,
})

# Paleta QuantBot — Preto & Dourado Premium
GOLD = "#D4A843"          # Dourado principal
GOLD_LIGHT = "#F0D68A"    # Dourado claro
GOLD_DARK = "#A07830"     # Dourado escuro
BLACK = "#0a0a0a"         # Preto profundo
SURFACE = "#111111"       # Superfície
TEXT = "#e8e0d0"          # Texto principal
TEXT_MUTED = "#8a7d6b"    # Texto secundário

COLORS = {
    "buy": "#2ECC71",         # Verde esmeralda (compra)
    "sell": "#E74C3C",        # Vermelho rubi (venda)
    "price": GOLD,            # Dourado (preço)
    "equity": GOLD_LIGHT,     # Dourado claro (equity)
    "sma_short": "#F0D68A",   # Dourado claro (SMA curta)
    "sma_long": "#A07830",    # Dourado escuro (SMA longa)
    "title": GOLD,            # Títulos
    "accent": "#C9B458",      # Acento
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Sinais de compra/venda para detecção
from config.settings import Signal
BUY_SIGNALS = {Signal.BUY, Signal.STRONG_BUY}
SELL_SIGNALS = {Signal.SELL, Signal.STRONG_SELL}


def plot_backtest_signals(
    df: pd.DataFrame, ticker: str, strategy_name: str,
    total_return_pct: float = 0, win_rate: float = 0, sharpe_ratio: float = 0,
    save_path: Optional[Path] = None,
) -> Path:
    """Gráfico principal: preço + sinais + volume + RSI + MACD."""
    if df is None or df.empty:
        return Path()

    fig, axes = plt.subplots(4, 1, figsize=(16, 14),
                              gridspec_kw={"height_ratios": [3, 1, 1, 1]}, sharex=True)
    fig.suptitle(
        f"{strategy_name} - {ticker}  |  Retorno: {total_return_pct:+.2f}%  |  "
        f"Win Rate: {win_rate:.0%}  |  Sharpe: {sharpe_ratio:.2f}",
        fontsize=14, fontweight="bold", color=GOLD)

    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], color=COLORS["price"], linewidth=1.2, label="Preco")

    for col, color, label in [
        ("SMA_20", COLORS["sma_short"], "SMA 20"),
        ("SMA_50", COLORS["sma_long"], "SMA 50"),
        ("SMA_short", COLORS["sma_short"], "SMA Curta"),
        ("SMA_long", COLORS["sma_long"], "SMA Longa"),
    ]:
        if col in df.columns and col not in [c for c, _, _ in [("SMA_20", "", ""), ("SMA_50", "", "")][:0]]:
            if col in df.columns:
                ax1.plot(df.index, df[col], color=color, linewidth=0.8, alpha=0.7, label=label)
                break  # Usa só um par

    if "BB_upper" in df.columns and "BB_lower" in df.columns:
        ax1.fill_between(df.index, df["BB_lower"], df["BB_upper"],
                         alpha=0.08, color=GOLD, label="Bollinger")

    if "signal" in df.columns:
        buys = df[df["signal"].isin(BUY_SIGNALS)]
        sells = df[df["signal"].isin(SELL_SIGNALS)]
        ax1.scatter(buys.index, buys["Close"], marker="^", color=COLORS["buy"],
                    s=80, zorder=5, label=f"Compra ({len(buys)})", edgecolors="white", linewidth=0.5)
        ax1.scatter(sells.index, sells["Close"], marker="v", color=COLORS["sell"],
                    s=80, zorder=5, label=f"Venda ({len(sells)})", edgecolors="white", linewidth=0.5)

    ax1.set_ylabel("Preco ($)")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax1.grid(True)

    ax2 = axes[1]
    if "Volume" in df.columns:
        colors_vol = [COLORS["buy"] if df["Close"].iloc[i] >= df["Close"].iloc[max(0, i-1)]
                       else COLORS["sell"] for i in range(len(df))]
        ax2.bar(df.index, df["Volume"], color=colors_vol, alpha=0.6, width=0.8)
    ax2.set_ylabel("Volume")
    ax2.grid(True)

    ax3 = axes[2]
    if "RSI" in df.columns:
        ax3.plot(df.index, df["RSI"], color=GOLD_LIGHT, linewidth=1)
        ax3.axhline(y=70, color=COLORS["sell"], linestyle="--", alpha=0.5)
        ax3.axhline(y=30, color=COLORS["buy"], linestyle="--", alpha=0.5)
        ax3.fill_between(df.index, 30, 70, alpha=0.05, color=TEXT_MUTED)
        ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI")
    ax3.grid(True)

    ax4 = axes[3]
    if "MACD" in df.columns and "MACD_signal" in df.columns:
        ax4.plot(df.index, df["MACD"], color=GOLD, linewidth=1, label="MACD")
        ax4.plot(df.index, df["MACD_signal"], color="#ffa657", linewidth=1, label="Signal")
        if "MACD_hist" in df.columns:
            colors_h = [COLORS["buy"] if v >= 0 else COLORS["sell"] for v in df["MACD_hist"]]
            ax4.bar(df.index, df["MACD_hist"], color=colors_h, alpha=0.5, width=0.8)
        ax4.axhline(y=0, color=TEXT_MUTED, linewidth=0.5)
        ax4.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax4.set_ylabel("MACD")
    ax4.grid(True)

    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path is None:
        safe = strategy_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        save_path = RESULTS_DIR / f"backtest_{ticker}_{safe}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_equity_curve(
    equity_curve: pd.Series, ticker: str, strategy_name: str,
    initial_capital: float = 100_000, save_path: Optional[Path] = None,
) -> Path:
    """Curva de equity com fill verde/vermelho."""
    if equity_curve is None or len(equity_curve) == 0:
        return Path()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(equity_curve.index, equity_curve, color=COLORS["equity"], linewidth=1.5)
    ax.axhline(y=initial_capital, color=TEXT_MUTED, linestyle="--", linewidth=0.8,
               label=f"Capital Inicial (${initial_capital:,.0f})")
    ax.fill_between(equity_curve.index, initial_capital, equity_curve,
                     where=equity_curve >= initial_capital, alpha=0.15, color=COLORS["buy"])
    ax.fill_between(equity_curve.index, initial_capital, equity_curve,
                     where=equity_curve < initial_capital, alpha=0.15, color=COLORS["sell"])

    final = equity_curve.iloc[-1]
    ax.set_title(f"Equity - {strategy_name} - {ticker}  |  Final: ${final:,.0f}",
                  fontsize=12, fontweight="bold", color=GOLD_LIGHT)
    ax.set_ylabel("Patrimonio ($)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.3)
    ax.grid(True)
    plt.tight_layout()

    if save_path is None:
        safe = strategy_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        save_path = RESULTS_DIR / f"equity_{ticker}_{safe}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_strategy_comparison(
    results: List[dict], ticker: str, save_path: Optional[Path] = None,
) -> Path:
    """Comparativo de estratégias: barras + tabela."""
    if not results:
        return Path()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.5]})

    ax1 = axes[0]
    names = [r["strategy_name"] for r in results]
    returns = [r.get("total_return_pct", 0) for r in results]
    colors = [COLORS["buy"] if r >= 0 else COLORS["sell"] for r in returns]

    bars = ax1.barh(names, returns, color=colors, edgecolor="#2a2a2a", height=0.5)
    ax1.axvline(x=0, color=TEXT_MUTED, linewidth=0.8)
    for bar, val in zip(bars, returns):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.1f}%", va="center", fontsize=10, fontweight="bold",
                 color=COLORS["buy"] if val >= 0 else COLORS["sell"])
    ax1.set_title(f"Retorno Total - {ticker}", fontweight="bold", color=GOLD)
    ax1.set_xlabel("Retorno (%)")
    ax1.grid(True, axis="x")

    ax2 = axes[1]
    ax2.axis("off")
    headers = ["Estrategia", "Trades", "Win Rate", "Retorno", "Sharpe", "Drawdown"]
    table_data = [[
        r["strategy_name"], str(r.get("total_trades", 0)), f"{r.get('win_rate', 0):.0%}",
        f"{r.get('total_return_pct', 0):+.1f}%", f"{r.get('sharpe_ratio', 0):.2f}",
        f"-{r.get('max_drawdown_pct', 0):.1f}%",
    ] for r in results]

    table = ax2.table(cellText=table_data, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#30363d")
        if row == 0:
            cell.set_facecolor("#1a1a0a")
            cell.set_text_props(fontweight="bold", color=GOLD)
        else:
            cell.set_facecolor("#111111")
            cell.set_text_props(color="#e8e0d0")

    ax2.set_title("Comparativo de Metricas", fontweight="bold", color=GOLD, pad=20)
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / f"comparison_{ticker}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_feature_importance(importance: pd.Series, ticker: str, save_path: Optional[Path] = None) -> Path:
    """Feature importance do modelo ML."""
    if importance.empty:
        return Path()
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", ax=ax, color=GOLD_LIGHT, edgecolor="#2a2a2a")
    ax.set_title(f"Feature Importance (ML) - {ticker}", fontweight="bold", color=GOLD_LIGHT)
    ax.set_xlabel("Importancia")
    ax.grid(True, axis="x")
    plt.tight_layout()
    if save_path is None:
        save_path = RESULTS_DIR / f"feature_importance_{ticker}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
