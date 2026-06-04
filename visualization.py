"""Geração de gráficos de backtest, equity e comparação de estratégias.

Usa o backend não interativo Agg (seguro em servidores/headless). Cada
função salva um PNG em `output/charts/` e retorna o Path do arquivo.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config.settings import Signal  # noqa: E402

_OUTPUT_DIR = Path("output") / "charts"


def _ensure_dir() -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


def plot_backtest_signals(df: pd.DataFrame, symbol: str, strategy_name: str) -> Path:
    """Plota o preço com marcadores de compra/venda."""
    out = _ensure_dir() / f"signals_{symbol}_{strategy_name}.png"
    close = df["Close"] if "Close" in df.columns else df["close"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, close, label="Preço", color="#222", linewidth=1.0)

    if "signal" in df.columns:
        buys = df[df["signal"].isin([Signal.BUY, Signal.STRONG_BUY])]
        sells = df[df["signal"].isin([Signal.SELL, Signal.STRONG_SELL])]
        bclose = buys["Close"] if "Close" in buys.columns else buys.get("close")
        sclose = sells["Close"] if "Close" in sells.columns else sells.get("close")
        if bclose is not None and len(buys):
            ax.scatter(buys.index, bclose, marker="^", color="green", label="Compra")
        if sclose is not None and len(sells):
            ax.scatter(sells.index, sclose, marker="v", color="red", label="Venda")

    ax.set_title(f"{symbol} — {strategy_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def plot_equity_curve(equity: pd.Series, symbol: str, name: str) -> Path:
    """Plota a curva de capital ao longo do tempo."""
    out = _ensure_dir() / f"equity_{symbol}_{name}.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity.values, color="#b8860b", linewidth=1.5)
    ax.set_title(f"Curva de Capital — {symbol} ({name})")
    ax.set_ylabel("Capital")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def plot_strategy_comparison(data: List[dict], symbol: str) -> Path:
    """Plota um comparativo de retorno/sharpe entre estratégias."""
    out = _ensure_dir() / f"comparison_{symbol}.png"
    names = [d.get("strategy_name", f"S{i}") for i, d in enumerate(data)]
    returns = [d.get("total_return_pct", 0.0) for d in data]
    sharpes = [d.get("sharpe_ratio", 0.0) for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.bar(names, returns, color="#b8860b")
    ax1.set_title("Retorno total (%)")
    ax1.tick_params(axis="x", rotation=45)
    ax2.bar(names, sharpes, color="#222")
    ax2.set_title("Sharpe ratio")
    ax2.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Comparação de Estratégias — {symbol}")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out
