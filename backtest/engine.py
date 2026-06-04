"""Motor de backtest vetorizado simples (long/flat).

Interpreta o sinal como posição-alvo por barra (1 = comprado, 0 = fora) e
aplica retorno do ativo com defasagem de 1 barra para evitar lookahead.
Para custos realistas da B3, use backtest/realistic_backtester.py — este
motor é o caminho rápido para sanity-checks e comparações.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    equity_curve: pd.Series
    benchmark_return: float = 0.0
    benchmark_curve: Optional[pd.Series] = None


class BacktestEngine:
    """Backtest long/flat sobre preços de fechamento."""

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> BacktestResult:
        if df is None or len(df) == 0:
            raise ValueError("DataFrame de preços vazio")

        close = df["close"] if "close" in df.columns else df["Close"]
        close = close.astype(float)

        # Posição-alvo por barra, executada na barra seguinte (sem lookahead).
        position = pd.Series(signals, index=df.index).reindex(df.index).fillna(0)
        position = (position > 0).astype(float)
        asset_ret = close.pct_change().fillna(0.0)
        strat_ret = position.shift(1).fillna(0.0) * asset_ret

        equity = self.initial_capital * (1 + strat_ret).cumprod()
        total_return = float(equity.iloc[-1] / self.initial_capital - 1.0)

        std = float(strat_ret.std(ddof=1))
        sharpe = (
            float(strat_ret.mean() / std * np.sqrt(_TRADING_DAYS))
            if std > 0 else 0.0
        )

        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_dd = float(drawdown.min())

        # Conta trades por transições de posição (0->1 ou 1->0).
        changes = position.diff().fillna(position).abs()
        total_trades = int((changes > 0).sum())

        # Win rate por trade fechado (entradas seguidas de saída).
        trade_returns = self._trade_returns(position, asset_ret)
        win_rate = (
            float((np.array(trade_returns) > 0).mean()) if trade_returns else 0.0
        )

        bench_return = 0.0
        bench_curve = None
        if benchmark_prices is not None and len(benchmark_prices) > 0:
            bp = pd.Series(benchmark_prices).astype(float)
            bench_curve = self.initial_capital * (bp / bp.iloc[0])
            bench_return = float(bp.iloc[-1] / bp.iloc[0] - 1.0)

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_trades=total_trades,
            win_rate=win_rate,
            equity_curve=equity,
            benchmark_return=bench_return,
            benchmark_curve=bench_curve,
        )

    @staticmethod
    def _trade_returns(position: pd.Series, asset_ret: pd.Series) -> list[float]:
        """Retorno acumulado de cada período em que esteve posicionado."""
        trades: list[float] = []
        in_trade = False
        cum = 1.0
        exec_pos = position.shift(1).fillna(0.0)
        for i in range(len(exec_pos)):
            if exec_pos.iloc[i] > 0:
                cum *= 1 + asset_ret.iloc[i]
                in_trade = True
            elif in_trade:
                trades.append(cum - 1.0)
                cum = 1.0
                in_trade = False
        if in_trade:
            trades.append(cum - 1.0)
        return trades
