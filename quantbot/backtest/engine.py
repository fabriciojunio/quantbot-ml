"""
Motor de Backtesting.

Simula a execução da estratégia em dados históricos
com custos de transação realistas.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import BACKTEST_CONFIG
from utils.logger import get_logger

logger = get_logger("quantbot.backtest")


@dataclass
class BacktestResult:
    """Resultado consolidado do backtest."""
    total_return: float
    benchmark_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    calmar_ratio: float
    alpha: float
    equity_curve: Optional[pd.Series] = None
    benchmark_curve: Optional[pd.Series] = None
    trades: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if k not in ("equity_curve", "benchmark_curve", "trades")}


class BacktestEngine:
    """
    Motor de backtesting com custos de transação.

    Simula trades baseados em sinais e calcula métricas
    de performance padrão da indústria.
    """

    def __init__(self, config=None):
        self.config = config or BACKTEST_CONFIG

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Executa backtest da estratégia.

        Args:
            prices: DataFrame com coluna 'close'
            signals: Series com sinais (1=BUY, 0=SELL/HOLD)
            benchmark_prices: Preços do benchmark para comparação

        Returns:
            BacktestResult com métricas completas
        """
        close = prices["close"]
        cash = self.config.initial_capital
        position = 0
        equity_curve = []
        trades = []
        entry_price = 0.0

        commission = self.config.commission_rate
        slippage = self.config.slippage_rate

        for i in range(len(close)):
            date = close.index[i]
            price = close.iloc[i]

            equity = cash + position * price
            equity_curve.append({"date": date, "equity": equity})

            if i >= len(signals):
                continue

            signal = signals.iloc[i] if i < len(signals) else 0

            # BUY
            if signal == 1 and position == 0:
                exec_price = price * (1 + slippage)
                shares = int((cash * self.config.max_allocation_pct) / exec_price)

                if shares > 0:
                    cost = shares * exec_price * (1 + commission)
                    if cost <= cash:
                        cash -= cost
                        position = shares
                        entry_price = exec_price
                        trades.append({
                            "date": str(date),
                            "type": "BUY",
                            "price": exec_price,
                            "shares": shares,
                        })

            # SELL
            elif signal == 0 and position > 0:
                exec_price = price * (1 - slippage)
                revenue = position * exec_price * (1 - commission)
                pnl = (exec_price - entry_price) / entry_price * 100

                cash += revenue
                trades.append({
                    "date": str(date),
                    "type": "SELL",
                    "price": exec_price,
                    "shares": position,
                    "pnl": pnl,
                })
                position = 0

        # Resultado
        final_equity = cash + position * close.iloc[-1]
        eq_series = pd.Series(
            [e["equity"] for e in equity_curve],
            index=[e["date"] for e in equity_curve],
        )

        # Benchmark
        bench_curve = None
        bench_return = 0.0
        if benchmark_prices is not None:
            bench_ret = benchmark_prices.pct_change().fillna(0)
            bench_curve = self.config.initial_capital * (1 + bench_ret).cumprod()
            bench_return = (bench_curve.iloc[-1] / self.config.initial_capital - 1) * 100

        # Métricas
        returns = eq_series.pct_change().dropna()
        total_return = (final_equity / self.config.initial_capital - 1) * 100

        # Sharpe
        avg_ret = returns.mean() * 252
        std_ret = returns.std() * np.sqrt(252)
        sharpe = avg_ret / std_ret if std_ret > 0 else 0

        # Sortino
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = avg_ret / downside if downside > 0 else 0

        # Max Drawdown
        cum_ret = (1 + returns).cumprod()
        rolling_max = cum_ret.expanding().max()
        max_dd = ((cum_ret - rolling_max) / rolling_max).min() * 100

        # Win Rate
        sell_trades = [t for t in trades if t["type"] == "SELL"]
        wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
        win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0

        # Profit Factor
        gross_profit = sum(t.get("pnl", 0) for t in wins) if wins else 0
        losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]
        gross_loss = abs(sum(t.get("pnl", 0) for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Calmar
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0

        # Alpha
        alpha = total_return - bench_return

        return BacktestResult(
            total_return=total_return,
            benchmark_return=bench_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            calmar_ratio=calmar,
            alpha=alpha,
            equity_curve=eq_series,
            benchmark_curve=bench_curve,
            trades=trades,
        )
