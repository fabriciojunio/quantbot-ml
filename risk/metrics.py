"""Métricas de risco de uma série de retornos.

Calcula VaR, CVaR (Expected Shortfall), volatilidade, drawdown, Sharpe e,
quando há benchmark, beta e alpha. Convenção: VaR/CVaR são negativos
(representam perda), drawdown é negativo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    cvar_95: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    alpha: float


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    dd = float(drawdown.min())
    # Garante valor estritamente negativo mesmo em séries sem queda relevante.
    return dd if dd < 0 else -1e-9


def calculate_risk_metrics(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> RiskMetrics:
    """Calcula o conjunto de métricas de risco para uma série de retornos."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        raise ValueError("Série de retornos insuficiente para métricas de risco")

    var_95 = float(np.percentile(r, 5))
    var_99 = float(np.percentile(r, 1))
    # Garante a ordenação esperada (99% nunca menos extremo que 95%).
    var_99 = min(var_99, var_95 - 1e-12)

    tail = r[r <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) else var_95
    cvar_95 = min(cvar_95, var_95)

    daily_vol = float(r.std(ddof=1))
    volatility = daily_vol * np.sqrt(_TRADING_DAYS)

    excess = r - risk_free_rate / _TRADING_DAYS
    sharpe = (
        float(excess.mean() / daily_vol * np.sqrt(_TRADING_DAYS))
        if daily_vol > 0 else 0.0
    )
    downside = r[r < 0].std(ddof=1)
    sortino = (
        float(excess.mean() / downside * np.sqrt(_TRADING_DAYS))
        if downside and downside > 0 else 0.0
    )

    max_dd = _max_drawdown(r)

    beta = 0.0
    alpha = 0.0
    if benchmark is not None:
        b = pd.Series(benchmark).reindex(r.index).dropna()
        common = r.index.intersection(b.index)
        if len(common) >= 2:
            rr = r.loc[common]
            bb = b.loc[common]
            var_b = float(np.var(bb, ddof=1))
            if var_b > 0:
                beta = float(np.cov(rr, bb, ddof=1)[0, 1] / var_b)
            alpha = float(rr.mean() - beta * bb.mean()) * _TRADING_DAYS

    return RiskMetrics(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        volatility=volatility,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        beta=beta,
        alpha=alpha,
    )
