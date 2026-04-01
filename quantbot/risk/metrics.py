"""
Cálculo de métricas de risco para portfólios.

Implementa Value at Risk (VaR), Conditional VaR, Beta, Alpha,
e outras métricas essenciais para gestão de risco.

Referências:
- Jorion, P. (2006). Value at Risk. McGraw-Hill.
- Sharpe, W. F. (1966). Mutual Fund Performance.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional

from utils.logger import get_logger

logger = get_logger("quantbot.risk.metrics")


@dataclass
class RiskMetrics:
    """Métricas de risco consolidadas."""
    var_95: float        # Value at Risk 95%
    var_99: float        # Value at Risk 99%
    cvar_95: float       # Conditional VaR 95% (Expected Shortfall)
    beta: float          # Sensibilidade ao mercado
    alpha: float         # Retorno acima do benchmark (anualizado)
    volatility: float    # Volatilidade anualizada
    max_drawdown: float  # Maior queda do pico
    current_drawdown: float  # Queda atual do pico
    sharpe_ratio: float  # Retorno ajustado ao risco
    sortino_ratio: float  # Sharpe usando apenas downside volatility

    def to_dict(self) -> dict:
        return {k: round(v, 6) for k, v in self.__dict__.items()}


def calculate_risk_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> RiskMetrics:
    """
    Calcula métricas de risco completas.

    Args:
        returns: Série de retornos diários
        benchmark_returns: Retornos do benchmark (opcional)
        risk_free_rate: Taxa livre de risco anualizada

    Returns:
        RiskMetrics com todas as métricas calculadas
    """
    returns_clean = returns.dropna()

    if len(returns_clean) < 30:
        logger.warning("Dados insuficientes para cálculo de risco confiável")

    # ── VaR (Value at Risk) ────────────────────────────
    # Percentil dos retornos: perda máxima esperada com X% de confiança
    var_95 = float(np.percentile(returns_clean, 5))
    var_99 = float(np.percentile(returns_clean, 1))

    # ── CVaR (Conditional VaR / Expected Shortfall) ────
    # Média dos retornos abaixo do VaR: "quando dá ruim, quão ruim fica?"
    cvar_95 = float(returns_clean[returns_clean <= var_95].mean())
    if np.isnan(cvar_95):
        cvar_95 = var_95

    # ── Volatilidade ───────────────────────────────────
    # Desvio padrão anualizado dos retornos
    daily_vol = returns_clean.std()
    volatility = float(daily_vol * np.sqrt(252))

    # ── Drawdown ───────────────────────────────────────
    # Queda percentual do pico histórico
    cum_returns = (1 + returns_clean).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())
    current_drawdown = float(drawdowns.iloc[-1]) if len(drawdowns) > 0 else 0.0

    # ── Sharpe Ratio ───────────────────────────────────
    # (Retorno - Rf) / Volatilidade
    daily_rf = risk_free_rate / 252
    excess_return = returns_clean.mean() - daily_rf
    sharpe = float(excess_return / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0.0

    # ── Sortino Ratio ──────────────────────────────────
    # Usa apenas volatilidade negativa (downside deviation)
    downside_returns = returns_clean[returns_clean < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else daily_vol
    sortino = float(excess_return * 252 / downside_vol) if downside_vol > 0 else 0.0

    # ── Beta e Alpha ───────────────────────────────────
    beta = 0.0
    alpha = 0.0

    if benchmark_returns is not None:
        bench_clean = benchmark_returns.dropna()
        # Alinha séries pelo índice
        aligned = pd.concat(
            [returns_clean, bench_clean], axis=1, keys=["port", "bench"]
        ).dropna()

        if len(aligned) > 30:
            cov_matrix = np.cov(aligned["port"], aligned["bench"])
            bench_var = cov_matrix[1, 1]

            if bench_var > 0:
                # Beta = Cov(R_p, R_b) / Var(R_b)
                beta = float(cov_matrix[0, 1] / bench_var)
                # Alpha = R_p - Beta * R_b (anualizado)
                alpha = float(
                    (aligned["port"].mean() - beta * aligned["bench"].mean()) * 252
                )

    return RiskMetrics(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        beta=beta,
        alpha=alpha,
        volatility=volatility,
        max_drawdown=max_drawdown,
        current_drawdown=current_drawdown,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
    )
