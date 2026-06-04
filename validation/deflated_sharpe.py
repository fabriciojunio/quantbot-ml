"""
Probabilistic Sharpe Ratio (PSR) e Deflated Sharpe Ratio (DSR).

O problema do Sharpe ratio "cru"
--------------------------------
Um Sharpe de 2.0 num backtest parece ótimo. Mas:
    1. Ele ignora que retornos financeiros têm caudas gordas (curtose alta)
       e assimetria (skew) — o Sharpe assume normalidade.
    2. Ele ignora QUANTAS estratégias você testou. Se você testou 100
       variações e escolheu a melhor, é quase garantido que o melhor Sharpe
       seja inflado por sorte (selection bias / data snooping).

PSR (Bailey & López de Prado, 2012) responde: "qual a probabilidade de que
o Sharpe verdadeiro seja maior que um benchmark, dado o tamanho da amostra,
skew e curtose?"

DSR (Bailey & López de Prado, 2014) vai além: usa como benchmark o Sharpe
MÁXIMO ESPERADO sob a hipótese nula (sem skill), dado o número de tentativas.
Se o seu Sharpe não supera o que você esperaria obter só por sorte testando
N estratégias, o DSR será baixo — sinal de overfitting.

Regra prática: queremos DSR > 0.95 (95% de confiança de que há skill real).

Referências:
    - Bailey, D. & López de Prado, M. (2012). "The Sharpe Ratio Efficient
      Frontier." Journal of Risk, 15(2).
    - Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
      Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
      Journal of Portfolio Management, 40(5).
"""

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Sharpe ratio anualizado (sem taxa livre de risco para simplificar)."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    std = returns.std(ddof=1) if returns.size >= 2 else 0.0
    # Guard: retornos constantes têm std ~0 (a menos de erro de ponto
    # flutuante). Sem skill mensurável -> Sharpe indefinido, retornamos 0.
    if returns.size < 2 or std < 1e-10:
        return 0.0
    sr = returns.mean() / std
    return sr * np.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """
    Probabilistic Sharpe Ratio.

    Retorna P(SR_verdadeiro > benchmark_sr), corrigindo por skew e curtose.

    Args:
        observed_sr: Sharpe observado (NÃO anualizado — por período).
        benchmark_sr: Sharpe de referência (NÃO anualizado).
        n_observations: Número de retornos (T).
        skewness: Assimetria dos retornos.
        kurtosis: Curtose (não em excesso; normal = 3).

    Returns:
        Probabilidade em [0, 1].
    """
    if n_observations < 2:
        return 0.0
    denom = 1 - skewness * observed_sr + ((kurtosis - 1) / 4.0) * observed_sr ** 2
    if denom <= 0:
        return 0.0
    numerator = (observed_sr - benchmark_sr) * np.sqrt(n_observations - 1)
    psr = stats.norm.cdf(numerator / np.sqrt(denom))
    return float(psr)


def expected_max_sharpe(
    n_trials: int,
    sr_variance: float,
) -> float:
    """
    Sharpe máximo esperado sob a hipótese nula (sem skill), dado N tentativas.

    Este é o "benchmark" do Deflated Sharpe. Se você testou muitas estratégias,
    o melhor Sharpe esperado SÓ POR SORTE cresce — e seu Sharpe real precisa
    superar isso para ser considerado skill.

    Fórmula (Bailey & López de Prado, 2014):
        E[max SR] ≈ sqrt(Var(SR)) * [ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)) ]
    onde γ é a constante de Euler-Mascheroni e Z⁻¹ é a inversa da normal padrão.

    Args:
        n_trials: Número de estratégias/configurações testadas (N).
        sr_variance: Variância dos Sharpe ratios das tentativas.

    Returns:
        Sharpe máximo esperado por acaso (NÃO anualizado).
    """
    if n_trials < 1:
        return 0.0
    if n_trials == 1:
        return 0.0
    sr_std = np.sqrt(sr_variance)
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sr_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_variance: float,
    n_trials: int,
    n_observations: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """
    Deflated Sharpe Ratio (DSR).

    Probabilidade de que a estratégia tenha skill REAL, depois de corrigir por:
        - número de tentativas (selection bias / data snooping)
        - não-normalidade (skew, curtose)
        - tamanho da amostra

    Args:
        observed_sr: Sharpe observado da melhor estratégia (NÃO anualizado).
        sr_variance: Variância dos Sharpe das tentativas testadas.
        n_trials: Número de estratégias/configs testadas.
        n_observations: Número de retornos (T).
        skewness: Assimetria dos retornos.
        kurtosis: Curtose (normal = 3).

    Returns:
        DSR em [0, 1]. Interpretação: P(skill real). Queremos > 0.95.
    """
    sr_benchmark = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(
        observed_sr=observed_sr,
        benchmark_sr=sr_benchmark,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def evaluate_strategy(
    returns: np.ndarray,
    n_trials: int,
    sr_variance: float,
    periods_per_year: int = 252,
) -> dict:
    """
    Avaliação completa e honesta de uma estratégia.

    Retorna um dicionário com Sharpe anualizado, PSR, DSR e um veredito
    legível em português.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    n = returns.size

    if n < 2 or returns.std(ddof=1) == 0:
        return {
            "n_observations": int(n),
            "sharpe_annualized": 0.0,
            "psr": 0.0,
            "dsr": 0.0,
            "verdict": "DADOS INSUFICIENTES",
            "has_skill": False,
        }

    sr_per_period = returns.mean() / returns.std(ddof=1)
    sr_annual = sr_per_period * np.sqrt(periods_per_year)
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))  # normal = 3

    psr = probabilistic_sharpe_ratio(sr_per_period, 0.0, n, skew, kurt)
    dsr = deflated_sharpe_ratio(
        sr_per_period, sr_variance, n_trials, n, skew, kurt
    )

    if dsr > 0.95:
        verdict = "SKILL PROVAVEL (DSR > 0.95)"
    elif dsr > 0.90:
        verdict = "SKILL POSSIVEL, MAS INCERTO (0.90 < DSR < 0.95)"
    else:
        verdict = "PROVAVEL OVERFITTING / SORTE (DSR <= 0.90)"

    return {
        "n_observations": int(n),
        "sharpe_annualized": round(sr_annual, 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "psr": round(psr, 4),
        "dsr": round(dsr, 4),
        "n_trials": n_trials,
        "verdict": verdict,
        "has_skill": bool(dsr > 0.95),
    }
