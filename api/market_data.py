"""Serviço de dados de mercado para a API: indicadores e sinais.

`compute_indicators` deriva indicadores técnicos de uma lista de preços;
`compute_signal` os converte num sinal legível (COMPRA/VENDA/NEUTRO) com
score e confiança. As funções `fetch_*` são pontos de integração externos
(BCB, brapi, etc.) — em testes elas são mockadas.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

_MIN_POINTS = 50


def compute_indicators(prices: List[float]) -> Dict[str, float]:
    """Calcula indicadores técnicos. Retorna {} se faltarem dados."""
    if prices is None or len(prices) < _MIN_POINTS:
        return {}

    arr = np.asarray(prices, dtype=float)
    returns = np.diff(arr) / arr[:-1]

    # RSI (14 períodos)
    deltas = np.diff(arr[-15:])
    gains = deltas[deltas > 0].sum() / 14
    losses = -deltas[deltas < 0].sum() / 14
    rs = gains / losses if losses > 0 else np.inf
    rsi = 100.0 if np.isinf(rs) else 100 - (100 / (1 + rs))

    sma20 = float(np.mean(arr[-20:]))
    sma50 = float(np.mean(arr[-50:]))

    # MACD (EMA12 - EMA26)
    ema12 = _ema(arr, 12)
    ema26 = _ema(arr, 26)
    macd = float(ema12 - ema26)

    vol = float(np.std(returns, ddof=1) * np.sqrt(252) * 100) if len(returns) > 1 else 0.0
    mom = float((arr[-1] / arr[-10] - 1) * 100) if len(arr) >= 10 else 0.0
    std_r = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / std_r * np.sqrt(252)) if std_r > 0 else 0.0

    return {
        "rsi": round(float(rsi), 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "macd": round(macd, 4),
        "vol": round(vol, 2),
        "mom": round(mom, 2),
        "sharpe": round(sharpe, 2),
    }


def _ema(arr: np.ndarray, span: int) -> float:
    """Última EMA de uma janela."""
    alpha = 2 / (span + 1)
    ema = arr[0]
    for price in arr[1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema


def compute_signal(indicators: Dict[str, float]) -> Dict[str, object]:
    """Converte indicadores num sinal legível com score (0-100) e confiança."""
    rsi = indicators.get("rsi", 50.0)
    macd = indicators.get("macd", 0.0)
    mom = indicators.get("mom", 0.0)

    rsi_score = (50 - rsi) * 0.8           # RSI baixo => comprador
    macd_score = float(np.clip(macd * 2.0, -15, 15))
    mom_score = float(np.clip(mom * 1.0, -15, 15))

    score = 50 + rsi_score + macd_score + mom_score
    score = float(np.clip(score, 5, 95))

    if score >= 70:
        signal = "COMPRA_FORTE"
    elif score >= 58:
        signal = "COMPRA"
    elif score <= 30:
        signal = "VENDA_FORTE"
    elif score <= 42:
        signal = "VENDA"
    else:
        signal = "NEUTRO"

    confidence = float(min(100.0, abs(score - 50) * 2))

    return {
        "score": round(score, 1),
        "signal": signal,
        "confidence": round(confidence, 1),
    }


# ── Pontos de integração externos (mockados em testes) ────────────────

def fetch_all_assets() -> Dict[str, dict]:  # pragma: no cover - integração
    """Retorna dados brutos dos ativos do universo. Stub para produção."""
    return {}


def fetch_selic_rate() -> Optional[float]:  # pragma: no cover - integração
    """Taxa Selic atual (via BCB). Stub para produção."""
    return None


def fetch_cdi_history() -> List[dict]:  # pragma: no cover - integração
    """Histórico do CDI. Stub para produção."""
    return []


def fetch_crypto_market_data() -> Dict[str, dict]:  # pragma: no cover - integração
    """Dados de mercado de criptomoedas. Stub para produção."""
    return {}
