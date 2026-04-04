"""
Serviço de dados de mercado — busca dados reais de múltiplas fontes.

Fontes:
- Yahoo Finance (yfinance): ações BR, US e crypto
- Banco Central do Brasil (BCB): CDI/Selic
- CoinGecko: dados complementares de crypto
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from utils.logger import get_logger

logger = get_logger("quantbot.api.market_data")

# Rate limiting global
_last_request: Dict[str, float] = {}
_MIN_INTERVAL = 1.0  # segundos entre requisições por fonte


def _rate_limit(source: str):
    now = time.time()
    last = _last_request.get(source, 0)
    wait = _MIN_INTERVAL - (now - last)
    if wait > 0:
        time.sleep(wait)
    _last_request[source] = time.time()


# ═══════════════════════════════════════════════════════════════
# YAHOO FINANCE — Ações e Crypto
# ═══════════════════════════════════════════════════════════════

ASSETS = {
    "B3": [
        {"symbol": "PETR4.SA", "name": "Petrobras", "sector": "Energia"},
        {"symbol": "VALE3.SA", "name": "Vale", "sector": "Mineração"},
        {"symbol": "ITUB4.SA", "name": "Itaú", "sector": "Financeiro"},
        {"symbol": "WEGE3.SA", "name": "WEG", "sector": "Industrial"},
        {"symbol": "BBDC4.SA", "name": "Bradesco", "sector": "Financeiro"},
    ],
    "US": [
        {"symbol": "AAPL", "name": "Apple", "sector": "Tech"},
        {"symbol": "MSFT", "name": "Microsoft", "sector": "Tech"},
        {"symbol": "NVDA", "name": "NVIDIA", "sector": "Tech"},
        {"symbol": "GOOGL", "name": "Alphabet", "sector": "Tech"},
        {"symbol": "AMZN", "name": "Amazon", "sector": "Tech"},
    ],
    "Crypto": [
        {"symbol": "BTC-USD", "name": "Bitcoin", "sector": "Crypto"},
        {"symbol": "ETH-USD", "name": "Ethereum", "sector": "Crypto"},
        {"symbol": "SOL-USD", "name": "Solana", "sector": "Crypto"},
    ],
}

# Display name (sem sufixo .SA e -USD)
DISPLAY_NAMES = {
    "PETR4.SA": "PETR4", "VALE3.SA": "VALE3", "ITUB4.SA": "ITUB4",
    "WEGE3.SA": "WEGE3", "BBDC4.SA": "BBDC4",
    "BTC-USD": "BTC", "ETH-USD": "ETH", "SOL-USD": "SOL",
}


def fetch_asset_history(symbol: str, period_days: int = 90) -> Optional[pd.DataFrame]:
    """Busca histórico de preços de um ativo via yfinance."""
    _rate_limit("yfinance")
    try:
        end = datetime.now()
        start = end - timedelta(days=int(period_days * 1.6))
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            logger.warning(f"Sem dados para {symbol}")
            return None
        hist = hist.tail(period_days)
        logger.info(f"  ✓ {symbol}: {len(hist)} pontos")
        return hist
    except Exception as e:
        logger.error(f"  ✗ {symbol}: {e}")
        return None


def fetch_all_assets(period_days: int = 90) -> Dict:
    """Busca dados de todos os ativos do universo."""
    result = {}
    all_assets = []
    for market, assets in ASSETS.items():
        all_assets.extend([(market, a) for a in assets])

    for market, asset in all_assets:
        sym = asset["symbol"]
        display = DISPLAY_NAMES.get(sym, sym)
        hist = fetch_asset_history(sym, period_days)
        if hist is not None and len(hist) >= 2:
            closes = hist["Close"].values.tolist()
            volumes = hist["Volume"].values.tolist() if "Volume" in hist else []
            dates = [d.strftime("%Y-%m-%d") for d in hist.index]
            current_price = closes[-1]
            prev_price = closes[-2] if len(closes) > 1 else closes[-1]
            change_pct = ((current_price - prev_price) / prev_price) * 100

            result[display] = {
                "symbol": display,
                "yf_symbol": sym,
                "name": asset["name"],
                "sector": asset["sector"],
                "market": market,
                "current_price": round(current_price, 2),
                "change_pct": round(change_pct, 2),
                "prices": [round(p, 2) for p in closes],
                "volumes": [int(v) for v in volumes] if volumes else [],
                "dates": dates,
            }

    return result


# ═══════════════════════════════════════════════════════════════
# INDICADORES TÉCNICOS
# ═══════════════════════════════════════════════════════════════

def compute_indicators(prices: List[float]) -> Dict:
    """Calcula indicadores técnicos a partir de preços de fechamento."""
    if len(prices) < 20:
        return {}

    arr = np.array(prices, dtype=float)
    returns = np.diff(arr) / arr[:-1]

    # SMA
    sma20 = float(np.mean(arr[-20:]))
    sma50 = float(np.mean(arr[-50:])) if len(arr) >= 50 else sma20

    # RSI (14 períodos)
    gains = returns[-14:][returns[-14:] > 0]
    losses = np.abs(returns[-14:][returns[-14:] < 0])
    avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.001
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # MACD simplificado
    macd = ((sma20 - sma50) / sma50) * 100 if sma50 != 0 else 0.0

    # Volatilidade anualizada
    recent_ret = returns[-20:] if len(returns) >= 20 else returns
    vol = float(np.std(recent_ret) * np.sqrt(252) * 100)

    # Momentum 20d
    mom = ((arr[-1] - arr[-21]) / arr[-21]) * 100 if len(arr) > 21 else 0.0

    # Sharpe
    mean_ret = float(np.mean(recent_ret))
    std_ret = float(np.std(recent_ret))
    sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    return {
        "rsi": round(min(100, max(0, rsi)), 1),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "macd": round(macd, 2),
        "vol": round(vol, 1),
        "mom": round(mom, 2),
        "sharpe": round(sharpe, 2),
    }


# ═══════════════════════════════════════════════════════════════
# SINAIS ML (simplificado baseado em indicadores)
# ═══════════════════════════════════════════════════════════════

def compute_signal(indicators: Dict) -> Dict:
    """Gera sinal de trading baseado em indicadores técnicos."""
    if not indicators:
        return {"score": 50, "signal": "NEUTRO", "confidence": 50}

    score = 50.0
    rsi = indicators.get("rsi", 50)
    macd = indicators.get("macd", 0)
    mom = indicators.get("mom", 0)

    if rsi < 30:
        score += 15
    elif rsi > 70:
        score -= 15

    if macd > 0:
        score += 10
    else:
        score -= 10

    if mom > 0:
        score += 8
    elif mom < -5:
        score -= 8

    score = max(5, min(95, score))

    if score > 75:
        signal = "COMPRA_FORTE"
    elif score > 65:
        signal = "COMPRA"
    elif score > 35:
        signal = "NEUTRO"
    elif score > 25:
        signal = "VENDA"
    else:
        signal = "VENDA_FORTE"

    confidence = min(95, 55 + abs(score - 50))

    return {
        "score": round(score),
        "signal": signal,
        "confidence": round(confidence),
    }


# ═══════════════════════════════════════════════════════════════
# BCB — CDI / SELIC
# ═══════════════════════════════════════════════════════════════

BCB_SELIC_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados/ultimos/1?formato=json"
BCB_CDI_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados/ultimos/30?formato=json"


def fetch_selic_rate() -> Optional[float]:
    """Busca taxa Selic atual do Banco Central."""
    try:
        _rate_limit("bcb")
        resp = requests.get(BCB_SELIC_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data and len(data) > 0:
            rate = float(data[0]["valor"])
            logger.info(f"  ✓ Selic: {rate}%")
            return rate
    except Exception as e:
        logger.warning(f"  ✗ Selic: {e}")
    return None


def fetch_cdi_history(days: int = 30) -> Optional[List[Dict]]:
    """Busca histórico CDI diário do BCB."""
    try:
        _rate_limit("bcb")
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados/ultimos/{days}?formato=json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [{"date": d["data"], "rate": float(d["valor"])} for d in data]
    except Exception as e:
        logger.warning(f"  ✗ CDI: {e}")
    return None


# ═══════════════════════════════════════════════════════════════
# COINGECKO — Dados complementares crypto
# ═══════════════════════════════════════════════════════════════

COINGECKO_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def fetch_crypto_market_data() -> Dict:
    """Busca dados de mercado crypto do CoinGecko."""
    result = {}
    ids = ",".join(COINGECKO_IDS.values())
    try:
        _rate_limit("coingecko")
        resp = requests.get(
            f"{COINGECKO_BASE}/simple/price",
            params={
                "ids": ids,
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        for sym, cg_id in COINGECKO_IDS.items():
            if cg_id in data:
                d = data[cg_id]
                result[sym] = {
                    "price": d.get("usd", 0),
                    "market_cap": d.get("usd_market_cap", 0),
                    "volume_24h": d.get("usd_24h_vol", 0),
                    "change_24h": d.get("usd_24h_change", 0),
                }
    except Exception as e:
        logger.warning(f"  ✗ CoinGecko: {e}")
    return result
