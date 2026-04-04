"""
QuantBot ML — API Server (FastAPI)

Serve dados reais de mercado para o frontend React.

Endpoints:
    GET /api/health         — health check
    GET /api/assets         — todos os ativos com preços e indicadores
    GET /api/asset/{symbol} — dados de um ativo específico
    GET /api/selic          — taxa Selic atual
    GET /api/cdi            — histórico CDI
    GET /api/crypto         — dados CoinGecko
    GET /api/dashboard      — dados consolidados para o dashboard
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.market_data import (
    fetch_all_assets,
    fetch_selic_rate,
    fetch_cdi_history,
    fetch_crypto_market_data,
    compute_indicators,
    compute_signal,
)
from utils.logger import get_logger

logger = get_logger("quantbot.api.server")


# ═══════════════════════════════════════════════════════════════
# CACHE EM MEMÓRIA (evita chamadas repetidas às APIs)
# ═══════════════════════════════════════════════════════════════

class DataCache:
    """Cache thread-safe com TTL para dados de mercado."""

    def __init__(self):
        self._store: Dict[str, dict] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, key: str, ttl_seconds: int = 300) -> Optional[dict]:
        with self._lock:
            ts = self._timestamps.get(key, 0)
            if time.time() - ts < ttl_seconds:
                return self._store.get(key)
        return None

    def set(self, key: str, data: dict):
        with self._lock:
            self._store[key] = data
            self._timestamps[key] = time.time()


cache = DataCache()

# TTLs em segundos
TTL_ASSETS = 300       # 5 min
TTL_SELIC = 3600       # 1 hora
TTL_CDI = 3600         # 1 hora
TTL_CRYPTO = 120       # 2 min
TTL_DASHBOARD = 300    # 5 min


# ═══════════════════════════════════════════════════════════════
# RATE LIMITING POR IP
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    """Rate limiter simples por IP."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: Dict[str, list] = {}
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            reqs = self._requests.get(ip, [])
            reqs = [t for t in reqs if now - t < self.window]
            if len(reqs) >= self.max_requests:
                return False
            reqs.append(now)
            self._requests[ip] = reqs
            return True


rate_limiter = RateLimiter(max_requests=60, window_seconds=60)


# ═══════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="QuantBot ML API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

# CORS — permite apenas origens locais do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Aplica rate limiting em todas as requisições."""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests. Try again later."},
        )
    response = await call_next(request)
    return response


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/assets")
async def get_assets():
    """Retorna todos os ativos com preços históricos."""
    cached = cache.get("assets", TTL_ASSETS)
    if cached:
        return cached

    data = fetch_all_assets(period_days=90)
    if not data:
        raise HTTPException(status_code=503, detail="Failed to fetch market data")

    # Adiciona indicadores e sinais
    for sym, asset in data.items():
        prices = asset.get("prices", [])
        indicators = compute_indicators(prices)
        signal = compute_signal(indicators)
        asset["indicators"] = indicators
        asset["signal"] = signal

    result = {"assets": data, "updated_at": datetime.now().isoformat()}
    cache.set("assets", result)
    return result


@app.get("/api/asset/{symbol}")
async def get_asset(symbol: str):
    """Retorna dados de um ativo específico."""
    # Valida input
    if not symbol.isalnum() and "-" not in symbol and "." not in symbol:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    if len(symbol) > 15:
        raise HTTPException(status_code=400, detail="Symbol too long")

    cached = cache.get("assets", TTL_ASSETS)
    if cached and symbol in cached.get("assets", {}):
        return cached["assets"][symbol]

    # Se não está no cache, busca todos
    assets_resp = await get_assets()
    asset = assets_resp.get("assets", {}).get(symbol)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset {symbol} not found")
    return asset


@app.get("/api/selic")
async def get_selic():
    """Retorna taxa Selic atual."""
    cached = cache.get("selic", TTL_SELIC)
    if cached:
        return cached

    rate = fetch_selic_rate()
    if rate is None:
        raise HTTPException(status_code=503, detail="Failed to fetch Selic rate")

    result = {"selic": rate, "updated_at": datetime.now().isoformat()}
    cache.set("selic", result)
    return result


@app.get("/api/cdi")
async def get_cdi():
    """Retorna histórico CDI dos últimos 30 dias."""
    cached = cache.get("cdi", TTL_CDI)
    if cached:
        return cached

    data = fetch_cdi_history(30)
    if data is None:
        raise HTTPException(status_code=503, detail="Failed to fetch CDI data")

    result = {"cdi": data, "updated_at": datetime.now().isoformat()}
    cache.set("cdi", result)
    return result


@app.get("/api/crypto")
async def get_crypto():
    """Retorna dados de mercado crypto do CoinGecko."""
    cached = cache.get("crypto", TTL_CRYPTO)
    if cached:
        return cached

    data = fetch_crypto_market_data()
    if not data:
        raise HTTPException(status_code=503, detail="Failed to fetch crypto data")

    result = {"crypto": data, "updated_at": datetime.now().isoformat()}
    cache.set("crypto", result)
    return result


@app.get("/api/dashboard")
async def get_dashboard():
    """
    Endpoint consolidado — retorna todos os dados para o dashboard.
    Busca ativos, selic, CDI e crypto em uma única chamada.
    """
    cached = cache.get("dashboard", TTL_DASHBOARD)
    if cached:
        return cached

    # Busca todos os dados
    assets_data = fetch_all_assets(period_days=90)

    # Enriquece com indicadores e sinais
    for sym, asset in assets_data.items():
        prices = asset.get("prices", [])
        indicators = compute_indicators(prices)
        signal = compute_signal(indicators)
        asset["indicators"] = indicators
        asset["signal"] = signal

    selic = fetch_selic_rate()
    cdi = fetch_cdi_history(30)
    crypto_extra = fetch_crypto_market_data()

    # Portfolio simulado (baseado nos preços reais)
    portfolio = []
    total_value = 0
    for sym, asset in assets_data.items():
        cp = asset["current_price"]
        market = asset["market"]
        if market == "Crypto":
            qty = 0.15 if sym == "BTC" else (1.5 if sym == "ETH" else 25.0)
        else:
            qty = 100
        val = cp * qty
        total_value += val
        portfolio.append({
            **asset,
            "qty": qty,
            "value": round(val, 2),
        })

    # Calcula pesos
    for p in portfolio:
        p["weight"] = round((p["value"] / total_value) * 100, 2) if total_value > 0 else 0

    # Alocação por mercado
    allocation = {}
    for p in portfolio:
        m = p["market"]
        allocation[m] = allocation.get(m, 0) + p["value"]
    for m in allocation:
        allocation[m] = round((allocation[m] / total_value) * 100, 2) if total_value > 0 else 0

    result = {
        "assets": assets_data,
        "portfolio": portfolio,
        "total_value": round(total_value, 2),
        "allocation": allocation,
        "selic": selic,
        "cdi": cdi,
        "crypto_extra": crypto_extra,
        "updated_at": datetime.now().isoformat(),
    }

    cache.set("dashboard", result)
    cache.set("assets", {"assets": assets_data, "updated_at": result["updated_at"]})
    return result
