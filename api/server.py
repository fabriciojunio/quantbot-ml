"""Servidor da API REST do QuantBot (FastAPI).

Expõe dados de mercado, taxas (Selic/CDI), cripto e sinais para o dashboard.
Inclui cache em memória com TTL, rate limiting por IP e CORS restrito às
origens do front-end. Apenas leitura (GET) — não executa ordens.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.market_data import (
    compute_indicators,
    compute_signal,
    fetch_all_assets,
    fetch_cdi_history,
    fetch_crypto_market_data,
    fetch_selic_rate,
)

# Símbolo válido: curto, alfanumérico com ponto/hífen.
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,14}$")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


# ═══════════════════════════════════════════════════════════════
# CACHE EM MEMÓRIA COM TTL
# ═══════════════════════════════════════════════════════════════

class DataCache:
    def __init__(self) -> None:
        self._store: Dict[str, object] = {}
        self._timestamps: Dict[str, float] = {}

    def set(self, key: str, value: object) -> None:
        self._store[key] = value
        self._timestamps[key] = time.time()

    def get(self, key: str, ttl_seconds: float = 60.0):
        if key not in self._store:
            return None
        if time.time() - self._timestamps.get(key, 0) > ttl_seconds:
            return None
        return self._store[key]


# ═══════════════════════════════════════════════════════════════
# RATE LIMITER POR IP (janela deslizante simples)
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        window = self._hits[ip]
        # Remove hits fora da janela.
        self._hits[ip] = [t for t in window if now - t < self.window_seconds]
        if len(self._hits[ip]) >= self.max_requests:
            return False
        self._hits[ip].append(now)
        return True


cache = DataCache()
rate_limiter = RateLimiter(max_requests=120, window_seconds=60.0)

app = FastAPI(title="QuantBot API", version="2.0", docs_url="/api/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


def _validate_symbol(symbol: str) -> str:
    if not _SYMBOL_RE.match(symbol):
        raise HTTPException(status_code=400, detail="Símbolo inválido")
    return symbol.upper()


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/assets")
def get_assets() -> dict:
    cached = cache.get("assets", ttl_seconds=300)
    if cached is not None:
        return {"assets": cached}

    raw = fetch_all_assets()
    assets = {}
    for symbol, info in raw.items():
        prices = info.get("prices", [])
        indicators = compute_indicators(prices)
        signal = compute_signal(indicators)
        assets[symbol] = {**info, "indicators": indicators, "signal": signal}

    cache.set("assets", assets)
    return {"assets": assets}


@app.get("/api/asset/{symbol}")
def get_asset(symbol: str) -> dict:
    symbol = _validate_symbol(symbol)
    raw = fetch_all_assets()
    info = raw.get(symbol)
    if info is None:
        raise HTTPException(status_code=404, detail="Ativo não encontrado")
    indicators = compute_indicators(info.get("prices", []))
    return {"symbol": symbol, **info, "indicators": indicators,
            "signal": compute_signal(indicators)}


@app.get("/api/selic")
def get_selic() -> dict:
    cached = cache.get("selic", ttl_seconds=3600)
    if cached is not None:
        return {"selic": cached}

    rate = fetch_selic_rate()
    if rate is None:
        raise HTTPException(status_code=503, detail="Selic indisponível")
    cache.set("selic", rate)
    return {"selic": rate}


@app.get("/api/cdi")
def get_cdi() -> dict:
    cached = cache.get("cdi", ttl_seconds=3600)
    if cached is not None:
        return {"cdi": cached}

    history = fetch_cdi_history()
    cache.set("cdi", history)
    return {"cdi": history}


@app.get("/api/crypto")
def get_crypto() -> dict:
    cached = cache.get("crypto", ttl_seconds=120)
    if cached is not None:
        return {"crypto": cached}

    data = fetch_crypto_market_data()
    cache.set("crypto", data)
    return {"crypto": data}
