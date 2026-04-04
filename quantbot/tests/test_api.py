"""
Testes unitários da API QuantBot ML.

Testa:
- Endpoints da API (health, assets, selic, cdi, crypto, dashboard)
- Serviço de dados de mercado (indicadores, sinais)
- Cache e rate limiting
- Segurança (CORS, validação de input)
"""

import sys
import os
import time
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# TESTES DO SERVIÇO DE DADOS DE MERCADO
# ═══════════════════════════════════════════════════════════════

class TestComputeIndicators:
    """Testa cálculo de indicadores técnicos."""

    def test_basic_indicators(self):
        from api.market_data import compute_indicators
        # Preços crescentes simples
        prices = [100 + i * 0.5 for i in range(60)]
        ind = compute_indicators(prices)
        assert "rsi" in ind
        assert "sma20" in ind
        assert "sma50" in ind
        assert "macd" in ind
        assert "vol" in ind
        assert "mom" in ind
        assert "sharpe" in ind

    def test_rsi_range(self):
        from api.market_data import compute_indicators
        prices = [100 + i * 0.5 for i in range(60)]
        ind = compute_indicators(prices)
        assert 0 <= ind["rsi"] <= 100

    def test_volatility_positive(self):
        from api.market_data import compute_indicators
        np.random.seed(42)
        prices = list(100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 60)))
        ind = compute_indicators(prices)
        assert ind["vol"] > 0

    def test_insufficient_data(self):
        from api.market_data import compute_indicators
        ind = compute_indicators([100, 101, 102])
        assert ind == {}

    def test_sma_values(self):
        from api.market_data import compute_indicators
        prices = [float(i) for i in range(1, 61)]
        ind = compute_indicators(prices)
        # SMA20 deve ser a média dos últimos 20 valores
        expected_sma20 = np.mean(prices[-20:])
        assert abs(ind["sma20"] - expected_sma20) < 0.01


class TestComputeSignal:
    """Testa geração de sinais ML."""

    def test_neutral_signal(self):
        from api.market_data import compute_signal
        sig = compute_signal({"rsi": 50, "macd": 0, "mom": 0})
        assert sig["signal"] == "NEUTRO"
        assert 5 <= sig["score"] <= 95

    def test_buy_signal(self):
        from api.market_data import compute_signal
        sig = compute_signal({"rsi": 25, "macd": 5, "mom": 10})
        assert sig["signal"] in ("COMPRA", "COMPRA_FORTE")
        assert sig["score"] > 60

    def test_sell_signal(self):
        from api.market_data import compute_signal
        sig = compute_signal({"rsi": 80, "macd": -5, "mom": -10})
        assert sig["signal"] in ("VENDA", "VENDA_FORTE")
        assert sig["score"] < 40

    def test_empty_indicators(self):
        from api.market_data import compute_signal
        sig = compute_signal({})
        assert sig["score"] == 50
        assert sig["signal"] == "NEUTRO"

    def test_confidence_range(self):
        from api.market_data import compute_signal
        sig = compute_signal({"rsi": 25, "macd": 5, "mom": 10})
        assert 0 <= sig["confidence"] <= 100


# ═══════════════════════════════════════════════════════════════
# TESTES DA API (FastAPI TestClient)
# ═══════════════════════════════════════════════════════════════

class TestAPIEndpoints:
    """Testa endpoints da API."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    @patch("api.server.fetch_all_assets")
    @patch("api.server.compute_indicators")
    @patch("api.server.compute_signal")
    def test_get_assets(self, mock_signal, mock_ind, mock_fetch, client):
        mock_fetch.return_value = {
            "PETR4": {
                "symbol": "PETR4",
                "name": "Petrobras",
                "sector": "Energia",
                "market": "B3",
                "current_price": 38.50,
                "change_pct": 1.2,
                "prices": [37 + i * 0.02 for i in range(90)],
                "volumes": [1000000] * 90,
                "dates": [f"2026-01-{i+1:02d}" for i in range(90)],
            }
        }
        mock_ind.return_value = {"rsi": 55, "sma20": 38, "sma50": 37, "macd": 0.5, "vol": 22, "mom": 3, "sharpe": 1.1}
        mock_signal.return_value = {"score": 65, "signal": "COMPRA", "confidence": 72}

        # Limpa cache
        from api.server import cache
        cache._store.clear()
        cache._timestamps.clear()

        resp = client.get("/api/assets")
        assert resp.status_code == 200
        data = resp.json()
        assert "assets" in data
        assert "PETR4" in data["assets"]

    def test_get_asset_invalid_symbol(self, client):
        resp = client.get("/api/asset/" + "A" * 20)
        assert resp.status_code == 400

    def test_get_asset_injection_attempt(self, client):
        resp = client.get("/api/asset/'; DROP TABLE--")
        assert resp.status_code == 400

    @patch("api.server.fetch_selic_rate")
    def test_get_selic(self, mock_selic, client):
        mock_selic.return_value = 13.25
        from api.server import cache
        cache._store.pop("selic", None)
        cache._timestamps.pop("selic", None)

        resp = client.get("/api/selic")
        assert resp.status_code == 200
        data = resp.json()
        assert data["selic"] == 13.25

    @patch("api.server.fetch_selic_rate")
    def test_get_selic_unavailable(self, mock_selic, client):
        mock_selic.return_value = None
        from api.server import cache
        cache._store.pop("selic", None)
        cache._timestamps.pop("selic", None)

        resp = client.get("/api/selic")
        assert resp.status_code == 503

    @patch("api.server.fetch_cdi_history")
    def test_get_cdi(self, mock_cdi, client):
        mock_cdi.return_value = [{"date": "01/04/2026", "rate": 0.0492}]
        from api.server import cache
        cache._store.pop("cdi", None)
        cache._timestamps.pop("cdi", None)

        resp = client.get("/api/cdi")
        assert resp.status_code == 200
        data = resp.json()
        assert "cdi" in data

    @patch("api.server.fetch_crypto_market_data")
    def test_get_crypto(self, mock_crypto, client):
        mock_crypto.return_value = {
            "BTC": {"price": 85000, "market_cap": 1.6e12, "volume_24h": 30e9, "change_24h": 2.3}
        }
        from api.server import cache
        cache._store.pop("crypto", None)
        cache._timestamps.pop("crypto", None)

        resp = client.get("/api/crypto")
        assert resp.status_code == 200
        data = resp.json()
        assert "BTC" in data["crypto"]


# ═══════════════════════════════════════════════════════════════
# TESTES DE CACHE
# ═══════════════════════════════════════════════════════════════

class TestCache:
    """Testa sistema de cache."""

    def test_cache_set_get(self):
        from api.server import DataCache
        c = DataCache()
        c.set("test", {"value": 42})
        assert c.get("test", ttl_seconds=10) == {"value": 42}

    def test_cache_ttl_expired(self):
        from api.server import DataCache
        c = DataCache()
        c.set("test", {"value": 42})
        c._timestamps["test"] = time.time() - 100
        assert c.get("test", ttl_seconds=10) is None

    def test_cache_miss(self):
        from api.server import DataCache
        c = DataCache()
        assert c.get("nonexistent", ttl_seconds=10) is None


# ═══════════════════════════════════════════════════════════════
# TESTES DE RATE LIMITING
# ═══════════════════════════════════════════════════════════════

class TestRateLimiter:
    """Testa rate limiter."""

    def test_allows_within_limit(self):
        from api.server import RateLimiter
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.is_allowed("127.0.0.1") is True

    def test_blocks_over_limit(self):
        from api.server import RateLimiter
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.is_allowed("127.0.0.1")
        assert rl.is_allowed("127.0.0.1") is False

    def test_different_ips_independent(self):
        from api.server import RateLimiter
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.is_allowed("1.1.1.1")
        rl.is_allowed("1.1.1.1")
        assert rl.is_allowed("1.1.1.1") is False
        assert rl.is_allowed("2.2.2.2") is True


# ═══════════════════════════════════════════════════════════════
# TESTES DE SEGURANÇA
# ═══════════════════════════════════════════════════════════════

class TestSecurity:
    """Testa medidas de segurança da API."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app)

    def test_cors_allowed_origin(self, client):
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200

    def test_only_get_allowed(self, client):
        resp = client.post("/api/health")
        assert resp.status_code == 405

    def test_invalid_endpoint_404(self, client):
        resp = client.get("/api/nonexistent")
        assert resp.status_code == 404
