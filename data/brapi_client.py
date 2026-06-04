"""
Cliente brapi.dev — dados reais da B3 para FIIs e ações.

A brapi.dev fornece dados oficiais da CVM e do Banco Central do Brasil:
    - Cotações de ações e FIIs em tempo real
    - Dividendos históricos e agendados
    - Indicadores de FIIs (DY, P/VP, vacância, patrimônio)
    - Dados fundamentalistas de ações (P/L, ROE, DY, payout)
    - Macroeconomia (Selic, IPCA, CDI)

Design:
    - Cache local (evita bater na API repetidamente — respeita rate limits)
    - Retry com backoff exponencial para falhas de rede
    - Fallback gracioso (retorna None + log, nunca crasha o sistema)
    - Funciona sem token (ações de teste) e com token (produção)

Documentação: https://brapi.dev/docs
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger("quantbot.data.brapi_client")

_BASE_URL = "https://brapi.dev/api"
_FII_URL = "https://brapi.dev/api/v2/fiis"
_MACRO_URL = "https://brapi.dev/api/v2/macro"


class BrapiCache:
    """Cache simples em disco — JSON por chave, com TTL."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("?", "_").replace("&", "_")
        return self.cache_dir / f"{safe}.json"

    def get(self, key: str, ttl_minutes: int = 60) -> Optional[Dict]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            if datetime.now(timezone.utc) - cached_at > timedelta(minutes=ttl_minutes):
                p.unlink(missing_ok=True)
                return None
            return data.get("payload")
        except Exception:
            return None

    def set(self, key: str, payload: Any) -> None:
        p = self._path(key)
        try:
            p.write_text(
                json.dumps({
                    "_cached_at": datetime.now(timezone.utc).isoformat(),
                    "payload": payload,
                }, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Cache write falhou para {key}: {e}")


class BrapiClient:
    """
    Cliente HTTP para a API brapi.dev.

    Configure BRAPI_TOKEN no .env para acessar todos os endpoints.
    Sem token: funciona apenas para PETR4, VALE3, MGLU3, ITUB4 (modo demo).

    Uso:
        client = BrapiClient()
        dados = client.get_quote("ITUB4", include_dividends=True)
        fiis = client.get_fii_list(limit=50)
    """

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: str = "data/cache",
        timeout_seconds: int = 15,
        max_retries: int = 3,
    ):
        self.token = token or os.getenv("BRAPI_TOKEN", "")
        self.cache = BrapiCache(cache_dir)
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self._last_request_time = 0.0
        self._min_interval = 0.5  # 500ms entre requests — respeita rate limit

    # ── Ações ────────────────────────────────────────────────────────────────

    def get_quote(
        self,
        ticker: str,
        include_dividends: bool = True,
        include_fundamentals: bool = True,
        range_period: str = "1y",
        cache_ttl: int = 30,
    ) -> Optional[Dict]:
        """
        Cotação atual + dados fundamentalistas + histórico de dividendos.

        Args:
            ticker: ex "ITUB4", "HGLG11"
            include_dividends: inclui histórico de dividendos
            include_fundamentals: inclui P/L, ROE, DY, etc
            range_period: período histórico ("1d","5d","1mo","3mo","6mo","1y","2y")
            cache_ttl: minutos de cache (30min padrão para cotações)
        """
        modules = []
        if include_fundamentals:
            modules += ["defaultKeyStatistics", "financialData", "summaryProfile"]
        params = {
            "range": range_period,
            "interval": "1d",
        }
        if include_dividends:
            params["dividends"] = "true"
        if modules:
            params["modules"] = ",".join(modules)
        if self.token:
            params["token"] = self.token

        endpoint = f"quote/{ticker.upper()}"
        cache_key = f"quote_{ticker}_{range_period}_{include_dividends}"
        cached = self.cache.get(cache_key, ttl_minutes=cache_ttl)
        if cached is not None:
            return cached

        data = self._get(endpoint, params)
        if data and data.get("results"):
            result = data["results"][0]
            self.cache.set(cache_key, result)
            return result
        return None

    def get_quotes_batch(
        self,
        tickers: List[str],
        include_dividends: bool = True,
        cache_ttl: int = 30,
    ) -> Dict[str, Optional[Dict]]:
        """Cotações de múltiplos tickers em lote."""
        results = {}
        # brapi aceita múltiplos tickers separados por vírgula
        chunk_size = 10  # não sobrecarregar
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            ticker_str = ",".join(t.upper() for t in chunk)
            params = {"dividends": "true" if include_dividends else "false"}
            if self.token:
                params["token"] = self.token

            data = self._get(f"quote/{ticker_str}", params)
            if data and data.get("results"):
                for item in data["results"]:
                    t = item.get("symbol", "").upper()
                    if t:
                        results[t] = item
                        self.cache.set(f"quote_{t}", item)
        return results

    # ── FIIs ─────────────────────────────────────────────────────────────────

    def get_fii_list(
        self,
        limit: int = 100,
        sort_by: str = "dividendYield",
        sort_order: str = "desc",
        cache_ttl: int = 120,
    ) -> Optional[List[Dict]]:
        """
        Lista de FIIs com indicadores completos.

        Campos retornados: ticker, lastDividend, dividendYield, priceToBook
        (P/VP), liquidity, patrimony, managementFee, vacancyRate, sector.
        """
        cache_key = f"fii_list_{limit}_{sort_by}_{sort_order}"
        cached = self.cache.get(cache_key, ttl_minutes=cache_ttl)
        if cached is not None:
            return cached

        params = {
            "limit": str(limit),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        if self.token:
            params["token"] = self.token

        data = self._get_v2("fiis", params)
        if data and data.get("data"):
            fiis = data["data"]
            self.cache.set(cache_key, fiis)
            return fiis
        return None

    def get_fii_indicators(
        self,
        ticker: str,
        cache_ttl: int = 60,
    ) -> Optional[Dict]:
        """Indicadores detalhados de um FII: DY, P/VP, vacância, carteira."""
        cache_key = f"fii_ind_{ticker}"
        cached = self.cache.get(cache_key, ttl_minutes=cache_ttl)
        if cached is not None:
            return cached

        params = {}
        if self.token:
            params["token"] = self.token

        data = self._get_v2(f"fiis/{ticker.upper()}/indicadores", params)
        if data and data.get("data"):
            result = data["data"]
            self.cache.set(cache_key, result)
            return result
        return None

    def get_fii_dividends(
        self,
        ticker: str,
        cache_ttl: int = 120,
    ) -> Optional[List[Dict]]:
        """Histórico de dividendos de um FII."""
        cache_key = f"fii_div_{ticker}"
        cached = self.cache.get(cache_key, ttl_minutes=cache_ttl)
        if cached is not None:
            return cached

        params = {}
        if self.token:
            params["token"] = self.token

        data = self._get_v2(f"fiis/{ticker.upper()}/dividendos", params)
        if data and data.get("data"):
            result = data["data"]
            self.cache.set(cache_key, result)
            return result
        return None

    # ── Macroeconomia ─────────────────────────────────────────────────────────

    def get_selic(self, cache_ttl: int = 240) -> Optional[float]:
        """Taxa Selic atual (% ao ano)."""
        cache_key = "macro_selic"
        cached = self.cache.get(cache_key, ttl_minutes=cache_ttl)
        if cached is not None:
            return cached

        params = {"dataType": "selic"}
        if self.token:
            params["token"] = self.token

        data = self._get_v2("macro", params)
        if data and data.get("data"):
            series = data["data"].get("selic", [])
            if series:
                valor = float(series[-1].get("value", 0))
                self.cache.set(cache_key, valor)
                return valor
        logger.warning("Não foi possível obter a Selic via API. Usando fallback 14.5%")
        return 14.5  # fallback com valor atual conhecido

    def get_ipca_12m(self, cache_ttl: int = 360) -> Optional[float]:
        """IPCA acumulado 12 meses (% ao ano)."""
        cache_key = "macro_ipca"
        cached = self.cache.get(cache_key, ttl_minutes=cache_ttl)
        if cached is not None:
            return cached

        params = {"dataType": "ipca"}
        if self.token:
            params["token"] = self.token

        data = self._get_v2("macro", params)
        if data and data.get("data"):
            series = data["data"].get("ipca", [])
            if len(series) >= 12:
                # Acumula últimos 12 meses
                acum = 1.0
                for obs in series[-12:]:
                    acum *= (1 + float(obs.get("value", 0)) / 100)
                valor = round((acum - 1) * 100, 2)
                self.cache.set(cache_key, valor)
                return valor
        return 5.0  # fallback

    # ── HTTP internals ────────────────────────────────────────────────────────

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Optional[Dict]:
        url = f"{_BASE_URL}/{endpoint}"
        return self._request(url, params)

    def _get_v2(
        self,
        path: str,
        params: Optional[Dict] = None,
    ) -> Optional[Dict]:
        url = f"https://brapi.dev/api/v2/{path}"
        return self._request(url, params)

    def _request(
        self,
        url: str,
        params: Optional[Dict] = None,
        attempt: int = 0,
    ) -> Optional[Dict]:
        """HTTP GET com retry, backoff e rate limiting."""
        # Rate limiting — mínimo 500ms entre requests
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        try:
            self._last_request_time = time.monotonic()
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "QuantBot-ML/2.0 (educational trading bot)"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited
                wait = 2 ** attempt * 2
                logger.warning(f"Rate limited. Aguardando {wait}s...")
                time.sleep(wait)
                if attempt < self.max_retries:
                    return self._request(url, None, attempt + 1)
            elif e.code in (401, 403):
                logger.error("Token inválido ou sem permissão para este endpoint.")
            else:
                logger.warning(f"HTTP {e.code} para {url}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < self.max_retries:
                wait = 2 ** attempt
                logger.warning(f"Falha de rede ({e}). Retry {attempt+1}/{self.max_retries} em {wait}s")
                time.sleep(wait)
                return self._request(url, None, attempt + 1)
            logger.error(f"Falha após {self.max_retries} tentativas para {url}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON inválido na resposta de {url}: {e}")
        return None
