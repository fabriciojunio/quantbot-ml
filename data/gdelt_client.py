"""
Cliente GDELT 2.0 (DOC API) — notícias do passado, gratuitas e sem chave.

O GDELT monitora a mídia mundial em tempo real e mantém um índice
pesquisável. A DOC API permite buscar notícias por palavra-chave dos
ÚLTIMOS ~3 MESES (e o dataset completo, desde 2017, está no BigQuery).
É a melhor fonte gratuita para dar ao sistema uma BASE histórica de
notícias — exatamente o "passado" que você quer para o bot aprender.

Limites honestos:
    - 1 requisição a cada ~5 segundos (o cliente respeita isso sozinho).
    - DOC API cobre ~3 meses; histórico profundo só via BigQuery (grátis
      com conta Google Cloud).
    - Tom (tone) do GDELT vai de -100 a +100; aqui priorizamos rodar o
      FinBERT-PT-BR sobre os títulos para manter o padrão do sistema.

Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

Uso:
    g = GDELTClient()
    artigos = g.buscar("Petrobras OR Vale OR Ibovespa", timespan="1w",
                       lingua="portuguese", max_registros=50)
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import List, Optional

from data.news_fetcher import NewsArticle

logger = logging.getLogger("quantbot.data.gdelt")

_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_MIN_INTERVAL = 5.2  # GDELT pede 1 req/5s


class GDELTClient:
    """Busca notícias históricas no GDELT respeitando o rate limit."""

    _last_request = 0.0  # compartilhado entre instâncias

    def __init__(self, timeout_seconds: int = 25):
        self.timeout = timeout_seconds

    def buscar(
        self,
        query: str,
        timespan: str = "1w",
        lingua: Optional[str] = "portuguese",
        sourcecountry: Optional[str] = None,
        max_registros: int = 50,
    ) -> List[NewsArticle]:
        """
        Busca artigos por palavra-chave.

        Args:
            query: termos (suporta OR/AND/aspas). Ex.: '"banco do brasil" OR BBAS3'
            timespan: janela ("1d","3d","1w","1m","3m").
            lingua: filtro de idioma do GDELT (ex.: "portuguese", "english").
            sourcecountry: país da fonte (ex.: "BR").
            max_registros: máximo de artigos.
        """
        q = query
        if lingua:
            q += f" sourcelang:{lingua}"
        if sourcecountry:
            q += f" sourcecountry:{sourcecountry}"

        params = {
            "query": q,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(min(max_registros, 250)),
            "timespan": timespan,
            "sort": "datedesc",
        }
        data = self._get(params)
        if not data:
            return []

        artigos: List[NewsArticle] = []
        for a in data.get("articles", []):
            try:
                artigos.append(NewsArticle(
                    title=a.get("title", "").strip(),
                    description="",
                    source=f"GDELT/{a.get('domain', '')}",
                    url=a.get("url", ""),
                    published=self._parse_date(a.get("seendate", "")),
                    category="Histórico",
                ))
            except Exception:
                continue
        logger.info(f"GDELT: {len(artigos)} artigos para '{query[:40]}' ({timespan})")
        return artigos

    # ── internals ───────────────────────────────────────────────────────

    def _get(self, params: dict) -> Optional[dict]:
        # respeita 1 req / 5s
        elapsed = time.monotonic() - GDELTClient._last_request
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

        url = _URL + "?" + urllib.parse.urlencode(params)
        try:
            GDELTClient._last_request = time.monotonic()
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (QuantBot research)"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", "ignore")
            return json.loads(raw) if raw.strip().startswith("{") else None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.warning("GDELT rate limit (429). Aguarde e tente de novo.")
            else:
                logger.warning(f"GDELT HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError) as e:
            logger.warning(f"GDELT falha de rede: {e}")
        except json.JSONDecodeError:
            logger.warning("GDELT retornou conteúdo não-JSON")
        return None

    @staticmethod
    def _parse_date(s: str) -> datetime:
        # GDELT usa formato 20260604T153000Z
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return datetime.now()
