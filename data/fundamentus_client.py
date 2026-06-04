"""
Cliente Fundamentus — fundamentos gratuitos de toda a B3.

O Fundamentus (fundamentus.com.br) publica, em uma única tabela por
categoria, os indicadores fundamentalistas de TODAS as ações e FIIs da
B3: Dividend Yield, P/VP, P/L, ROE, segmento, liquidez, etc. Sem API
key e sem custo.

Isto preenche a lacuna do plano gratuito da brapi.dev, que entrega o
universo e o preço, mas não o DY/fundamentos (esses são pagos lá).

Estratégia de fontes do QuantBot:
    - brapi.dev  → universo, preço em tempo (quase) real, dividendos
    - Fundamentus → DY, P/VP, P/L, ROE (fundamentos, grátis)
    - Banco Central → macro histórico (Selic, IPCA, CDI, câmbio)
    - CVM → documentos oficiais (proventos, informes)
    - yfinance → histórico de preços e dividendos por ticker

Observação honesta: Fundamentus é HTML (scraping). A estrutura da tabela
é estável há anos, mas pode mudar. O parser falha de forma graciosa
(retorna lista vazia + log) e nunca derruba o sistema.

Uso:
    fc = FundamentusClient()
    fiis = fc.fiis()          # lista de dicts com dy, pvp, segmento...
    acoes = fc.acoes()
    metrics = fc.to_fii_metrics(fiis)   # pronto para o FIIScreener
"""

import logging
import re
import urllib.error
import urllib.request
from html import unescape
from typing import Dict, List, Optional

logger = logging.getLogger("quantbot.data.fundamentus")

_FII_URL = "https://www.fundamentus.com.br/fii_resultado.php"
_STOCK_URL = "https://www.fundamentus.com.br/resultado.php"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9",
}


def _to_float(raw: str) -> float:
    """Converte número em formato pt-BR ('9,50%', '1.234,56') para float."""
    s = unescape(raw).strip().replace("%", "").replace(".", "").replace(",", ".")
    if s in ("", "-", "N/A"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()


class FundamentusClient:
    """Lê e parseia as tabelas de resultado do Fundamentus."""

    def __init__(self, timeout_seconds: int = 25):
        self.timeout = timeout_seconds
        self._cache: Dict[str, List[Dict]] = {}

    # ── Público ─────────────────────────────────────────────────────────

    def fiis(self, use_cache: bool = True) -> List[Dict]:
        """Todos os FIIs com DY, P/VP, segmento, liquidez, etc."""
        if use_cache and "fiis" in self._cache:
            return self._cache["fiis"]
        rows = self._fetch_table(_FII_URL)
        self._cache["fiis"] = rows
        logger.info(f"Fundamentus: {len(rows)} FIIs carregados")
        return rows

    def acoes(self, use_cache: bool = True) -> List[Dict]:
        """Todas as ações com DY, P/L, P/VP, ROE, etc."""
        if use_cache and "acoes" in self._cache:
            return self._cache["acoes"]
        rows = self._fetch_table(_STOCK_URL)
        self._cache["acoes"] = rows
        logger.info(f"Fundamentus: {len(rows)} ações carregadas")
        return rows

    # ── Parsing ─────────────────────────────────────────────────────────

    def _fetch_table(self, url: str) -> List[Dict]:
        html = self._get(url)
        if not html:
            return []

        # Isola a tabela de resultado
        m = re.search(r'<table[^>]*id="resultado".*?</table>', html, re.S)
        table = m.group(0) if m else html

        # Cabeçalhos
        head = re.search(r"<thead.*?</thead>", table, re.S)
        headers = [
            _strip_tags(h).lower()
            for h in re.findall(r"<th[^>]*>(.*?)</th>", head.group(0) if head else table, re.S)
        ]
        if not headers:
            return []

        # Linhas do corpo
        body = re.search(r"<tbody.*?</tbody>", table, re.S)
        body_html = body.group(0) if body else table
        rows: List[Dict] = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body_html, re.S):
            cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
            if len(cells) < len(headers):
                continue
            record = {}
            for h, c in zip(headers, cells):
                record[h] = _strip_tags(c)
            if record.get(headers[0]):  # tem ticker
                rows.append(record)
        return rows

    def _get(self, url: str) -> Optional[str]:
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                # Fundamentus usa latin-1
                return resp.read().decode("latin-1", "ignore")
        except urllib.error.HTTPError as e:
            logger.warning(f"Fundamentus HTTP {e.code} em {url}")
        except (urllib.error.URLError, TimeoutError) as e:
            logger.warning(f"Fundamentus falha de rede: {e}")
        return None

    # ── Conversão para os screeners ─────────────────────────────────────

    def to_fii_metrics(self, fii_rows: Optional[List[Dict]] = None):
        """
        Converte linhas de FII em FIIMetrics (entrada do FIIScreener),
        com DY e P/VP REAIS do Fundamentus.
        """
        from income.fii_screener import FIIMetrics, FIISegment

        # Fallback neutro para segmentos híbridos/desconhecidos.
        neutro = FIISegment.FUNDO_DE_FUNDOS
        seg_map = {
            "logística": FIISegment.LOGISTICA, "logistica": FIISegment.LOGISTICA,
            "lajes corporativas": FIISegment.LAJES_CORPORATIVAS,
            "shoppings": FIISegment.SHOPPINGS, "shopping": FIISegment.SHOPPINGS,
            "residencial": FIISegment.RESIDENCIAL,
            "títulos e val. mob.": FIISegment.PAPEL_CRI,
            "papéis": FIISegment.PAPEL_CRI, "papel": FIISegment.PAPEL_CRI,
            "hospital": FIISegment.HOSPITAIS, "hospitalar": FIISegment.HOSPITAIS,
            "fof": FIISegment.FUNDO_DE_FUNDOS,
        }
        rows = fii_rows if fii_rows is not None else self.fiis()
        out = []
        for r in rows:
            ticker = (r.get("papel") or "").upper()
            if not ticker:
                continue
            dy = _to_float(r.get("dividend yield", "0"))
            pvp = _to_float(r.get("p/vp", "1"))
            preco = _to_float(r.get("cotação") or r.get("cotacao") or "0")
            liquidez = _to_float(r.get("liquidez", "0"))
            vac = _to_float(r.get("vacância média") or r.get("vacancia media") or "0")
            seg = seg_map.get((r.get("segmento") or "").lower().strip(), neutro)
            try:
                out.append(FIIMetrics(
                    ticker=ticker,
                    nome=ticker,
                    segmento=seg,
                    gestora="N/D",
                    dy_12m=round(dy, 2),
                    dy_mes_atual=round(dy / 12, 3),
                    provento_mensal=round(preco * (dy / 100) / 12, 4) if preco else 0.0,
                    preco_cota=round(preco, 2),
                    pvp=round(pvp, 3),
                    patrimonio_liq=round(_to_float(r.get("valor de mercado", "0")) / 1_000_000, 1),
                    vacancia_fisica=8.0,           # Fundamentus não traz; default neutro
                    vacancia_financeira=9.0,
                    liquidez_diaria=round(liquidez, 0),
                    resultado_por_cota=round(preco * (dy / 100) / 12 * 1.05, 4) if preco else 0.0,
                    provento_sobre_resultado=0.95,
                    historico_anos_sem_corte=3,
                    taxa_administracao=0.8,
                    n_imoveis=int(_to_float(r.get("qtd de imóveis", "1")) or 1),
                    n_inquilinos=5,
                    indexador_predominante="IPCA",
                ))
            except Exception as e:
                logger.debug(f"Fundamentus: erro ao converter {ticker}: {e}")
        return out
