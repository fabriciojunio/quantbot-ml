"""
Coletor de Notícias via RSS Feeds.

Busca notícias financeiras de fontes gratuitas e públicas,
sem necessidade de API key ou conta paga.

Fontes:
- InfoMoney (BR)
- Valor Econômico (BR)
- Google News Finance (BR + Global)
- CoinDesk (Crypto)
- Investing.com (BR)
- Yahoo Finance (Global)

Uso:
    fetcher = NewsFetcher()
    news = fetcher.fetch_all()
    for article in news:
        print(article.title, article.source)
"""

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError
from html import unescape

from utils.logger import get_logger

logger = get_logger("quantbot.data.news")


# ═══════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════

@dataclass
class NewsArticle:
    """Representa uma notícia coletada."""
    title: str
    description: str
    source: str
    url: str
    published: datetime
    category: str = "Geral"
    symbols: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None       # preenchido pelo SentimentAnalyzer
    sentiment_score: Optional[float] = None
    impact: str = "média"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "url": self.url,
            "published": self.published.isoformat(),
            "category": self.category,
            "symbols": self.symbols,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "impact": self.impact,
        }


# ═══════════════════════════════════════════════════════════════
# RSS FEED SOURCES
# ═══════════════════════════════════════════════════════════════

RSS_FEEDS = [
    {
        "name": "Google News - Negócios BR",
        "url": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FuQjBHZ0pDVWlnQVAB?hl=pt-BR&gl=BR&ceid=BR:pt-419",
        "category": "Negócios",
        "lang": "pt",
    },
    {
        "name": "Google News - Economia BR",
        "url": "https://news.google.com/rss/topics/CAAqKggKIiRDQkFTRlFvSUwyMHZNR2RtZUhJU0JYQjBMVUpTS0FBUAE?hl=pt-BR&gl=BR&ceid=BR:pt-419",
        "category": "Economia",
        "lang": "pt",
    },
    {
        "name": "Yahoo Finance",
        "url": "https://finance.yahoo.com/news/rssindex",
        "category": "Mercado Global",
        "lang": "en",
    },
    {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "category": "Crypto",
        "lang": "en",
    },
    {
        "name": "Investing.com BR",
        "url": "https://br.investing.com/rss/news.rss",
        "category": "Investimentos",
        "lang": "pt",
    },
]

# Mapeamento de palavras-chave para símbolos de ativos
KEYWORD_TO_SYMBOLS = {
    # B3
    "petrobras": ["PETR4.SA"], "petr4": ["PETR4.SA"],
    "vale": ["VALE3.SA"], "vale3": ["VALE3.SA"],
    "itaú": ["ITUB4.SA"], "itau": ["ITUB4.SA"], "itub4": ["ITUB4.SA"],
    "bradesco": ["BBDC4.SA"], "bbdc4": ["BBDC4.SA"],
    "weg": ["WEGE3.SA"], "wege3": ["WEGE3.SA"],
    "ambev": ["ABEV3.SA"], "abev3": ["ABEV3.SA"],
    "banco do brasil": ["BBAS3.SA"], "bbas3": ["BBAS3.SA"],
    "b3 sa": ["B3SA3.SA"], "b3sa3": ["B3SA3.SA"],
    "suzano": ["SUZB3.SA"],
    "localiza": ["RENT3.SA"],
    # US
    "apple": ["AAPL"], "aapl": ["AAPL"],
    "microsoft": ["MSFT"], "msft": ["MSFT"],
    "nvidia": ["NVDA"], "nvda": ["NVDA"],
    "alphabet": ["GOOGL"], "google": ["GOOGL"], "googl": ["GOOGL"],
    "amazon": ["AMZN"], "amzn": ["AMZN"],
    "meta": ["META"], "facebook": ["META"],
    "tesla": ["TSLA"], "tsla": ["TSLA"],
    "jpmorgan": ["JPM"], "jp morgan": ["JPM"],
    # Crypto
    "bitcoin": ["BTC-USD"], "btc": ["BTC-USD"],
    "ethereum": ["ETH-USD"], "eth": ["ETH-USD"], "ether": ["ETH-USD"],
    "solana": ["SOL-USD"], "sol": ["SOL-USD"],
    "cardano": ["ADA-USD"], "ada": ["ADA-USD"],
    "polkadot": ["DOT-USD"], "dot": ["DOT-USD"],
    # Macro
    "selic": ["ITUB4.SA", "BBDC4.SA", "BBAS3.SA"],
    "copom": ["ITUB4.SA", "BBDC4.SA", "BBAS3.SA"],
    "fed": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "juros": ["ITUB4.SA", "BBDC4.SA"],
    "inflação": ["ITUB4.SA", "BBDC4.SA", "BBAS3.SA"],
    "ipca": ["ITUB4.SA", "BBDC4.SA"],
    "dólar": ["PETR4.SA", "VALE3.SA"],
    "petróleo": ["PETR4.SA"],
    "minério": ["VALE3.SA"],
    "commodities": ["PETR4.SA", "VALE3.SA"],
    "china": ["VALE3.SA", "PETR4.SA"],
    "eua": ["AAPL", "MSFT", "NVDA"],
    "cripto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "regulação cripto": ["BTC-USD", "ETH-USD"],
}

# Categorização por palavras-chave
CATEGORY_KEYWORDS = {
    "Política Monetária": ["selic", "copom", "juros", "fed", "taxa", "monetary", "rate", "interest"],
    "Resultados": ["lucro", "resultado", "balanço", "receita", "dividendo", "earnings", "profit", "revenue"],
    "Crypto": ["bitcoin", "btc", "ethereum", "eth", "cripto", "crypto", "blockchain", "token", "defi"],
    "Geopolítica": ["guerra", "conflito", "sanção", "china", "eua", "rússia", "tensão", "war", "geopolitical"],
    "Macro": ["pib", "gdp", "inflação", "emprego", "desemprego", "cpi", "ipca", "economia", "recession"],
    "Regulação": ["regulação", "regulament", "cvm", "sec", "compliance", "lei", "regulation"],
    "Corporativo": ["aquisição", "fusão", "ipo", "oferta", "expansion", "merger", "acquisition"],
    "Tech": ["inteligência artificial", "ai ", " ia ", "machine learning", "data", "cloud", "chip", "semicondutor"],
}


# ═══════════════════════════════════════════════════════════════
# NEWS FETCHER
# ═══════════════════════════════════════════════════════════════

class NewsFetcher:
    """
    Coleta notícias de feeds RSS gratuitos.

    Sem API keys, sem custo, sem limites significativos.
    Usa apenas bibliotecas padrão do Python (urllib + xml).
    """

    def __init__(self, max_age_days: int = 7, timeout: int = 10):
        """
        Args:
            max_age_days: Idade máxima das notícias em dias
            timeout: Timeout de conexão em segundos
        """
        self.max_age_days = max_age_days
        self.timeout = timeout
        self._cache: List[NewsArticle] = []

    def fetch_feed(self, feed_config: dict) -> List[NewsArticle]:
        """
        Busca notícias de um feed RSS individual.

        Args:
            feed_config: Dicionário com name, url, category, lang

        Returns:
            Lista de NewsArticle
        """
        articles = []
        url = feed_config["url"]
        source = feed_config["name"]

        try:
            # Request com User-Agent para evitar bloqueio
            req = Request(url, headers={
                "User-Agent": "QuantBot-ML/1.0 (Academic Research)"
            })

            with urlopen(req, timeout=self.timeout) as response:
                content = response.read()

            root = ET.fromstring(content)

            # Tenta RSS 2.0 e Atom
            items = root.findall(".//item")
            if not items:
                # Atom format
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                items = root.findall(".//atom:entry", ns)

            cutoff = datetime.now() - timedelta(days=self.max_age_days)

            for item in items[:20]:  # Max 20 por feed
                try:
                    article = self._parse_item(item, source, feed_config)
                    if article and article.published > cutoff:
                        articles.append(article)
                except Exception:
                    continue

            logger.info(f"  ✓ {source}: {len(articles)} notícias")

        except URLError as e:
            logger.warning(f"  ✗ {source}: erro de conexão — {e.reason}")
        except ET.ParseError:
            logger.warning(f"  ✗ {source}: erro ao parsear XML")
        except Exception as e:
            logger.warning(f"  ✗ {source}: {e}")

        return articles

    def fetch_all(self, feeds: List[dict] = None) -> List[NewsArticle]:
        """
        Busca notícias de todos os feeds configurados.

        Args:
            feeds: Lista de feeds (padrão: RSS_FEEDS)

        Returns:
            Lista de NewsArticle ordenada por data
        """
        if feeds is None:
            feeds = RSS_FEEDS

        logger.info(f"📰 Buscando notícias de {len(feeds)} fontes...")

        all_articles = []
        for feed in feeds:
            articles = self.fetch_feed(feed)
            all_articles.extend(articles)
            time.sleep(0.3)  # Rate limiting

        # Remove duplicatas por título similar
        unique = self._deduplicate(all_articles)

        # Detecta símbolos relacionados
        for article in unique:
            article.symbols = self._detect_symbols(article)
            article.category = self._detect_category(article)
            article.impact = self._estimate_impact(article)

        # Ordena por data (mais recente primeiro)
        unique.sort(key=lambda a: a.published, reverse=True)

        self._cache = unique
        logger.info(f"📰 Total: {len(unique)} notícias únicas coletadas")

        return unique

    def get_news_for_symbol(self, symbol: str) -> List[NewsArticle]:
        """Retorna notícias relevantes para um ativo específico."""
        return [a for a in self._cache if symbol in a.symbols]

    def get_cached(self) -> List[NewsArticle]:
        """Retorna notícias em cache."""
        return self._cache

    # ─── PARSING ──────────────────────────────────────────

    def _parse_item(
        self, item: ET.Element, source: str, feed_config: dict
    ) -> Optional[NewsArticle]:
        """Parseia um item RSS/Atom em NewsArticle."""

        # RSS 2.0
        title = self._get_text(item, "title")
        description = self._get_text(item, "description")
        link = self._get_text(item, "link")
        pub_date = self._get_text(item, "pubDate")

        # Atom fallback
        if not title:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            title = self._get_text(item, "atom:title", ns)
            description = self._get_text(item, "atom:summary", ns) or self._get_text(item, "atom:content", ns)
            link_elem = item.find("atom:link", ns)
            link = link_elem.get("href", "") if link_elem is not None else ""
            pub_date = self._get_text(item, "atom:published", ns) or self._get_text(item, "atom:updated", ns)

        if not title:
            return None

        # Limpa HTML do título e descrição
        title = self._clean_html(title).strip()
        description = self._clean_html(description or "").strip()[:500]

        # Parseia data
        published = self._parse_date(pub_date)

        return NewsArticle(
            title=title,
            description=description,
            source=source,
            url=link or "",
            published=published,
            category=feed_config.get("category", "Geral"),
        )

    @staticmethod
    def _get_text(elem: ET.Element, tag: str, ns: dict = None) -> str:
        """Extrai texto de um elemento XML."""
        if ns:
            child = elem.find(tag, ns)
        else:
            child = elem.find(tag)
        return child.text if child is not None and child.text else ""

    @staticmethod
    def _clean_html(text: str) -> str:
        """Remove tags HTML e decodifica entidades."""
        text = unescape(text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """Parseia data de várias formatos RSS."""
        if not date_str:
            return datetime.now()

        formats = [
            "%a, %d %b %Y %H:%M:%S %z",      # RFC 822
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",             # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y %H:%M",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.replace(tzinfo=None) if dt.tzinfo else dt
            except ValueError:
                continue

        return datetime.now()

    # ─── DETECÇÃO ─────────────────────────────────────────

    @staticmethod
    def _detect_symbols(article: NewsArticle) -> List[str]:
        """Detecta símbolos de ativos mencionados na notícia."""
        text = f"{article.title} {article.description}".lower()
        symbols = set()

        for keyword, syms in KEYWORD_TO_SYMBOLS.items():
            if keyword in text:
                symbols.update(syms)

        return list(symbols)

    @staticmethod
    def _detect_category(article: NewsArticle) -> str:
        """Detecta categoria da notícia por palavras-chave."""
        text = f"{article.title} {article.description}".lower()

        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return category

        return article.category  # Mantém a do feed

    @staticmethod
    def _estimate_impact(article: NewsArticle) -> str:
        """Estima impacto da notícia (alta/média/baixa)."""
        text = f"{article.title} {article.description}".lower()

        high_impact = [
            "recorde", "crise", "colapso", "guerra", "emergência",
            "bilhão", "trilhão", "histórico", "inédito", "crash",
            "record", "crisis", "crash", "billion", "trillion",
            "fed", "copom", "selic", "juros",
        ]

        low_impact = [
            "análise", "opinião", "coluna", "podcast",
            "analysis", "opinion", "column",
        ]

        if any(w in text for w in high_impact):
            return "alta"
        elif any(w in text for w in low_impact):
            return "baixa"
        return "média"

    @staticmethod
    def _deduplicate(articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove notícias duplicadas por título similar."""
        seen = set()
        unique = []

        for article in articles:
            # Normaliza título para comparação
            key = re.sub(r"[^a-zA-Z0-9àáâãéêíóôõúç]", "", article.title.lower())[:60]
            if key not in seen:
                seen.add(key)
                unique.append(article)

        return unique
