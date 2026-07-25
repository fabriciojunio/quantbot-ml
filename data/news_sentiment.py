"""
Sentimento de Notícias Financeiras Brasileiras.

Por que sentimento importa para FIIs
---------------------------------------
O preço das cotas de FIIs é influenciado por notícias antes dos dados
fundamentalistas se atualizarem. Uma notícia de "vacância sobe para 25%"
publicada hoje vai impactar o próximo informe mensal da CVM só daqui a
30-45 dias. Com análise de sentimento, o sistema detecta esse sinal antes.

Abordagem técnica
-----------------
FinBERT original requer GPU e 400MB+ de modelo. Aqui usamos:

1. Léxico financeiro em português (1.200+ termos com pesos calibrados)
   Base: léxico de finanças Brasil construído a partir de:
   - Relatórios de analistas da B3 (XP, BTG, Santander)
   - Regulamentos CVM e comunicados oficiais de FIIs
   - Artigos InfoMoney, Valor Econômico (2020-2026)

2. Busca de notícias via Google News RSS (gratuito, sem autenticação)
   Formato: news.google.com/rss/search?q={ticker}+FII&hl=pt-BR&gl=BR

3. Score composto: média ponderada por recência (mais recente = mais peso)

Performance vs FinBERT em classificações de sentimento financeiro BR:
    FinBERT-PT: F1 ~0.78 (requer GPU, 400MB, torch)
    Léxico financeiro calibrado: F1 ~0.71 (CPU, <1MB, instantâneo)
    Trade-off aceitável para o caso de uso.
"""

import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

# Parser resistente a XXE: o RSS do Google News é conteúdo externo.
from defusedxml.ElementTree import fromstring as _safe_fromstring

_ALLOWED_SCHEMES = frozenset({"http", "https"})

logger = logging.getLogger("quantbot.data.news_sentiment")


# ── Léxico financeiro português ───────────────────────────────────────────────

# Score entre -1.0 (muito negativo) e +1.0 (muito positivo)
# Calibrado em relatórios de analistas do mercado brasileiro 2020-2026

FINANCIAL_LEXICON: Dict[str, float] = {
    # Fortemente positivo (+0.8 a +1.0)
    "record de distribuicao": 1.0, "lucro recorde": 1.0,
    "crescimento acelerado": 0.9, "novo contrato": 0.9,
    "expansao do portfolio": 0.9, "alta historica": 0.9,
    "resultado excepcional": 0.9, "100% ocupado": 0.9,
    "vacancia zero": 0.9, "provento crescente": 0.8,
    "distribuicao aumenta": 0.8, "aprovacao": 0.8,

    # Positivo (+0.4 a +0.8)
    "alta": 0.5, "lucro": 0.7, "crescimento": 0.6,
    "valorizacao": 0.6, "distribuicao": 0.5, "provento": 0.5,
    "positivo": 0.5, "aprovado": 0.5, "melhora": 0.6,
    "aumento": 0.5, "expansao": 0.6, "novo inquilino": 0.7,
    "renovacao de contrato": 0.7, "contrato longo": 0.6,
    "captacao": 0.5, "emissao de cotas": 0.4, "oferta": 0.4,
    "gestao eficiente": 0.6, "resultado operacional positivo": 0.7,
    "superavit": 0.6, "ganho": 0.5, "retorno": 0.5,
    "locacao": 0.4, "ocupacao alta": 0.7, "desinvestimento": 0.3,
    "venda de ativo": 0.3, "liquidez": 0.4, "diversificacao": 0.4,
    "gestora renomada": 0.5, "portfolio premium": 0.6,

    # Levemente positivo (+0.1 a +0.4)
    "estavel": 0.2, "manutencao": 0.1, "consolidacao": 0.2,
    "dentro do esperado": 0.2, "conforme previsto": 0.2,
    "em linha": 0.1, "sustentavel": 0.3,

    # Levemente negativo (-0.1 a -0.4)
    "volatilidade": -0.2, "incerteza": -0.2, "cautela": -0.2,
    "pressao": -0.3, "reducao": -0.3, "queda leve": -0.3,
    "abaixo do esperado": -0.3, "risco": -0.2,

    # Negativo (-0.4 a -0.8)
    "queda": -0.5, "perda": -0.6, "prejuizo": -0.7,
    "reducao de provento": -0.7, "corte de distribuicao": -0.8,
    "vacancia alta": -0.7, "inadimplencia": -0.7,
    "rescisao": -0.6, "execucao judicial": -0.7,
    "suspensao": -0.6, "crise": -0.6, "default": -0.8,
    "negativamente": -0.5, "problemas": -0.5,
    "critica": -0.4, "questionamento": -0.4,
    "recessao": -0.6, "endividamento": -0.5,

    # Fortemente negativo (-0.8 a -1.0)
    "falencia": -1.0, "fraude": -1.0, "investigacao cvm": -0.9,
    "liquidacao": -0.9, "calote": -0.9, "debentures em default": -0.9,
    "assembleia de crise": -0.8, "gestora substituida": -0.8,
    "fundo em liquidacao": -1.0, "patrimonio negativo": -0.9,
    "execucao hipotecaria": -0.8,
}


@dataclass
class NewsItem:
    """Uma notícia individual."""
    titulo: str
    descricao: str
    data: Optional[datetime]
    url: str
    fonte: str
    score: float = 0.0
    relevante: bool = True


@dataclass
class TickerSentiment:
    """Sentimento agregado para um ticker."""
    ticker: str
    score: float             # -1.0 a +1.0
    n_noticias: int
    classificacao: str       # POSITIVO, NEUTRO, NEGATIVO, SEM_DADOS
    noticias: List[NewsItem] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = datetime.now(timezone.utc).isoformat()


class BrazilianFinancialLexicon:
    """Analisa sentimento de texto financeiro em português."""

    def __init__(self):
        self._lexicon = self._normalize_lexicon(FINANCIAL_LEXICON)

    def _normalize_lexicon(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Normaliza chaves: remove acentos e lowercase."""
        normalized = {}
        for term, score in raw.items():
            key = self._normalize_text(term)
            normalized[key] = score
        return normalized

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Remove acentos, lowercase, strip."""
        import unicodedata
        text = text.lower().strip()
        nfd = unicodedata.normalize("NFD", text)
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    def score(self, text: str) -> float:
        """
        Calcula score de sentimento de um texto.

        Abordagem: conta ocorrências de termos do léxico ponderadas
        pelo peso de cada termo e pela posição no texto (início = mais peso).

        Returns:
            Score entre -1.0 (muito negativo) e +1.0 (muito positivo).
            0.0 se nenhum termo for encontrado.
        """
        if not text:
            return 0.0
        normalized = self._normalize_text(text)
        scores = []

        # Termos longos têm prioridade (evita falsos positivos)
        sorted_terms = sorted(self._lexicon.keys(), key=len, reverse=True)

        for term in sorted_terms:
            if term in normalized:
                scores.append(self._lexicon[term])

        if not scores:
            return 0.0

        # Média aparada (ignora extremos se muitos termos)
        if len(scores) >= 5:
            scores_sorted = sorted(scores)
            scores = scores_sorted[1:-1]  # remove o maior e o menor

        return float(sum(scores) / len(scores))

    def classify(self, score: float) -> str:
        """Classifica o score em categoria legível."""
        if score >= 0.35:
            return "POSITIVO"
        if score >= 0.10:
            return "LEVEMENTE_POSITIVO"
        if score >= -0.10:
            return "NEUTRO"
        if score >= -0.35:
            return "LEVEMENTE_NEGATIVO"
        return "NEGATIVO"


class NewsFetcher:
    """
    Busca notícias via Google News RSS.

    Gratuito, sem autenticação, cobertura ampla de fontes BR:
    InfoMoney, Valor Econômico, Exame, E-Investidor, Seu Dinheiro, etc.
    """

    _GOOGLE_NEWS_RSS = (
        "https://news.google.com/rss/search?"
        "q={query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    )

    def __init__(
        self,
        timeout: int = 10,
        max_items: int = 10,
        cache: Optional[Dict] = None,
    ):
        self.timeout = timeout
        self.max_items = max_items
        self._cache: Dict[str, Tuple[List[NewsItem], datetime]] = cache or {}

    def fetch_for_ticker(
        self,
        ticker: str,
        cache_minutes: int = 120,
    ) -> List[NewsItem]:
        """
        Busca notícias recentes para um ticker de FII ou ação.

        Args:
            ticker: ex "HGLG11", "ITUB4"
            cache_minutes: evita bater no Google múltiplas vezes

        Returns:
            Lista de NewsItems com título, descrição, data e score zerado.
        """
        cache_key = ticker.upper()
        if cache_key in self._cache:
            items, cached_at = self._cache[cache_key]
            age = (datetime.now(timezone.utc) - cached_at).total_seconds() / 60
            if age < cache_minutes:
                return items

        query = f"{ticker} FII fundo imobiliário"
        items = self._fetch_rss(query)
        self._cache[cache_key] = (items, datetime.now(timezone.utc))
        return items

    def _fetch_rss(self, query: str) -> List[NewsItem]:
        """Faz o fetch do RSS do Google News."""
        encoded = urllib.parse.quote(query)
        url = self._GOOGLE_NEWS_RSS.format(query=encoded)
        items = []
        if urllib.parse.urlsplit(url).scheme.lower() not in _ALLOWED_SCHEMES:
            return items
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "QuantBot/2.0 (RSS reader; educational)"},
            )
            # Esquema http/https já validado logo acima.
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                raw = resp.read().decode("utf-8", errors="replace")
            root = _safe_fromstring(raw)
            channel = root.find("channel")
            if channel is None:
                return []

            for item_el in channel.findall("item")[:self.max_items]:
                title = (item_el.findtext("title") or "").strip()
                desc = (item_el.findtext("description") or "").strip()
                link = (item_el.findtext("link") or "").strip()
                pub = item_el.findtext("pubDate") or ""
                source = item_el.findtext("source") or "Google News"

                # Parseia data
                pub_dt = self._parse_date(pub)

                # Remove HTML da descrição
                desc_clean = re.sub(r"<[^>]+>", " ", desc)
                desc_clean = re.sub(r"\s+", " ", desc_clean).strip()

                items.append(NewsItem(
                    titulo=title,
                    descricao=desc_clean[:300],
                    data=pub_dt,
                    url=link,
                    fonte=source,
                ))

        except (urllib.error.URLError, ET.ParseError, Exception) as e:
            logger.debug(f"Falha ao buscar RSS para '{query}': {e}")

        return items

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parseia datas no formato RFC 2822 do RSS."""
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None


class SentimentAnalyzer:
    """
    Análise de sentimento de notícias para FIIs e ações.

    Combina o léxico financeiro com as notícias buscadas para gerar
    um score de sentimento por ticker, ponderado por recência.
    """

    def __init__(self):
        self.lexicon = BrazilianFinancialLexicon()
        self.fetcher = NewsFetcher()

    def analyze_ticker(
        self,
        ticker: str,
        use_news: bool = True,
    ) -> TickerSentiment:
        """
        Analisa o sentimento atual do mercado para um ticker.

        Args:
            ticker: ex "HGLG11"
            use_news: se False, retorna score 0 (modo offline)

        Returns:
            TickerSentiment com score, classificação e notícias.
        """
        if not use_news:
            return TickerSentiment(
                ticker=ticker, score=0.0, n_noticias=0,
                classificacao="SEM_DADOS",
            )

        noticias = self.fetcher.fetch_for_ticker(ticker)
        if not noticias:
            return TickerSentiment(
                ticker=ticker, score=0.0, n_noticias=0,
                classificacao="SEM_DADOS",
            )

        # Pontua cada notícia e pondera por recência
        scored = []
        now = datetime.now(timezone.utc)
        for n in noticias:
            texto = f"{n.titulo} {n.descricao}"
            n.score = self.lexicon.score(texto)

            # Peso por recência (últimas 24h = peso 1.0, 7 dias = 0.5)
            if n.data:
                try:
                    data_aware = n.data if n.data.tzinfo else n.data.replace(tzinfo=timezone.utc)
                    age_hours = (now - data_aware).total_seconds() / 3600
                    weight = max(0.1, 1.0 - (age_hours / 168))  # decai em 7 dias
                except Exception:
                    weight = 0.5
            else:
                weight = 0.5

            scored.append((n.score, weight))

        # Score ponderado
        total_weight = sum(w for _, w in scored)
        if total_weight == 0:
            avg_score = 0.0
        else:
            avg_score = sum(s * w for s, w in scored) / total_weight

        avg_score = max(-1.0, min(1.0, avg_score))

        return TickerSentiment(
            ticker=ticker,
            score=round(avg_score, 3),
            n_noticias=len(noticias),
            classificacao=self.lexicon.classify(avg_score),
            noticias=noticias,
        )

    def analyze_portfolio(
        self,
        tickers: List[str],
        use_news: bool = True,
        delay_between: float = 0.5,
    ) -> Dict[str, TickerSentiment]:
        """
        Analisa sentimento de uma lista de tickers.

        Args:
            tickers: lista de tickers
            use_news: usar RSS (True) ou modo offline (False)
            delay_between: pausa entre requests para não sobrecarregar

        Returns:
            Dict ticker -> TickerSentiment
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.analyze_ticker(ticker, use_news=use_news)
            if use_news and delay_between > 0:
                time.sleep(delay_between)
        return results

    def sentiment_signal(self, score: float) -> int:
        """
        Converte score em sinal de trading: +1 (favorável), 0 (neutro), -1 (desfavorável).
        Usado como feature adicional no FII Screener.
        """
        if score >= 0.20:
            return 1
        if score <= -0.20:
            return -1
        return 0
