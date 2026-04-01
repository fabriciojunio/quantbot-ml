"""Testes para módulos de notícias e sentimento."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from data.news_fetcher import NewsFetcher, NewsArticle, KEYWORD_TO_SYMBOLS, CATEGORY_KEYWORDS
from data.sentiment import (
    SentimentAnalyzer, LexiconAnalyzer, SentimentResult,
    analyze_news_sentiment, get_symbol_sentiment,
)
from datetime import datetime


class TestNewsFetcher:

    def test_detect_symbols_petrobras(self):
        article = NewsArticle(
            title="Petrobras anuncia dividendos recordes",
            description="A estatal de petróleo Petrobras divulgou resultados",
            source="Test", url="", published=datetime.now(),
        )
        fetcher = NewsFetcher()
        symbols = fetcher._detect_symbols(article)
        assert "PETR4.SA" in symbols

    def test_detect_symbols_bitcoin(self):
        article = NewsArticle(
            title="Bitcoin supera US$70 mil",
            description="O preço do Bitcoin atingiu novo recorde",
            source="Test", url="", published=datetime.now(),
        )
        fetcher = NewsFetcher()
        symbols = fetcher._detect_symbols(article)
        assert "BTC-USD" in symbols

    def test_detect_symbols_macro(self):
        article = NewsArticle(
            title="Copom decide manter taxa Selic em 13,75%",
            description="Banco Central manteve a taxa básica de juros",
            source="Test", url="", published=datetime.now(),
        )
        fetcher = NewsFetcher()
        symbols = fetcher._detect_symbols(article)
        assert any("ITUB4" in s for s in symbols)

    def test_detect_category(self):
        article = NewsArticle(
            title="Fed mantém juros inalterados",
            description="A taxa básica de juros americana permanece",
            source="Test", url="", published=datetime.now(),
        )
        fetcher = NewsFetcher()
        category = fetcher._detect_category(article)
        assert category == "Política Monetária"

    def test_estimate_impact_high(self):
        article = NewsArticle(
            title="Crise bancária histórica atinge mercados",
            description="Colapso sem precedentes",
            source="Test", url="", published=datetime.now(),
        )
        fetcher = NewsFetcher()
        impact = fetcher._estimate_impact(article)
        assert impact == "alta"

    def test_estimate_impact_low(self):
        article = NewsArticle(
            title="Coluna: opinião sobre investimentos",
            description="Análise de mercado semanal",
            source="Test", url="", published=datetime.now(),
        )
        fetcher = NewsFetcher()
        impact = fetcher._estimate_impact(article)
        assert impact == "baixa"

    def test_deduplicate(self):
        articles = [
            NewsArticle("Petrobras sobe 5%", "", "A", "", datetime.now()),
            NewsArticle("Petrobras sobe 5%", "", "B", "", datetime.now()),
            NewsArticle("Vale cai 3%", "", "A", "", datetime.now()),
        ]
        fetcher = NewsFetcher()
        unique = fetcher._deduplicate(articles)
        assert len(unique) == 2

    def test_clean_html(self):
        html = "<p>Texto <b>importante</b> &amp; relevante</p>"
        result = NewsFetcher._clean_html(html)
        assert result == "Texto importante & relevante"


class TestLexiconAnalyzer:

    def test_positive_sentiment(self):
        analyzer = LexiconAnalyzer()
        result = analyzer.analyze("Petrobras reporta lucro recorde e alta nas ações")
        assert result.label == "positivo"
        assert result.score > 0.5
        assert result.method == "lexicon"

    def test_negative_sentiment(self):
        analyzer = LexiconAnalyzer()
        result = analyzer.analyze("Crise e queda nas ações após prejuízo")
        assert result.label == "negativo"
        assert result.score > 0.5

    def test_neutral_sentiment(self):
        analyzer = LexiconAnalyzer()
        result = analyzer.analyze("Reunião marcada para amanhã às 14h")
        assert result.label == "neutro"

    def test_negation(self):
        analyzer = LexiconAnalyzer()
        result = analyzer.analyze("A empresa não lucrou e teve queda nos resultados")
        # "não" + "lucrou" → negativo, plus "queda"
        assert result.label == "negativo"

    def test_empty_text(self):
        analyzer = LexiconAnalyzer()
        result = analyzer.analyze("")
        assert result.label == "neutro"

    def test_english_text(self):
        analyzer = LexiconAnalyzer()
        result = analyzer.analyze("Stock market rally as profits beat expectations")
        assert result.label == "positivo"

    def test_mixed_sentiment(self):
        analyzer = LexiconAnalyzer()
        result = analyzer.analyze("Lucro subiu mas risco de crise permanece")
        # Tem palavras positivas e negativas
        assert result.label in ["positivo", "negativo", "neutro"]
        assert result.score > 0


class TestSentimentAnalyzer:

    def test_auto_fallback(self):
        """Deve usar léxico quando FinBERT não está instalado."""
        analyzer = SentimentAnalyzer(prefer_finbert=True)
        result = analyzer.analyze("Ações sobem com resultados fortes")
        assert result.label in ["positivo", "negativo", "neutro"]
        assert result.score > 0

    def test_force_lexicon(self):
        analyzer = SentimentAnalyzer(prefer_finbert=False)
        assert analyzer.get_method() == "lexicon"
        result = analyzer.analyze("Crash no mercado de ações")
        assert result.label == "negativo"

    def test_batch_analysis(self):
        analyzer = SentimentAnalyzer(prefer_finbert=False)
        texts = [
            "Lucro recorde da empresa",
            "Crise econômica se aprofunda",
            "Reunião agendada para amanhã",
        ]
        results = analyzer.analyze_batch(texts)
        assert len(results) == 3
        assert results[0].label == "positivo"
        assert results[1].label == "negativo"


class TestIntegration:

    def test_analyze_news_sentiment(self):
        articles = [
            NewsArticle("Petrobras lucro recorde", "Resultados fortes", "Test", "", datetime.now()),
            NewsArticle("Crise bancária global", "Risco de colapso", "Test", "", datetime.now()),
        ]
        analyzed = analyze_news_sentiment(articles)
        assert analyzed[0].sentiment == "positivo"
        assert analyzed[1].sentiment == "negativo"

    def test_get_symbol_sentiment(self):
        articles = [
            NewsArticle("Petrobras sobe com lucro", "Alta expressiva", "Test", "",
                        datetime.now(), symbols=["PETR4.SA"], sentiment="positivo",
                        sentiment_score=0.85, impact="alta"),
            NewsArticle("Petrobras enfrenta risco", "Pressão regulatória", "Test", "",
                        datetime.now(), symbols=["PETR4.SA"], sentiment="negativo",
                        sentiment_score=0.7, impact="média"),
        ]
        label, score = get_symbol_sentiment(articles, "PETR4.SA")
        assert label in ["positivo", "negativo", "neutro"]
        assert 0 <= score <= 1

    def test_no_news_for_symbol(self):
        articles = [
            NewsArticle("Apple sobe", "", "Test", "", datetime.now(),
                        symbols=["AAPL"], sentiment="positivo", sentiment_score=0.8),
        ]
        label, score = get_symbol_sentiment(articles, "PETR4.SA")
        assert label == "neutro"
        assert score == 0.5
