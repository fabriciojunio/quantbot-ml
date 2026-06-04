"""
Testes dos módulos de Sentimento de Notícias e CVM.
Verificam comportamento sem acesso à rede (offline/mockado).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.news_sentiment import BrazilianFinancialLexicon, SentimentAnalyzer, NewsFetcher
from data.cvm_client import CVMClient


# ═══════════════════════════════════════════════════════════════
# Léxico Financeiro
# ═══════════════════════════════════════════════════════════════

class TestBrazilianFinancialLexicon:
    def _lex(self):
        return BrazilianFinancialLexicon()

    def test_texto_claramente_positivo(self):
        """Notícia com lucro recorde deve ter score positivo."""
        lex = self._lex()
        score = lex.score("HGLG11 registra lucro recorde e aumenta distribuicao")
        assert score > 0

    def test_texto_claramente_negativo(self):
        """Notícia de queda e prejuizo deve ter score negativo."""
        lex = self._lex()
        score = lex.score("FII registra queda acentuada e prejuizo no trimestre")
        assert score < 0

    def test_texto_vazio(self):
        """Texto vazio deve retornar score zero."""
        lex = self._lex()
        assert lex.score("") == 0.0

    def test_texto_neutro(self):
        """Texto sem termos financeiros deve retornar próximo de zero."""
        lex = self._lex()
        score = lex.score("O gestor realizou reunião ordinária de cotistas")
        assert -0.5 <= score <= 0.5

    def test_classify_positivo(self):
        assert BrazilianFinancialLexicon.classify(None, 0.5) == "POSITIVO"

    def test_classify_negativo(self):
        assert BrazilianFinancialLexicon.classify(None, -0.5) == "NEGATIVO"

    def test_classify_neutro(self):
        assert BrazilianFinancialLexicon.classify(None, 0.0) == "NEUTRO"

    def test_score_bounds(self):
        """Score deve estar sempre entre -1 e +1."""
        lex = self._lex()
        textos = [
            "lucro recorde crescimento positivo alta distribuicao expansao",
            "falencia fraude investigacao cvm liquidacao calote default",
            "",
            "texto completamente aleatorio sem palavras financeiras xyzxyz",
        ]
        for t in textos:
            s = lex.score(t)
            assert -1.0 <= s <= 1.0, f"Score fora dos limites para '{t[:40]}': {s}"

    def test_acento_normalizado(self):
        """Palavras com e sem acento devem ter o mesmo score."""
        lex = self._lex()
        s1 = lex.score("queda e reducao de provento")
        s2 = lex.score("queda e redução de provento")
        assert abs(s1 - s2) < 0.01


# ═══════════════════════════════════════════════════════════════
# SentimentAnalyzer (offline)
# ═══════════════════════════════════════════════════════════════

class TestSentimentAnalyzer:
    def test_offline_retorna_sem_dados(self):
        """Modo offline (use_news=False) retorna SEM_DADOS."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_ticker("HGLG11", use_news=False)
        assert result.ticker == "HGLG11"
        assert result.classificacao == "SEM_DADOS"
        assert result.score == 0.0

    def test_signal_positivo(self):
        analyzer = SentimentAnalyzer()
        assert analyzer.sentiment_signal(0.5) == 1

    def test_signal_negativo(self):
        analyzer = SentimentAnalyzer()
        assert analyzer.sentiment_signal(-0.5) == -1

    def test_signal_neutro(self):
        analyzer = SentimentAnalyzer()
        assert analyzer.sentiment_signal(0.05) == 0

    def test_portfolio_offline(self):
        """Análise de portfolio offline deve retornar dict com todos os tickers."""
        analyzer = SentimentAnalyzer()
        tickers = ["HGLG11", "KNCR11", "PVBI11"]
        results = analyzer.analyze_portfolio(tickers, use_news=False, delay_between=0)
        assert len(results) == 3
        for t in tickers:
            assert t in results


# ═══════════════════════════════════════════════════════════════
# CVM Client (sem rede — testa parsing)
# ═══════════════════════════════════════════════════════════════

class TestCVMClient:
    def _client(self, tmp_path):
        return CVMClient(cache_dir=str(tmp_path / "cache"))

    def test_parse_cadastro_csv(self, tmp_path):
        """Parsing do CSV de cadastro com formato correto da CVM."""
        client = self._client(tmp_path)
        csv_sample = (
            "CNPJ_FUNDO;DENOM_SOCIAL;NOME_ADMIN;CNPJ_ADMIN;"
            "DT_REG;SIT;TP_FUNDO;PRAZO;MUNI_GESTOR;UF_GESTOR\n"
            "17.022.084/0001-53;CSHG LOGISTICA FII;Credit Suisse;60.746.948/0001-12;"
            "2010-05-14;EM FUNCIONAMENTO NORMAL;FII;INDETERMINADO;Sao Paulo;SP\n"
            "00.000.000/0001-00;FII CANCELADO;Admin X;00.000.000/0001-01;"
            "2015-01-01;CANCELADO;FII;INDETERMINADO;Rio de Janeiro;RJ\n"
        )
        regs = client._parse_cadastro_csv(csv_sample)
        assert len(regs) == 2
        assert regs[0].cnpj_fundo == "17.022.084/0001-53"
        assert regs[0].nome_fundo == "CSHG LOGISTICA FII"
        assert regs[0].situacao == "EM FUNCIONAMENTO NORMAL"

    def test_filtro_apenas_ativos(self, tmp_path):
        """Deve filtrar apenas fundos em funcionamento."""
        client = self._client(tmp_path)
        csv_sample = (
            "CNPJ_FUNDO;DENOM_SOCIAL;NOME_ADMIN;CNPJ_ADMIN;"
            "DT_REG;SIT;TP_FUNDO;PRAZO;MUNI_GESTOR;UF_GESTOR\n"
            "11.111.111/0001-11;ATIVO;Admin;00.000.000/0001-01;"
            "2020-01-01;EM FUNCIONAMENTO NORMAL;FII;INDETERMINADO;SP;SP\n"
            "22.222.222/0001-22;CANCELADO FII;Admin;00.000.000/0001-01;"
            "2015-01-01;CANCELADO;FII;INDETERMINADO;RJ;RJ\n"
        )
        regs = client._parse_cadastro_csv(csv_sample)
        ativos = [r for r in regs if "FUNCIONAMENTO" in r.situacao.upper()]
        assert len(ativos) == 1
        assert ativos[0].cnpj_fundo == "11.111.111/0001-11"

    def test_parse_informe_csv(self, tmp_path):
        """Parsing do CSV de informe mensal."""
        client = self._client(tmp_path)
        csv_sample = (
            "CNPJ_FUNDO;COMPETENCIA;VL_PATRIM_LIQ;NR_COTST;VL_COTA;"
            "VL_RENDTO_PAGO;VL_RENDTO_PAGO_COTA;VL_TOTAL_ATIVOS;VL_TOTAL_PASSIVOS;VL_RESULT_MES\n"
            "17.022.084/0001-53;2026-05;3500000000,00;120000;120,50;"
            "138000000,00;1,15;3600000000,00;100000000,00;145000000,00\n"
        )
        reports = client._parse_informe_csv(csv_sample)
        assert len(reports) == 1
        r = reports[0]
        assert r.cnpj_fundo == "17.022.084/0001-53"
        assert r.pl == 3_500_000_000.0
        assert r.n_cotistas == 120_000
        assert abs(r.rendimento_por_cota - 1.15) < 0.01

    def test_cross_validate_brapi_match(self, tmp_path):
        """Quando valores batem, validado deve ser True."""
        client = self._client(tmp_path)
        csv = (
            "CNPJ_FUNDO;COMPETENCIA;VL_PATRIM_LIQ;NR_COTST;VL_COTA;"
            "VL_RENDTO_PAGO;VL_RENDTO_PAGO_COTA;VL_TOTAL_ATIVOS;VL_TOTAL_PASSIVOS;VL_RESULT_MES\n"
            "17.022.084/0001-53;2026-05;3500000000;120000;120,50;"
            "138000000;1,15;3600000000;100000000;145000000\n"
        )
        reports = client._parse_informe_csv(csv)
        cnpj = "17.022.084/0001-53"

        class MockClient(CVMClient):
            def get_monthly_reports(self, year=None, month=None, cache_hours=48):
                return reports

        mc = MockClient(cache_dir=str(tmp_path / "cache2"))
        result = mc.cross_validate_brapi("HGLG11", 1.15, cnpj, 2026, 5)
        assert result["validado"] is True
        assert result["diferenca"] == 0.0

    def test_to_float_virgula(self, tmp_path):
        """Deve converter números com vírgula (padrão BR) corretamente."""
        client = self._client(tmp_path)
        assert client._to_float("3.500.000,50") == 3_500_000.50
        assert client._to_float("") == 0.0
        assert client._to_float("0") == 0.0

    def test_cache_sem_rede(self, tmp_path):
        """Sem rede, deve retornar lista vazia sem crash."""
        client = self._client(tmp_path)
        result = client.get_fii_registration()
        assert isinstance(result, list)
