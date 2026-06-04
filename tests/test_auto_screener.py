"""Testes do AutoScreener e integração de componentes."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from income.auto_screener import AutoScreener
from income.fii_screener import FIIMetrics, FIISegment, FIIScreener


def _mock_fii(ticker, dy=11.5, pvp=0.92, vacancia=3.0, liquidez=5_000_000, patrimonio=2000.0):
    return FIIMetrics(
        ticker=ticker, nome=f"Fundo {ticker}", segmento=FIISegment.LOGISTICA,
        gestora="CSHG", dy_12m=dy, dy_mes_atual=0.96, provento_mensal=1.15,
        preco_cota=120.0, pvp=pvp, patrimonio_liq=patrimonio,
        vacancia_fisica=vacancia, vacancia_financeira=vacancia * 1.1,
        liquidez_diaria=liquidez, resultado_por_cota=1.20,
        provento_sobre_resultado=0.96, historico_anos_sem_corte=5,
        taxa_administracao=0.60, n_imoveis=20, n_inquilinos=35,
        indexador_predominante="IPCA",
    )


class TestAutoScreener:
    def _screener(self):
        from income.fii_screener import FIIScreener
        return AutoScreener(fii_screener=FIIScreener(), selic_override=14.5)

    def test_recommendations_comprar_novo(self):
        """FIIs aprovados que não estão na carteira → COMPRAR."""
        screener = self._screener()
        fiis = [_mock_fii("HGLG11"), _mock_fii("KNCR11")]

        from income.fii_screener import ScreeningResult, ScreenerCriteria
        from income.fii_screener import FIIScreener
        result = FIIScreener().screen(fiis)

        recs = screener._generate_recommendations(result, current_portfolio={})
        compras = [r for r in recs if r["acao"] == "COMPRAR"]
        assert len(compras) > 0

    def test_recommendations_vender_reprovado(self):
        """FII reprovado que está na carteira → VENDER."""
        screener = self._screener()
        fii_ruim = _mock_fii("LIXO11", dy=22.0, pvp=1.5, vacancia=35.0,
                             liquidez=50_000, patrimonio=50.0)
        fii_ruim.taxa_administracao = 3.0
        fii_ruim.provento_sobre_resultado = 1.8

        from income.fii_screener import FIIScreener
        result = FIIScreener().screen([fii_ruim])

        recs = screener._generate_recommendations(result, current_portfolio={"LIXO11": 10.0})
        vendas = [r for r in recs if r["acao"] == "VENDER"]
        assert len(vendas) >= 1
        assert vendas[0]["ticker"] == "LIXO11"

    def test_manter_aprovado_na_carteira(self):
        """FII aprovado que já está na carteira → MANTER."""
        screener = self._screener()
        fiis = [_mock_fii("HGLG11")]

        from income.fii_screener import FIIScreener
        result = FIIScreener().screen(fiis)

        recs = screener._generate_recommendations(result, current_portfolio={"HGLG11": 5.0})
        mantencoes = [r for r in recs if r["acao"] == "MANTER"]
        assert len(mantencoes) >= 1

    def test_portfolio_health_alerts_vacancia(self):
        """FII com vacância alta na carteira → alerta."""
        screener = self._screener()
        fii_vac = _mock_fii("PROB11", vacancia=25.0)
        alerts = screener._portfolio_health_alerts([fii_vac], current_portfolio={"PROB11": 10.0})
        assert any("PROB11" in a and "vacância" in a.lower() for a in alerts)

    def test_portfolio_health_alerts_sem_dados(self):
        """FII na carteira sem dados na brapi → alerta diferente."""
        screener = self._screener()
        alerts = screener._portfolio_health_alerts([], current_portfolio={"MIST11": 5.0})
        assert any("MIST11" in a for a in alerts)

    def test_adjust_criteria_for_selic(self):
        """Critérios de DY ajustados para a Selic."""
        screener = self._screener()
        screener._adjust_criteria_for_selic(14.5)
        assert screener.screener.criteria.dy_min >= 7.0
        screener._adjust_criteria_for_selic(8.0)
        assert screener.screener.criteria.dy_min >= 7.0  # mínimo absoluto

    def test_convert_brapi_formato_basico(self):
        """Conversão de formato brapi.dev para FIIMetrics."""
        screener = self._screener()
        raw = [
            {
                "ticker": "HGLG11",
                "name": "CSHG Logística",
                "dividendYield": 11.5,
                "priceToBook": 0.92,
                "vacancyRate": 3.0,
                "liquidity": 8_000_000,
                "netAssets": 3_500,  # em R$ milhões
                "lastDividend": 1.15,
                "close": 120.0,
                "managementFee": 0.60,
                "segment": "logística",
            }
        ]
        metrics = screener._convert_brapi_to_metrics(raw)
        assert len(metrics) == 1
        assert metrics[0].ticker == "HGLG11"
        assert metrics[0].dy_12m == 11.5
        assert metrics[0].pvp == 0.92

    def test_format_report_text(self):
        """Relatório formatado deve conter texto legível."""
        screener = self._screener()
        from income.auto_screener import ScreeningReport
        report = ScreeningReport(
            timestamp="2026-06-01T18:00:00",
            fii_result=None,
            stock_result=None,
            recommendations=[
                {"acao": "COMPRAR", "ticker": "HGLG11",
                 "motivo": "Score 80 | DY 11.5%", "pontos_positivos": [], "prioridade": "ALTA"},
                {"acao": "VENDER", "ticker": "LIXO11",
                 "motivo": "Vacância 35%", "pontos_positivos": [], "prioridade": "ALTA"},
            ],
            alerts=["⚠️ PROB11: vacância 25%"],
            data_quality={"selic_usada": 14.5},
        )
        texto = screener.format_report_text(report)
        assert "COMPRAR" in texto
        assert "HGLG11" in texto
        assert "VENDER" in texto
        assert "LIXO11" in texto
