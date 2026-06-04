"""
Testes do módulo de renda passiva — FII Screener, Barsi Screener, Projector.
Verificam comportamento de negócio real, não implementação.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from income.fii_screener import (
    FIIMetrics, FIISegment, FIIRiskLevel,
    FIIScreener, ScreenerCriteria,
)
from income.barsi_screener import (
    StockBarsiMetrics, BarsiSector, BarsiScreener, BarsiScreenerCriteria,
    BESTIES_TICKERS,
)
from income.income_projector import (
    AssetPosition, IncomePortfolio, IncomeProjector,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fii_excelente() -> FIIMetrics:
    return FIIMetrics(
        ticker="HGLG11", nome="CSHG Logística", segmento=FIISegment.LOGISTICA,
        gestora="Credit Suisse Hedging-Griffo", dy_12m=11.5, dy_mes_atual=0.96,
        provento_mensal=1.15, preco_cota=120.0, pvp=0.92, patrimonio_liq=3500.0,
        vacancia_fisica=3.0, vacancia_financeira=2.5, liquidez_diaria=8_000_000,
        resultado_por_cota=1.20, provento_sobre_resultado=0.96,
        historico_anos_sem_corte=6, taxa_administracao=0.60,
        n_imoveis=25, n_inquilinos=40, indexador_predominante="IPCA",
    )


def _fii_ruim() -> FIIMetrics:
    return FIIMetrics(
        ticker="LIXO11", nome="Fundo Problemático", segmento=FIISegment.LAJES_CORPORATIVAS,
        gestora="Gestora Desconhecida", dy_12m=22.0, dy_mes_atual=1.8,
        provento_mensal=0.90, preco_cota=50.0, pvp=1.35, patrimonio_liq=80.0,
        vacancia_fisica=35.0, vacancia_financeira=40.0, liquidez_diaria=100_000,
        resultado_por_cota=0.5, provento_sobre_resultado=1.8,
        historico_anos_sem_corte=0, taxa_administracao=2.5,
        n_imoveis=2, n_inquilinos=3, indexador_predominante="FIXO",
    )


def _acao_barsi() -> StockBarsiMetrics:
    """EGIE3 com DY 9% > referência Bazin 8% → preço abaixo do teto."""
    return StockBarsiMetrics(
        ticker="EGIE3", nome="Engie Brasil", setor=BarsiSector.ENERGIA,
        dy_12m=9.0, dy_medio_5a=7.5, payout_ratio=75.0,
        historico_pagamentos_anos=10, frequencia_pagamento="trimestral",
        crescimento_dividendo_5a=6.5, preco_atual=42.0, lpa=4.2,
        p_l=10.0, divida_ebitda=2.1, roe=22.0, crescimento_lucro_5a=8.0,
    )


def _acao_ruim() -> StockBarsiMetrics:
    return StockBarsiMetrics(
        ticker="MGLU3", nome="Magazine Luiza", setor=BarsiSector.OUTRO,
        dy_12m=1.2, dy_medio_5a=0.5, payout_ratio=10.0,
        historico_pagamentos_anos=1, frequencia_pagamento="anual",
        crescimento_dividendo_5a=-5.0, preco_atual=3.50, lpa=0.1,
        p_l=35.0, divida_ebitda=8.5, roe=2.0, crescimento_lucro_5a=-15.0,
    )


# ═══════════════════════════════════════════════════════════════
# FII Screener
# ═══════════════════════════════════════════════════════════════

class TestFIIScreener:
    def test_fii_excelente_aprovado(self):
        """FII de alta qualidade deve ser aprovado."""
        screener = FIIScreener()
        resultado = screener.screen([_fii_excelente()])
        assert len(resultado.aprovados) == 1
        assert resultado.aprovados[0].ticker == "HGLG11"

    def test_fii_ruim_reprovado(self):
        """FII com múltiplos problemas deve ser reprovado."""
        screener = FIIScreener()
        resultado = screener.screen([_fii_ruim()])
        assert len(resultado.reprovados) == 1
        assert len(resultado.reprovados[0].motivos_reprovacao) > 0

    def test_ranking_por_score(self):
        """Aprovados devem estar ordenados por score decrescente."""
        screener = FIIScreener()
        fiis = [_fii_excelente()]
        # Adiciona uma versão levemente pior
        pior = _fii_excelente()
        pior.ticker = "PEOR11"
        pior.pvp = 1.10
        pior.vacancia_fisica = 8.0
        resultado = screener.screen(fiis + [pior])
        scores = [f.score for f in resultado.aprovados]
        assert scores == sorted(scores, reverse=True)

    def test_dy_muito_alto_reprovado(self):
        """DY acima de 20% deve levantar suspeita — sinal de insustentabilidade."""
        screener = FIIScreener()
        fii = _fii_excelente()
        fii.dy_12m = 25.0  # suspeito
        resultado = screener.screen([fii])
        assert len(resultado.reprovados) == 1

    def test_provento_insustentavel_reprovado(self):
        """Distribuição maior que o resultado operacional deve reprovar."""
        screener = FIIScreener()
        fii = _fii_excelente()
        fii.provento_sobre_resultado = 1.5  # pagando 50% a mais do que ganha
        resultado = screener.screen([fii])
        assert len(resultado.reprovados) == 1

    def test_summary_gerado(self):
        """O resumo deve ser gerado sem erros."""
        screener = FIIScreener()
        resultado = screener.screen([_fii_excelente(), _fii_ruim()])
        sumario = resultado.summary()
        assert "HGLG11" in sumario
        assert "TOP" in sumario

    def test_to_dict_list(self):
        """Conversão para dicionário deve incluir campos essenciais."""
        screener = FIIScreener()
        resultado = screener.screen([_fii_excelente()])
        dicts = resultado.to_dict_list()
        assert len(dicts) == 1
        assert "ticker" in dicts[0]
        assert "dy_12m" in dicts[0]
        assert "score" in dicts[0]


# ═══════════════════════════════════════════════════════════════
# Barsi Screener
# ═══════════════════════════════════════════════════════════════

class TestBarsiScreener:
    def test_acao_besties_aprovada(self):
        """Ação BESTIES com bons fundamentos deve ser aprovada."""
        screener = BarsiScreener()
        resultado = screener.screen([_acao_barsi()])
        assert len(resultado.aprovadas) == 1

    def test_acao_fora_besties_reprovada(self):
        """Ação fora dos setores BESTIES deve ser reprovada por padrão."""
        screener = BarsiScreener()
        resultado = screener.screen([_acao_ruim()])
        assert len(resultado.reprovadas) == 1

    def test_preco_teto_bazin_calculado(self):
        """O preço teto Bazin deve ser calculado corretamente."""
        screener = BarsiScreener()
        acao = _acao_barsi()
        screener._analyze(acao)
        # Teto = dividendo_anual / (dy_ref / 100)
        dividendo_anual = acao.preco_atual * (acao.dy_12m / 100)
        teto_esperado = dividendo_anual / (acao.dy_referencia_bazin / 100)
        assert abs(acao.preco_teto_bazin - teto_esperado) < 0.01

    def test_besties_tickers_mapeamento(self):
        """Os tickers BESTIES conhecidos devem estar mapeados."""
        assert "ITUB4" in BESTIES_TICKERS
        assert "EGIE3" in BESTIES_TICKERS
        assert "BBSE3" in BESTIES_TICKERS
        assert "VIVT3" in BESTIES_TICKERS
        assert "SAPR11" in BESTIES_TICKERS

    def test_sentimento_positivo_desempata_sem_mudar_aprovacao(self):
        """Sentimento positivo sobe o score, mas não muda a aprovação."""
        screener = BarsiScreener()
        resultado = screener.screen([_acao_barsi()])
        assert len(resultado.aprovadas) == 1
        score_antes = resultado.aprovadas[0].score

        screener.aplicar_sentimento_noticias(resultado, {"EGIE3": ("positivo", 0.9)})
        acao = resultado.aprovadas[0]
        assert len(resultado.aprovadas) == 1            # aprovação intacta
        assert acao.score > score_antes                 # desempate pra cima
        assert acao.score_sem_sentimento == score_antes # guarda o original
        assert acao.sentimento_noticias == "positivo"

    def test_sentimento_negativo_reduz_score(self):
        """Sentimento negativo derruba o score (mas não reprova)."""
        screener = BarsiScreener()
        resultado = screener.screen([_acao_barsi()])
        score_antes = resultado.aprovadas[0].score
        screener.aplicar_sentimento_noticias(resultado, {"EGIE3": ("negativo", 0.95)})
        assert resultado.aprovadas[0].score < score_antes
        assert len(resultado.aprovadas) == 1

    def test_sentimento_neutro_nao_altera_score(self):
        """Confiança 0.5 (indeciso) não deve mexer no score."""
        screener = BarsiScreener()
        resultado = screener.screen([_acao_barsi()])
        score_antes = resultado.aprovadas[0].score
        screener.aplicar_sentimento_noticias(resultado, {"EGIE3": ("neutro", 0.5)})
        assert resultado.aprovadas[0].score == score_antes

    def test_summary_barsi_gerado(self):
        screener = BarsiScreener()
        resultado = screener.screen([_acao_barsi(), _acao_ruim()])
        sumario = resultado.summary()
        assert "MÉTODO BARSI" in sumario
        assert "EGIE3" in sumario

    def test_sem_exigencia_besties(self):
        """Sem a exigência de BESTIES, ação boa de outro setor pode passar."""
        criteria = BarsiScreenerCriteria(apenas_besties=False, dy_min=1.0,
                                         payout_min=5.0, historico_min_anos=1,
                                         roe_min=1.0, preco_abaixo_teto=False)
        screener = BarsiScreener(criteria)
        acao = _acao_ruim()
        acao.dy_12m = 8.0
        acao.payout_ratio = 60.0
        acao.historico_pagamentos_anos = 3
        acao.roe = 15.0
        acao.divida_ebitda = 1.5
        resultado = screener.screen([acao])
        # Sem BESTIES obrigatório e com métricas corrigidas, deve passar
        assert len(resultado.aprovadas) == 1


# ═══════════════════════════════════════════════════════════════
# Income Projector
# ═══════════════════════════════════════════════════════════════

class TestIncomeProjector:
    def _portfolio_simples(self, valor=10_000.0, dy=10.0):
        p = IncomePortfolio()
        p.adicionar(AssetPosition(
            ticker="HGLG11", nome="CSHG Log", tipo="FII",
            valor_atual=valor, dy_anual=dy, pagamento="mensal",
            isento_ir=True,
        ))
        return p

    def test_renda_mensal_correta(self):
        """Renda mensal deve ser DY_anual * capital / 12."""
        p = self._portfolio_simples(10_000, 12.0)
        esperado = 10_000 * 0.12 / 12  # = 100 / mês
        assert abs(p.renda_mensal_total - esperado) < 0.01

    def test_projecao_atinge_meta_com_aportes(self):
        """Com aportes mensais constantes, deve atingir meta no horizonte."""
        projector = IncomeProjector()
        p = self._portfolio_simples(5_000, 11.0)
        resultado = projector.project(p, aporte_mensal=500, meta_renda_mensal=1_000,
                                      horizonte_meses=360, dy_anual_medio=11.0)
        assert resultado.meta_atingida

    def test_sem_aporte_demora_mais(self):
        """Sem aportes, demora mais para atingir a meta."""
        projector = IncomeProjector()
        p = self._portfolio_simples(5_000, 10.0)
        res_com = projector.project(p, 500, 1_000, 360)
        res_sem = projector.project(p, 0, 1_000, 360)
        if res_com.meta_atingida and res_sem.meta_atingida:
            assert res_sem.meses_para_meta > res_com.meses_para_meta

    def test_capital_necessario_formula(self):
        """Fórmula de capital necessário: meta * 12 / (DY / 100)."""
        projector = IncomeProjector()
        capital = projector.capital_necessario(meta_renda_mensal=3_000, dy_anual=12.0)
        esperado = 3_000 * 12 / 0.12  # = 300_000
        assert abs(capital - esperado) < 1.0

    def test_multi_meta_retorna_todas(self):
        """multi_meta deve retornar resultado para cada meta solicitada."""
        projector = IncomeProjector()
        p = self._portfolio_simples(10_000, 10.0)
        metas = [500, 1_000, 2_000]
        resultados = projector.multi_meta(p, 300, metas)
        assert len(resultados) == 3
        for r in resultados:
            assert "meta_mensal" in r

    def test_dividendos_acumulados_positivos(self):
        """Total de dividendos recebidos deve crescer ao longo do tempo."""
        projector = IncomeProjector()
        p = self._portfolio_simples(20_000, 10.0)
        resultado = projector.project(p, 200, 5_000, 120)
        assert resultado.total_dividendos_recebidos > 0

    def test_bola_de_neve_funciona(self):
        """Com reinvestimento, o capital final deve superar sem reinvestimento."""
        projector = IncomeProjector()
        p = self._portfolio_simples(10_000, 10.0)
        com = projector.project(p, 300, 5_000, 240, reinvestir_dividendos=True)
        sem = projector.project(p, 300, 5_000, 240, reinvestir_dividendos=False)
        assert com.capital_no_final > sem.capital_no_final

    def test_resumo_formatado_gerado(self):
        projector = IncomeProjector()
        p = self._portfolio_simples(20_000, 11.0)
        texto = projector.resumo_formatado(p, aporte_mensal=500)
        assert "PROJEÇÃO" in texto
        assert "R$" in texto
