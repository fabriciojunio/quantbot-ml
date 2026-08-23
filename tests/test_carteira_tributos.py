"""
Testes do motor de imposto de renda.

Este é o módulo que mais precisa de teste do pacote, porque errar imposto não
gera exceção: gera um backtest otimista que só aparece na declaração. Cada
regra da lei vira pelo menos um teste, e as pegadinhas (ETF de renda variável
não ter a isenção de R$ 20 mil, ETF de FII não herdar a isenção do provento)
viram testes com nome explícito.
"""

from datetime import date

import pytest

from carteira.tributos import (
    ALIQUOTA_ACAO_COMUM,
    ALIQUOTA_ACAO_DAYTRADE,
    ALIQUOTA_ETF_RV,
    ALIQUOTA_FII_GANHO,
    ISENCAO_MENSAL_ACOES,
    ISENCAO_MENSAL_CRIPTO,
    ApuradorIR,
    Venda,
    aliquota_cripto,
    aliquota_regressiva,
    custo_fiscal_do_giro,
    equivalente_bruto,
    imposto_sobre_provento,
)
from carteira.universo import Regime


def venda(regime, valor, custo, dia=15, mes=3, ano=2026, dias=100, day_trade=False):
    return Venda(
        ticker="TESTE",
        regime=regime,
        quando=date(ano, mes, dia),
        valor_alienado=valor,
        custo_aquisicao=custo,
        dias_carregado=dias,
        day_trade=day_trade,
    )


class TestTabelaRegressiva:
    @pytest.mark.parametrize(
        "dias,esperado",
        [
            (0, 0.225), (1, 0.225), (90, 0.225), (180, 0.225),
            (181, 0.20), (200, 0.20), (360, 0.20),
            (361, 0.175), (500, 0.175), (720, 0.175),
            (721, 0.15), (1000, 0.15), (3650, 0.15),
        ],
    )
    def test_aliquota_por_prazo(self, dias, esperado):
        assert aliquota_regressiva(dias) == esperado

    def test_prazo_negativo_e_recusado(self):
        with pytest.raises(ValueError, match="negativo"):
            aliquota_regressiva(-1)

    def test_aliquota_so_cai_com_o_tempo(self):
        anterior = 1.0
        for dias in (0, 100, 200, 400, 800, 2000):
            atual = aliquota_regressiva(dias)
            assert atual <= anterior
            anterior = atual

    def test_piso_da_tabela_e_quinze_por_cento(self):
        assert aliquota_regressiva(10**6) == 0.15


class TestAliquotaCripto:
    @pytest.mark.parametrize(
        "ganho,esperado",
        [
            (1_000.0, 0.15), (5_000_000.0, 0.15),
            (5_000_001.0, 0.175), (10_000_000.0, 0.175),
            (10_000_001.0, 0.20), (30_000_000.0, 0.20),
            (30_000_001.0, 0.225), (10**9, 0.225),
        ],
    )
    def test_faixas(self, ganho, esperado):
        assert aliquota_cripto(ganho) == esperado

    def test_aliquota_so_sobe_com_o_ganho(self):
        anterior = 0.0
        for g in (1_000, 6_000_000, 11_000_000, 40_000_000):
            atual = aliquota_cripto(g)
            assert atual >= anterior
            anterior = atual


class TestIsencaoDeAcoes:
    def test_venda_ate_vinte_mil_e_isenta(self):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.ACAO_INDIVIDUAL, 19_000, 15_000)])
        assert r.imposto_devido == 0.0
        assert r.ganho_isento == 4_000

    def test_exatamente_vinte_mil_ainda_e_isento(self):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.ACAO_INDIVIDUAL, ISENCAO_MENSAL_ACOES, 15_000)])
        assert r.imposto_devido == 0.0

    def test_um_real_acima_perde_a_isencao_inteira(self):
        """A isenção é sobre o total vendido, não sobre o excedente: passar de
        R$ 20 mil tributa o ganho inteiro, não só a parte que passou."""
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.ACAO_INDIVIDUAL, 20_001, 15_000)])
        assert r.imposto_devido == pytest.approx(5_001 * ALIQUOTA_ACAO_COMUM, rel=1e-6)

    def test_isencao_olha_o_total_do_mes_e_nao_cada_venda(self):
        a = ApuradorIR()
        vendas = [
            venda(Regime.ACAO_INDIVIDUAL, 12_000, 10_000, dia=5),
            venda(Regime.ACAO_INDIVIDUAL, 12_000, 10_000, dia=20),
        ]
        r = a.apurar_mes(2026, 3, vendas)
        assert r.imposto_devido > 0  # somadas dão R$ 24 mil

    def test_day_trade_nao_tem_isencao(self):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.ACAO_INDIVIDUAL, 5_000, 4_000, day_trade=True)])
        assert r.imposto_devido == pytest.approx(1_000 * ALIQUOTA_ACAO_DAYTRADE)


class TestETFNaoTemIsencao:
    """A pegadinha mais cara do varejo brasileiro."""

    def test_etf_de_renda_variavel_paga_mesmo_abaixo_de_vinte_mil(self):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.ETF_RENDA_VARIAVEL, 5_000, 4_000)])
        assert r.imposto_devido == pytest.approx(1_000 * ALIQUOTA_ETF_RV)

    def test_acao_isenta_e_etf_tributado_no_mesmo_valor(self):
        acao = ApuradorIR().apurar_mes(2026, 3, [venda(Regime.ACAO_INDIVIDUAL, 5_000, 4_000)])
        etf = ApuradorIR().apurar_mes(2026, 3, [venda(Regime.ETF_RENDA_VARIAVEL, 5_000, 4_000)])
        assert acao.imposto_devido == 0.0
        assert etf.imposto_devido > 0.0


class TestFII:
    def test_provento_de_fii_e_isento(self):
        assert imposto_sobre_provento(Regime.FII, 1_000.0) == 0.0

    def test_provento_de_etf_de_fii_e_tributado(self):
        """XFIX11 é ETF, não FII: o rendimento dele não herda a isenção."""
        assert imposto_sobre_provento(Regime.ETF_RENDA_VARIAVEL, 1_000.0) > 0.0

    def test_ganho_de_capital_em_fii_paga_vinte_por_cento(self):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.FII, 15_000, 10_000)])
        assert r.imposto_devido == pytest.approx(5_000 * ALIQUOTA_FII_GANHO)

    def test_fii_nao_tem_isencao_de_vinte_mil(self):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.FII, 5_000, 4_000)])
        assert r.imposto_devido > 0

    def test_provento_negativo_ou_zero_nao_gera_imposto(self):
        assert imposto_sobre_provento(Regime.ETF_RENDA_VARIAVEL, 0.0) == 0.0
        assert imposto_sobre_provento(Regime.ETF_RENDA_VARIAVEL, -50.0) == 0.0


class TestIsencaoDeCripto:
    def test_venda_ate_trinta_e_cinco_mil_e_isenta(self):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.CRIPTO, 30_000, 20_000)])
        assert r.imposto_devido == 0.0

    def test_acima_de_trinta_e_cinco_mil_tributa(self):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.CRIPTO, ISENCAO_MENSAL_CRIPTO + 1, 30_000)])
        assert r.imposto_devido > 0


class TestCompensacaoDePrejuizo:
    def test_prejuizo_de_um_mes_compensa_ganho_do_seguinte(self):
        a = ApuradorIR()
        a.apurar_mes(2026, 1, [venda(Regime.ETF_RENDA_VARIAVEL, 8_000, 10_000, mes=1)])
        r = a.apurar_mes(2026, 2, [venda(Regime.ETF_RENDA_VARIAVEL, 12_000, 10_000, mes=2)])
        assert r.imposto_devido == 0.0  # prejuízo de 2.000 cobre o ganho de 2.000

    def test_compensacao_parcial_tributa_a_sobra(self):
        a = ApuradorIR()
        a.apurar_mes(2026, 1, [venda(Regime.ETF_RENDA_VARIAVEL, 9_000, 10_000, mes=1)])
        r = a.apurar_mes(2026, 2, [venda(Regime.ETF_RENDA_VARIAVEL, 13_000, 10_000, mes=2)])
        assert r.imposto_devido == pytest.approx(2_000 * ALIQUOTA_ETF_RV)

    def test_prejuizo_de_fii_nao_compensa_ganho_de_acao(self):
        """A lei separa por modalidade. Misturar é o erro que faz o backtest
        pagar menos imposto do que a declaração vai cobrar."""
        a = ApuradorIR()
        a.apurar_mes(2026, 1, [venda(Regime.FII, 8_000, 10_000, mes=1)])
        r = a.apurar_mes(2026, 2, [venda(Regime.ETF_RENDA_VARIAVEL, 12_000, 10_000, mes=2)])
        assert r.imposto_devido > 0

    def test_prejuizo_de_fii_compensa_ganho_de_fii(self):
        a = ApuradorIR()
        a.apurar_mes(2026, 1, [venda(Regime.FII, 8_000, 10_000, mes=1)])
        r = a.apurar_mes(2026, 2, [venda(Regime.FII, 12_000, 10_000, mes=2)])
        assert r.imposto_devido == 0.0

    def test_prejuizo_acumula_entre_meses(self):
        a = ApuradorIR()
        a.apurar_mes(2026, 1, [venda(Regime.ETF_RENDA_VARIAVEL, 9_000, 10_000, mes=1)])
        r = a.apurar_mes(2026, 2, [venda(Regime.ETF_RENDA_VARIAVEL, 9_000, 10_000, mes=2)])
        assert r.prejuizo_acumulado == pytest.approx(2_000)

    def test_prejuizo_dentro_da_isencao_nao_gera_credito(self):
        """Operação isenta não gera crédito de prejuízo — é o outro lado da moeda."""
        a = ApuradorIR()
        a.apurar_mes(2026, 1, [venda(Regime.ACAO_INDIVIDUAL, 9_000, 10_000, mes=1)])
        assert a.prejuizo.get("renda_variavel", 0.0) == 0.0


class TestApuracaoMensal:
    def test_venda_de_outro_mes_e_ignorada(self):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.ETF_RENDA_VARIAVEL, 12_000, 10_000, mes=4)])
        assert r.imposto_devido == 0.0

    def test_mes_sem_venda_nao_gera_imposto(self):
        assert ApuradorIR().apurar_mes(2026, 3, []).imposto_devido == 0.0

    def test_detalhe_explica_cada_linha(self):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.ETF_RENDA_VARIAVEL, 12_000, 10_000)])
        assert r.detalhe
        assert all("motivo" in d for d in r.detalhe)

    def test_regime_isento_nunca_paga(self):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.ISENTO, 50_000, 10_000)])
        assert r.imposto_devido == 0.0


class TestRendaFixa:
    @pytest.mark.parametrize("dias,aliquota", [(100, 0.225), (300, 0.20), (500, 0.175), (900, 0.15)])
    def test_etf_de_renda_fixa_usa_a_regressiva(self, dias, aliquota):
        a = ApuradorIR()
        r = a.apurar_mes(2026, 3, [venda(Regime.ETF_RENDA_FIXA, 11_000, 10_000, dias=dias)])
        assert r.imposto_devido == pytest.approx(1_000 * aliquota)

    def test_carregar_mais_tempo_paga_menos(self):
        curto = ApuradorIR().apurar_mes(
            2026, 3, [venda(Regime.TESOURO, 11_000, 10_000, dias=100)]
        )
        longo = ApuradorIR().apurar_mes(
            2026, 3, [venda(Regime.TESOURO, 11_000, 10_000, dias=900)]
        )
        assert longo.imposto_devido < curto.imposto_devido


class TestEquivalenteBruto:
    def test_isento_exige_menos_que_tributado(self):
        assert equivalente_bruto(10.0, dias=365) > 10.0

    @pytest.mark.parametrize("dias", [100, 300, 500, 900])
    def test_prazo_maior_reduz_o_bruto_necessario(self, dias):
        assert equivalente_bruto(10.0, dias) >= equivalente_bruto(10.0, 10_000)

    def test_fii_a_nove_e_meio_equivale_a_cdb_acima_de_onze(self):
        assert 11.0 < equivalente_bruto(9.5, dias=365) < 12.0


class TestCustoFiscalDoGiro:
    def test_sem_giro_nao_ha_imposto_no_ano(self):
        r = custo_fiscal_do_giro(12.0, 10_000, 0, Regime.ETF_RENDA_FIXA)
        assert r["imposto_girando"] == 0.0
        assert r["custo_extra_do_giro"] == 0.0

    def test_girar_mais_encarece_em_renda_fixa(self):
        pouco = custo_fiscal_do_giro(12.0, 10_000, 1, Regime.ETF_RENDA_FIXA)
        muito = custo_fiscal_do_giro(12.0, 10_000, 12, Regime.ETF_RENDA_FIXA)
        assert muito["imposto_girando"] > pouco["imposto_girando"]

    def test_em_renda_variavel_a_aliquota_nao_muda_com_o_giro(self):
        pouco = custo_fiscal_do_giro(12.0, 10_000, 1, Regime.ETF_RENDA_VARIAVEL)
        muito = custo_fiscal_do_giro(12.0, 10_000, 12, Regime.ETF_RENDA_VARIAVEL)
        assert pouco["aliquota_girando"] == muito["aliquota_girando"]

    def test_giros_negativos_sao_recusados(self):
        with pytest.raises(ValueError, match="negativo"):
            custo_fiscal_do_giro(12.0, 10_000, -1)

    @pytest.mark.parametrize("giros", [0, 1, 2, 4, 12, 52, 252])
    def test_nunca_devolve_imposto_negativo(self, giros):
        r = custo_fiscal_do_giro(12.0, 10_000, giros)
        assert r["imposto_girando"] >= 0


class TestModalidade:
    @pytest.mark.parametrize(
        "regime,esperada",
        [
            (Regime.FII, "fii"),
            (Regime.CRIPTO, "cripto"),
            (Regime.ETF_RENDA_FIXA, "renda_fixa"),
            (Regime.TESOURO, "renda_fixa"),
            (Regime.ETF_RENDA_VARIAVEL, "renda_variavel"),
            (Regime.ACAO_INDIVIDUAL, "renda_variavel"),
        ],
    )
    def test_classificacao(self, regime, esperada):
        assert ApuradorIR.modalidade(venda(regime, 1_000, 900)) == esperada

    def test_day_trade_e_modalidade_propria(self):
        v = venda(Regime.ACAO_INDIVIDUAL, 1_000, 900, day_trade=True)
        assert ApuradorIR.modalidade(v) == "day_trade"


class TestVenda:
    def test_ganho_e_a_diferenca(self):
        assert venda(Regime.FII, 1_500, 1_000).ganho == 500

    def test_ganho_negativo_em_prejuizo(self):
        assert venda(Regime.FII, 800, 1_000).ganho == -200
