"""
Testes do teto de risco pelo juro real.

Esta é a guarda mais forte do sistema e a única que se mostrou capaz de decidir
o resultado, então é também a que mais precisa de trava. Os testes protegem
três coisas:

* o teto **nunca aumenta** o risco de ninguém — só reduz
* a conta usa Fisher e não subtração, porque a diferença muda a decisão perto
  do limite
* o retorno esperado de bolsa é constante de literatura e não parâmetro
  ajustável, porque ajustá-lo contra o próprio histórico transformaria a regra
  em profecia auto-realizável
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from carteira import juro_real as jr

SEM_SAUDE = settings(
    max_examples=100, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


# ─────────────────────────────────────────────────────────────────────────────
# Fisher
# ─────────────────────────────────────────────────────────────────────────────

class TesteFisher:
    def test_sem_inflacao_o_real_e_o_nominal(self):
        assert jr.fisher(0.10, 0.0) == pytest.approx(0.10)

    def test_inflacao_igual_ao_juro_da_real_zero(self):
        assert jr.fisher(0.10, 0.10) == pytest.approx(0.0)

    def test_e_menor_que_a_subtracao(self):
        """Com CDI de 23% e IPCA de 9%, subtrair diz 14% e a conta certa diz 12,8%."""
        assert jr.fisher(0.23, 0.09) < 0.23 - 0.09
        assert jr.fisher(0.23, 0.09) == pytest.approx(0.1284, abs=0.001)

    def test_inflacao_acima_do_juro_da_real_negativo(self):
        assert jr.fisher(0.05, 0.10) < 0

    def test_inflacao_impossivel_e_recusada(self):
        with pytest.raises(ValueError, match="implausível"):
            jr.fisher(0.10, -1.5)

    @SEM_SAUDE
    @given(nominal=st.floats(-0.2, 0.5), inflacao=st.floats(-0.1, 0.4))
    def test_real_nunca_passa_do_nominal_com_inflacao_positiva(self, nominal, inflacao):
        if inflacao > 0:
            assert jr.fisher(nominal, inflacao) <= nominal + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Imposto
# ─────────────────────────────────────────────────────────────────────────────

class TesteImposto:
    def test_prazo_longo_cobra_quinze(self):
        assert jr.liquido_de_ir(0.10, 800) == pytest.approx(0.085)

    def test_prazo_curto_cobra_mais(self):
        assert jr.liquido_de_ir(0.10, 100) < jr.liquido_de_ir(0.10, 800)

    def test_juro_negativo_nao_e_tributado(self):
        assert jr.liquido_de_ir(-0.02, 800) == pytest.approx(-0.02)

    def test_juro_zero(self):
        assert jr.liquido_de_ir(0.0, 800) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Leitura
# ─────────────────────────────────────────────────────────────────────────────

class TesteLeitura:
    def test_juro_baixo_libera_risco(self):
        """2020: CDI 2,77%, IPCA 4,52%. Juro real negativo, bolsa é o caminho."""
        leitura = jr.ler(0.0277, 0.0452)
        assert leitura.juro_real_liquido < 0
        assert leitura.teto_de_risco_sugerido == 100.0

    def test_juro_altissimo_derruba_o_teto(self):
        """2003: CDI 23,15%, IPCA ~9,3%. Renda fixa pagando mais que bolsa."""
        leitura = jr.ler(0.2315, 0.093)
        assert leitura.premio_da_bolsa < 0
        assert leitura.teto_de_risco_sugerido == 20.0

    def test_hoje_esta_no_regime_de_juro_alto(self):
        """CDI 14,5% com IPCA 4,4% dá prêmio negativo. É o cenário de agosto de 2026."""
        leitura = jr.ler(0.145, 0.0444)
        assert leitura.premio_da_bolsa < 0
        assert leitura.teto_de_risco_sugerido == 20.0

    def test_teto_nunca_passa_de_cem(self):
        assert jr.ler(-0.02, 0.10).teto_de_risco_sugerido <= 100.0

    def test_teto_nunca_fica_abaixo_de_vinte(self):
        assert jr.ler(0.50, 0.02).teto_de_risco_sugerido >= 20.0

    @pytest.mark.parametrize("cdi", [0.02, 0.05, 0.08, 0.11, 0.14, 0.18, 0.23])
    def test_teto_nunca_sobe_quando_o_juro_sobe(self, cdi):
        anterior = jr.ler(cdi - 0.01, 0.045).teto_de_risco_sugerido
        atual = jr.ler(cdi, 0.045).teto_de_risco_sugerido
        assert atual <= anterior

    @pytest.mark.parametrize("ipca", [0.02, 0.04, 0.06, 0.09, 0.12])
    def test_inflacao_maior_libera_mais_risco(self, ipca):
        """Inflação alta corrói a renda fixa e aumenta o prêmio relativo da bolsa."""
        baixa = jr.ler(0.14, ipca)
        alta = jr.ler(0.14, ipca + 0.02)
        assert alta.teto_de_risco_sugerido >= baixa.teto_de_risco_sugerido

    def test_veredito_explica_o_numero(self):
        for cdi, ipca in [(0.03, 0.05), (0.09, 0.045), (0.14, 0.044), (0.23, 0.09)]:
            leitura = jr.ler(cdi, ipca)
            assert len(leitura.veredito) > 40
            assert "%" in leitura.veredito

    def test_resumo_e_serializavel(self):
        import json

        json.dumps(jr.ler(0.14, 0.044).resumo())

    @SEM_SAUDE
    @given(cdi=st.floats(-0.02, 0.40), ipca=st.floats(-0.02, 0.25))
    def test_teto_sempre_num_dos_quatro_niveis(self, cdi, ipca):
        assert jr.ler(cdi, ipca).teto_de_risco_sugerido in (20.0, 40.0, 70.0, 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Aplicação do teto
# ─────────────────────────────────────────────────────────────────────────────

class TesteAplicarTeto:
    def test_nunca_aumenta_o_risco(self):
        """A regra reduz. Se algum dia aumentar, virou aposta em vez de guarda."""
        leitura = jr.ler(0.02, 0.06)  # juro real negativo, teto 100
        risco, aviso = jr.aplicar_teto(30.0, leitura)
        assert risco == 30.0
        assert aviso is None

    def test_reduz_quando_o_pedido_passa_do_teto(self):
        leitura = jr.ler(0.23, 0.09)
        risco, aviso = jr.aplicar_teto(100.0, leitura)
        assert risco == 20.0
        assert aviso and "reduzido" in aviso

    def test_pedido_igual_ao_teto_nao_gera_aviso(self):
        leitura = jr.ler(0.23, 0.09)
        risco, aviso = jr.aplicar_teto(20.0, leitura)
        assert risco == 20.0
        assert aviso is None

    @SEM_SAUDE
    @given(
        pedido=st.floats(0, 100),
        cdi=st.floats(-0.02, 0.35),
        ipca=st.floats(-0.02, 0.20),
    )
    def test_resultado_nunca_maior_que_o_pedido(self, pedido, cdi, ipca):
        risco, _ = jr.aplicar_teto(pedido, jr.ler(cdi, ipca))
        assert risco <= pedido + 1e-9
        assert risco >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Constantes que não podem virar parâmetro ajustado
# ─────────────────────────────────────────────────────────────────────────────

class TesteConstantes:
    def test_retorno_de_bolsa_esta_na_faixa_da_literatura(self):
        """Se alguém ajustar isto contra o histórico, a regra vira profecia."""
        assert 0.04 <= jr.RETORNO_REAL_DE_BOLSA <= 0.09

    def test_premio_minimo_e_modesto(self):
        assert 0.005 <= jr.PREMIO_MINIMO <= 0.04

    def test_prazo_longo_alcanca_a_menor_aliquota(self):
        from carteira.tributos import aliquota_regressiva

        assert aliquota_regressiva(jr.PRAZO_LONGO_DIAS) == pytest.approx(0.15)


# ─────────────────────────────────────────────────────────────────────────────
# Série histórica
# ─────────────────────────────────────────────────────────────────────────────

def cdi_e_ipca(dias=1500, cdi_dia=0.0005, ipca_mes=0.4):
    idx = pd.bdate_range("2015-01-01", periods=dias)
    cdi = pd.Series(np.full(dias, cdi_dia), index=idx)
    meses = pd.date_range("2013-01-01", periods=dias // 21 + 30, freq="MS")
    ipca = pd.Series(np.full(len(meses), ipca_mes), index=meses)
    return cdi, ipca


class TesteSerieHistorica:
    def test_produz_as_colunas_esperadas(self):
        cdi, ipca = cdi_e_ipca()
        df = jr.serie_historica(cdi, ipca)
        assert {"cdi_anual", "ipca_12m", "juro_real",
                "juro_real_liquido", "premio_da_bolsa", "teto_de_risco"} <= set(df.columns)

    def test_nao_comeca_antes_de_ter_janela_cheia(self):
        cdi, ipca = cdi_e_ipca(1500)
        df = jr.serie_historica(cdi, ipca, janela_dias=252)
        assert len(df) <= 1500 - 252

    def test_serie_curta_e_recusada(self):
        cdi, ipca = cdi_e_ipca(100)
        with pytest.raises(ValueError, match="curto demais"):
            jr.serie_historica(cdi, ipca)

    def test_juro_real_liquido_fica_abaixo_do_bruto(self):
        cdi, ipca = cdi_e_ipca()
        df = jr.serie_historica(cdi, ipca)
        assert (df["juro_real_liquido"] <= df["juro_real"] + 1e-12).all()

    def test_teto_so_assume_os_quatro_niveis(self):
        cdi, ipca = cdi_e_ipca()
        df = jr.serie_historica(cdi, ipca)
        assert set(df["teto_de_risco"].unique()) <= {20.0, 40.0, 70.0, 100.0}

    def test_cdi_maior_derruba_o_teto(self):
        baixo = jr.serie_historica(*cdi_e_ipca(cdi_dia=0.0001))
        alto = jr.serie_historica(*cdi_e_ipca(cdi_dia=0.0007))
        assert alto["teto_de_risco"].median() <= baixo["teto_de_risco"].median()

    def test_a_serie_olha_so_para_tras(self):
        """Nenhum valor pode depender de dado posterior à própria data."""
        cdi, ipca = cdi_e_ipca(1500)
        completa = jr.serie_historica(cdi, ipca)
        corte = completa.index[len(completa) // 2]
        parcial = jr.serie_historica(cdi.loc[:corte], ipca.loc[:corte])
        comum = parcial.index.intersection(completa.index)
        assert np.allclose(
            parcial.loc[comum, "juro_real"].to_numpy(),
            completa.loc[comum, "juro_real"].to_numpy(),
        )
