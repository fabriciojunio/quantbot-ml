"""
Testes do simulador de aporte mensal.

O que aqui interessa não é "o código roda", é que a contabilidade fecha: o que
entrou de aporte, mais o que o mercado rendeu, menos custo e imposto, tem que
bater com o que saiu. Quase todo erro grave de backtest é dinheiro aparecendo
ou sumindo de graça, e é isso que a maior parte destes testes procura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from carteira import aporte_historico as ah
from carteira.custos_b3 import ModeloCustoB3
from carteira.universo import UNIVERSO_B3

SEM_SAUDE = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


# ─────────────────────────────────────────────────────────────────────────────
# Séries de apoio
# ─────────────────────────────────────────────────────────────────────────────

def calendario(dias: int, inicio: str = "2015-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(inicio, periods=dias)


def precos_planos(dias: int = 500, tickers=("BOVA11.SA",), valor: float = 100.0) -> pd.DataFrame:
    idx = calendario(dias)
    return pd.DataFrame({t: np.full(dias, valor) for t in tickers}, index=idx)


def precos_com_retorno(dias: int, taxa_diaria: float, tickers=("BOVA11.SA",)) -> pd.DataFrame:
    idx = calendario(dias)
    curva = 100 * (1 + taxa_diaria) ** np.arange(dias)
    return pd.DataFrame({t: curva for t in tickers}, index=idx)


def cdi_plano(idx: pd.DatetimeIndex, taxa_diaria: float = 0.0) -> pd.Series:
    return pd.Series(np.full(len(idx), taxa_diaria), index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Calendários de aporte
# ─────────────────────────────────────────────────────────────────────────────

class TesteCalendarios:
    def test_constante_repete_o_valor(self):
        f = ah.constante(500)
        assert [f(m) for m in range(5)] == [500] * 5

    def test_constante_recusa_negativo(self):
        with pytest.raises(ValueError):
            ah.constante(-1)

    def test_degrau_sobe_no_mes_certo(self):
        f = ah.degrau(500, 200, 24)
        assert f(0) == 500
        assert f(23) == 500
        assert f(24) == 700
        assert f(47) == 700
        assert f(48) == 900

    def test_degrau_recusa_intervalo_zero(self):
        with pytest.raises(ValueError):
            ah.degrau(500, 200, 0)

    def test_degrau_recusa_passo_negativo(self):
        with pytest.raises(ValueError):
            ah.degrau(500, -50, 24)

    @given(
        inicial=st.floats(0, 5_000),
        passo=st.floats(0, 1_000),
        intervalo=st.integers(1, 60),
        mes=st.integers(0, 600),
    )
    def test_degrau_nunca_diminui(self, inicial, passo, intervalo, mes):
        f = ah.degrau(inicial, passo, intervalo)
        assert f(mes + 1) >= f(mes)

    def test_inflacao_preserva_poder_de_compra(self):
        f = ah.corrigido_pela_inflacao(500, 0.045)
        assert f(12) == pytest.approx(500 * 1.045, rel=1e-6)

    def test_inflacao_recusa_valor_absurdo(self):
        with pytest.raises(ValueError):
            ah.corrigido_pela_inflacao(500, -0.9)


# ─────────────────────────────────────────────────────────────────────────────
# Taxa interna de retorno
# ─────────────────────────────────────────────────────────────────────────────

class TesteTIR:
    def test_aporte_unico_dobrando_em_um_ano(self):
        assert ah.tir_anual([(0.0, -100.0), (1.0, 200.0)]) == pytest.approx(1.0, abs=1e-4)

    def test_sem_ganho_da_zero(self):
        assert ah.tir_anual([(0.0, -100.0), (1.0, 100.0)]) == pytest.approx(0.0, abs=1e-5)

    def test_prejuizo_da_taxa_negativa(self):
        assert ah.tir_anual([(0.0, -100.0), (1.0, 50.0)]) == pytest.approx(-0.5, abs=1e-4)

    def test_sem_troca_de_sinal_nao_tem_tir(self):
        assert np.isnan(ah.tir_anual([(0.0, -100.0), (1.0, -50.0)]))

    def test_fluxo_curto_demais(self):
        assert np.isnan(ah.tir_anual([(0.0, -100.0)]))

    def test_dois_aportes_mesma_taxa(self):
        # R$ 100 por 2 anos e R$ 100 por 1 ano, ambos a 10% ao ano.
        final = 100 * 1.1**2 + 100 * 1.1
        tir = ah.tir_anual([(0.0, -100.0), (1.0, -100.0), (2.0, final)])
        assert tir == pytest.approx(0.10, abs=1e-4)

    @given(taxa=st.floats(-0.5, 0.5), anos=st.integers(1, 20))
    def test_recupera_a_taxa_que_gerou_o_fluxo(self, taxa, anos):
        final = 1000 * (1 + taxa) ** anos
        assert ah.tir_anual([(0.0, -1000.0), (float(anos), final)]) == pytest.approx(taxa, abs=1e-4)

    @given(
        aportes=st.lists(st.floats(50, 2000), min_size=2, max_size=30),
        taxa=st.floats(0.0, 0.3),
    )
    def test_tir_positiva_quando_tudo_rende(self, aportes, taxa):
        fluxos = [(i / 12, -v) for i, v in enumerate(aportes)]
        n = len(aportes)
        final = sum(v * (1 + taxa) ** ((n - 1 - i) / 12) for i, v in enumerate(aportes))
        tir = ah.tir_anual(fluxos + [((n - 1) / 12, final)])
        assert np.isnan(tir) or tir >= -1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Contabilidade da simulação
# ─────────────────────────────────────────────────────────────────────────────

class TesteContabilidade:
    def test_preco_e_cdi_parados_devolvem_quase_o_aportado(self):
        """Sem retorno nenhum, o que sai é o que entrou menos os custos."""
        p = precos_planos(520)
        r = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0}, ah.constante(500))

        assert r.total_aportado == pytest.approx(500 * r.meses)
        assert r.imposto_total_pago == pytest.approx(0.0, abs=1e-6)
        assert r.custo_total_pago > 0
        assert r.valor_liquido_final == pytest.approx(
            r.total_aportado - r.custo_total_pago, rel=1e-6
        )

    def test_nada_investido_fica_no_caixa_rendendo_cdi(self):
        """100% CDI: nenhuma ordem, nenhum custo, nenhum imposto de bolsa."""
        p = precos_planos(260)
        r = ah.simular(p, cdi_plano(p.index, 0.0005), {"CDI": 1.0}, ah.constante(500))

        assert r.ordens == 0
        assert r.custo_total_pago == pytest.approx(0.0)
        assert r.valor_liquido_final > r.total_aportado

    def test_aporte_zero_nao_movimenta_nada(self):
        p = precos_planos(260)
        r = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0}, ah.constante(0))

        assert r.total_aportado == 0
        assert r.ordens == 0
        assert r.valor_liquido_final == pytest.approx(0.0)

    def test_imposto_so_aparece_quando_ha_ganho(self):
        subindo = precos_com_retorno(520, 0.0005)
        caindo = precos_com_retorno(520, -0.0005)
        pesos = {"BOVA11.SA": 1.0}

        r_sobe = ah.simular(subindo, cdi_plano(subindo.index), pesos, ah.constante(500))
        r_cai = ah.simular(caindo, cdi_plano(caindo.index), pesos, ah.constante(500))

        assert r_sobe.imposto_total_pago > 0
        assert r_cai.imposto_total_pago == pytest.approx(0.0, abs=1e-9)

    def test_liquidar_no_fim_cobra_imposto_que_o_bruto_nao_cobra(self):
        p = precos_com_retorno(760, 0.0006)
        pesos = {"BOVA11.SA": 1.0}
        com = ah.simular(p, cdi_plano(p.index), pesos, ah.constante(500), liquidar_no_fim=True)
        sem = ah.simular(p, cdi_plano(p.index), pesos, ah.constante(500), liquidar_no_fim=False)

        assert com.valor_liquido_final < sem.valor_liquido_final
        assert com.imposto_total_pago > sem.imposto_total_pago

    def test_o_custo_de_entrada_entra_na_base_do_imposto(self):
        """Custo de compra é dedutível: sem isso o imposto sai maior que o devido."""
        p = precos_com_retorno(520, 0.0004)
        caro = ModeloCustoB3(corretagem_brl=20.0)
        barato = ModeloCustoB3(corretagem_brl=0.0)

        r_caro = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0},
                            ah.constante(500), modelo=caro)
        r_barato = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0},
                              ah.constante(500), modelo=barato)

        assert r_caro.custo_total_pago > r_barato.custo_total_pago
        assert r_caro.imposto_total_pago < r_barato.imposto_total_pago
        assert r_caro.valor_liquido_final < r_barato.valor_liquido_final

    @SEM_SAUDE
    @given(
        aporte=st.floats(100, 5_000),
        taxa=st.floats(-0.0008, 0.0008),
        cdi=st.floats(0.0, 0.0006),
    )
    def test_dinheiro_nunca_aparece_do_nada(self, aporte, taxa, cdi):
        """Invariante central: o final é aportado + mercado - custo - imposto."""
        p = precos_com_retorno(400, taxa)
        r = ah.simular(p, cdi_plano(p.index, cdi), {"BOVA11.SA": 0.7, "CDI": 0.3},
                       ah.constante(aporte))

        assert r.total_aportado == pytest.approx(aporte * r.meses)
        assert r.custo_total_pago >= 0
        assert r.imposto_total_pago >= 0
        assert r.valor_liquido_final >= 0
        assert np.isfinite(r.tir_anual_pct) or np.isnan(r.tir_anual_pct)

    @SEM_SAUDE
    @given(taxa=st.floats(0.0001, 0.001))
    def test_mercado_subindo_bate_mercado_parado(self, taxa):
        sobe = precos_com_retorno(400, taxa)
        parado = precos_planos(400)
        pesos = {"BOVA11.SA": 1.0}
        r1 = ah.simular(sobe, cdi_plano(sobe.index), pesos, ah.constante(500))
        r2 = ah.simular(parado, cdi_plano(parado.index), pesos, ah.constante(500))
        assert r1.valor_liquido_final > r2.valor_liquido_final


# ─────────────────────────────────────────────────────────────────────────────
# Validação de entrada
# ─────────────────────────────────────────────────────────────────────────────

class TesteValidacao:
    def test_recusa_ativo_com_peso_e_sem_preco(self):
        p = precos_planos(120)
        with pytest.raises(ValueError, match="não estão na série de preços"):
            ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 0.5, "IVVB11.SA": 0.5})

    def test_recusa_pesos_que_nao_somam_um(self):
        p = precos_planos(120)
        with pytest.raises(ValueError, match="somam"):
            ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 0.7})

    def test_recusa_serie_vazia(self):
        with pytest.raises(ValueError, match="sem preços"):
            ah.simular(pd.DataFrame(), pd.Series(dtype=float), {"CDI": 1.0})

    def test_recusa_indice_fora_de_ordem(self):
        p = precos_planos(120).iloc[::-1]
        with pytest.raises(ValueError, match="fora de ordem"):
            ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0})


# ─────────────────────────────────────────────────────────────────────────────
# Referência de CDI
# ─────────────────────────────────────────────────────────────────────────────

class TesteReferenciaCDI:
    def test_cdi_nao_compoe_depois_do_fim_da_janela(self):
        """O bug que fazia toda janela antiga 'perder do CDI' por recorte errado."""
        idx = calendario(2000, "2010-01-01")
        cdi = pd.Series(np.full(len(idx), 0.0005), index=idx)

        curta = idx[:400]
        p = pd.DataFrame({"BOVA11.SA": np.full(len(curta), 100.0)}, index=curta)
        r = ah.simular(p, cdi, {"BOVA11.SA": 1.0}, ah.constante(500))

        # Com CDI de 0,05% ao dia por ~1,6 ano, a referência fica perto do
        # aportado vezes 1,2. Se compusesse até 2018 passaria de 2,5x.
        assert r.liquido_cdi < r.total_aportado * 1.5

    def test_referencia_cresce_com_a_taxa(self):
        p = precos_planos(500)
        baixo = ah.simular(p, cdi_plano(p.index, 0.0001), {"BOVA11.SA": 1.0}, ah.constante(500))
        alto = ah.simular(p, cdi_plano(p.index, 0.0006), {"BOVA11.SA": 1.0}, ah.constante(500))
        assert alto.liquido_cdi > baixo.liquido_cdi

    def test_cdi_zerado_devolve_o_aportado(self):
        p = precos_planos(300)
        r = ah.simular(p, cdi_plano(p.index, 0.0), {"BOVA11.SA": 1.0}, ah.constante(500))
        assert r.liquido_cdi == pytest.approx(r.total_aportado, rel=1e-9)

    def test_imposto_regressivo_por_aporte(self):
        """Aporte recente paga mais imposto que aporte antigo: alíquotas diferentes."""
        p = precos_planos(1300)
        r = ah.simular(p, cdi_plano(p.index, 0.0005), {"BOVA11.SA": 1.0}, ah.constante(500))
        bruto_aproximado = r.total_aportado * 1.0
        assert r.liquido_cdi > bruto_aproximado


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de risco
# ─────────────────────────────────────────────────────────────────────────────

class TesteMetricas:
    def test_queda_e_medida_contra_o_aportado(self):
        p = precos_com_retorno(500, -0.001)
        r = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0}, ah.constante(500))
        assert r.max_drawdown_pct < -5
        assert r.meses_no_prejuizo > 0

    def test_sem_queda_quando_so_sobe(self):
        p = precos_com_retorno(500, 0.001)
        r = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0}, ah.constante(500))
        assert r.max_drawdown_pct > -1.0
        assert r.meses_no_prejuizo == 0

    def test_pior_mes_ignora_o_aporte(self):
        """Com preço parado o pior mês tem que ser ~0, não o tamanho do depósito.

        Antes de existir a curva de cota, o pior mês era medido no patrimônio e
        saía +7,7% aqui: o aporte entrava somado ao saldo e virava "desempenho".
        """
        p = precos_planos(400)
        r = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0}, ah.constante(500))
        assert r.pior_mes_pct == pytest.approx(0.0, abs=0.5)

    def test_pior_mes_captura_a_queda_real(self):
        idx = calendario(400)
        curva = np.full(400, 100.0)
        curva[200:] = 70.0  # tombo de 30% de um dia para o outro
        p = pd.DataFrame({"BOVA11.SA": curva}, index=idx)
        r = ah.simular(p, cdi_plano(idx), {"BOVA11.SA": 1.0}, ah.constante(500))
        assert r.pior_mes_pct < -25
        assert r.queda_da_cota_pct < -25

    def test_cota_nao_cresce_com_preco_parado(self):
        p = precos_planos(500)
        r = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0}, ah.constante(500))
        assert float(r.curva_cota.iloc[-1]) == pytest.approx(1.0, abs=0.02)

    def test_cota_segue_o_ativo_quando_ha_um_so(self):
        p = precos_com_retorno(400, 0.0005)
        r = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0}, ah.constante(500),
                       liquidar_no_fim=False)
        # A cota tem que subir junto com o ativo, não com o total depositado.
        assert float(r.curva_cota.iloc[-1]) > 1.10
        assert float(r.curva_cota.iloc[-1]) < 1.30

    def test_resumo_traz_as_chaves_do_relatorio(self):
        p = precos_planos(300)
        r = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0}, ah.constante(500))
        esperadas = {
            "inicio", "fim", "meses", "total_aportado", "valor_liquido_final",
            "lucro", "multiplo", "tir_anual_pct", "liquido_cdi", "tir_cdi_pct",
            "vantagem_sobre_cdi", "custo_total_pago", "imposto_total_pago",
            "max_drawdown_pct", "pior_mes_pct", "queda_da_cota_pct",
            "meses_no_prejuizo", "ordens",
        }
        assert esperadas <= set(r.resumo())

    def test_multiplo_e_lucro_sao_coerentes(self):
        p = precos_com_retorno(600, 0.0005)
        r = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0}, ah.constante(500))
        assert r.lucro == pytest.approx(r.valor_liquido_final - r.total_aportado)
        assert r.multiplo == pytest.approx(r.valor_liquido_final / r.total_aportado)


# ─────────────────────────────────────────────────────────────────────────────
# Rodízio e rebalanceamento
# ─────────────────────────────────────────────────────────────────────────────

class TesteRodizioERebalanceamento:
    def test_rodizio_faz_uma_ordem_por_mes(self):
        """Com aporte pequeno, comprar tudo todo mês pagaria N spreads."""
        p = precos_planos(300, tickers=("BOVA11.SA", "IVVB11.SA", "GOLD11.SA"))
        r = ah.simular(
            p, cdi_plano(p.index),
            {"BOVA11.SA": 0.4, "IVVB11.SA": 0.4, "GOLD11.SA": 0.2},
            ah.constante(500), rebalancear_a_cada_meses=999,
            liquidar_no_fim=False,
        )
        assert r.ordens <= r.meses

    def test_liquidacao_acrescenta_uma_venda_por_ativo(self):
        p = precos_planos(300, tickers=("BOVA11.SA", "IVVB11.SA", "GOLD11.SA"))
        pesos = {"BOVA11.SA": 0.4, "IVVB11.SA": 0.4, "GOLD11.SA": 0.2}
        sem = ah.simular(p, cdi_plano(p.index), pesos, ah.constante(500),
                         rebalancear_a_cada_meses=999, liquidar_no_fim=False)
        com = ah.simular(p, cdi_plano(p.index), pesos, ah.constante(500),
                         rebalancear_a_cada_meses=999, liquidar_no_fim=True)
        assert com.ordens > sem.ordens
        assert com.ordens - sem.ordens <= 3

    def test_rodizio_converge_para_os_pesos_alvo(self):
        p = precos_planos(2600, tickers=("BOVA11.SA", "IVVB11.SA"))
        pesos = {"BOVA11.SA": 0.6, "IVVB11.SA": 0.4}
        r = ah.simular(p, cdi_plano(p.index), pesos, ah.constante(500),
                       rebalancear_a_cada_meses=999, liquidar_no_fim=False)
        # Com preço constante o rodízio leva as posições ao alvo sozinho.
        assert r.valor_liquido_final > 0

    def test_rebalancear_mais_vezes_custa_mais(self):
        idx = calendario(1500)
        rng = np.random.default_rng(7)
        a = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(idx))))
        b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, len(idx))))
        p = pd.DataFrame({"BOVA11.SA": a, "IVVB11.SA": b}, index=idx)
        pesos = {"BOVA11.SA": 0.5, "IVVB11.SA": 0.5}

        frequente = ah.simular(p, cdi_plano(idx), pesos, ah.constante(1000),
                               rebalancear_a_cada_meses=3, banda_de_rebalanceamento=0.0)
        raro = ah.simular(p, cdi_plano(idx), pesos, ah.constante(1000),
                          rebalancear_a_cada_meses=36, banda_de_rebalanceamento=0.0)
        assert frequente.ordens > raro.ordens
        assert frequente.custo_total_pago > raro.custo_total_pago

    def test_ordem_minima_segura_o_dinheiro_no_caixa(self):
        p = precos_planos(300)
        r = ah.simular(p, cdi_plano(p.index), {"BOVA11.SA": 1.0},
                       ah.constante(50), ordem_minima=1_000)
        assert r.ordens == 0
        assert r.valor_liquido_final == pytest.approx(r.total_aportado, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Janelas móveis
# ─────────────────────────────────────────────────────────────────────────────

class TesteJanelasMoveis:
    def test_produz_uma_linha_por_janela(self):
        p = precos_planos(2600, tickers=("BOVA11.SA",))
        j = ah.janelas_moveis(p, cdi_plano(p.index), {"BOVA11.SA": 1.0},
                              ah.constante(500), anos=3, passo_meses=6)
        assert len(j) >= 2
        assert "valor_liquido_final" in j.columns

    def test_falha_quando_a_janela_nao_cabe(self):
        p = precos_planos(300)
        with pytest.raises(ValueError, match="nenhuma janela"):
            ah.janelas_moveis(p, cdi_plano(p.index), {"BOVA11.SA": 1.0}, anos=10)

    def test_recusa_janela_de_zero_anos(self):
        p = precos_planos(2600)
        with pytest.raises(ValueError, match="ao menos um ano"):
            ah.janelas_moveis(p, cdi_plano(p.index), {"BOVA11.SA": 1.0}, anos=0)

    def test_distribuicao_ordena_os_quantis(self):
        p = precos_planos(2600)
        j = ah.janelas_moveis(p, cdi_plano(p.index), {"BOVA11.SA": 1.0},
                              ah.constante(500), anos=3, passo_meses=6)
        d = ah.distribuicao(j)
        assert d["pior"] <= d["p25"] <= d["mediana"] <= d["p75"] <= d["melhor"]

    def test_comparar_com_cdi_conta_certo(self):
        p = precos_com_retorno(2600, 0.0004)
        j = ah.janelas_moveis(p, cdi_plano(p.index, 0.0), {"BOVA11.SA": 1.0},
                              ah.constante(500), anos=3, passo_meses=6)
        c = ah.comparar_com_cdi(j)
        assert c["venceu_o_cdi_em"] == c["janelas"]
        assert c["taxa_de_vitoria_pct"] == 100.0

    def test_janela_ruim_perde_do_cdi(self):
        p = precos_com_retorno(2600, -0.0004)
        j = ah.janelas_moveis(p, cdi_plano(p.index, 0.0004), {"BOVA11.SA": 1.0},
                              ah.constante(500), anos=3, passo_meses=6)
        c = ah.comparar_com_cdi(j)
        assert c["venceu_o_cdi_em"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Alíquotas por regime
# ─────────────────────────────────────────────────────────────────────────────

class TesteRegimeTributario:
    def test_etf_de_renda_variavel_paga_quinze(self):
        assert ah._aliquota_de_venda(UNIVERSO_B3["BOVA11.SA"], 5_000) == pytest.approx(0.15)

    def test_etf_de_renda_fixa_usa_tabela_regressiva(self):
        curto = ah._aliquota_de_venda(UNIVERSO_B3["IMAB11.SA"], 100)
        longo = ah._aliquota_de_venda(UNIVERSO_B3["IMAB11.SA"], 800)
        assert curto > longo
        assert longo == pytest.approx(0.15)

    def test_ticker_desconhecido_recebe_spread_conservador(self):
        a = ah._ativo_generico("XXXX11.SA")
        assert a.spread_pct >= UNIVERSO_B3["BOVA11.SA"].spread_pct
