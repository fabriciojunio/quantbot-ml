"""
Testes do backtest de carteira.

Os mais importantes são os de conservação (o dinheiro tem que fechar), os de
custo (toda operação paga) e os de ausência de olhar-o-futuro. Um backtest de
carteira errado não quebra: ele só devolve um número bom demais.
"""

import numpy as np
import pandas as pd
import pytest

from carteira import alocacao, backtest
from carteira.custos_b3 import ModeloCustoB3
from carteira.rebalanceamento import ConfigRebalanceamento, Politica
from carteira.universo import UNIVERSO_B3

TICKERS = ["BOVA11.SA", "IVVB11.SA", "IMAB11.SA", "GOLD11.SA"]
POLITICAS = [
    Politica.MENSAL, Politica.TRIMESTRAL, Politica.SEMESTRAL,
    Politica.ANUAL, Politica.BANDA, Politica.BANDA_COM_CUSTO,
]


def _idx(n):
    return pd.bdate_range("2020-01-01", periods=n)


def precos(n=756, semente=13, deriva=0.0004):
    rng = np.random.default_rng(semente)
    return pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(deriva, 0.011, n))) for t in TICKERS},
        index=_idx(n),
    )


def parados(n=756, preco=100.0):
    return pd.DataFrame({t: np.full(n, preco) for t in TICKERS}, index=_idx(n))


def cdi(n=756, taxa=0.0005):
    return pd.Series(taxa, index=_idx(n))


def rodar(px=None, alvo=None, **kw):
    px = precos() if px is None else px
    alvo = alvo or {"BOVA11.SA": 0.25, "IVVB11.SA": 0.25, "IMAB11.SA": 0.25, "GOLD11.SA": 0.25}
    pesos = alocacao.pesos_fixos(px, alvo)
    cfg = backtest.ConfigBacktest(**kw)
    return backtest.executar(px, pesos, cfg, cdi=cdi(len(px)), ativos=UNIVERSO_B3)


class TestConservacao:
    def test_patrimonio_comeca_no_capital_inicial(self):
        r = rodar(capital_inicial=10_000)
        assert float(r.patrimonio.iloc[0]) == pytest.approx(10_000, rel=0.02)

    def test_patrimonio_nunca_fica_negativo(self):
        r = rodar()
        assert (r.patrimonio > 0).all()

    def test_serie_tem_uma_linha_por_pregao(self):
        px = precos()
        assert len(rodar(px).patrimonio) == len(px)

    def test_aportes_entram_no_capital_investido(self):
        r = rodar(capital_inicial=10_000, aporte_mensal=500)
        assert r.aportes_totais > 0
        assert r.capital_investido == pytest.approx(10_000 + r.aportes_totais)

    def test_sem_aporte_o_capital_investido_e_so_o_inicial(self):
        r = rodar(capital_inicial=10_000, aporte_mensal=0)
        assert r.capital_investido == 10_000

    def test_retorno_e_medido_sobre_o_investido_e_nao_sobre_o_inicial(self):
        """Com aporte, dividir pelo capital inicial infla o retorno."""
        r = rodar(capital_inicial=1_000, aporte_mensal=1_000)
        ingenuo = (float(r.patrimonio.iloc[-1]) / 1_000 - 1) * 100
        assert r.retorno_total_pct < ingenuo


class TestCusto:
    def test_toda_operacao_paga(self):
        r = rodar()
        assert r.operacoes
        assert all(o["custo"] >= 0 for o in r.operacoes)
        assert r.custo_total > 0

    def test_girar_mais_custa_mais(self):
        mensal = rodar(config_rebal=ConfigRebalanceamento(politica=Politica.MENSAL))
        anual = rodar(config_rebal=ConfigRebalanceamento(politica=Politica.ANUAL))
        assert mensal.custo_total > anual.custo_total

    def test_corretagem_aumenta_o_custo(self):
        sem = rodar(modelo_custo=ModeloCustoB3(corretagem_brl=0.0))
        com = rodar(modelo_custo=ModeloCustoB3(corretagem_brl=5.0))
        assert com.custo_total > sem.custo_total

    def test_taxa_de_administracao_reduz_o_patrimonio(self):
        com = rodar(cobrar_taxa_adm=True)
        sem = rodar(cobrar_taxa_adm=False)
        assert float(com.patrimonio.iloc[-1]) < float(sem.patrimonio.iloc[-1])

    def test_mercado_parado_so_perde_custo(self):
        r = rodar(px=parados(), caixa_rende_cdi=False)
        assert float(r.patrimonio.iloc[-1]) <= r.config.capital_inicial

    def test_atrito_e_custo_mais_imposto(self):
        r = rodar()
        resumo = r.resumo()
        assert resumo["atrito_total"] == pytest.approx(
            resumo["custo_total"] + resumo["imposto_total"], abs=0.01
        )


class TestImposto:
    def test_apuracao_registra_os_meses(self):
        r = rodar(config_rebal=ConfigRebalanceamento(politica=Politica.MENSAL))
        assert isinstance(r.imposto_por_mes, list)

    def test_desligar_imposto_zera_a_apuracao(self):
        r = rodar(cobrar_imposto=False)
        assert r.imposto_por_mes == []
        assert r.imposto_total == 0.0

    def test_imposto_nunca_e_negativo(self):
        r = rodar(config_rebal=ConfigRebalanceamento(politica=Politica.MENSAL))
        assert all(m["imposto_devido"] >= 0 for m in r.imposto_por_mes)

    def test_com_imposto_o_patrimonio_e_menor_ou_igual(self):
        com = rodar(cobrar_imposto=True)
        sem = rodar(cobrar_imposto=False)
        assert float(com.patrimonio.iloc[-1]) <= float(sem.patrimonio.iloc[-1]) + 1e-6


class TestCaixa:
    def test_caixa_rende_cdi(self):
        alvo = {"BOVA11.SA": 0.10}  # 90% em caixa
        com = rodar(px=parados(), alvo=alvo, caixa_rende_cdi=True)
        sem = rodar(px=parados(), alvo=alvo, caixa_rende_cdi=False)
        assert float(com.patrimonio.iloc[-1]) > float(sem.patrimonio.iloc[-1])

    def test_carteira_toda_em_caixa_acompanha_o_cdi(self):
        px = parados()
        pesos = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        cfg = backtest.ConfigBacktest(capital_inicial=10_000)
        r = backtest.executar(px, pesos, cfg, cdi=cdi(len(px)), ativos=UNIVERSO_B3)
        esperado = 10_000 * (1.0005 ** len(px))
        assert float(r.patrimonio.iloc[-1]) == pytest.approx(esperado, rel=0.01)


class TestMetricas:
    def test_drawdown_nunca_e_positivo(self):
        assert rodar().max_drawdown_pct <= 0.0

    def test_volatilidade_nunca_e_negativa(self):
        assert rodar().vol_anual_pct >= 0.0

    def test_sharpe_e_finito(self):
        assert np.isfinite(rodar().sharpe)

    def test_cagr_e_finito(self):
        assert np.isfinite(rodar().cagr_pct)

    def test_resumo_traz_os_campos_esperados(self):
        r = rodar().resumo()
        for campo in (
            "patrimonio_final", "retorno_sobre_investido_pct", "cagr_pct",
            "max_drawdown_pct", "sharpe", "custo_total", "imposto_total",
        ):
            assert campo in r

    def test_comparacao_gera_alpha(self):
        px = precos()
        r = rodar(px)
        base = pd.Series(
            10_000 * np.cumprod(np.full(len(px), 1.0005)), index=px.index
        )
        resumo = r.resumo({"cdi": base})
        assert "vs_cdi_pct" in resumo
        assert "alpha_vs_cdi_pp" in resumo

    def test_alpha_e_a_diferenca(self):
        px = precos()
        r = rodar(px)
        base = pd.Series(
            10_000 * np.cumprod(np.full(len(px), 1.0005)), index=px.index
        )
        resumo = r.resumo({"cdi": base})
        assert resumo["alpha_vs_cdi_pp"] == pytest.approx(
            resumo["retorno_sobre_investido_pct"] - resumo["vs_cdi_pct"], abs=0.01
        )


class TestPoliticas:
    @pytest.mark.parametrize("politica", POLITICAS)
    def test_toda_politica_roda(self, politica):
        r = rodar(config_rebal=ConfigRebalanceamento(politica=politica))
        assert len(r.patrimonio) > 0

    @pytest.mark.parametrize("politica", POLITICAS)
    def test_pesos_reais_somam_no_maximo_um(self, politica):
        r = rodar(config_rebal=ConfigRebalanceamento(politica=politica))
        assert (r.pesos.sum(axis=1) <= 1.0001).all()

    @pytest.mark.parametrize("politica", POLITICAS)
    def test_nenhum_peso_negativo(self, politica):
        r = rodar(config_rebal=ConfigRebalanceamento(politica=politica))
        assert (r.pesos >= -1e-9).all().all()


class TestSemOlharOFuturo:
    def test_truncar_a_serie_nao_muda_o_passado(self):
        completa = precos(n=756)
        curta = completa.iloc[:500]
        a = rodar(completa)
        b = rodar(curta)
        np.testing.assert_allclose(
            a.patrimonio.iloc[:500].to_numpy(),
            b.patrimonio.to_numpy(),
            rtol=1e-9,
        )

    def test_operacoes_do_passado_sao_identicas(self):
        completa = rodar(precos(n=756))
        curta = rodar(precos(n=756).iloc[:500])
        corte = str(precos(n=756).index[499])[:10]
        a = [o for o in completa.operacoes if o["quando"] <= corte]
        b = [o for o in curta.operacoes if o["quando"] <= corte]
        assert len(a) == len(b)


class TestDeterminismo:
    def test_rodar_duas_vezes_da_o_mesmo(self):
        a, b = rodar(), rodar()
        assert float(a.patrimonio.iloc[-1]) == pytest.approx(float(b.patrimonio.iloc[-1]))

    def test_nao_altera_a_serie_de_precos(self):
        px = precos()
        antes = px.copy()
        rodar(px)
        pd.testing.assert_frame_equal(px, antes)


class TestCasosLimite:
    def test_capital_muito_pequeno_nao_quebra(self):
        r = rodar(capital_inicial=50.0)
        assert (r.patrimonio > 0).all()

    def test_carteira_de_um_ativo_so(self):
        r = rodar(alvo={"BOVA11.SA": 1.0})
        assert float(r.patrimonio.iloc[-1]) > 0

    def test_serie_curta_nao_quebra(self):
        r = rodar(px=precos(n=60))
        assert len(r.patrimonio) == 60

    def test_aporte_alto_domina_o_patrimonio(self):
        r = rodar(capital_inicial=100, aporte_mensal=1_000)
        assert r.aportes_totais > r.config.capital_inicial * 10
