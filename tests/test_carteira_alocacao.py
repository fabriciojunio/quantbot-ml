"""
Testes dos motores de alocação.

Invariantes que valem para todo motor, e que os testes cobrem por
parametrização: peso nunca negativo, soma nunca acima do limite, teto por
ativo respeitado, e nenhum motor enxerga o futuro.
"""

import numpy as np
import pandas as pd
import pytest

from carteira import alocacao
from carteira.alocacao import ConfigAlocacao

TICKERS = ["A", "B", "C", "D"]
MOTORES_SIMPLES = [
    alocacao.peso_igual,
    alocacao.inverso_da_volatilidade,
    alocacao.momentum_absoluto_puro,
    alocacao.dual_momentum,
]


def _idx(n):
    return pd.bdate_range("2020-01-01", periods=n)


def precos(n=500, semente=11):
    rng = np.random.default_rng(semente)
    dados = {}
    for i, t in enumerate(TICKERS):
        dados[t] = 100 * np.exp(np.cumsum(rng.normal(0.0003 * (i - 1), 0.012, n)))
    return pd.DataFrame(dados, index=_idx(n))


def rampa(n=500, inclinacoes=(0.5, 0.2, -0.2, -0.5)):
    return pd.DataFrame(
        {t: np.maximum(100 + np.arange(n) * i, 1.0) for t, i in zip(TICKERS, inclinacoes)},
        index=_idx(n),
    )


def cdi_acumulado(px, taxa_diaria=0.0005):
    return pd.Series(np.exp(np.arange(len(px)) * taxa_diaria), index=px.index)


class TestPesosFixos:
    def test_mantem_o_peso_pedido(self):
        p = alocacao.pesos_fixos(precos(), {"A": 0.6, "B": 0.4})
        assert float(p["A"].iloc[-1]) == pytest.approx(0.6)
        assert float(p["B"].iloc[-1]) == pytest.approx(0.4)

    def test_ativo_fora_da_lista_fica_zerado(self):
        p = alocacao.pesos_fixos(precos(), {"A": 1.0})
        assert (p["B"] == 0).all()

    def test_soma_acima_de_um_e_recusada(self):
        with pytest.raises(ValueError, match="não pode passar"):
            alocacao.pesos_fixos(precos(), {"A": 0.7, "B": 0.7})

    def test_soma_zero_e_recusada(self):
        with pytest.raises(ValueError, match="mais que zero"):
            alocacao.pesos_fixos(precos(), {"A": 0.0})

    def test_ticker_inexistente_e_ignorado(self):
        p = alocacao.pesos_fixos(precos(), {"A": 0.5, "ZZZ": 0.5})
        assert "ZZZ" not in p.columns
        assert float(p["A"].iloc[-1]) == pytest.approx(0.5)

    def test_o_peso_nao_muda_ao_longo_do_tempo(self):
        p = alocacao.pesos_fixos(precos(), {"A": 0.5, "B": 0.5})
        assert p["A"].nunique() == 1


class TestPesoIgual:
    def test_divide_igualmente(self):
        p = alocacao.peso_igual(precos())
        assert np.allclose(p.iloc[-1], 0.25)

    def test_soma_um(self):
        p = alocacao.peso_igual(precos())
        assert float(p.iloc[-1].sum()) == pytest.approx(1.0)

    @pytest.mark.parametrize("n_ativos", [1, 2, 3, 4])
    def test_funciona_com_qualquer_numero_de_ativos(self, n_ativos):
        """Sem teto de concentração, cada ativo recebe 1/n."""
        px = precos()[TICKERS[:n_ativos]]
        p = alocacao.peso_igual(px, ConfigAlocacao(min_por_ativo=0.0, max_por_ativo=1.0))
        assert np.allclose(p.iloc[-1], 1.0 / n_ativos)

    @pytest.mark.parametrize("n_ativos", [1, 2])
    def test_teto_de_concentracao_corta_carteira_curta(self, n_ativos):
        """Com poucos ativos o peso igual estoura o teto de 35%, e o teto vence.
        A sobra vira caixa: é o motor recusando concentrar demais."""
        px = precos()[TICKERS[:n_ativos]]
        p = alocacao.peso_igual(px)
        assert (p.iloc[-1] <= 0.35001).all()
        assert float(p.iloc[-1].sum()) < 1.0


class TestInversoDaVolatilidade:
    def test_ativo_menos_volatil_recebe_mais(self):
        rng = np.random.default_rng(2)
        n = 400
        px = pd.DataFrame(
            {
                "calmo": 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))),
                "agitado": 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))),
            },
            index=_idx(n),
        )
        p = alocacao.inverso_da_volatilidade(px).iloc[-1]
        assert p["calmo"] > p["agitado"]

    def test_soma_nao_passa_de_um(self):
        p = alocacao.inverso_da_volatilidade(precos())
        assert (p.sum(axis=1) <= 1.0001).all()

    @pytest.mark.parametrize("dias", [21, 42, 63, 126])
    def test_varias_janelas(self, dias):
        p = alocacao.inverso_da_volatilidade(precos(), dias=dias)
        assert p.shape[1] == 4


class TestMomentumAbsolutoPuro:
    def test_so_fica_no_que_sobe(self):
        p = alocacao.momentum_absoluto_puro(rampa()).iloc[-1]
        assert p["A"] > 0 and p["B"] > 0
        assert p["C"] == 0 and p["D"] == 0

    def test_tudo_caindo_deixa_a_carteira_em_caixa(self):
        px = rampa(inclinacoes=(-0.5, -0.4, -0.3, -0.2))
        p = alocacao.momentum_absoluto_puro(px).iloc[-1]
        assert float(p.sum()) == 0.0

    def test_tudo_subindo_investe_tudo(self):
        px = rampa(inclinacoes=(0.5, 0.4, 0.3, 0.2))
        p = alocacao.momentum_absoluto_puro(px).iloc[-1]
        assert float(p.sum()) == pytest.approx(1.0)

    def test_descontar_cdi_reduz_a_exposicao(self):
        px = rampa(inclinacoes=(0.02, 0.02, 0.02, 0.02))  # alta fraca
        sem = float(alocacao.momentum_absoluto_puro(px).iloc[-1].sum())
        com = float(
            alocacao.momentum_absoluto_puro(px, cdi_acumulado(px, 0.002)).iloc[-1].sum()
        )
        assert com <= sem


class TestDualMomentum:
    @pytest.mark.parametrize("top_n", [1, 2, 3, 4])
    def test_seleciona_no_maximo_top_n(self, top_n):
        p = alocacao.dual_momentum(rampa(inclinacoes=(0.5, 0.4, 0.3, 0.2)), top_n=top_n)
        assert int((p.iloc[-1] > 0).sum()) <= top_n

    def test_top_n_invalido_e_recusado(self):
        with pytest.raises(ValueError, match="pelo menos 1"):
            alocacao.dual_momentum(precos(), top_n=0)

    def test_escolhe_os_melhores(self):
        p = alocacao.dual_momentum(rampa(), top_n=2).iloc[-1]
        assert p["A"] > 0 and p["B"] > 0
        assert p["C"] == 0 and p["D"] == 0

    def test_filtro_absoluto_impede_comprar_o_menos_pior(self):
        """Sem o filtro de tendência, o motor compraria o melhor de um mercado
        que está todo caindo. É assim que estratégia relativa perde dinheiro."""
        px = rampa(inclinacoes=(-0.1, -0.2, -0.3, -0.4))
        p = alocacao.dual_momentum(px, top_n=2).iloc[-1]
        assert float(p.sum()) == 0.0


class TestNucleoComTrend:
    def test_sem_protecao_e_igual_ao_nucleo(self):
        px = rampa()
        nucleo = {"A": 0.5, "B": 0.5}
        p = alocacao.nucleo_com_trend(px, nucleo, fatia_protegida=0.0).iloc[-1]
        assert float(p["A"]) == pytest.approx(0.5)

    def test_protecao_total_zera_o_que_esta_caindo(self):
        px = rampa()
        nucleo = {"A": 0.5, "C": 0.5}
        p = alocacao.nucleo_com_trend(px, nucleo, fatia_protegida=1.0).iloc[-1]
        assert float(p["A"]) == pytest.approx(0.5)
        assert float(p["C"]) == pytest.approx(0.0)

    def test_protecao_parcial_reduz_pela_metade(self):
        px = rampa()
        nucleo = {"A": 0.5, "C": 0.5}
        p = alocacao.nucleo_com_trend(px, nucleo, fatia_protegida=0.5).iloc[-1]
        assert float(p["C"]) == pytest.approx(0.25)

    @pytest.mark.parametrize("fatia", [-0.1, 1.1, 2.0])
    def test_fatia_fora_do_intervalo_e_recusada(self, fatia):
        with pytest.raises(ValueError, match="de 0 a 1"):
            alocacao.nucleo_com_trend(precos(), {"A": 1.0}, fatia_protegida=fatia)

    def test_no_aquecimento_mantem_o_nucleo(self):
        px = rampa()
        p = alocacao.nucleo_com_trend(px, {"A": 0.5, "B": 0.5})
        assert float(p["A"].iloc[0]) == pytest.approx(0.5)


class TestLimites:
    def test_teto_por_ativo_e_respeitado(self):
        cfg = ConfigAlocacao(max_por_ativo=0.20, min_por_ativo=0.0)
        p = alocacao.peso_igual(precos(), cfg)
        assert (p <= 0.2001).all().all()

    def test_piso_zera_posicao_irrelevante(self):
        cfg = ConfigAlocacao(min_por_ativo=0.30, max_por_ativo=1.0)
        p = alocacao.peso_igual(precos(), cfg)
        assert (p.iloc[-1] == 0).all()  # 25% cada, todos abaixo de 30%

    def test_caixa_minimo_e_reservado(self):
        cfg = ConfigAlocacao(caixa_minimo=0.20, min_por_ativo=0.0)
        p = alocacao.peso_igual(precos(), cfg)
        assert (p.sum(axis=1) <= 0.8001).all()

    @pytest.mark.parametrize("motor", MOTORES_SIMPLES, ids=lambda f: f.__name__)
    def test_nenhum_peso_e_negativo(self, motor):
        p = motor(precos())
        assert (p >= 0).all().all()

    @pytest.mark.parametrize("motor", MOTORES_SIMPLES, ids=lambda f: f.__name__)
    def test_soma_nunca_passa_de_um(self, motor):
        p = motor(precos())
        assert (p.sum(axis=1) <= 1.0001).all()

    @pytest.mark.parametrize("motor", MOTORES_SIMPLES, ids=lambda f: f.__name__)
    def test_devolve_uma_linha_por_data(self, motor):
        px = precos()
        assert len(motor(px)) == len(px)

    @pytest.mark.parametrize("motor", MOTORES_SIMPLES, ids=lambda f: f.__name__)
    def test_nao_altera_a_entrada(self, motor):
        px = precos()
        antes = px.copy()
        motor(px)
        pd.testing.assert_frame_equal(px, antes)


class TestVolAlvo:
    def test_reduz_exposicao_quando_a_vol_passa_do_alvo(self):
        rng = np.random.default_rng(4)
        n = 400
        px = pd.DataFrame(
            {t: 100 * np.exp(np.cumsum(rng.normal(0, 0.04, n))) for t in TICKERS},
            index=_idx(n),
        )
        base = alocacao.peso_igual(px)
        escalado = alocacao.escalar_para_vol_alvo(base, px, vol_alvo_anual=0.05)
        assert float(escalado.iloc[-1].sum()) < float(base.iloc[-1].sum())

    def test_nunca_alavanca_com_max_escala_um(self):
        px = precos()
        base = alocacao.peso_igual(px)
        escalado = alocacao.escalar_para_vol_alvo(base, px, vol_alvo_anual=0.50)
        assert (escalado.sum(axis=1) <= base.sum(axis=1) + 1e-9).all()

    @pytest.mark.parametrize("alvo", [0.0, -0.1])
    def test_alvo_invalido_e_recusado(self, alvo):
        px = precos()
        with pytest.raises(ValueError, match="positiva"):
            alocacao.escalar_para_vol_alvo(alocacao.peso_igual(px), px, alvo)


class TestSemOlharOFuturo:
    @pytest.mark.parametrize("motor", MOTORES_SIMPLES, ids=lambda f: f.__name__)
    def test_truncar_a_serie_nao_muda_os_pesos_do_passado(self, motor):
        completa = precos(n=500)
        curta = completa.iloc[:400]
        a = motor(completa).iloc[:400]
        b = motor(curta)
        pd.testing.assert_frame_equal(a, b, check_freq=False)


class TestRegistroDeMotores:
    def test_todos_os_motores_estao_registrados(self):
        assert set(alocacao.MOTORES) == {
            "peso_igual", "inverso_vol", "momentum_absoluto", "dual_momentum"
        }

    @pytest.mark.parametrize("nome", ["peso_igual", "inverso_vol", "momentum_absoluto", "dual_momentum"])
    def test_motor_registrado_e_chamavel(self, nome):
        p = alocacao.MOTORES[nome](precos())
        assert isinstance(p, pd.DataFrame)
