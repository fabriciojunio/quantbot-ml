"""
Testes dos perfis de risco.

O que estes testes protegem: que o orçamento de risco seja respeitado ao pé da
letra, que os pesos sempre fechem em 100%, que o satélite não vaze para a
carteira sem aprovação, e que mais risco produza mais queda — porque é essa
relação que o usuário precisa enxergar antes de escolher a fatia.
"""

import numpy as np
import pandas as pd
import pytest

from carteira.perfil import (
    COMPOSICAO,
    FAIXAS_DE_RISCO,
    PERFIS,
    Faixa,
    Perfil,
    medir_perfis,
    serie_do_perfil,
    sugerir,
)

TICKERS = ["BOVA11.SA", "SMAL11.SA", "IVVB11.SA", "IMAB11.SA", "B5P211.SA", "GOLD11.SA", "XFIX11.SA"]
NOMES = list(PERFIS)


def _idx(n):
    return pd.bdate_range("2020-01-01", periods=n)


def precos(n=756, semente=17):
    rng = np.random.default_rng(semente)
    vols = {"BOVA11.SA": 0.018, "SMAL11.SA": 0.022, "IVVB11.SA": 0.013,
            "IMAB11.SA": 0.005, "B5P211.SA": 0.002, "GOLD11.SA": 0.020,
            "XFIX11.SA": 0.008}
    return pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0003, v, n))) for t, v in vols.items()},
        index=_idx(n),
    )


def cdi(n=756, taxa=0.0005):
    return pd.Series(taxa, index=_idx(n))


class TestConstrucaoDePerfil:
    @pytest.mark.parametrize("risco", [0, 10, 25, 50, 75, 90, 100])
    def test_risco_valido_e_aceito(self, risco):
        assert Perfil("teste", risco, "x").risco_pct == risco

    @pytest.mark.parametrize("risco", [-1, -0.1, 100.1, 150])
    def test_risco_fora_do_intervalo_e_recusado(self, risco):
        with pytest.raises(ValueError, match="de 0 a 100"):
            Perfil("teste", risco, "x")

    def test_pesos_do_risco_que_nao_somam_um_sao_recusados(self):
        with pytest.raises(ValueError, match="dentro do risco"):
            Perfil("t", 50, "x", dentro_do_risco={Faixa.BOLSA: 0.5})

    def test_pesos_do_seguro_que_nao_somam_um_sao_recusados(self):
        with pytest.raises(ValueError, match="dentro do seguro"):
            Perfil("t", 50, "x", dentro_do_seguro={Faixa.CAIXA: 0.3})

    def test_perfil_e_imutavel(self):
        with pytest.raises(Exception):
            PERFIS["moderado"].risco_pct = 99


class TestOrcamentoDeRisco:
    @pytest.mark.parametrize("nome", NOMES)
    def test_pesos_por_faixa_somam_um(self, nome):
        assert sum(PERFIS[nome].pesos_por_faixa().values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("nome", NOMES)
    def test_pesos_por_ativo_somam_um(self, nome):
        assert sum(PERFIS[nome].pesos_por_ativo().values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("nome", NOMES)
    def test_fatia_de_risco_bate_com_o_orcamento(self, nome):
        p = PERFIS[nome]
        faixas = p.pesos_por_faixa()
        risco = sum(faixas.get(f, 0.0) for f in FAIXAS_DE_RISCO)
        assert risco == pytest.approx(p.risco_pct / 100)

    @pytest.mark.parametrize("nome", NOMES)
    def test_nenhum_peso_negativo(self, nome):
        assert all(p >= 0 for p in PERFIS[nome].pesos_por_ativo().values())

    @pytest.mark.parametrize("risco", [0, 20, 40, 60, 80, 100])
    def test_risco_declarado_e_respeitado_em_qualquer_nivel(self, risco):
        p = Perfil("t", risco, "x")
        faixas = p.pesos_por_faixa()
        assert sum(faixas.get(f, 0.0) for f in FAIXAS_DE_RISCO) == pytest.approx(risco / 100)

    def test_risco_zero_fica_todo_em_seguro(self):
        faixas = Perfil("t", 0, "x").pesos_por_faixa()
        assert faixas[Faixa.CAIXA] + faixas[Faixa.RENDA_FIXA] == pytest.approx(1.0)

    def test_risco_cem_nao_deixa_caixa(self):
        faixas = Perfil("t", 100, "x").pesos_por_faixa()
        assert faixas.get(Faixa.CAIXA, 0.0) == pytest.approx(0.0)

    def test_mais_risco_significa_menos_caixa(self):
        anterior = 1.0
        for risco in (0, 25, 50, 75, 100):
            caixa = Perfil("t", risco, "x").pesos_por_faixa().get(Faixa.CAIXA, 0.0)
            assert caixa <= anterior + 1e-9
            anterior = caixa


class TestSatelite:
    def test_satelite_bloqueado_volta_para_dolar(self):
        p = Perfil(
            "t", 60, "x",
            dentro_do_risco={Faixa.DOLAR: 0.4, Faixa.BOLSA: 0.4, Faixa.SATELITE: 0.2},
        )
        sem = p.pesos_por_ativo(satelite_liberado=False)
        assert sum(sem.values()) == pytest.approx(1.0)
        # A fatia do satélite foi somada ao dólar.
        assert sem["IVVB11.SA"] > 0.4 * 0.6

    def test_liberar_satelite_vazio_nao_faz_dinheiro_sumir(self):
        """Bug encontrado por este teste: com o satélite liberado mas sem nada
        para comprar, 12% da carteira evaporava sem erro nenhum."""
        p = Perfil(
            "t", 60, "x",
            dentro_do_risco={Faixa.DOLAR: 0.4, Faixa.BOLSA: 0.4, Faixa.SATELITE: 0.2},
        )
        for liberado in (True, False):
            pesos = p.pesos_por_ativo(satelite_liberado=liberado)
            assert sum(pesos.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("risco", [10, 30, 50, 70, 90, 100])
    @pytest.mark.parametrize("liberado", [True, False])
    def test_pesos_fecham_em_qualquer_combinacao(self, risco, liberado):
        p = Perfil(
            "t", risco, "x",
            dentro_do_risco={Faixa.DOLAR: 0.4, Faixa.BOLSA: 0.3, Faixa.SATELITE: 0.3},
        )
        assert sum(p.pesos_por_ativo(liberado).values()) == pytest.approx(1.0)

    def test_composicao_do_satelite_esta_vazia(self):
        """Enquanto nada passa na validação, o satélite não tem o que comprar."""
        assert COMPOSICAO[Faixa.SATELITE] == {}


class TestExposicaoADolar:
    @pytest.mark.parametrize("nome", NOMES)
    def test_todo_perfil_com_risco_tem_dolar(self, nome):
        p = PERFIS[nome]
        if p.risco_pct > 0:
            assert p.pesos_por_ativo().get("IVVB11.SA", 0) > 0

    @pytest.mark.parametrize("risco", [20, 40, 60, 80, 100])
    def test_dolar_cresce_com_o_risco(self, risco):
        anterior = 0.0
        for r in (20, 40, 60, 80, 100):
            atual = Perfil("t", r, "x").pesos_por_ativo().get("IVVB11.SA", 0.0)
            assert atual >= anterior - 1e-9
            anterior = atual

    def test_sugerir_informa_a_exposicao(self):
        s = sugerir(60)
        assert s["exposicao_a_dolar_pct"] > 0


class TestSugerir:
    @pytest.mark.parametrize("risco", [0, 15, 30, 45, 60, 75, 90, 100])
    def test_soma_dos_ativos_e_cem(self, risco):
        s = sugerir(risco)
        assert sum(s["por_ativo"].values()) == pytest.approx(100.0, abs=0.2)

    @pytest.mark.parametrize("risco", [0, 25, 50, 75, 100])
    def test_soma_das_faixas_e_cem(self, risco):
        s = sugerir(risco)
        assert sum(s["por_faixa"].values()) == pytest.approx(100.0, abs=0.2)

    def test_sempre_traz_aviso_sobre_a_simetria(self):
        assert "andam juntas" in sugerir(80)["aviso"]

    def test_risco_invalido_e_recusado(self):
        with pytest.raises(ValueError):
            sugerir(150)


class TestSerieDoPerfil:
    @pytest.mark.parametrize("nome", NOMES)
    def test_gera_curva_do_tamanho_certo(self, nome):
        px = precos()
        curva = serie_do_perfil(PERFIS[nome], px, cdi(len(px)))
        assert len(curva) == len(px)

    @pytest.mark.parametrize("nome", NOMES)
    def test_curva_e_sempre_positiva(self, nome):
        px = precos()
        assert (serie_do_perfil(PERFIS[nome], px, cdi(len(px))) > 0).all()

    def test_perfil_conservador_oscila_menos(self):
        px = precos()
        c = serie_do_perfil(PERFIS["conservador"], px, cdi(len(px))).pct_change().std()
        a = serie_do_perfil(PERFIS["muito_agressivo"], px, cdi(len(px))).pct_change().std()
        assert c < a

    def test_risco_zero_acompanha_o_cdi(self):
        px = precos()
        p = Perfil("t", 0, "x", dentro_do_seguro={Faixa.CAIXA: 1.0, Faixa.RENDA_FIXA: 0.0})
        curva = serie_do_perfil(p, px, cdi(len(px)))
        esperado = 1.0005 ** len(px)
        assert float(curva.iloc[-1]) == pytest.approx(esperado, rel=0.02)

    def test_falta_pequena_de_ativo_ainda_roda(self):
        """Sem GOLD11 falta pouco peso: renormalizar é aceitável."""
        px = precos().drop(columns=["GOLD11.SA"])
        curva = serie_do_perfil(PERFIS["conservador"], px, cdi(len(px)))
        assert len(curva) == len(px)

    def test_universo_incompleto_demais_levanta(self):
        """Bug encontrado por este teste: com quase nada do perfil presente, a
        carteira virava 100% CDI em silêncio e a curva saía bonita e errada."""
        px = precos()[["BOVA11.SA"]]
        with pytest.raises(ValueError, match="do peso do perfil"):
            serie_do_perfil(PERFIS["muito_agressivo"], px, cdi(len(px)))

    def test_sem_nenhum_ativo_conhecido_levanta(self):
        px = pd.DataFrame({"XXXX": [1.0] * 100}, index=_idx(100))
        with pytest.raises(ValueError, match="peso do perfil|nenhum ativo"):
            serie_do_perfil(PERFIS["muito_agressivo"], px, cdi(100))

    def test_nao_altera_a_serie_de_precos(self):
        px = precos()
        antes = px.copy()
        serie_do_perfil(PERFIS["moderado"], px, cdi(len(px)))
        pd.testing.assert_frame_equal(px, antes)


class TestMedirPerfis:
    def test_mede_todos_os_perfis(self):
        px = precos()
        assert len(medir_perfis(px, cdi(len(px)))) == len(PERFIS)

    def test_traz_as_colunas_que_importam(self):
        px = precos()
        df = medir_perfis(px, cdi(len(px)))
        for coluna in ("cagr_pct", "max_drawdown_pct", "pior_12_meses_pct", "vol_anual_pct"):
            assert coluna in df.columns

    def test_mais_risco_produz_queda_maior(self):
        """A relação que o usuário precisa ver antes de escolher a fatia."""
        px = precos()
        df = medir_perfis(px, cdi(len(px))).sort_values("risco_pct")
        quedas = df["max_drawdown_pct"].tolist()
        assert quedas == sorted(quedas, reverse=True)

    def test_mais_risco_produz_volatilidade_maior_nos_extremos(self):
        px = precos()
        df = medir_perfis(px, cdi(len(px))).set_index("perfil")
        assert df.loc["muito_agressivo", "vol_anual_pct"] > df.loc["conservador", "vol_anual_pct"]

    def test_drawdown_nunca_e_positivo(self):
        px = precos()
        assert (medir_perfis(px, cdi(len(px)))["max_drawdown_pct"] <= 0).all()

    def test_tempo_a_frente_do_cdi_fica_entre_zero_e_cem(self):
        px = precos()
        col = medir_perfis(px, cdi(len(px)))["tempo_a_frente_do_cdi_pct"]
        assert ((col >= 0) & (col <= 100)).all()


class TestComposicao:
    @pytest.mark.parametrize("faixa", [f for f in Faixa if f != Faixa.SATELITE])
    def test_toda_faixa_soma_um(self, faixa):
        assert sum(COMPOSICAO[faixa].values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("faixa", list(Faixa))
    def test_toda_faixa_esta_declarada(self, faixa):
        assert faixa in COMPOSICAO

    def test_caixa_e_so_cdi(self):
        assert COMPOSICAO[Faixa.CAIXA] == {"CDI": 1.0}

    def test_dolar_e_so_ivvb(self):
        assert list(COMPOSICAO[Faixa.DOLAR]) == ["IVVB11.SA"]


class TestPerfisPadrao:
    @pytest.mark.parametrize("nome", NOMES)
    def test_tem_descricao(self, nome):
        assert PERFIS[nome].descricao

    def test_estao_ordenados_por_risco(self):
        riscos = [PERFIS[n].risco_pct for n in NOMES]
        assert riscos == sorted(riscos)

    def test_vao_de_vinte_a_cem(self):
        assert PERFIS["conservador"].risco_pct == 20
        assert PERFIS["muito_agressivo"].risco_pct == 100
