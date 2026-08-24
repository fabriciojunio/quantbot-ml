"""
Testes da camada de moeda e cripto.

O teto de 5% e a regra de tirar a fatia do risco (nunca da parte segura) são
as duas decisões que este arquivo protege. As duas parecem detalhe e não são:
sem o teto, um número digitado errado transforma a carteira num fundo de
bitcoin; sem a regra de origem, o orçamento de risco que o usuário escolheu
deixa de significar o que ele acha que significa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from carteira import multimoeda as mm
from carteira.perfil import PERFIS, Perfil
from carteira.serie_longa import COMPOSICAO_LONGA

SEM_SAUDE = settings(
    max_examples=80, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

BASE = {"IVVB11.SA": 0.36, "BOVA11.SA": 0.28, "GOLD11.SA": 0.16, "CDI": 0.20}


# ─────────────────────────────────────────────────────────────────────────────
# Exposição cambial
# ─────────────────────────────────────────────────────────────────────────────

class TesteExposicaoCambial:
    def test_separa_local_de_estrangeiro(self):
        e = mm.exposicao_cambial(BASE)
        assert e.estrangeira == pytest.approx(0.52)
        assert e.local == pytest.approx(0.48)

    def test_ivvb_conta_como_dolar_apesar_de_negociar_em_real(self):
        """O ETF é cotado em real, mas o que está dentro dele é dólar."""
        e = mm.exposicao_cambial({"IVVB11.SA": 1.0})
        assert e.por_moeda[mm.Moeda.USD] == pytest.approx(1.0)

    def test_ouro_conta_como_dolar(self):
        e = mm.exposicao_cambial({"GOLD11.SA": 1.0})
        assert e.estrangeira == pytest.approx(1.0)

    def test_renda_fixa_local_e_real(self):
        e = mm.exposicao_cambial({"IMAB11.SA": 0.5, "CDI": 0.5})
        assert e.local == pytest.approx(1.0)

    def test_ticker_desconhecido_vira_real_e_nao_quebra(self):
        e = mm.exposicao_cambial({"NADA11.SA": 1.0})
        assert e.local == pytest.approx(1.0)

    def test_soma_das_moedas_bate_com_a_soma_dos_pesos(self):
        e = mm.exposicao_cambial(BASE)
        assert sum(e.por_moeda.values()) == pytest.approx(1.0)

    def test_resumo_e_percentual(self):
        r = mm.exposicao_cambial(BASE).resumo()
        assert r["estrangeira_pct"] == pytest.approx(52.0)
        assert set(r["por_moeda"]) == {"USD", "BRL"}

    @pytest.mark.parametrize("nome", list(PERFIS))
    def test_todo_perfil_tem_exposicao_somando_um(self, nome):
        pesos = PERFIS[nome].pesos_por_ativo(composicao=COMPOSICAO_LONGA)
        assert sum(mm.exposicao_cambial(pesos).por_moeda.values()) == pytest.approx(1.0)

    def test_risco_maior_aumenta_a_exposicao_estrangeira(self):
        baixo = Perfil("b", 20.0, "x").pesos_por_ativo(composicao=COMPOSICAO_LONGA)
        alto = Perfil("a", 100.0, "x").pesos_por_ativo(composicao=COMPOSICAO_LONGA)
        assert mm.exposicao_cambial(alto).estrangeira > mm.exposicao_cambial(baixo).estrangeira

    def test_todo_ativo_do_universo_tem_moeda_declarada(self):
        from carteira.universo import UNIVERSO_B3

        for ticker in UNIVERSO_B3:
            assert ticker in mm.MOEDA_DE_FATO, ticker


# ─────────────────────────────────────────────────────────────────────────────
# Fatia de cripto
# ─────────────────────────────────────────────────────────────────────────────

class TesteComCripto:
    def test_fatia_zero_nao_muda_nada(self):
        assert mm.com_cripto(BASE, 0.0) == BASE

    def test_fatia_entra_com_o_tamanho_pedido(self):
        novos = mm.com_cripto(BASE, 0.03)
        assert novos["BTC-BRL"] == pytest.approx(0.03)

    def test_pesos_continuam_somando_um(self):
        for fatia in (0.01, 0.02, 0.03, 0.05):
            assert sum(mm.com_cripto(BASE, fatia).values()) == pytest.approx(1.0)

    def test_a_parte_segura_nao_e_tocada(self):
        """Tirar do CDI aumentaria o risco por duas vias ao mesmo tempo."""
        novos = mm.com_cripto(BASE, 0.05)
        assert novos["CDI"] == pytest.approx(BASE["CDI"])

    def test_sai_proporcionalmente_dos_ativos_de_risco(self):
        novos = mm.com_cripto(BASE, 0.04)
        proporcao_antes = BASE["IVVB11.SA"] / BASE["BOVA11.SA"]
        proporcao_depois = novos["IVVB11.SA"] / novos["BOVA11.SA"]
        assert proporcao_antes == pytest.approx(proporcao_depois)

    def test_teto_e_respeitado(self):
        with pytest.raises(ValueError, match="passa do teto"):
            mm.com_cripto(BASE, 0.20)

    def test_fatia_negativa_e_recusada(self):
        with pytest.raises(ValueError, match="negativa"):
            mm.com_cripto(BASE, -0.01)

    def test_teto_padrao_e_cinco_por_cento(self):
        assert mm.TETO_DE_CRIPTO == pytest.approx(0.05)

    def test_carteira_sem_risco_suficiente_e_recusada(self):
        conservadora = {"CDI": 0.98, "BOVA11.SA": 0.02}
        with pytest.raises(ValueError, match="não dá para tirar"):
            mm.com_cripto(conservadora, 0.05)

    def test_renda_fixa_tambem_e_preservada(self):
        pesos = {"CDI": 0.2, "IMAB11.SA": 0.2, "BOVA11.SA": 0.3, "IVVB11.SA": 0.3}
        novos = mm.com_cripto(pesos, 0.05)
        assert novos["IMAB11.SA"] == pytest.approx(0.2)
        assert novos["CDI"] == pytest.approx(0.2)

    def test_cripto_desconhecida_pode_ser_nomeada(self):
        novos = mm.com_cripto(BASE, 0.02, cripto="ETH-BRL")
        assert "ETH-BRL" in novos

    @SEM_SAUDE
    @given(fatia=st.floats(0.0, mm.TETO_DE_CRIPTO))
    def test_soma_um_para_qualquer_fatia_valida(self, fatia):
        assert sum(mm.com_cripto(BASE, fatia).values()) == pytest.approx(1.0, abs=1e-9)

    @SEM_SAUDE
    @given(fatia=st.floats(0.0, mm.TETO_DE_CRIPTO))
    def test_nenhum_peso_fica_negativo(self, fatia):
        assert all(p >= 0 for p in mm.com_cripto(BASE, fatia).values())


# ─────────────────────────────────────────────────────────────────────────────
# Série de cripto em reais
# ─────────────────────────────────────────────────────────────────────────────

def historico_falso(dias=400, valor=50_000.0):
    def _h(simbolo, intervalo="1d", dias_=None, **kw):
        idx = pd.date_range("2020-01-01", periods=dias, freq="D")
        return pd.DataFrame({"fechamento": np.full(dias, valor)}, index=idx)
    return _h


def cambista_falso(valor=5.0, dias=500):
    def _c(anos=9, moeda="USD_BRL"):
        idx = pd.date_range("2019-12-01", periods=dias, freq="D")
        return pd.Series(np.full(dias, valor), index=idx)
    return _c


class TesteSerieCripto:
    def test_converte_pelo_cambio(self):
        s = mm.serie_cripto_em_reais(
            "BTC-BRL", anos=1, historico=historico_falso(), cambista=cambista_falso(5.0)
        )
        assert float(s.iloc[0]) == pytest.approx(250_000.0)

    def test_cambio_maior_da_preco_maior_em_real(self):
        barato = mm.serie_cripto_em_reais(
            "BTC-BRL", anos=1, historico=historico_falso(), cambista=cambista_falso(4.0))
        caro = mm.serie_cripto_em_reais(
            "BTC-BRL", anos=1, historico=historico_falso(), cambista=cambista_falso(6.0))
        assert float(caro.iloc[0]) > float(barato.iloc[0])

    def test_ticker_desconhecido_e_recusado(self):
        with pytest.raises(ValueError, match="desconhecida"):
            mm.serie_cripto_em_reais("DOGE-BRL")

    def test_exchange_vazia_cai_para_a_fonte_alternativa(self):
        """A Binance devolve HTTP 451 de datacenter; sem fallback o robô morre na nuvem."""
        def vazio(*a, **k):
            return pd.DataFrame()

        def yahoo(tickers, anos=8):
            idx = pd.date_range("2020-01-01", periods=400, freq="D")
            return pd.DataFrame({tickers[0]: np.full(400, 60_000.0)}, index=idx)

        s = mm.serie_cripto_em_reais(
            "BTC-BRL", anos=1, historico=vazio,
            cambista=cambista_falso(5.0), fonte_alternativa=yahoo,
        )
        assert float(s.iloc[0]) == pytest.approx(300_000.0)

    def test_falha_da_exchange_cai_para_a_fonte_alternativa(self):
        def explode(*a, **k):
            raise RuntimeError("HTTP Error 451")

        def yahoo(tickers, anos=8):
            idx = pd.date_range("2020-01-01", periods=400, freq="D")
            return pd.DataFrame({tickers[0]: np.full(400, 50_000.0)}, index=idx)

        s = mm.serie_cripto_em_reais(
            "BTC-BRL", anos=1, historico=explode,
            cambista=cambista_falso(5.0), fonte_alternativa=yahoo,
        )
        assert len(s) > 0

    def test_as_duas_fontes_falhando_vira_erro_e_nao_dado_inventado(self):
        def explode(*a, **k):
            raise RuntimeError("HTTP Error 451")

        def yahoo_tambem_falha(*a, **k):
            raise RuntimeError("sem rede")

        with pytest.raises(mm.ErroMultimoeda, match="todas as fontes"):
            mm.serie_cripto_em_reais(
                "BTC-BRL", historico=explode,
                cambista=cambista_falso(), fonte_alternativa=yahoo_tambem_falha,
            )

    def test_fonte_alternativa_vazia_tambem_vira_erro(self):
        def explode(*a, **k):
            raise RuntimeError("451")

        def yahoo_vazio(*a, **k):
            return pd.DataFrame()

        with pytest.raises(mm.ErroMultimoeda, match="todas as fontes"):
            mm.serie_cripto_em_reais(
                "BTC-BRL", historico=explode,
                cambista=cambista_falso(), fonte_alternativa=yahoo_vazio,
            )

    def test_toda_cripto_conhecida_tem_fonte_alternativa(self):
        assert set(mm.SIMBOLOS_CRIPTO) == set(mm.SIMBOLOS_YAHOO)

    def test_sem_dia_em_comum_vira_erro(self):
        def cambio_de_outra_epoca(anos=9, moeda="USD_BRL"):
            idx = pd.date_range("1995-01-01", periods=100, freq="D")
            return pd.Series(np.full(100, 1.0), index=idx)

        with pytest.raises(mm.ErroMultimoeda, match="nenhum dia"):
            mm.serie_cripto_em_reais(
                "BTC-BRL", historico=historico_falso(), cambista=cambio_de_outra_epoca)

    def test_o_nome_da_serie_e_o_ticker(self):
        s = mm.serie_cripto_em_reais(
            "BTC-BRL", anos=1, historico=historico_falso(), cambista=cambista_falso())
        assert s.name == "BTC-BRL"


# ─────────────────────────────────────────────────────────────────────────────
# Estudo e veredito
# ─────────────────────────────────────────────────────────────────────────────

def montar_precos(dias=2600, semente=11, com_btc_subindo=True):
    idx = pd.bdate_range("2016-01-01", periods=dias)
    rng = np.random.default_rng(semente)
    dados = {
        "IVVB11.SA": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.010, dias))),
        "BOVA11.SA": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.014, dias))),
        "GOLD11.SA": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, dias))),
    }
    deriva = 0.0020 if com_btc_subindo else -0.0015
    dados["BTC-BRL"] = 1000 * np.exp(np.cumsum(rng.normal(deriva, 0.040, dias)))
    return pd.DataFrame(dados, index=idx)


def cdi_serie(idx, taxa=0.0004):
    return pd.Series(np.full(len(idx), taxa), index=idx)


class TesteEstudo:
    def test_produz_uma_linha_por_fatia(self):
        p = montar_precos()
        e = mm.estudar_fatia(p, cdi_serie(p.index), BASE, fatias=(0.0, 0.02, 0.05))
        assert len(e) == 3
        assert list(e["fatia_cripto_pct"]) == [0.0, 2.0, 5.0]

    def test_recusa_quando_a_cripto_nao_esta_nos_precos(self):
        p = montar_precos().drop(columns=["BTC-BRL"])
        with pytest.raises(ValueError, match="não está na série"):
            mm.estudar_fatia(p, cdi_serie(p.index), BASE)

    def test_cripto_subindo_melhora_o_retorno(self):
        p = montar_precos(com_btc_subindo=True)
        e = mm.estudar_fatia(p, cdi_serie(p.index), BASE, fatias=(0.0, 0.05))
        assert e["tir_anual_pct"].iloc[1] > e["tir_anual_pct"].iloc[0]

    def test_cripto_caindo_piora_o_retorno(self):
        p = montar_precos(com_btc_subindo=False)
        e = mm.estudar_fatia(p, cdi_serie(p.index), BASE, fatias=(0.0, 0.05))
        assert e["tir_anual_pct"].iloc[1] < e["tir_anual_pct"].iloc[0]

    def test_janelas_moveis_multiplicam_as_linhas(self):
        p = montar_precos()
        j = mm.estudar_fatia_em_janelas(
            p, cdi_serie(p.index), BASE, fatias=(0.0, 0.02), anos=3, passo_meses=12)
        assert set(j["fatia_cripto_pct"]) == {0.0, 2.0}
        assert len(j) > 2


class TesteVeredito:
    def _janelas(self, com_btc_subindo=True, semente=11):
        p = montar_precos(semente=semente, com_btc_subindo=com_btc_subindo)
        return mm.estudar_fatia_em_janelas(
            p, cdi_serie(p.index), BASE,
            fatias=(0.0, 0.01, 0.03, 0.05), anos=3, passo_meses=6,
        )

    def test_exige_a_referencia_de_zero(self):
        j = self._janelas()
        with pytest.raises(ValueError, match="0%"):
            mm.veredito_da_fatia(j[j["fatia_cripto_pct"] > 0])

    def test_exige_a_coluna_de_fatia(self):
        with pytest.raises(ValueError, match="fatia_cripto_pct"):
            mm.veredito_da_fatia(pd.DataFrame({"x": [1]}))

    def test_cripto_que_so_cai_nao_e_recomendada(self):
        v = mm.veredito_da_fatia(self._janelas(com_btc_subindo=False))
        assert v["recomendada_pct"] == 0.0
        assert "nenhuma fatia" in v["conclusao"]

    def test_entre_aceitaveis_escolhe_a_menor(self):
        v = mm.veredito_da_fatia(self._janelas(com_btc_subindo=True))
        aceitaveis = [c["fatia_pct"] for c in v["candidatas"] if c["aceitavel"]]
        if aceitaveis:
            assert v["recomendada_pct"] == min(aceitaveis)

    def test_traz_pior_caso_e_nao_so_mediana(self):
        v = mm.veredito_da_fatia(self._janelas())
        for c in v["candidatas"]:
            assert "ganho_no_pior_caso_pp" in c
            assert "piora_maxima_da_queda_pp" in c

    def test_avisos_dizem_o_que_o_numero_nao_cobre(self):
        v = mm.veredito_da_fatia(self._janelas())
        texto = " ".join(v["avisos"]).lower()
        assert "sobrepõem" in texto
        assert "operar" in texto

    def test_recomendacao_nunca_passa_do_teto(self):
        v = mm.veredito_da_fatia(self._janelas())
        assert v["recomendada_pct"] <= mm.TETO_DE_CRIPTO * 100

    def test_taxa_de_vitoria_fica_entre_zero_e_cem(self):
        v = mm.veredito_da_fatia(self._janelas())
        for c in v["candidatas"]:
            assert 0.0 <= c["taxa_de_vitoria_pct"] <= 100.0
