"""
Testes do universo de ativos.

O universo carrega a informação que o motor de imposto e o de custo consomem.
Um regime tributário errado aqui não gera exceção: gera um backtest que paga
menos imposto do que a vida real. Estes testes fecham essa porta.
"""

import pytest

from carteira.universo import (
    CAIXA,
    CARTEIRAS_REFERENCIA,
    UNIVERSO_B3,
    UNIVERSO_CRIPTO,
    Ativo,
    Classe,
    Regime,
    cabe_a_ordem,
    por_classe,
    universo_completo,
)

TODOS = list(UNIVERSO_B3.values())


class TestIntegridadeDoUniverso:
    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_spread_e_positivo_e_plausivel(self, ativo):
        assert 0 < ativo.spread_pct < 0.01

    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_volume_e_positivo(self, ativo):
        assert ativo.volume_medio_brl > 0

    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_tem_nome_e_classe(self, ativo):
        assert ativo.nome
        assert isinstance(ativo.classe, Classe)

    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_tem_regime_tributario(self, ativo):
        assert isinstance(ativo.regime, Regime)

    @pytest.mark.parametrize("ticker,ativo", list(UNIVERSO_B3.items()))
    def test_chave_bate_com_o_ticker(self, ticker, ativo):
        assert ticker == ativo.ticker

    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_dy_nunca_e_negativo(self, ativo):
        assert ativo.dy_anual_pct >= 0

    def test_ativo_e_imutavel(self):
        with pytest.raises(Exception):
            TODOS[0].spread_pct = 0.5


class TestRegimesTributarios:
    def test_etfs_de_acao_sao_renda_variavel(self):
        for t in ("BOVA11.SA", "IVVB11.SA", "SMAL11.SA", "DIVO11.SA"):
            assert UNIVERSO_B3[t].regime == Regime.ETF_RENDA_VARIAVEL

    def test_etfs_de_renda_fixa_usam_a_regressiva(self):
        for t in ("IMAB11.SA", "B5P211.SA", "FIXA11.SA"):
            assert UNIVERSO_B3[t].regime == Regime.ETF_RENDA_FIXA

    def test_etf_de_fii_nao_e_tratado_como_fii(self):
        """XFIX11 é ETF: o rendimento dele é tributado, ao contrário do FII."""
        assert UNIVERSO_B3["XFIX11.SA"].regime == Regime.ETF_RENDA_VARIAVEL
        assert UNIVERSO_B3["XFIX11.SA"].regime != Regime.FII

    def test_o_aviso_sobre_o_etf_de_fii_esta_documentado(self):
        assert "isenção" in UNIVERSO_B3["XFIX11.SA"].observacao

    def test_cripto_tem_regime_proprio(self):
        for a in UNIVERSO_CRIPTO.values():
            assert a.regime == Regime.CRIPTO


class TestMeioSpread:
    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_e_metade_do_spread(self, ativo):
        assert ativo.custo_meio_spread == pytest.approx(ativo.spread_pct / 2)


class TestUniversoCompleto:
    def test_sem_cripto_por_padrao(self):
        assert "BTC-USD" not in universo_completo()

    def test_com_cripto_quando_pedido(self):
        assert "BTC-USD" in universo_completo(incluir_cripto=True)

    def test_tamanho_bate(self):
        assert len(universo_completo(True)) == len(UNIVERSO_B3) + len(UNIVERSO_CRIPTO)


class TestPorClasse:
    @pytest.mark.parametrize("classe", list(Classe))
    def test_devolve_lista(self, classe):
        assert isinstance(por_classe(classe), list)

    def test_acoes_br_traz_bova_e_divo(self):
        tickers = {a.ticker for a in por_classe(Classe.ACOES_BR)}
        assert {"BOVA11.SA", "DIVO11.SA"} <= tickers

    def test_cripto_traz_btc(self):
        assert "BTC-USD" in {a.ticker for a in por_classe(Classe.CRIPTO)}

    def test_classe_sem_ativo_devolve_vazio(self):
        assert por_classe(Classe.CAIXA) == []


class TestCabeAOrdem:
    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_ordem_de_varejo_sempre_cabe(self, ativo):
        assert cabe_a_ordem(ativo, 10_000)

    def test_ordem_gigante_nao_cabe(self):
        assert not cabe_a_ordem(UNIVERSO_B3["XFIX11.SA"], 10_000_000)

    def test_fatia_maior_permite_ordem_maior(self):
        ativo = UNIVERSO_B3["XFIX11.SA"]
        valor = ativo.volume_medio_brl * 0.05
        assert not cabe_a_ordem(ativo, valor, fatia_maxima=0.01)
        assert cabe_a_ordem(ativo, valor, fatia_maxima=0.10)


class TestCarteirasDeReferencia:
    @pytest.mark.parametrize("nome", list(CARTEIRAS_REFERENCIA))
    def test_pesos_somam_um(self, nome):
        assert sum(CARTEIRAS_REFERENCIA[nome].values()) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("nome", list(CARTEIRAS_REFERENCIA))
    def test_nenhum_peso_negativo(self, nome):
        assert all(p >= 0 for p in CARTEIRAS_REFERENCIA[nome].values())

    @pytest.mark.parametrize("nome", list(CARTEIRAS_REFERENCIA))
    def test_tickers_existem_no_universo(self, nome):
        for t in CARTEIRAS_REFERENCIA[nome]:
            assert t in UNIVERSO_B3 or t == "CDI"

    def test_diversificada_tem_mais_ativos_que_sessenta_quarenta(self):
        assert len(CARTEIRAS_REFERENCIA["diversificada"]) > len(CARTEIRAS_REFERENCIA["60_40"])

    def test_so_cdi_e_cem_por_cento_caixa(self):
        assert CARTEIRAS_REFERENCIA["so_cdi"] == {"CDI": 1.0}


class TestCaixa:
    def test_caixa_nao_tem_spread(self):
        assert CAIXA.spread_pct == 0.0

    def test_caixa_e_classe_caixa(self):
        assert CAIXA.classe == Classe.CAIXA

    def test_caixa_tem_liquidez_infinita(self):
        assert CAIXA.volume_medio_brl == float("inf")
