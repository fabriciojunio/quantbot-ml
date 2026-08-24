"""
Testes do custo de operar na B3.

O ponto que estes testes protegem é o mesmo que derrubou os dois sistemas
anteriores: custo que não é cobrado vira retorno que não existe. Aqui se
verifica que toda ponta paga, que o spread entra no preço executado e que
girar mais custa mais — de forma proporcional, não aproximada.
"""

import pytest

from carteira.custos_b3 import (
    EMOLUMENTOS_PCT,
    TAXA_ADM_ANUAL,
    ModeloCustoB3,
    comparar_frequencias,
)
from carteira.universo import UNIVERSO_B3, Ativo, Classe, Regime

BOVA = UNIVERSO_B3["BOVA11.SA"]
XFIX = UNIVERSO_B3["XFIX11.SA"]
TODOS = list(UNIVERSO_B3.values())


@pytest.fixture
def modelo():
    return ModeloCustoB3()


class TestCustoPorPonta:
    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_toda_ponta_custa_alguma_coisa(self, modelo, ativo):
        assert modelo.custo_ponta(ativo, 10_000) > 0

    @pytest.mark.parametrize("valor", [0, -1, -1000])
    def test_valor_nao_positivo_nao_custa(self, modelo, valor):
        assert modelo.custo_ponta(BOVA, valor) == 0.0

    def test_custo_cresce_com_o_valor(self, modelo):
        assert modelo.custo_ponta(BOVA, 20_000) > modelo.custo_ponta(BOVA, 10_000)

    def test_ida_e_volta_e_o_dobro_da_ponta(self, modelo):
        assert modelo.custo_ida_volta(BOVA, 10_000) == pytest.approx(
            2 * modelo.custo_ponta(BOVA, 10_000)
        )

    def test_emolumentos_estao_embutidos(self, modelo):
        sem_spread = Ativo(
            "X", "x", Classe.ACOES_BR, Regime.ETF_RENDA_VARIAVEL,
            spread_pct=0.0, volume_medio_brl=1e9,
        )
        assert modelo.custo_ponta(sem_spread, 10_000) == pytest.approx(10_000 * EMOLUMENTOS_PCT)

    def test_corretagem_entra_no_custo(self):
        com = ModeloCustoB3(corretagem_brl=10.0)
        sem = ModeloCustoB3(corretagem_brl=0.0)
        assert com.custo_ponta(BOVA, 10_000) - sem.custo_ponta(BOVA, 10_000) == pytest.approx(10.0)

    def test_corretagem_fixa_pesa_mais_em_ordem_pequena(self):
        m = ModeloCustoB3(corretagem_brl=10.0)
        assert m.custo_ponta_pct(BOVA, 500) > m.custo_ponta_pct(BOVA, 50_000)


class TestSpread:
    def test_ativo_menos_liquido_custa_mais(self, modelo):
        assert modelo.custo_ponta_pct(XFIX, 10_000) > modelo.custo_ponta_pct(BOVA, 10_000)

    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_meio_spread_e_metade_do_spread(self, ativo):
        assert ativo.custo_meio_spread == pytest.approx(ativo.spread_pct / 2)

    def test_comprando_paga_acima_do_meio(self, modelo):
        assert modelo.preco_efetivo(BOVA, 100.0, comprando=True) > 100.0

    def test_vendendo_recebe_abaixo_do_meio(self, modelo):
        assert modelo.preco_efetivo(BOVA, 100.0, comprando=False) < 100.0

    def test_sem_spread_o_preco_nao_muda(self, modelo):
        sem = Ativo("X", "x", Classe.ACOES_BR, Regime.ETF_RENDA_VARIAVEL,
                    spread_pct=0.0, volume_medio_brl=1e9)
        assert modelo.preco_efetivo(sem, 100.0, True) == pytest.approx(100.0)

    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_ida_e_volta_no_preco_sempre_perde(self, modelo, ativo):
        compra = modelo.preco_efetivo(ativo, 100.0, True)
        venda = modelo.preco_efetivo(ativo, 100.0, False)
        assert venda < compra


class TestTaxaDeAdministracao:
    @pytest.mark.parametrize("ticker", list(TAXA_ADM_ANUAL))
    def test_taxa_diaria_e_positiva(self, modelo, ticker):
        assert modelo.taxa_administracao_diaria(ticker) > 0

    @pytest.mark.parametrize("ticker", list(TAXA_ADM_ANUAL))
    def test_taxa_diaria_vezes_252_bate_com_a_anual(self, modelo, ticker):
        assert modelo.taxa_administracao_diaria(ticker) * 252 == pytest.approx(
            TAXA_ADM_ANUAL[ticker]
        )

    def test_ticker_desconhecido_nao_cobra(self, modelo):
        assert modelo.taxa_administracao_diaria("NAOEXISTE11.SA") == 0.0

    def test_bova_e_mais_barato_que_smal(self):
        assert TAXA_ADM_ANUAL["BOVA11.SA"] < TAXA_ADM_ANUAL["SMAL11.SA"]


class TestCustoAnualDoGiro:
    @pytest.mark.parametrize("giros", [0, 1, 2, 4, 12, 52, 252])
    def test_custo_cresce_com_o_giro(self, modelo, giros):
        r = modelo.custo_anual_do_giro(BOVA, 10_000, giros)
        assert r["custo_anual_brl"] == pytest.approx(r["custo_por_giro_brl"] * giros)

    def test_sem_giro_nao_ha_custo(self, modelo):
        assert modelo.custo_anual_do_giro(BOVA, 10_000, 0)["custo_anual_brl"] == 0.0

    def test_giro_negativo_e_recusado(self, modelo):
        with pytest.raises(ValueError, match="negativo"):
            modelo.custo_anual_do_giro(BOVA, 10_000, -1)

    def test_diario_custa_mais_de_vinte_por_cento_ao_ano(self, modelo):
        """O número que encerra a discussão sobre rebalancear todo dia."""
        r = modelo.custo_anual_do_giro(BOVA, 10_000, 252)
        assert r["custo_anual_pct"] > 20

    def test_anual_custa_menos_de_meio_por_cento(self, modelo):
        r = modelo.custo_anual_do_giro(BOVA, 10_000, 1)
        assert r["custo_anual_pct"] < 0.5

    def test_percentual_independe_do_valor_quando_nao_ha_corretagem(self, modelo):
        pequeno = modelo.custo_anual_do_giro(BOVA, 1_000, 12)["custo_anual_pct"]
        grande = modelo.custo_anual_do_giro(BOVA, 100_000, 12)["custo_anual_pct"]
        assert pequeno == pytest.approx(grande, rel=1e-9)


class TestCompararFrequencias:
    def test_devolve_uma_linha_por_frequencia(self, modelo):
        assert len(comparar_frequencias(BOVA, 10_000, modelo)) == 8

    def test_esta_ordenado_do_mais_caro_para_o_mais_barato(self, modelo):
        linhas = comparar_frequencias(BOVA, 10_000, modelo)
        custos = [linha["custo_anual_brl"] for linha in linhas]
        assert custos == sorted(custos, reverse=True)

    def test_nunca_custa_menos_que_zero(self, modelo):
        for linha in comparar_frequencias(BOVA, 10_000, modelo):
            assert linha["custo_anual_brl"] >= 0

    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_funciona_para_todo_ativo_do_universo(self, modelo, ativo):
        linhas = comparar_frequencias(ativo, 5_000, modelo)
        assert all("custo_anual_pct" in linha for linha in linhas)


class TestComparacaoEntreAtivos:
    @pytest.mark.parametrize("ativo", TODOS, ids=lambda a: a.ticker)
    def test_custo_percentual_e_menor_que_um_por_cento_por_ponta(self, modelo, ativo):
        """Nenhum ativo do universo deveria custar mais de 1% para entrar."""
        assert modelo.custo_ponta_pct(ativo, 10_000) < 0.01

    def test_ordenacao_por_liquidez_bate_com_o_spread(self):
        ordenado = sorted(TODOS, key=lambda a: a.spread_pct)
        assert ordenado[0].ticker == "BOVA11.SA"
