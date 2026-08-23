"""Testes do modelo de custo.

O custo é o que decide se a estratégia sobrevive, então ele merece teste tão
rigoroso quanto o indicador. Em especial: o alvo de US$ 0,10 do painel antigo
tem que ser reprovado aqui em todos os tamanhos de posição realistas.
"""

import pytest

from cripto.custos import TAXAS_SPOT, ModeloCusto, resumo_viabilidade


class TestTaxas:
    def test_da_exchange_conhecida_usa_a_taxa_dela(self):
        m = ModeloCusto.da_exchange("gateio")
        assert m.taxa_taker == TAXAS_SPOT["gateio"]["taker"]

    def test_exchange_desconhecida_cai_no_padrao_conservador(self):
        m = ModeloCusto.da_exchange("corretora_que_nao_existe")
        assert m.taxa_taker == TAXAS_SPOT["binance"]["taker"]


class TestSlippage:
    def test_ordem_zerada_nao_tem_slippage(self):
        assert ModeloCusto().slippage_pct(0) == 0.0

    def test_slippage_cresce_com_o_tamanho(self):
        m = ModeloCusto()
        assert m.slippage_pct(10_000) > m.slippage_pct(1_000)

    def test_slippage_cresce_menos_que_proporcional(self):
        """Dobrar o tamanho não dobra o impacto: a forma é de raiz."""
        m = ModeloCusto()
        assert m.slippage_pct(2_000) < 2 * m.slippage_pct(1_000)


class TestCustoPonta:
    def test_taker_custa_mais_que_maker_quando_as_taxas_diferem(self):
        m = ModeloCusto(taxa_maker=0.0002, taxa_taker=0.0010)
        assert m.custo_ponta_pct(1_000, taker=True) > m.custo_ponta_pct(1_000, taker=False)

    def test_ida_e_volta_e_a_soma_das_duas_pontas(self):
        m = ModeloCusto()
        assert m.custo_ida_volta_pct(1_000) == pytest.approx(2 * m.custo_ponta_pct(1_000))

    def test_custo_em_dolar_bate_com_o_percentual(self):
        m = ModeloCusto()
        assert m.custo_ida_volta_usd(1_000) == pytest.approx(
            1_000 * m.custo_ida_volta_pct(1_000)
        )


class TestPrecoEfetivo:
    def test_comprando_paga_acima_do_meio(self):
        m = ModeloCusto()
        assert m.preco_efetivo(100.0, 1_000, comprando=True) > 100.0

    def test_vendendo_recebe_abaixo_do_meio(self):
        m = ModeloCusto()
        assert m.preco_efetivo(100.0, 1_000, comprando=False) < 100.0

    def test_sem_spread_nem_impacto_o_preco_nao_muda(self):
        m = ModeloCusto(spread_pct=0.0, impacto_base=0.0)
        assert m.preco_efetivo(100.0, 1_000, comprando=True) == pytest.approx(100.0)


class TestAcertoMinimo:
    def test_alvo_e_stop_simetricos_pedem_mais_que_metade(self):
        """Com custo, empatar exige acertar mais que 50% — não é 50/50."""
        m = ModeloCusto()
        assert m.acerto_minimo(0.02, 0.02, 1_000) > 0.5

    def test_alvo_menor_que_o_custo_e_impossivel(self):
        m = ModeloCusto()
        # Alvo de 0,01% com custo de ida e volta de ~0,23%: nem 100% empata.
        assert m.acerto_minimo(0.0001, 0.0001, 1_000) == float("inf")

    def test_payoff_alto_derruba_o_acerto_necessario(self):
        m = ModeloCusto()
        exigente = m.acerto_minimo(0.02, 0.02, 1_000)
        folgado = m.acerto_minimo(0.06, 0.02, 1_000)
        assert folgado < exigente


class TestAlvoDeCentavos:
    """A regra que o painel antigo usava: fechar em +US$ 0,10."""

    @pytest.mark.parametrize("notional", [50, 100, 200, 500, 1_000, 3_000])
    def test_alvo_de_dez_centavos_nao_paga_o_proprio_custo(self, notional):
        r = resumo_viabilidade(alvo_usd=0.10, notional_usd=notional)
        assert r["viavel"] is False
        assert r["resultado_liquido_usd"] < 0

    def test_quanto_maior_a_posicao_pior_fica(self):
        pequena = resumo_viabilidade(0.10, 100)
        grande = resumo_viabilidade(0.10, 3_000)
        assert grande["custo_usd"] > pequena["custo_usd"]
        assert grande["quantas_vezes_o_alvo_o_custo_come"] > pequena["quantas_vezes_o_alvo_o_custo_come"]

    def test_alvo_grande_o_bastante_passa(self):
        """O modelo não é pessimista por decreto: alvo suficiente é aprovado."""
        r = resumo_viabilidade(alvo_usd=50.0, notional_usd=1_000)
        assert r["viavel"] is True
