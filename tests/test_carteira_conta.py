"""
Testes da contabilidade.

Esta é a classe que o backtest e a operação diária compartilham, então um erro
aqui aparece nos dois lugares ao mesmo tempo — e, pior, aparece igual nos dois,
o que faz a comparação entre eles parecer coerente. Por isso a maior parte
destes testes confere invariante de dinheiro: o que entrou menos o que saiu tem
que bater, sempre.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from carteira.conta import Conta, Posicao, aliquota_de_venda, ativo_generico
from carteira.custos_b3 import ModeloCustoB3
from carteira.universo import UNIVERSO_B3, Regime

SEM_SAUDE = settings(
    max_examples=100, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

SEM_CUSTO = ModeloCustoB3(emolumentos_pct=0.0, corretagem_brl=0.0)


def conta_sem_custo(caixa=10_000.0) -> Conta:
    ativos = {
        t: type(a)(**{**a.__dict__, "spread_pct": 0.0})
        for t, a in UNIVERSO_B3.items()
    }
    return Conta(caixa=caixa, modelo=SEM_CUSTO, ativos=ativos)


# ─────────────────────────────────────────────────────────────────────────────
# Posição
# ─────────────────────────────────────────────────────────────────────────────

class TestePosicao:
    def test_custo_medio_de_posicao_vazia_e_zero(self):
        assert Posicao().custo_medio == 0.0

    def test_custo_medio_divide_pela_quantidade(self):
        assert Posicao(cotas=10, custo_total=1000).custo_medio == pytest.approx(100.0)

    def test_dias_medios_de_posicao_vazia(self):
        assert Posicao().dias_medios(100) == 0.0

    def test_dias_medios_com_uma_compra(self):
        p = Posicao(cotas=10, custo_total=1000, idade_ponderada=10 * 30)
        assert p.dias_medios(100) == pytest.approx(70.0)

    def test_dias_medios_com_duas_compras(self):
        # 10 cotas no dia 0, 10 cotas no dia 100. Idade média no dia 100: 50.
        p = Posicao(cotas=20, custo_total=2000, idade_ponderada=10 * 0 + 10 * 100)
        assert p.dias_medios(100) == pytest.approx(50.0)

    def test_nunca_devolve_idade_negativa(self):
        p = Posicao(cotas=10, custo_total=100, idade_ponderada=10 * 500)
        assert p.dias_medios(100) == 0.0

    def test_ida_e_volta_por_dicionario(self):
        p = Posicao(cotas=3.5, custo_total=350.0, idade_ponderada=17.5)
        voltou = Posicao.de_dict(p.para_dict())
        assert voltou.cotas == pytest.approx(p.cotas)
        assert voltou.custo_total == pytest.approx(p.custo_total)


# ─────────────────────────────────────────────────────────────────────────────
# Compra
# ─────────────────────────────────────────────────────────────────────────────

class TesteCompra:
    def test_compra_tira_do_caixa_e_soma_em_cota(self):
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 500, preco=100.0, dia=0)
        assert c.caixa == pytest.approx(500.0)
        assert c.posicoes["BOVA11.SA"].cotas == pytest.approx(5.0)

    def test_custo_reduz_as_cotas_compradas(self):
        c = Conta(caixa=1000, modelo=ModeloCustoB3(corretagem_brl=10.0))
        c.comprar("BOVA11.SA", 500, preco=100.0, dia=0)
        assert c.posicoes["BOVA11.SA"].cotas < 5.0
        assert c.custo_pago >= 10.0

    def test_custo_de_entrada_entra_na_base(self):
        """É dedutível na venda: sem isso o imposto sai maior que o devido."""
        c = Conta(caixa=1000, modelo=ModeloCustoB3(corretagem_brl=10.0))
        c.comprar("BOVA11.SA", 500, preco=100.0, dia=0)
        assert c.posicoes["BOVA11.SA"].custo_total == pytest.approx(500.0)

    def test_nao_compra_mais_do_que_o_caixa(self):
        c = conta_sem_custo(100)
        with pytest.raises(ValueError, match="não cabe no caixa"):
            c.comprar("BOVA11.SA", 500, preco=100.0, dia=0)

    def test_valor_zero_nao_faz_nada(self):
        c = conta_sem_custo()
        assert c.comprar("BOVA11.SA", 0, preco=100.0, dia=0) is None
        assert not c.posicoes

    def test_preco_invalido_e_recusado(self):
        c = conta_sem_custo()
        with pytest.raises(ValueError, match="preço inválido"):
            c.comprar("BOVA11.SA", 100, preco=0.0, dia=0)

    def test_ordem_menor_que_o_proprio_custo_e_ignorada(self):
        c = Conta(caixa=1000, modelo=ModeloCustoB3(corretagem_brl=50.0))
        assert c.comprar("BOVA11.SA", 10, preco=100.0, dia=0) is None

    def test_duas_compras_acumulam(self):
        c = conta_sem_custo(2000)
        c.comprar("BOVA11.SA", 500, preco=100.0, dia=0)
        c.comprar("BOVA11.SA", 500, preco=50.0, dia=10)
        p = c.posicoes["BOVA11.SA"]
        assert p.cotas == pytest.approx(15.0)
        assert p.custo_medio == pytest.approx(1000 / 15)

    def test_registra_o_movimento_com_motivo(self):
        c = conta_sem_custo()
        mov = c.comprar("BOVA11.SA", 500, preco=100.0, dia=3, motivo="rodízio")
        assert mov.motivo == "rodízio"
        assert c.movimentos[-1] is mov


# ─────────────────────────────────────────────────────────────────────────────
# Venda
# ─────────────────────────────────────────────────────────────────────────────

class TesteVenda:
    def test_venda_devolve_ao_caixa(self):
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        c.vender("BOVA11.SA", 500, preco=100.0, dia=1)
        assert c.caixa == pytest.approx(500.0)

    def test_venda_com_lucro_paga_imposto(self):
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        c.vender("BOVA11.SA", 2000, preco=200.0, dia=400)
        # Ganho de 1000, alíquota 15%.
        assert c.imposto_pago == pytest.approx(150.0)

    def test_venda_com_prejuizo_nao_paga_imposto(self):
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        c.vender("BOVA11.SA", 500, preco=50.0, dia=400)
        assert c.imposto_pago == pytest.approx(0.0)

    def test_prejuizo_compensa_lucro_da_mesma_modalidade(self):
        c = conta_sem_custo(4000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        c.comprar("SMAL11.SA", 1000, preco=100.0, dia=0)
        c.vender("BOVA11.SA", 500, preco=50.0, dia=100)     # prejuízo de 500
        c.vender("SMAL11.SA", 1500, preco=150.0, dia=100)   # lucro de 500
        assert c.imposto_pago == pytest.approx(0.0)

    def test_prejuizo_nao_compensa_modalidade_diferente(self):
        """Renda variável e renda fixa são modalidades separadas."""
        c = conta_sem_custo(4000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        c.comprar("IMAB11.SA", 1000, preco=100.0, dia=0)
        c.vender("BOVA11.SA", 500, preco=50.0, dia=100)      # prejuízo em RV
        c.vender("IMAB11.SA", 1500, preco=150.0, dia=100)    # lucro em RF
        assert c.imposto_pago > 0

    def test_venda_parcial_reduz_a_base_proporcionalmente(self):
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        c.vender("BOVA11.SA", 500, preco=100.0, dia=1)
        p = c.posicoes["BOVA11.SA"]
        assert p.cotas == pytest.approx(5.0)
        assert p.custo_total == pytest.approx(500.0)
        assert p.custo_medio == pytest.approx(100.0)

    def test_venda_total_remove_a_posicao(self):
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        c.vender("BOVA11.SA", 1000, preco=100.0, dia=1)
        assert "BOVA11.SA" not in c.posicoes

    def test_nao_vende_mais_do_que_tem(self):
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 500, preco=100.0, dia=0)
        c.vender("BOVA11.SA", 99_999, preco=100.0, dia=1)
        assert "BOVA11.SA" not in c.posicoes
        assert c.caixa == pytest.approx(1000.0)

    def test_vender_o_que_nao_tem_nao_faz_nada(self):
        c = conta_sem_custo()
        assert c.vender("BOVA11.SA", 500, preco=100.0, dia=0) is None

    def test_venda_de_renda_fixa_usa_tabela_regressiva(self):
        curto = conta_sem_custo(2000)
        curto.comprar("IMAB11.SA", 1000, preco=100.0, dia=0)
        curto.vender("IMAB11.SA", 2000, preco=200.0, dia=100)

        longo = conta_sem_custo(2000)
        longo.comprar("IMAB11.SA", 1000, preco=100.0, dia=0)
        longo.vender("IMAB11.SA", 2000, preco=200.0, dia=800)

        assert curto.imposto_pago > longo.imposto_pago


# ─────────────────────────────────────────────────────────────────────────────
# Invariantes
# ─────────────────────────────────────────────────────────────────────────────

class TesteInvariantes:
    def test_liquidar_zera_as_posicoes(self):
        c = conta_sem_custo(3000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        c.comprar("IVVB11.SA", 1000, preco=200.0, dia=0)
        final = c.liquidar({"BOVA11.SA": 100.0, "IVVB11.SA": 200.0}, dia=10)
        assert not c.posicoes
        assert final == pytest.approx(3000.0)

    def test_liquidar_sem_preco_falha(self):
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 500, preco=100.0, dia=0)
        with pytest.raises(ValueError, match="sem preço"):
            c.liquidar({}, dia=1)

    def test_pesos_somam_um(self):
        c = conta_sem_custo(2000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        assert sum(c.pesos({"BOVA11.SA": 100.0}).values()) == pytest.approx(1.0)

    def test_pesos_de_conta_vazia(self):
        assert Conta().pesos({}) == {}

    def test_conferir_reprova_caixa_negativo(self):
        c = Conta(caixa=-10.0)
        with pytest.raises(ValueError, match="alavancagem"):
            c.conferir({})

    def test_conferir_reprova_posicao_sem_preco(self):
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 500, preco=100.0, dia=0)
        with pytest.raises(ValueError, match="sem preço"):
            c.conferir({})

    def test_conferir_passa_em_conta_saudavel(self):
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 500, preco=100.0, dia=0)
        c.conferir({"BOVA11.SA": 100.0})

    def test_aporte_negativo_e_recusado(self):
        with pytest.raises(ValueError, match="negativo"):
            Conta().aportar(-1)

    def test_render_caixa_devolve_o_rendimento(self):
        c = Conta(caixa=1000)
        assert c.render_caixa(0.001) == pytest.approx(1.0)
        assert c.caixa == pytest.approx(1001.0)

    @SEM_SAUDE
    @given(
        valores=st.lists(st.floats(150, 3000), min_size=1, max_size=12),
        preco=st.floats(10, 500),
    )
    def test_dinheiro_nao_some_nem_aparece(self, valores, preco):
        """Aportado = caixa + valor investido + custo + imposto, sempre."""
        c = conta_sem_custo(0.0)
        for v in valores:
            c.aportar(v)
            c.comprar("BOVA11.SA", min(v, c.caixa), preco=preco, dia=0)

        investido = c.valor_investido({"BOVA11.SA": preco})
        total = c.caixa + investido + c.custo_pago + c.imposto_pago
        assert total == pytest.approx(c.total_aportado, rel=1e-6)

    @SEM_SAUDE
    @given(preco_final=st.floats(1, 1000))
    def test_comprar_e_liquidar_sem_custo_devolve_o_valor_de_mercado(self, preco_final):
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        cotas = c.posicoes["BOVA11.SA"].cotas
        final = c.liquidar({"BOVA11.SA": preco_final}, dia=400)

        bruto = cotas * preco_final
        imposto = max(0.0, bruto - 1000) * 0.15
        assert final == pytest.approx(bruto - imposto, rel=1e-6)

    @SEM_SAUDE
    @given(n=st.integers(1, 20))
    def test_caixa_nunca_fica_negativo_comprando_o_disponivel(self, n):
        c = conta_sem_custo(0.0)
        for i in range(n):
            c.aportar(500)
            c.comprar("BOVA11.SA", c.caixa, preco=100.0 + i, dia=i)
        assert c.caixa >= -1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Persistência
# ─────────────────────────────────────────────────────────────────────────────

class TestePersistencia:
    def test_ida_e_volta_preserva_tudo(self):
        c = conta_sem_custo(2000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        c.vender("BOVA11.SA", 200, preco=50.0, dia=10)

        voltou = Conta.de_dict(c.para_dict())
        assert voltou.caixa == pytest.approx(c.caixa)
        assert voltou.total_aportado == pytest.approx(c.total_aportado)
        assert voltou.imposto_pago == pytest.approx(c.imposto_pago)
        assert voltou.prejuizo_a_compensar == c.prejuizo_a_compensar
        assert set(voltou.posicoes) == set(c.posicoes)

    def test_prejuizo_acumulado_sobrevive_a_retomada(self):
        """Sem isso, reiniciar o robô apagaria prejuízo compensável."""
        c = conta_sem_custo(1000)
        c.comprar("BOVA11.SA", 1000, preco=100.0, dia=0)
        c.vender("BOVA11.SA", 500, preco=50.0, dia=10)
        assert c.prejuizo_a_compensar["etf_rv"] > 0

        voltou = Conta.de_dict(c.para_dict())
        assert voltou.prejuizo_a_compensar["etf_rv"] == pytest.approx(
            c.prejuizo_a_compensar["etf_rv"]
        )

    def test_conta_vazia_ida_e_volta(self):
        voltou = Conta.de_dict(Conta().para_dict())
        assert voltou.caixa == 0.0
        assert voltou.posicoes == {}


# ─────────────────────────────────────────────────────────────────────────────
# Regimes
# ─────────────────────────────────────────────────────────────────────────────

class TesteRegimes:
    def test_etf_de_renda_variavel_paga_quinze(self):
        assert aliquota_de_venda(UNIVERSO_B3["BOVA11.SA"], 5000) == pytest.approx(0.15)

    def test_fii_paga_vinte(self):
        from carteira.universo import Ativo, Classe

        fii = Ativo("X11.SA", "x", Classe.IMOBILIARIO, Regime.FII, 0.001, 1e6)
        assert aliquota_de_venda(fii, 5000) == pytest.approx(0.20)

    def test_isento_nao_paga(self):
        from carteira.universo import Ativo, Classe

        lci = Ativo("LCI", "x", Classe.RENDA_FIXA_POS, Regime.ISENTO, 0.0, 1e9)
        assert aliquota_de_venda(lci, 100) == pytest.approx(0.0)

    @pytest.mark.parametrize("dias,esperado", [(100, 0.225), (200, 0.20), (500, 0.175), (800, 0.15)])
    def test_tabela_regressiva_por_prazo(self, dias, esperado):
        assert aliquota_de_venda(UNIVERSO_B3["IMAB11.SA"], dias) == pytest.approx(esperado)

    def test_ativo_generico_e_conservador(self):
        a = ativo_generico("NOVO11.SA")
        assert a.spread_pct >= UNIVERSO_B3["BOVA11.SA"].spread_pct
        assert a.regime is Regime.ETF_RENDA_VARIAVEL

    def test_modalidade_separa_renda_fixa_de_variavel(self):
        c = Conta()
        assert c.modalidade("BOVA11.SA") != c.modalidade("IMAB11.SA")
