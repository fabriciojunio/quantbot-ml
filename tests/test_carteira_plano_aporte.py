"""
Testes do plano de aporte.

A regra que estes testes mais protegem é a ordem das fases: enquanto a reserva
não fecha, nenhum real vai para renda variável. É a regra que impede a carteira
de ser liquidada no pior momento por causa de um imprevisto.
"""

import pytest

from carteira.plano_aporte import (
    MESES_DE_RESERVA,
    NUCLEO_PADRAO,
    Fase,
    Situacao,
    escolher_por_rodizio,
    montar,
    projetar_fases,
)


def sit(**kw):
    base = dict(aporte_mensal=500.0, despesa_mensal=2_000.0, reserva_atual=0.0, vinculo="pj")
    base.update(kw)
    return Situacao(**base)


class TestReserva:
    @pytest.mark.parametrize("vinculo,meses", list(MESES_DE_RESERVA.items()))
    def test_alvo_depende_do_vinculo(self, vinculo, meses):
        assert sit(vinculo=vinculo).reserva_alvo == 2_000.0 * meses

    def test_pj_precisa_do_dobro_de_clt(self):
        assert sit(vinculo="pj").reserva_alvo == 2 * sit(vinculo="clt").reserva_alvo

    def test_falta_e_a_diferenca(self):
        assert sit(reserva_atual=5_000).reserva_falta == 19_000.0

    def test_reserva_acima_do_alvo_nao_da_falta_negativa(self):
        assert sit(reserva_atual=100_000).reserva_falta == 0.0

    def test_reserva_completa_e_detectada(self):
        assert sit(reserva_atual=24_000).reserva_completa

    def test_meses_para_completar(self):
        assert sit(reserva_atual=0).meses_para_completar == 48

    def test_reserva_pronta_devolve_zero_meses(self):
        assert sit(reserva_atual=24_000).meses_para_completar == 0

    def test_sem_aporte_nunca_completa(self):
        assert sit(aporte_mensal=0).meses_para_completar is None


class TestOrdemDasFases:
    def test_sem_reserva_tudo_vai_para_renda_fixa(self):
        p = montar(sit())
        assert len(p["destinos"]) == 1
        assert p["destinos"][0]["fase"] == Fase.RESERVA.value
        assert p["destinos"][0]["valor"] == 500.0

    def test_com_reserva_pronta_o_aporte_vai_para_o_nucleo(self):
        p = montar(sit(reserva_atual=24_000))
        assert p["destinos"][0]["fase"] == Fase.NUCLEO.value

    def test_mes_de_transicao_divide_o_aporte(self):
        """No mês em que a reserva fecha, a sobra já vai para o núcleo."""
        p = montar(sit(reserva_atual=23_800))
        assert len(p["destinos"]) == 2
        assert p["destinos"][0]["valor"] == pytest.approx(200.0)
        assert p["destinos"][1]["valor"] == pytest.approx(300.0)

    def test_nenhum_real_some(self):
        for reserva in (0, 10_000, 23_800, 24_000, 50_000):
            p = montar(sit(reserva_atual=reserva))
            soma = sum(d["valor"] for d in p["destinos"]) + p["sobrou"]
            assert soma == pytest.approx(500.0, abs=0.01)

    def test_fase_atual_e_declarada(self):
        assert montar(sit())["fase_atual"] == Fase.RESERVA.value
        assert montar(sit(reserva_atual=24_000))["fase_atual"] == Fase.NUCLEO.value

    def test_satelite_so_aparece_quando_pedido(self):
        p = montar(sit(reserva_atual=24_000))
        assert all(d["fase"] != Fase.SATELITE.value for d in p["destinos"])

    def test_satelite_recebe_a_fatia_declarada(self):
        p = montar(sit(reserva_atual=24_000), fatia_satelite=0.20)
        sat = [d for d in p["destinos"] if d["fase"] == Fase.SATELITE.value]
        assert sat and sat[0]["valor"] == pytest.approx(100.0)


class TestRodizio:
    def test_carteira_vazia_comeca_pelo_maior_peso(self):
        escolhido, motivo = escolher_por_rodizio({}, NUCLEO_PADRAO)
        assert escolhido == "IVVB11.SA"
        assert "vazia" in motivo

    def test_escolhe_o_mais_atrasado(self):
        carteira = {"IVVB11.SA": 5_000, "IMAB11.SA": 100, "BOVA11.SA": 100}
        escolhido, _ = escolher_por_rodizio(carteira, NUCLEO_PADRAO)
        assert escolhido != "IVVB11.SA"

    def test_motivo_explica_a_lacuna(self):
        carteira = {"IVVB11.SA": 5_000}
        _, motivo = escolher_por_rodizio(carteira, NUCLEO_PADRAO)
        assert "abaixo do peso alvo" in motivo

    def test_concentra_em_um_ativo_so(self):
        """Com aporte pequeno, uma ordem por mês em vez de cinco."""
        p = montar(sit(reserva_atual=24_000))
        do_nucleo = [d for d in p["destinos"] if d["fase"] == Fase.NUCLEO.value]
        assert len(do_nucleo) == 1

    def test_rodizio_converge_para_o_alvo(self):
        """Comprando o mais atrasado todo mês, a carteira chega no peso alvo."""
        carteira: dict[str, float] = {}
        for _ in range(60):
            escolhido, _ = escolher_por_rodizio(carteira, NUCLEO_PADRAO)
            carteira[escolhido] = carteira.get(escolhido, 0.0) + 500.0

        total = sum(carteira.values())
        for ticker, alvo in NUCLEO_PADRAO.items():
            real = carteira.get(ticker, 0.0) / total
            assert abs(real - alvo) < 0.03  # dentro de 3 pontos percentuais


class TestExposicaoADolar:
    def test_nucleo_padrao_tem_dolar(self):
        assert montar(sit(reserva_atual=24_000))["exposicao_a_dolar_no_nucleo_pct"] > 0

    def test_percentual_bate_com_o_nucleo(self):
        p = montar(sit(reserva_atual=24_000))
        assert p["exposicao_a_dolar_no_nucleo_pct"] == pytest.approx(
            NUCLEO_PADRAO["IVVB11.SA"] * 100
        )

    def test_nucleo_sem_dolar_gera_aviso(self):
        p = montar(sit(reserva_atual=24_000), nucleo={"BOVA11.SA": 1.0})
        assert any("moeda só" in a for a in p["avisos"])

    def test_nucleo_padrao_soma_um(self):
        assert sum(NUCLEO_PADRAO.values()) == pytest.approx(1.0)


class TestAvisos:
    def test_pj_recebe_o_aviso_do_alvo_dobrado(self):
        assert any("FGTS" in a for a in montar(sit(vinculo="pj"))["avisos"])

    def test_clt_nao_recebe_esse_aviso(self):
        assert not any("FGTS" in a for a in montar(sit(vinculo="clt"))["avisos"])

    def test_sem_reserva_avisa_para_nao_ir_para_bolsa(self):
        assert any("renda variável" in a for a in montar(sit())["avisos"])

    def test_despesa_zerada_avisa(self):
        assert any("Despesa mensal" in a for a in montar(sit(despesa_mensal=0))["avisos"])

    def test_aporte_pequeno_avisa_sobre_o_custo(self):
        assert any("rodízio" in a for a in montar(sit(aporte_mensal=200))["avisos"])


class TestProjecao:
    def test_projeta_o_numero_de_meses_pedido(self):
        linhas = projetar_fases(sit(), meses=60)
        assert linhas[-1]["mes"] == 60

    def test_reserva_cresce_ate_o_alvo_e_para(self):
        linhas = projetar_fases(sit(), meses=72)
        assert linhas[-1]["reserva"] == pytest.approx(sit().reserva_alvo)

    def test_nucleo_so_comeca_depois_da_reserva(self):
        linhas = projetar_fases(sit(), meses=60)
        for linha in linhas:
            if linha["fase"] == Fase.RESERVA.value:
                assert linha["nucleo"] == 0.0

    def test_marca_a_fase_de_cada_momento(self):
        fases = {linha["fase"] for linha in projetar_fases(sit(), meses=72)}
        assert Fase.RESERVA.value in fases
        assert Fase.NUCLEO.value in fases

    def test_conta_meses_de_despesa_cobertos(self):
        linhas = projetar_fases(sit(), meses=24)
        assert linhas[-1]["meses_de_despesa"] == pytest.approx(6.0, abs=0.1)
