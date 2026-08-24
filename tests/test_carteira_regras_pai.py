"""
Testes da camada de risco herdada do painel Sentinela.

As regras de gestão eram a melhor parte do sistema do painel, e são a parte
que não depende de acertar previsão. Estes testes garantem que elas foram
portadas com a lógica intacta: convicção adaptativa, risco fixo por operação,
trailing em dois estágios e guarda de regime.
"""

import pytest

from carteira.regras_pai import (
    Avaliacao,
    Decisao,
    LimitesRisco,
    PoliticaReversao,
    PortaoDeRisco,
    Posicao,
    guarda_de_regime,
)


@pytest.fixture
def portao():
    return PortaoDeRisco()


def pos(ticker="A", entrada=100.0, atual=100.0, stop=95.0, atr=5.0, pregoes=0, estagio=0):
    return Posicao(
        ticker=ticker, preco_entrada=entrada, preco_atual=atual,
        quantidade=10.0, stop_atual=stop, atr=atr,
        pregoes_aberta=pregoes, estagio_trailing=estagio,
    )


class TestConviccaoAdaptativa:
    """O filtro sniper do painel: cada posição aberta encarece a próxima."""

    @pytest.mark.parametrize(
        "abertas,esperado", [(0, 55.0), (1, 58.5), (2, 62.0), (3, 65.5)]
    )
    def test_sobe_a_cada_posicao(self, portao, abertas, esperado):
        assert portao.conviccao_exigida(abertas) == pytest.approx(esperado)

    def test_respeita_o_teto(self, portao):
        assert portao.conviccao_exigida(100) == portao.limites.conviccao_teto

    def test_e_monotonica(self, portao):
        valores = [portao.conviccao_exigida(n) for n in range(10)]
        assert valores == sorted(valores)

    def test_configuravel(self):
        p = PortaoDeRisco(LimitesRisco(conviccao_base=70.0, conviccao_incremento=10.0))
        assert p.conviccao_exigida(1) == pytest.approx(75.0)  # bateu no teto


class TestTamanhoPorRisco:
    def test_stop_mais_longe_da_posicao_menor(self, portao):
        perto = portao.tamanho_por_risco(10_000, 100.0, 98.0)
        longe = portao.tamanho_por_risco(10_000, 100.0, 90.0)
        assert longe < perto

    def test_respeita_o_teto_de_concentracao(self, portao):
        # Stop colado no preço pediria posição gigantesca.
        tamanho = portao.tamanho_por_risco(10_000, 100.0, 99.99)
        assert tamanho <= 10_000 * portao.limites.max_por_posicao_pct / 100 + 1e-6

    def test_risco_declarado_bate_com_a_perda_no_stop(self, portao):
        """Se o stop for atingido, a perda tem que ser os 2% prometidos.

        Stop a 10% do preço: o tamanho por risco dá R$ 2.000, abaixo do teto de
        concentração de 30%, então quem manda é a regra de risco.
        """
        patrimonio, preco, stop = 10_000.0, 100.0, 90.0
        valor = portao.tamanho_por_risco(patrimonio, preco, stop)
        perda = (valor / preco) * (preco - stop)
        assert perda == pytest.approx(patrimonio * 0.02, rel=1e-6)

    def test_quando_o_teto_de_concentracao_morde_a_perda_fica_menor(self, portao):
        """Stop curto pediria posição acima de 30% do patrimônio. O teto corta,
        e o resultado é arriscar MENOS que os 2% — nunca mais."""
        patrimonio, preco, stop = 10_000.0, 100.0, 95.0
        valor = portao.tamanho_por_risco(patrimonio, preco, stop)
        perda = (valor / preco) * (preco - stop)
        assert valor == pytest.approx(patrimonio * 0.30)
        assert perda < patrimonio * 0.02

    @pytest.mark.parametrize("preco,stop", [(0, 95), (-10, 5), (100, 100)])
    def test_entradas_invalidas_devolvem_zero(self, portao, preco, stop):
        assert portao.tamanho_por_risco(10_000, preco, stop) == 0.0

    def test_patrimonio_zerado_devolve_zero(self, portao):
        assert portao.tamanho_por_risco(0, 100.0, 95.0) == 0.0

    @pytest.mark.parametrize("risco_pct", [0.5, 1.0, 2.0, 5.0])
    def test_risco_maior_da_posicao_maior(self, risco_pct):
        p = PortaoDeRisco(LimitesRisco(risco_por_operacao_pct=risco_pct, max_por_posicao_pct=100))
        assert p.tamanho_por_risco(10_000, 100.0, 95.0) == pytest.approx(
            10_000 * risco_pct / 100 * 100 / 5
        )


class TestAvaliarEntrada:
    def test_ordem_boa_e_aprovada(self, portao):
        a = portao.avaliar_entrada("A", 10_000, 10_000, 100.0, 95.0, 60.0, {})
        assert a.decisao == Decisao.APROVADA
        assert a.valor_aprovado > 0

    def test_conviccao_baixa_bloqueia(self, portao):
        a = portao.avaliar_entrada("A", 10_000, 10_000, 100.0, 95.0, 50.0, {})
        assert a.decisao == Decisao.BLOQUEADA
        assert "convicção" in a.motivos[0]

    def test_conviccao_que_bastava_para_a_primeira_nao_basta_para_a_terceira(self, portao):
        aberta = {"X": pos("X"), "Y": pos("Y")}
        a = portao.avaliar_entrada("A", 10_000, 10_000, 100.0, 95.0, 56.0, aberta)
        assert a.decisao == Decisao.BLOQUEADA

    def test_posicao_duplicada_e_bloqueada(self, portao):
        a = portao.avaliar_entrada("A", 10_000, 10_000, 100.0, 95.0, 80.0, {"A": pos("A")})
        assert a.decisao == Decisao.BLOQUEADA
        assert "já existe" in a.motivos[0]

    def test_limite_de_posicoes_bloqueia(self, portao):
        abertas = {t: pos(t) for t in ("X", "Y", "Z")}
        a = portao.avaliar_entrada("A", 10_000, 10_000, 100.0, 95.0, 90.0, abertas)
        assert a.decisao == Decisao.BLOQUEADA
        assert "limite" in a.motivos[0]

    def test_regime_desfavoravel_bloqueia(self, portao):
        a = portao.avaliar_entrada(
            "A", 10_000, 10_000, 100.0, 95.0, 90.0, {}, regime_favoravel=False
        )
        assert a.decisao == Decisao.BLOQUEADA
        assert "regime" in a.motivos[0]

    def test_sem_dupla_chancela_bloqueia(self, portao):
        a = portao.avaliar_entrada(
            "A", 10_000, 10_000, 100.0, 95.0, 90.0, {}, confirmacao_secundaria=False
        )
        assert a.decisao == Decisao.BLOQUEADA
        assert "dupla chancela" in a.motivos[0]

    def test_dupla_chancela_pode_ser_desligada(self):
        p = PortaoDeRisco(LimitesRisco(exigir_dupla_chancela=False))
        a = p.avaliar_entrada(
            "A", 10_000, 10_000, 100.0, 95.0, 90.0, {}, confirmacao_secundaria=False
        )
        assert a.decisao == Decisao.APROVADA

    def test_stop_acima_do_preco_e_bloqueado(self, portao):
        a = portao.avaliar_entrada("A", 10_000, 10_000, 100.0, 105.0, 90.0, {})
        assert a.decisao == Decisao.BLOQUEADA
        assert "stop" in a.motivos[0]

    def test_caixa_curto_reduz_em_vez_de_bloquear(self, portao):
        a = portao.avaliar_entrada("A", 10_000, 500.0, 100.0, 95.0, 90.0, {})
        assert a.decisao == Decisao.REDUZIDA
        assert a.valor_aprovado == pytest.approx(500.0)

    def test_sem_caixa_bloqueia(self, portao):
        a = portao.avaliar_entrada("A", 10_000, 0.5, 100.0, 95.0, 90.0, {})
        assert a.decisao == Decisao.BLOQUEADA

    def test_avaliacao_sempre_traz_motivo(self, portao):
        for conviccao in (10, 50, 60, 90):
            a = portao.avaliar_entrada("A", 10_000, 10_000, 100.0, 95.0, conviccao, {})
            assert a.motivos

    def test_avaliacao_informa_a_conviccao_exigida(self, portao):
        a = portao.avaliar_entrada("A", 10_000, 10_000, 100.0, 95.0, 60.0, {})
        assert a.conviccao_exigida == 55.0


class TestTrailing:
    def test_sem_lucro_o_stop_nao_muda(self, portao):
        p = pos(atual=100.0, stop=95.0)
        novo, nota = portao.atualizar_trailing(p)
        assert novo == 95.0
        assert nota is None

    def test_estagio_um_trava_acima_da_entrada(self, portao):
        p = pos(entrada=100.0, atual=105.0, stop=95.0, atr=5.0)  # lucro de 1 ATR
        novo, nota = portao.atualizar_trailing(p)
        assert novo > 100.0
        assert p.estagio_trailing == 1
        assert "estágio 1" in nota

    def test_estagio_dois_segue_de_perto(self, portao):
        p = pos(entrada=100.0, atual=115.0, stop=95.0, atr=5.0)  # lucro de 3 ATR
        novo, nota = portao.atualizar_trailing(p)
        assert novo == pytest.approx(115.0 - 1.5 * 5.0)
        assert p.estagio_trailing == 2
        assert "estágio 2" in nota

    def test_no_estagio_dois_o_stop_acompanha_a_alta(self, portao):
        p = pos(entrada=100.0, atual=115.0, stop=95.0, atr=5.0)
        portao.atualizar_trailing(p)
        p.preco_atual = 125.0
        novo, _ = portao.atualizar_trailing(p)
        assert novo == pytest.approx(125.0 - 1.5 * 5.0)

    def test_o_stop_nunca_anda_para_tras(self, portao):
        p = pos(entrada=100.0, atual=115.0, stop=95.0, atr=5.0)
        portao.atualizar_trailing(p)
        travado = p.stop_atual
        p.preco_atual = 101.0  # preço caiu
        novo, _ = portao.atualizar_trailing(p)
        assert novo >= travado

    def test_atr_zerado_nao_quebra(self, portao):
        p = pos(atr=0.0)
        novo, _ = portao.atualizar_trailing(p)
        assert novo == p.stop_atual

    @pytest.mark.parametrize("lucro_atr", [0.0, 0.5, 0.99])
    def test_abaixo_do_gatilho_nao_dispara(self, portao, lucro_atr):
        p = pos(entrada=100.0, atual=100.0 + lucro_atr * 5.0, stop=95.0, atr=5.0)
        _, nota = portao.atualizar_trailing(p)
        assert nota is None


class TestAvaliarSaida:
    def test_stop_atingido_fecha(self, portao):
        p = pos(atual=94.0, stop=95.0)
        sai, motivo = portao.avaliar_saida(p, True, True)
        assert sai and "stop" in motivo

    def test_prazo_maximo_fecha(self, portao):
        p = pos(pregoes=300)
        sai, motivo = portao.avaliar_saida(p, True, True)
        assert sai and "prazo" in motivo

    def test_sinal_invalido_fecha(self, portao):
        sai, motivo = portao.avaliar_saida(pos(), False, True)
        assert sai and "sinal" in motivo

    def test_posicao_saudavel_fica(self, portao):
        sai, motivo = portao.avaliar_saida(pos(atual=110.0), True, True)
        assert not sai and "mantida" in motivo

    def test_politica_fechar_zera_no_regime_ruim(self):
        p = PortaoDeRisco(LimitesRisco(politica_reversao=PoliticaReversao.FECHAR))
        sai, motivo = p.avaliar_saida(pos(atual=110.0), True, False)
        assert sai and "regime" in motivo

    def test_politica_apertar_traz_o_stop_para_a_entrada(self):
        p = PortaoDeRisco(LimitesRisco(politica_reversao=PoliticaReversao.APERTAR_STOP))
        posicao = pos(entrada=100.0, atual=110.0, stop=95.0)
        sai, _ = p.avaliar_saida(posicao, True, False)
        assert not sai
        assert posicao.stop_atual == 100.0

    def test_politica_so_avisar_nao_mexe_em_nada(self):
        p = PortaoDeRisco(LimitesRisco(politica_reversao=PoliticaReversao.SO_AVISAR))
        posicao = pos(entrada=100.0, atual=110.0, stop=95.0)
        sai, motivo = p.avaliar_saida(posicao, True, False)
        assert not sai
        assert posicao.stop_atual == 95.0
        assert "registrado" in motivo


class TestGuardaDeRegime:
    def test_indice_batendo_cdi_e_acima_da_media_libera(self):
        favoravel, d = guarda_de_regime(0.20, 0.14, True)
        assert favoravel
        assert d["indice_bateu_cdi_12m"]

    def test_indice_abaixo_da_media_bloqueia(self):
        favoravel, _ = guarda_de_regime(0.20, 0.14, False)
        assert not favoravel

    def test_subir_menos_que_o_cdi_conta_como_desfavoravel(self):
        """O ponto brasileiro: 8% ao ano com CDI a 14% não é mercado de alta."""
        _, d = guarda_de_regime(0.08, 0.14, True)
        assert d["indice_bateu_cdi_12m"] is False
        assert d["excesso_sobre_cdi_pp"] < 0

    def test_peso_da_tendencia_e_configuravel(self):
        # Com peso zero na tendência, só o excesso sobre CDI decide.
        favoravel, _ = guarda_de_regime(0.20, 0.14, False, peso_tendencia=0.0)
        assert favoravel

    @pytest.mark.parametrize("peso", [0.0, 0.25, 0.5, 0.86, 1.0])
    def test_score_fica_entre_zero_e_cem(self, peso):
        _, d = guarda_de_regime(0.20, 0.14, True, peso_tendencia=peso)
        assert 0.0 <= d["score"] <= 100.0

    def test_leitura_sempre_presente(self):
        for excesso in (-0.10, 0.0, 0.10):
            _, d = guarda_de_regime(0.14 + excesso, 0.14, True)
            assert d["leitura"]


class TestPosicao:
    def test_valor_e_quantidade_vezes_preco(self):
        assert pos(atual=110.0).valor == pytest.approx(1_100.0)

    @pytest.mark.parametrize("atual,esperado", [(100.0, 0.0), (105.0, 1.0), (115.0, 3.0)])
    def test_lucro_em_atr(self, atual, esperado):
        assert pos(entrada=100.0, atual=atual, atr=5.0).lucro_em_atr == pytest.approx(esperado)

    def test_atr_zerado_da_lucro_zero(self):
        assert pos(atr=0.0).lucro_em_atr == 0.0
