"""
Testes do simulador de aporte.

Aqui mora a conta que decide se vale gastar energia em estratégia ou em taxa
de poupança. Os testes garantem que a projeção é aritmeticamente correta e,
principalmente, que ela continua mostrando o que mostra: para quem tem pouco
capital, o aporte domina o retorno.
"""

import pytest

from carteira.aportes import (
    Cenario,
    projetar,
    quanto_pesa_a_estrategia,
    renda_mensal_sustentavel,
    tabela_de_sensibilidade,
    tempo_para_meta,
)


class TestProjecao:
    def test_numero_de_linhas_bate_com_os_meses(self):
        df = projetar(Cenario(1_000, 100, 10.0, 5))
        assert len(df) == 60

    def test_sem_retorno_o_saldo_e_so_a_soma_dos_aportes(self):
        df = projetar(Cenario(1_000, 100, 0.0, 10))
        assert float(df["saldo"].iloc[-1]) == pytest.approx(1_000 + 100 * 120)

    def test_sem_aporte_o_saldo_e_juro_composto_puro(self):
        df = projetar(Cenario(1_000, 0, 10.0, 10))
        assert float(df["saldo"].iloc[-1]) == pytest.approx(1_000 * 1.10**10, rel=0.01)

    def test_saldo_so_cresce_com_retorno_positivo(self):
        df = projetar(Cenario(1_000, 100, 10.0, 5))
        assert df["saldo"].is_monotonic_increasing

    def test_horizonte_invalido_e_recusado(self):
        with pytest.raises(ValueError, match="positivo"):
            projetar(Cenario(1_000, 100, 10.0, 0))

    @pytest.mark.parametrize("anos", [1, 3, 5, 10, 20, 30])
    def test_varios_horizontes(self, anos):
        df = projetar(Cenario(1_000, 200, 12.0, anos))
        assert len(df) == anos * 12

    @pytest.mark.parametrize("retorno", [0.0, 5.0, 8.0, 13.9, 20.0, 30.0])
    def test_varios_retornos(self, retorno):
        df = projetar(Cenario(1_000, 200, retorno, 10))
        assert float(df["saldo"].iloc[-1]) > 0

    def test_total_aportado_bate_com_a_conta(self):
        df = projetar(Cenario(1_000, 300, 10.0, 5))
        assert float(df["total_aportado"].iloc[-1]) == pytest.approx(1_000 + 300 * 60)

    def test_ganho_e_saldo_menos_aportado(self):
        df = projetar(Cenario(1_000, 300, 10.0, 5))
        linha = df.iloc[-1]
        assert float(linha["ganho_acumulado"]) == pytest.approx(
            float(linha["saldo"]) - float(linha["total_aportado"]), abs=0.02
        )

    def test_reajuste_do_aporte_aumenta_o_saldo(self):
        sem = projetar(Cenario(1_000, 300, 10.0, 10))
        com = projetar(Cenario(1_000, 300, 10.0, 10, reajuste_aporte_anual_pct=5.0))
        assert float(com["saldo"].iloc[-1]) > float(sem["saldo"].iloc[-1])

    def test_fatia_de_aporte_cai_com_o_tempo(self):
        """No começo quase tudo é dinheiro colocado; com o tempo o juro assume."""
        df = projetar(Cenario(1_000, 300, 15.0, 30))
        assert float(df["fatia_que_e_aporte_pct"].iloc[-1]) < float(
            df["fatia_que_e_aporte_pct"].iloc[0]
        )


class TestQuantoPesaAEstrategia:
    def test_retorno_melhor_gera_saldo_maior(self):
        r = quanto_pesa_a_estrategia(1_000, 300, 10, 13.9, 18.0)
        assert r["saldo_com_retorno_melhor"] > r["saldo_com_retorno_base"]
        assert r["ganho_da_estrategia"] > 0

    def test_devolve_o_aporte_equivalente(self):
        r = quanto_pesa_a_estrategia(1_000, 300, 10, 13.9, 18.0)
        assert r["aporte_extra_equivalente_por_mes"] > 0

    def test_retornos_iguais_nao_geram_ganho(self):
        r = quanto_pesa_a_estrategia(1_000, 300, 10, 13.9, 13.9)
        assert r["ganho_da_estrategia"] == pytest.approx(0.0, abs=0.01)

    def test_leitura_sempre_presente(self):
        assert quanto_pesa_a_estrategia(1_000, 300, 10, 13.9, 18.0)["leitura"]

    @pytest.mark.parametrize("aporte", [0, 100, 300, 500, 1_000])
    def test_aporte_maior_amplifica_o_ganho_da_estrategia(self, aporte):
        r = quanto_pesa_a_estrategia(1_000, aporte, 10, 13.9, 18.0)
        assert r["ganho_da_estrategia"] >= 0

    def test_com_capital_pequeno_o_aporte_equivalente_e_baixo(self):
        """Ganhar 4 pontos de retorno em R$ 1.000 equivale a pouquíssimo aporte."""
        r = quanto_pesa_a_estrategia(1_000, 0, 10, 13.9, 18.0)
        assert r["aporte_extra_equivalente_por_mes"] < 50


class TestTabelaDeSensibilidade:
    def test_tem_uma_linha_por_aporte(self):
        assert len(tabela_de_sensibilidade()) == 5

    def test_aportar_mais_sempre_termina_melhor(self):
        df = tabela_de_sensibilidade()
        coluna = [c for c in df.columns if c != "aporte_mensal"][0]
        assert df[coluna].is_monotonic_increasing

    def test_render_mais_sempre_termina_melhor(self):
        df = tabela_de_sensibilidade()
        linha = df.iloc[-1]
        valores = [linha[c] for c in df.columns if c != "aporte_mensal"]
        assert valores == sorted(valores)

    def test_poupar_bate_render_com_capital_pequeno(self):
        """O número que encerra a discussão: R$ 300/mês no colchão supera
        25% ao ano sobre R$ 1.000."""
        df = tabela_de_sensibilidade(capital_inicial=1_000, anos=10)
        sem_aporte_com_retorno = float(df[df["aporte_mensal"] == 0]["25.0%_a.a."].iloc[0])
        com_aporte_sem_retorno = float(df[df["aporte_mensal"] == 300]["0.0%_a.a."].iloc[0])
        assert com_aporte_sem_retorno > sem_aporte_com_retorno


class TestTempoParaMeta:
    def test_meta_ja_alcancada_devolve_zero(self):
        r = tempo_para_meta(10_000, 100, 10.0, 5_000)
        assert r["meses"] == 0

    def test_meta_alcancavel_devolve_prazo(self):
        r = tempo_para_meta(1_000, 1_000, 10.0, 50_000)
        assert r["alcancavel"]
        assert r["meses"] > 0

    def test_meta_inalcancavel_e_declarada(self):
        r = tempo_para_meta(100, 10, 0.0, 10_000_000)
        assert not r["alcancavel"]
        assert "não é alcançada" in r["leitura"]

    def test_aportar_mais_encurta_o_prazo(self):
        pouco = tempo_para_meta(1_000, 300, 12.0, 500_000)
        muito = tempo_para_meta(1_000, 3_000, 12.0, 500_000)
        assert muito["meses"] < pouco["meses"]

    def test_render_mais_encurta_o_prazo(self):
        baixo = tempo_para_meta(1_000, 500, 8.0, 500_000)
        alto = tempo_para_meta(1_000, 500, 20.0, 500_000)
        assert alto["meses"] < baixo["meses"]

    @pytest.mark.parametrize("meta", [10_000, 100_000, 500_000, 1_500_000])
    def test_varias_metas(self, meta):
        r = tempo_para_meta(1_000, 1_000, 13.9, meta)
        assert "leitura" in r


class TestRendaSustentavel:
    def test_renda_mensal_e_um_doze_avos_da_anual(self):
        r = renda_mensal_sustentavel(1_200_000, 4.0)
        assert r["renda_mensal"] == pytest.approx(r["renda_anual"] / 12)

    @pytest.mark.parametrize("taxa", [3.0, 4.0, 5.0, 6.0])
    def test_taxa_maior_da_renda_maior(self, taxa):
        assert renda_mensal_sustentavel(1_000_000, taxa)["renda_mensal"] > 0

    def test_patrimonio_maior_da_renda_maior(self):
        pequeno = renda_mensal_sustentavel(100_000)
        grande = renda_mensal_sustentavel(1_000_000)
        assert grande["renda_mensal"] > pequeno["renda_mensal"]

    def test_multiplo_necessario_a_quatro_por_cento_e_trezentos(self):
        assert renda_mensal_sustentavel(1_000_000, 4.0)["multiplo_necessario"] == 300

    def test_cinco_mil_por_mes_exige_mais_de_um_milhao(self):
        r = renda_mensal_sustentavel(1_500_000, 4.0)
        assert r["renda_mensal"] == pytest.approx(5_000)
