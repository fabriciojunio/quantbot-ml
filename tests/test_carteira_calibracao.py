"""
Testes da calibração walk-forward.

O teste mais importante deste arquivo é o que exige que a calibração **reprove**
o aprendizado quando os dados são ruído puro. Um calibrador que sempre acha algo
para aprender é um gerador de overfitting com aparência de ciência, e a única
forma de saber de que lado o nosso está é dar ruído para ele e conferir que ele
diz não.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auditoria import TipoEvento, abrir, verificar
from carteira import calibracao as cal


def indice(anos: float, inicio: str = "2000-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(inicio, periods=int(anos * 252))


def precos_ruido(idx, semente=1, tickers=("BOVA11.SA", "IVVB11.SA")):
    rng = np.random.default_rng(semente)
    return pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(idx)))) for t in tickers},
        index=idx,
    )


def cdi_plano(idx, taxa=0.0004):
    return pd.Series(np.full(len(idx), taxa), index=idx)


def aplicar_pesos(p):
    """Converte {'peso_bova': x} nos argumentos da simulação."""
    x = p.get("peso_bova", 0.5)
    return {
        "pesos": {"BOVA11.SA": x, "IVVB11.SA": round(1 - x, 10)},
        "rebalancear_a_cada_meses": p.get("rebalanceamento", 12),
    }


GRADES = [
    cal.Grade("peso_bova", (0.3, 0.5, 0.7)),
    cal.Grade("rebalanceamento", (6, 12)),
]
PADRAO = {"peso_bova": 0.5, "rebalanceamento": 12}


# ─────────────────────────────────────────────────────────────────────────────
# Grades
# ─────────────────────────────────────────────────────────────────────────────

class TesteGrade:
    def test_grade_vazia_e_recusada(self):
        with pytest.raises(ValueError, match="vazia"):
            cal.Grade("x", ())

    def test_grade_com_repetido_e_recusada(self):
        with pytest.raises(ValueError, match="repetidos"):
            cal.Grade("x", (1, 2, 1))

    def test_combinacoes_fazem_o_produto(self):
        c = cal.combinacoes([cal.Grade("a", (1, 2)), cal.Grade("b", ("x", "y", "z"))])
        assert len(c) == 6
        assert {"a": 1, "b": "x"} in c

    def test_sem_grade_da_uma_combinacao_vazia(self):
        assert cal.combinacoes([]) == [{}]

    def test_uma_grade_de_um_valor(self):
        assert cal.combinacoes([cal.Grade("a", (7,))]) == [{"a": 7}]


# ─────────────────────────────────────────────────────────────────────────────
# Blocos
# ─────────────────────────────────────────────────────────────────────────────

class TesteBlocos:
    def test_ajuste_e_uso_nao_se_sobrepoem(self):
        idx = indice(20)
        for b in cal.montar_blocos(idx, anos_de_ajuste=8, anos_de_uso=2):
            assert b.uso_de > b.ajuste_ate

    def test_embargo_separa_os_dois_periodos(self):
        idx = indice(20)
        b = cal.montar_blocos(idx, 8, 2, embargo_dias=30)[0]
        assert (b.uso_de - b.ajuste_ate).days == 30

    def test_blocos_avancam_no_tempo(self):
        idx = indice(24)
        blocos = cal.montar_blocos(idx, 8, 2)
        assert len(blocos) >= 2
        for a, b in zip(blocos, blocos[1:]):
            assert b.ajuste_de > a.ajuste_de
            assert b.uso_de > a.uso_de

    def test_todo_bloco_cabe_no_historico(self):
        idx = indice(20)
        for b in cal.montar_blocos(idx, 8, 2):
            assert b.uso_ate <= idx[-1]

    def test_historico_curto_e_recusado(self):
        with pytest.raises(ValueError, match="curto demais"):
            cal.montar_blocos(indice(0.5), 8, 2)

    def test_historico_que_nao_cabe_um_bloco(self):
        with pytest.raises(ValueError, match="não cabe"):
            cal.montar_blocos(indice(5), 8, 2)

    def test_periodo_nao_positivo_e_recusado(self):
        with pytest.raises(ValueError, match="positivos"):
            cal.montar_blocos(indice(20), 0, 2)

    def test_bloco_com_sobreposicao_e_recusado(self):
        with pytest.raises(ValueError, match="sobrepõem"):
            cal.Bloco(
                pd.Timestamp("2010-01-01"), pd.Timestamp("2018-01-01"),
                pd.Timestamp("2017-01-01"), pd.Timestamp("2019-01-01"),
            )


# ─────────────────────────────────────────────────────────────────────────────
# A métrica
# ─────────────────────────────────────────────────────────────────────────────

class TesteMetrica:
    def _resultado(self, tir, tir_cdi, queda):
        from carteira.aporte_historico import ResultadoAporte

        return ResultadoAporte(
            inicio=pd.Timestamp("2020-01-01"), fim=pd.Timestamp("2025-01-01"), meses=60,
            total_aportado=30000, valor_bruto_final=40000, custo_total_pago=100,
            imposto_total_pago=500, valor_liquido_final=39500,
            tir_anual_pct=tir, liquido_cdi=35000, tir_cdi_pct=tir_cdi,
            max_drawdown_pct=-5, pior_mes_pct=-8, queda_da_cota_pct=queda,
            meses_no_prejuizo=2,
        )

    def test_premia_vantagem_sobre_o_cdi(self):
        pouca = cal.vantagem_sobre_cdi_por_risco(self._resultado(11, 10, -10))
        muita = cal.vantagem_sobre_cdi_por_risco(self._resultado(15, 10, -10))
        assert muita > pouca

    def test_penaliza_queda_maior(self):
        suave = cal.vantagem_sobre_cdi_por_risco(self._resultado(14, 10, -10))
        violenta = cal.vantagem_sobre_cdi_por_risco(self._resultado(14, 10, -40))
        assert suave > violenta

    def test_nao_divide_por_zero(self):
        assert np.isfinite(cal.vantagem_sobre_cdi_por_risco(self._resultado(14, 10, 0.0)))

    def test_perder_do_cdi_da_metrica_negativa(self):
        assert cal.vantagem_sobre_cdi_por_risco(self._resultado(6, 10, -10)) < 0


# ─────────────────────────────────────────────────────────────────────────────
# O veredito
# ─────────────────────────────────────────────────────────────────────────────

def bloco_falso(i, dentro, fora, padrao, escolhido=None):
    b = cal.Bloco(
        pd.Timestamp("2000-01-01") + pd.DateOffset(years=i * 2),
        pd.Timestamp("2008-01-01") + pd.DateOffset(years=i * 2),
        pd.Timestamp("2008-02-01") + pd.DateOffset(years=i * 2),
        pd.Timestamp("2010-02-01") + pd.DateOffset(years=i * 2),
    )
    return cal.ResultadoDoBloco(b, escolhido or {"a": 1}, dentro, fora, padrao, 6)


class TesteVeredito:
    def test_poucos_blocos_reprova(self):
        v = cal._julgar([bloco_falso(0, 1.0, 0.9, 0.5)], 3, 0.6, 0.5)
        assert not v.aprender
        assert "bloco" in v.motivos[0]

    def test_aprovacao_quando_ganha_sempre_com_pouca_degradacao(self):
        blocos = [bloco_falso(i, 0.9, 0.85, 0.5) for i in range(5)]
        v = cal._julgar(blocos, 3, 0.6, 0.5)
        assert v.aprender
        assert v.vitorias == 5

    def test_reprova_quando_perde_do_padrao_na_maioria(self):
        blocos = [bloco_falso(i, 1.0, 0.3, 0.5) for i in range(5)]
        v = cal._julgar(blocos, 3, 0.6, 0.5)
        assert not v.aprender
        assert any("padrão" in m for m in v.motivos)

    def test_reprova_por_degradacao_alta(self):
        blocos = [bloco_falso(i, 5.0, 0.6, 0.5) for i in range(5)]
        v = cal._julgar(blocos, 3, 0.6, 0.5)
        assert not v.aprender
        assert any("degradação" in m for m in v.motivos)

    def test_reprova_quando_cada_bloco_escolhe_coisa_diferente(self):
        blocos = [
            bloco_falso(i, 0.9, 0.85, 0.5, escolhido={"a": i})
            for i in range(5)
        ]
        v = cal._julgar(blocos, 3, 0.6, 0.5)
        assert not v.aprender
        assert any("ruído, não regra" in m for m in v.motivos)

    def test_reprova_vitoria_que_uma_moeda_honesta_daria(self):
        """4 de 6 é resultado de acaso: 34% de chance sem habilidade nenhuma."""
        blocos = [bloco_falso(i, 0.9, 0.85, 0.5) for i in range(4)]
        blocos += [bloco_falso(i + 10, 0.9, 0.4, 0.5) for i in range(2)]
        v = cal._julgar(blocos, 3, 0.5, 0.5)
        assert not v.aprender
        assert any("moeda honesta" in m for m in v.motivos)

    def test_aprova_quando_ganha_em_todos(self):
        blocos = [bloco_falso(i, 0.9, 0.85, 0.5) for i in range(6)]
        v = cal._julgar(blocos, 3, 0.6, 0.5)
        assert v.aprender


class TesteBinomial:
    def test_ganhar_todas_e_improvavel(self):
        assert cal.p_valor_binomial(6, 6) == pytest.approx(1 / 64)

    def test_ganhar_metade_e_provavel(self):
        assert cal.p_valor_binomial(3, 6) > 0.5

    def test_quatro_de_seis_e_acaso_comum(self):
        assert cal.p_valor_binomial(4, 6) == pytest.approx(22 / 64)

    def test_zero_vitorias_da_probabilidade_um(self):
        assert cal.p_valor_binomial(0, 5) == pytest.approx(1.0)

    def test_sem_blocos_nao_conclui_nada(self):
        assert cal.p_valor_binomial(0, 0) == 1.0

    def test_p_cai_conforme_a_evidencia_cresce(self):
        anterior = 1.0
        for n in range(2, 12):
            p = cal.p_valor_binomial(n, n)
            assert p < anterior
            anterior = p

    def test_mais_vitorias_reduz_o_p(self):
        assert cal.p_valor_binomial(8, 10) < cal.p_valor_binomial(6, 10)

    def test_degradacao_mediana_e_calculada(self):
        blocos = [bloco_falso(i, 1.0, 0.5, 0.4) for i in range(3)]
        v = cal._julgar(blocos, 3, 0.6, 10.0)
        assert v.degradacao_mediana == pytest.approx(0.5)

    def test_resumo_e_serializavel(self):
        import json

        v = cal._julgar([bloco_falso(i, 0.9, 0.85, 0.5) for i in range(4)], 3, 0.6, 0.5)
        json.dumps(v.resumo())


# ─────────────────────────────────────────────────────────────────────────────
# Parâmetro estável
# ─────────────────────────────────────────────────────────────────────────────

class TesteParametroEstavel:
    def test_maioria_clara_devolve_o_parametro(self):
        blocos = [bloco_falso(i, 1, 1, 1, {"x": 5}) for i in range(4)]
        blocos.append(bloco_falso(9, 1, 1, 1, {"x": 7}))
        assert cal.parametro_estavel(blocos) == {"x": 5}

    def test_sem_maioria_devolve_nada(self):
        blocos = [bloco_falso(i, 1, 1, 1, {"x": i}) for i in range(4)]
        assert cal.parametro_estavel(blocos) is None

    def test_empate_na_metade_nao_e_maioria(self):
        blocos = [bloco_falso(i, 1, 1, 1, {"x": 1}) for i in range(2)]
        blocos += [bloco_falso(i + 5, 1, 1, 1, {"x": 2}) for i in range(2)]
        assert cal.parametro_estavel(blocos) is None

    def test_lista_vazia(self):
        assert cal.parametro_estavel([]) is None


# ─────────────────────────────────────────────────────────────────────────────
# Fim a fim
# ─────────────────────────────────────────────────────────────────────────────

class TesteFimAFim:
    def test_ruido_puro_nao_produz_aprendizado(self):
        """O teste que decide se este módulo presta.

        Duas séries de ruído sem estrutura nenhuma. Não existe peso ótimo
        estável entre elas. Um calibrador honesto tem que dizer não.
        """
        idx = indice(22)
        p = precos_ruido(idx, semente=42)
        v = cal.calibrar(p, cdi_plano(idx), GRADES, aplicar_pesos, PADRAO,
                         anos_de_ajuste=8, anos_de_uso=2)
        assert len(v.blocos) >= 3
        assert not v.aprender, f"aprendeu de ruído: {v.resumo()}"

    @pytest.mark.parametrize("semente", [1, 7, 13, 99, 2026])
    def test_ruido_nao_aprende_em_varias_sementes(self, semente):
        idx = indice(22)
        p = precos_ruido(idx, semente=semente)
        v = cal.calibrar(p, cdi_plano(idx), GRADES, aplicar_pesos, PADRAO,
                         anos_de_ajuste=8, anos_de_uso=2)
        assert not v.aprender

    def test_produz_um_resultado_por_bloco(self):
        idx = indice(22)
        p = precos_ruido(idx, semente=5)
        v = cal.calibrar(p, cdi_plano(idx), GRADES, aplicar_pesos, PADRAO,
                         anos_de_ajuste=8, anos_de_uso=2)
        esperados = len(cal.montar_blocos(idx, 8, 2))
        assert len(v.blocos) == esperados

    def test_conta_as_combinacoes_testadas(self):
        idx = indice(22)
        p = precos_ruido(idx, semente=5)
        v = cal.calibrar(p, cdi_plano(idx), GRADES, aplicar_pesos, PADRAO,
                         anos_de_ajuste=8, anos_de_uso=2)
        assert all(b.combinacoes_testadas == 6 for b in v.blocos)

    def test_vantagem_estrutural_e_encontrada(self):
        """Controle positivo: com um ativo claramente melhor, tem que achar.

        Sem este teste, "nunca aprende" poderia ser um calibrador quebrado em
        vez de um calibrador honesto.
        """
        idx = indice(22)
        rng = np.random.default_rng(3)
        bom = 100 * np.exp(np.cumsum(rng.normal(0.0009, 0.010, len(idx))))
        ruim = 100 * np.exp(np.cumsum(rng.normal(-0.0002, 0.010, len(idx))))
        p = pd.DataFrame({"BOVA11.SA": bom, "IVVB11.SA": ruim}, index=idx)

        v = cal.calibrar(p, cdi_plano(idx, 0.0), GRADES, aplicar_pesos, PADRAO,
                         anos_de_ajuste=8, anos_de_uso=2)
        estavel = cal.parametro_estavel(v.blocos)
        assert estavel is not None
        assert estavel["peso_bova"] == 0.7


# ─────────────────────────────────────────────────────────────────────────────
# Registro no livro
# ─────────────────────────────────────────────────────────────────────────────

class TesteRegistro:
    def test_grava_cada_bloco_e_o_veredito(self, tmp_path):
        livro = abrir(tmp_path / "l.sqlite3")
        v = cal._julgar([bloco_falso(i, 0.9, 0.85, 0.5) for i in range(4)], 3, 0.6, 0.5)
        eventos = cal.registrar_no_livro(livro, v, "teste")

        assert len(eventos) == 5
        assert len(livro.historico_do_parametro("teste")) == 4
        assert len(livro.eventos(TipoEvento.VALIDACAO)) == 1
        assert verificar(livro).intacta
        livro.fechar()

    def test_reprovacao_tambem_e_registrada(self, tmp_path):
        livro = abrir(tmp_path / "l.sqlite3")
        v = cal._julgar([bloco_falso(i, 5.0, 0.1, 0.9) for i in range(4)], 3, 0.6, 0.5)
        cal.registrar_no_livro(livro, v, "teste")

        validacao = livro.eventos(TipoEvento.VALIDACAO)[0]
        assert "reprovada" in validacao.resumo
        livro.fechar()

    def test_parametro_gravado_respeita_a_vigencia(self, tmp_path):
        livro = abrir(tmp_path / "l.sqlite3")
        v = cal._julgar([bloco_falso(i, 0.9, 0.85, 0.5) for i in range(4)], 3, 0.6, 0.5)
        cal.registrar_no_livro(livro, v, "teste")

        for linha in livro.historico_do_parametro("teste"):
            assert linha["vale_a_partir"] >= linha["ajustado_ate"]
        livro.fechar()
