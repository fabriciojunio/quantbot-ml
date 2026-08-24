"""
Testes do livro-razão.

Metade destes testes adultera o banco de propósito, por dentro do SQLite, e
exige que a verificação acuse. Um registro de auditoria que não detecta
alteração não é auditoria, é log.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from auditoria import TipoEvento, abrir, verificar
from auditoria.livro import GENESE, calcular_hash
from auditoria.verificacao import ancora, conferir_ancora

SEM_SAUDE = settings(
    max_examples=50, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


@pytest.fixture
def livro(tmp_path):
    lv = abrir(tmp_path / "teste.sqlite3")
    yield lv
    lv.fechar()


def povoar(livro, n: int = 10):
    for i in range(n):
        livro.registrar(
            TipoEvento.DECISAO, "teste", f"decisão número {i}",
            {"indice": i, "valor": i * 1.5},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Abertura
# ─────────────────────────────────────────────────────────────────────────────

class TesteAbertura:
    def test_livro_novo_ja_tem_evento_de_abertura(self, livro):
        assert livro.total_de_eventos() == 1
        assert livro.eventos()[0].tipo is TipoEvento.ABERTURA

    def test_primeiro_evento_aponta_para_a_genese(self, livro):
        assert livro.eventos()[0].hash_anterior == GENESE

    def test_reabrir_nao_duplica_a_abertura(self, tmp_path):
        caminho = tmp_path / "l.sqlite3"
        a = abrir(caminho)
        a.registrar(TipoEvento.DECISAO, "t", "algo")
        a.fechar()

        b = abrir(caminho)
        assert b.total_de_eventos() == 2
        assert sum(1 for e in b.percorrer() if e.tipo is TipoEvento.ABERTURA) == 1
        b.fechar()

    def test_cria_o_diretorio_se_nao_existir(self, tmp_path):
        caminho = tmp_path / "fundo" / "do" / "poco" / "l.sqlite3"
        lv = abrir(caminho)
        assert caminho.exists()
        lv.fechar()


# ─────────────────────────────────────────────────────────────────────────────
# Encadeamento
# ─────────────────────────────────────────────────────────────────────────────

class TesteEncadeamento:
    def test_cada_evento_aponta_para_o_anterior(self, livro):
        povoar(livro, 8)
        eventos = list(livro.percorrer())
        for anterior, atual in zip(eventos, eventos[1:]):
            assert atual.hash_anterior == anterior.hash

    def test_hashes_sao_todos_distintos(self, livro):
        povoar(livro, 30)
        hashes = [e.hash for e in livro.percorrer()]
        assert len(set(hashes)) == len(hashes)

    def test_dois_eventos_iguais_tem_hash_diferente(self, livro):
        """O elo com o anterior entra no hash, então nem repetição colide."""
        a = livro.registrar(TipoEvento.DECISAO, "t", "mesma coisa", {"x": 1})
        b = livro.registrar(TipoEvento.DECISAO, "t", "mesma coisa", {"x": 1})
        assert a.hash != b.hash

    def test_cadeia_recem_criada_esta_intacta(self, livro):
        povoar(livro, 20)
        r = verificar(livro)
        assert r.intacta
        assert r.eventos_conferidos == 21
        assert r.quebras == ()

    def test_hash_e_estavel_para_a_mesma_entrada(self):
        a = calcular_hash("2026-01-01T00:00:00", "decisao", "t", "r", {"b": 2, "a": 1}, GENESE)
        b = calcular_hash("2026-01-01T00:00:00", "decisao", "t", "r", {"a": 1, "b": 2}, GENESE)
        assert a == b

    def test_hash_muda_com_qualquer_campo(self):
        base = ("2026-01-01T00:00:00", "decisao", "origem", "resumo", {"a": 1}, GENESE)
        h = calcular_hash(*base)
        variacoes = [
            ("2026-01-02T00:00:00", "decisao", "origem", "resumo", {"a": 1}, GENESE),
            ("2026-01-01T00:00:00", "ordem", "origem", "resumo", {"a": 1}, GENESE),
            ("2026-01-01T00:00:00", "decisao", "outra", "resumo", {"a": 1}, GENESE),
            ("2026-01-01T00:00:00", "decisao", "origem", "outro", {"a": 1}, GENESE),
            ("2026-01-01T00:00:00", "decisao", "origem", "resumo", {"a": 2}, GENESE),
            ("2026-01-01T00:00:00", "decisao", "origem", "resumo", {"a": 1}, "f" * 64),
        ]
        for v in variacoes:
            assert calcular_hash(*v) != h

    @SEM_SAUDE
    @given(textos=st.lists(st.text(min_size=1, max_size=40), min_size=1, max_size=25))
    def test_cadeia_fecha_para_qualquer_conteudo(self, tmp_path_factory, textos):
        caminho = tmp_path_factory.mktemp("h") / "l.sqlite3"
        lv = abrir(caminho)
        for t in textos:
            lv.registrar(TipoEvento.NOTICIA, "fuzz", "evento", {"texto": t})
        assert verificar(lv).intacta
        lv.fechar()


# ─────────────────────────────────────────────────────────────────────────────
# Adulteração
# ─────────────────────────────────────────────────────────────────────────────

class TesteAdulteracao:
    def test_mudar_o_conteudo_de_um_evento_e_detectado(self, tmp_path):
        caminho = tmp_path / "l.sqlite3"
        lv = abrir(caminho)
        povoar(lv, 10)
        lv.fechar()

        # Edita direto no SQLite, como faria quem quisesse maquiar o histórico.
        con = sqlite3.connect(caminho)
        con.execute("UPDATE evento SET dados = ? WHERE id = 5",
                    (json.dumps({"indice": 999}),))
        con.commit()
        con.close()

        lv = abrir(caminho)
        r = verificar(lv)
        assert not r.intacta
        assert any(q.evento_id == 5 for q in r.quebras)
        lv.fechar()

    def test_apagar_um_evento_do_meio_e_detectado(self, tmp_path):
        caminho = tmp_path / "l.sqlite3"
        lv = abrir(caminho)
        povoar(lv, 10)
        lv.fechar()

        con = sqlite3.connect(caminho)
        con.execute("DELETE FROM evento WHERE id = 6")
        con.commit()
        con.close()

        lv = abrir(caminho)
        r = verificar(lv)
        assert not r.intacta
        assert any("elo" in q.motivo for q in r.quebras)
        lv.fechar()

    def test_mudar_a_data_de_um_evento_e_detectado(self, tmp_path):
        caminho = tmp_path / "l.sqlite3"
        lv = abrir(caminho)
        povoar(lv, 6)
        lv.fechar()

        con = sqlite3.connect(caminho)
        con.execute("UPDATE evento SET quando = '1999-01-01T00:00:00' WHERE id = 3")
        con.commit()
        con.close()

        lv = abrir(caminho)
        assert not verificar(lv).intacta
        lv.fechar()

    def test_quebra_no_meio_derruba_tudo_que_vem_depois(self, tmp_path):
        """Mostrar o tamanho do estrago, não só a primeira linha."""
        caminho = tmp_path / "l.sqlite3"
        lv = abrir(caminho)
        povoar(lv, 20)
        lv.fechar()

        con = sqlite3.connect(caminho)
        con.execute("UPDATE evento SET resumo = 'mexido' WHERE id = 5")
        con.commit()
        con.close()

        lv = abrir(caminho)
        r = verificar(lv)
        assert len(r.quebras) >= 2
        lv.fechar()

    def test_parar_na_primeira_devolve_uma_quebra_so(self, tmp_path):
        caminho = tmp_path / "l.sqlite3"
        lv = abrir(caminho)
        povoar(lv, 20)
        lv.fechar()

        con = sqlite3.connect(caminho)
        con.execute("UPDATE evento SET resumo = 'mexido' WHERE id = 5")
        con.commit()
        con.close()

        lv = abrir(caminho)
        assert len(verificar(lv, parar_na_primeira=True).quebras) == 1
        lv.fechar()

    def test_reescrever_a_cadeia_inteira_passa_e_isso_esta_documentado(self, tmp_path):
        """Limite conhecido: quem tem o arquivo pode refazer todos os hashes.

        O teste existe para que o limite fique registrado em código, e para
        justificar por que `ancora()` precisa existir.
        """
        caminho = tmp_path / "l.sqlite3"
        lv = abrir(caminho)
        povoar(lv, 5)
        carimbo = ancora(lv)
        lv.fechar()

        con = sqlite3.connect(caminho)
        con.row_factory = sqlite3.Row
        linhas = [dict(r) for r in con.execute("SELECT * FROM evento ORDER BY id")]
        linhas[3]["dados"] = json.dumps({"indice": 42})
        anterior = GENESE
        for linha in linhas:
            novo = calcular_hash(linha["quando"], linha["tipo"], linha["origem"],
                                 linha["resumo"], json.loads(linha["dados"]), anterior)
            con.execute(
                "UPDATE evento SET dados = ?, hash_anterior = ?, hash = ? WHERE id = ?",
                (linha["dados"], anterior, novo, linha["id"]),
            )
            anterior = novo
        con.commit()
        con.close()

        lv = abrir(caminho)
        # A cadeia interna fecha de novo...
        assert verificar(lv).intacta
        # ...mas o carimbo externo denuncia.
        ok, mensagem = conferir_ancora(lv, carimbo)
        assert not ok
        assert "mudou" in mensagem
        lv.fechar()


# ─────────────────────────────────────────────────────────────────────────────
# Âncora
# ─────────────────────────────────────────────────────────────────────────────

class TesteAncora:
    def test_carimbo_traz_hash_e_contagem(self, livro):
        povoar(livro, 5)
        c = ancora(livro)
        assert c["eventos"] == 6
        assert len(c["hash_final"]) == 64
        assert c["intacta"] is True

    def test_livro_que_so_cresceu_continua_valido(self, livro):
        povoar(livro, 5)
        c = ancora(livro)
        povoar(livro, 5)
        ok, msg = conferir_ancora(livro, c)
        assert ok
        assert "cresceu" in msg

    def test_livro_encolhido_e_recusado(self, tmp_path):
        caminho = tmp_path / "l.sqlite3"
        lv = abrir(caminho)
        povoar(lv, 10)
        c = ancora(lv)
        lv.fechar()

        con = sqlite3.connect(caminho)
        con.execute("DELETE FROM evento WHERE id > 5")
        con.commit()
        con.close()

        lv = abrir(caminho)
        ok, msg = conferir_ancora(lv, c)
        assert not ok
        assert "menos" in msg
        lv.fechar()


# ─────────────────────────────────────────────────────────────────────────────
# Correção
# ─────────────────────────────────────────────────────────────────────────────

class TesteCorrecao:
    def test_correcao_nao_apaga_o_original(self, livro):
        errado = livro.registrar(TipoEvento.ORDEM, "robo", "comprou 10", {"q": 10})
        livro.registrar(TipoEvento.CORRECAO, "robo", "eram 12, não 10",
                        {"q": 12}, corrige=errado.id)

        assert livro.evento(errado.id) is not None
        assert errado.id in livro.corrigidos()
        assert verificar(livro).intacta

    def test_evento_sem_correcao_nao_aparece_como_corrigido(self, livro):
        e = livro.registrar(TipoEvento.ORDEM, "robo", "comprou 10")
        assert e.id not in livro.corrigidos()

    def test_resumo_vazio_e_recusado(self, livro):
        with pytest.raises(ValueError, match="resumo"):
            livro.registrar(TipoEvento.DECISAO, "t", "   ")


# ─────────────────────────────────────────────────────────────────────────────
# Fechamento diário
# ─────────────────────────────────────────────────────────────────────────────

class TesteFechamento:
    def test_grava_patrimonio_e_posicoes(self, livro):
        livro.registrar_fechamento(
            "2026-08-21", caixa=100.0,
            posicoes={
                "BOVA11.SA": {"quantidade": 10, "preco": 120.0, "valor": 1200.0, "custo_medio": 100.0},
                "IVVB11.SA": {"quantidade": 5, "preco": 300.0, "valor": 1500.0, "custo_medio": 280.0},
            },
            aportado=2500.0, cota=1.12, cdi_acumulado=1.08,
        )
        p = livro.patrimonio()
        assert len(p) == 1
        assert p[0]["total"] == pytest.approx(2800.0)
        assert p[0]["investido"] == pytest.approx(2700.0)

    def test_reescrever_o_mesmo_dia_substitui(self, livro):
        for total in (1000.0, 2000.0):
            livro.registrar_fechamento(
                "2026-08-21", caixa=total,
                posicoes={}, aportado=500.0, cota=1.0, cdi_acumulado=1.0,
            )
        p = livro.patrimonio()
        assert len(p) == 1
        assert p[0]["total"] == pytest.approx(2000.0)
        # Os dois eventos continuam registrados.
        assert len(livro.eventos(TipoEvento.FECHAMENTO)) == 2

    def test_ordena_por_data(self, livro):
        for d in ("2026-03-01", "2026-01-01", "2026-02-01"):
            livro.registrar_fechamento(d, 0.0, {}, 0.0, 1.0, 1.0)
        datas = [x["data"] for x in livro.patrimonio()]
        assert datas == sorted(datas)

    def test_filtra_a_partir_de_uma_data(self, livro):
        for d in ("2026-01-01", "2026-02-01", "2026-03-01"):
            livro.registrar_fechamento(d, 0.0, {}, 0.0, 1.0, 1.0)
        assert len(livro.patrimonio(desde="2026-02-01")) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Parâmetros aprendidos
# ─────────────────────────────────────────────────────────────────────────────

class TesteParametros:
    def test_registra_com_as_tres_datas(self, livro):
        livro.registrar_parametro(
            "janela_de_tendencia", 200,
            ajustado_de="2010-01-01", ajustado_ate="2015-01-01",
            vale_a_partir="2015-01-02", metrica_dentro=1.2, metrica_fora=0.8,
        )
        h = livro.historico_do_parametro("janela_de_tendencia")
        assert len(h) == 1
        assert h[0]["metrica_fora"] == pytest.approx(0.8)

    def test_recusa_parametro_que_olha_para_o_futuro(self, livro):
        """Ajustado com dados até 2026 e valendo desde 2015 seria vazamento."""
        with pytest.raises(ValueError, match="futuro"):
            livro.registrar_parametro(
                "x", 1, ajustado_de="2010-01-01", ajustado_ate="2026-01-01",
                vale_a_partir="2015-01-01",
            )

    def test_vigente_pega_o_mais_recente_ate_a_data(self, livro):
        for ate, vale, valor in [
            ("2015-01-01", "2015-01-02", 100),
            ("2020-01-01", "2020-01-02", 200),
            ("2025-01-01", "2025-01-02", 300),
        ]:
            livro.registrar_parametro("janela", valor, "2010-01-01", ate, vale)

        assert livro.parametros_vigentes("2018-06-01")["janela"] == 100
        assert livro.parametros_vigentes("2022-06-01")["janela"] == 200
        assert livro.parametros_vigentes("2026-06-01")["janela"] == 300

    def test_vigente_nao_enxerga_parametro_futuro(self, livro):
        livro.registrar_parametro("janela", 100, "2010-01-01", "2020-01-01", "2020-01-02")
        assert livro.parametros_vigentes("2019-01-01") == {}

    def test_valor_composto_sobrevive_a_ida_e_volta(self, livro):
        valor = {"pesos": {"BOVA11.SA": 0.6, "IVVB11.SA": 0.4}, "banda": 0.05}
        livro.registrar_parametro("alocacao", valor, "2010-01-01", "2020-01-01", "2020-01-02")
        assert livro.parametros_vigentes("2021-01-01")["alocacao"] == valor


# ─────────────────────────────────────────────────────────────────────────────
# Consulta e exportação
# ─────────────────────────────────────────────────────────────────────────────

class TesteConsulta:
    def test_filtra_por_tipo(self, livro):
        povoar(livro, 5)
        livro.registrar(TipoEvento.ALERTA, "t", "algo estranho")
        assert len(livro.eventos(TipoEvento.ALERTA)) == 1
        assert len(livro.eventos(TipoEvento.DECISAO)) == 5

    def test_filtra_por_periodo(self, livro):
        for d in ("2020-01-01T00:00:00", "2022-01-01T00:00:00", "2024-01-01T00:00:00"):
            livro.registrar(TipoEvento.DECISAO, "t", "x", quando=d)
        assert len(livro.eventos(desde="2021-01-01", ate="2023-01-01")) == 1

    def test_limite_corta_o_resultado(self, livro):
        povoar(livro, 20)
        assert len(livro.eventos(limite=5)) == 5

    def test_evento_inexistente_devolve_nada(self, livro):
        assert livro.evento(99999) is None

    def test_exporta_json_legivel(self, livro, tmp_path):
        povoar(livro, 4)
        destino = livro.exportar(tmp_path / "saida.json")
        conteudo = json.loads(destino.read_text(encoding="utf-8"))
        assert conteudo["total_de_eventos"] == 5
        assert len(conteudo["eventos"]) == 5
        assert conteudo["eventos"][0]["hash_anterior"] == GENESE

    def test_export_preserva_o_conteudo_dos_dados(self, livro, tmp_path):
        livro.registrar(TipoEvento.DECISAO, "t", "r", {"lista": [1, 2, 3], "texto": "ação"})
        conteudo = json.loads(livro.exportar(tmp_path / "s.json").read_text(encoding="utf-8"))
        assert conteudo["eventos"][-1]["dados"] == {"lista": [1, 2, 3], "texto": "ação"}
