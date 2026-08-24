"""
Regressão do accrual de proventos no paper book de FIIs.

Contexto: até 23/08/2026 a `marcar_mercado` lia `ultima_avaliacao` para saber
quantos dias tinham passado desde o último accrual, mas nunca atualizava esse
campo — só a `avaliar()` atualizava, e apenas de 30 em 30 dias. Cada marcação
relançava o provento de todos os dias desde a entrada, então em n dias entravam
n(n+1)/2 dias de rendimento. Em 61 dias o provento saiu 14 vezes maior, e o
track record registrado (+17,14%) virou, depois de corrigido, −6,32%.

O teste central aqui é `test_marcar_todo_dia_da_o_mesmo_que_marcar_uma_vez`:
enquanto ele passar, o accrual não pode voltar a ser quadrático.
"""

from datetime import date, timedelta

import pytest

from income.paper_book import CUSTO_PONTA, PaperBook


class _Aprovado:
    """Substitui o objeto do screener, que só precisa destes quatro campos."""

    def __init__(self, ticker, score=80.0, dy_12m=12.0, pvp=0.9):
        self.ticker = ticker
        self.score = score
        self.dy_12m = dy_12m
        self.pvp = pvp


@pytest.fixture
def book(tmp_path):
    return PaperBook(arquivo=str(tmp_path / "book.json"), capital_inicial=10_000.0, max_posicoes=4)


def _avanca(monkeypatch, dias):
    """Empurra o 'hoje' do módulo, sem mexer no relógio do sistema."""
    alvo = date.today() + timedelta(days=dias)
    monkeypatch.setattr("income.paper_book._hoje", lambda: alvo.isoformat())
    return alvo


class TestAccrualDeProvento:
    def test_marcar_todo_dia_da_o_mesmo_que_marcar_uma_vez(self, tmp_path, monkeypatch):
        """O coração da regressão: marcar diariamente por 60 dias tem que render
        o mesmo que marcar uma única vez no fim dos 60 dias."""
        precos = {"XPTO11": 100.0}

        diario = PaperBook(arquivo=str(tmp_path / "a.json"), capital_inicial=10_000.0)
        diario.decidir([_Aprovado("XPTO11")], precos)
        for d in range(1, 61):
            _avanca(monkeypatch, d)
            diario.marcar_mercado(precos)
        div_diario = diario.estado["posicoes"]["XPTO11"]["dividendos"]

        monkeypatch.undo()
        unico = PaperBook(arquivo=str(tmp_path / "b.json"), capital_inicial=10_000.0)
        unico.decidir([_Aprovado("XPTO11")], precos)
        _avanca(monkeypatch, 60)
        unico.marcar_mercado(precos)
        div_unico = unico.estado["posicoes"]["XPTO11"]["dividendos"]

        assert div_diario == pytest.approx(div_unico, rel=0.02)

    def test_provento_de_60_dias_bate_com_a_formula(self, book, monkeypatch):
        """DY de 12% ao ano, em 60 dias, tem que render ~1,97% do investido."""
        precos = {"XPTO11": 100.0}
        book.decidir([_Aprovado("XPTO11", dy_12m=12.0)], precos)
        pos = book.estado["posicoes"]["XPTO11"]
        investido = pos["cotas"] * pos["preco_entrada"]

        _avanca(monkeypatch, 60)
        book.marcar_mercado(precos)

        esperado = investido * 0.12 * (60 / 365)
        assert book.estado["posicoes"]["XPTO11"]["dividendos"] == pytest.approx(esperado, rel=0.01)

    def test_provento_nunca_passa_do_dy_anual(self, book, monkeypatch):
        """Um ano inteiro não pode render mais que o DY de um ano. Era esta a
        invariante que o bug quebrava — em 61 dias já rendia 28% de um DY de 12%."""
        precos = {"XPTO11": 100.0}
        book.decidir([_Aprovado("XPTO11", dy_12m=12.0)], precos)
        pos = book.estado["posicoes"]["XPTO11"]
        investido = pos["cotas"] * pos["preco_entrada"]

        for d in range(1, 366):
            _avanca(monkeypatch, d)
            book.marcar_mercado(precos)

        acumulado = book.estado["posicoes"]["XPTO11"]["dividendos"]
        assert acumulado <= investido * 0.12 * 1.02

    def test_campo_de_accrual_e_atualizado_a_cada_marcacao(self, book, monkeypatch):
        precos = {"XPTO11": 100.0}
        book.decidir([_Aprovado("XPTO11")], precos)
        alvo = _avanca(monkeypatch, 10)
        book.marcar_mercado(precos)
        assert book.estado["posicoes"]["XPTO11"]["ultimo_provento"] == alvo.isoformat()

    def test_accrual_e_avaliacao_usam_campos_diferentes(self, book, monkeypatch):
        """A `avaliar()` só dispara de 30 em 30 dias. Se ela dividisse o campo
        com o accrual, voltaríamos ao bug original."""
        precos = {"XPTO11": 100.0}
        book.decidir([_Aprovado("XPTO11")], precos)
        _avanca(monkeypatch, 5)
        book.marcar_mercado(precos)
        pos = book.estado["posicoes"]["XPTO11"]
        assert pos["ultimo_provento"] != pos["ultima_avaliacao"]

    def test_dy_zerado_nao_gera_provento(self, book, monkeypatch):
        precos = {"XPTO11": 100.0}
        book.decidir([_Aprovado("XPTO11", dy_12m=0.0)], precos)
        _avanca(monkeypatch, 90)
        book.marcar_mercado(precos)
        assert book.estado["posicoes"]["XPTO11"]["dividendos"] == 0.0


class TestCustoDeOperacao:
    def test_compra_paga_custo(self, book):
        precos = {"XPTO11": 100.0}
        book.decidir([_Aprovado("XPTO11")], precos)
        assert book.estado["posicoes"]["XPTO11"]["custo_entrada"] > 0

    def test_comprar_e_vender_no_mesmo_preco_da_prejuizo(self, book):
        """Sem o preço se mexer, o resultado é exatamente o custo. Era isto que
        faltava: o book antigo devolvia zero e fingia que operar é de graça."""
        precos = {"XPTO11": 100.0}
        book.decidir([_Aprovado("XPTO11")], precos)
        book.decidir([], precos)  # some da lista de aprovados → vende
        fechada = book.estado["fechadas"][0]
        assert fechada["retorno_total_pct"] < 0
        assert fechada["custo_total"] > 0

    def test_custo_de_ida_e_volta_bate_com_a_constante(self, book):
        precos = {"XPTO11": 100.0}
        book.decidir([_Aprovado("XPTO11")], precos)
        book.decidir([], precos)
        fechada = book.estado["fechadas"][0]
        assert fechada["custo_pct"] == pytest.approx(CUSTO_PONTA * 2 * 100, rel=0.05)

    def test_caixa_cai_pelo_custo_alem_do_preco(self, book):
        precos = {"XPTO11": 100.0}
        caixa_antes = book.caixa
        book.decidir([_Aprovado("XPTO11")], precos)
        pos = book.estado["posicoes"]["XPTO11"]
        gasto = caixa_antes - book.caixa
        assert gasto > pos["cotas"] * pos["preco_entrada"]


class TestResumo:
    def test_retorno_sem_movimento_de_preco_e_negativo(self, book):
        """Compra, marca um dia, e nada rende além do provento de um dia:
        o custo tem que deixar a carteira abaixo do capital inicial."""
        precos = {"XPTO11": 100.0}
        book.decidir([_Aprovado("XPTO11")], precos)
        r = book.resumo(precos)
        assert r["valor_total"] < r["capital_inicial"]

    def test_valor_total_soma_caixa_e_posicoes(self, book):
        precos = {"XPTO11": 100.0}
        book.decidir([_Aprovado("XPTO11")], precos)
        pos = book.estado["posicoes"]["XPTO11"]
        esperado = book.caixa + pos["cotas"] * 100.0
        assert book.valor_total(precos) == pytest.approx(esperado, abs=0.02)
