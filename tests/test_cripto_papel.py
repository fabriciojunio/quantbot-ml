"""
Testes do paper trading.

O ponto que estes testes protegem: o simulado tem que doer igual ao real. Se
o paper fechar posição sem pagar taxa, ou entrar no preço do meio em vez do
ask, ele vira o mesmo teatro do painel antigo — números bons que não
sobrevivem ao primeiro dia com dinheiro de verdade.
"""

import json

import pytest

from cripto import papel
from cripto.custos import ModeloCusto
from cripto.exchanges.base import Book, NivelBook


class ClienteFalso:
    """Book fixo, sem rede. `nome` existe porque o operador registra a fonte."""

    def __init__(self, bid=100.0, ask=100.2):
        self.bid = bid
        self.ask = ask
        self.ultima_fonte = "falso"
        self.clientes = []

    def book(self, simbolo, profundidade=20):
        return Book(
            simbolo=simbolo,
            bids=(NivelBook(self.bid, 10.0),),
            asks=(NivelBook(self.ask, 10.0),),
            timestamp_ms=0,
        )


@pytest.fixture
def operador(tmp_path, monkeypatch):
    """Operador isolado: book, estado de risco e log de decisões em tmp_path."""
    monkeypatch.setattr(papel, "ARQUIVO", tmp_path / "paper_book.json")

    book = papel.Book(capital_inicial=1_000.0, caixa=1_000.0, criado_em="2026-08-23T00:00:00Z")
    op = papel.OperadorPapel(
        book=book,
        cliente=ClienteFalso(),
        modelo_custo=ModeloCusto(),
    )
    # Redireciona o estado de risco e o log de decisões para a pasta temporária.
    op.risco.state_file = tmp_path / "risk_state.json"
    op.decisoes.log_dir = tmp_path / "decisoes"
    op.decisoes.log_dir.mkdir(parents=True, exist_ok=True)
    return op


class TestAbertura:
    def test_abre_posicao_e_desconta_do_caixa(self, operador):
        operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0)
        assert "BTCUSDT" in operador.book.posicoes
        assert operador.book.caixa < 1_000.0

    def test_compra_paga_o_ask_e_nao_o_meio(self, operador):
        operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0)
        pos = operador.book.posicoes["BTCUSDT"]
        meio = (100.0 + 100.2) / 2
        assert pos.preco_entrada > meio

    def test_taxa_de_entrada_e_cobrada(self, operador):
        operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0)
        assert operador.book.posicoes["BTCUSDT"].taxa_entrada_usd > 0

    def test_nao_abre_duas_posicoes_no_mesmo_par(self, operador):
        operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0)
        assert operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0) is None

    def test_respeita_o_limite_de_posicoes_simultaneas(self, operador):
        operador.max_posicoes = 2
        for par in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            operador.abrir(par, 1, stop=95.0, alvo=110.0, score=70.0)
        assert len(operador.book.posicoes) == 2

    def test_caixa_insuficiente_nao_abre(self, operador):
        operador.book.caixa = 5.0
        assert operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0) is None


class TestFechamento:
    def test_venda_recebe_o_bid_e_nao_o_meio(self, operador):
        operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0)
        fechada = operador.fechar("BTCUSDT", "manual")
        meio = (100.0 + 100.2) / 2
        assert fechada.preco_saida < meio

    def test_abrir_e_fechar_no_mesmo_preco_da_prejuizo(self, operador):
        """Sem movimento de preço, o resultado é exatamente o custo — negativo.
        É o teste que o simulador antigo reprovaria."""
        operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0)
        fechada = operador.fechar("BTCUSDT", "manual")
        assert fechada.pnl_liquido_usd < 0
        assert fechada.custo_total_usd > 0

    def test_alta_suficiente_cobre_o_custo(self, operador):
        operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0)
        operador.cliente.bid, operador.cliente.ask = 110.0, 110.2
        fechada = operador.fechar("BTCUSDT", "alvo")
        assert fechada.pnl_liquido_usd > 0

    def test_posicao_sai_do_book_ao_fechar(self, operador):
        operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0)
        operador.fechar("BTCUSDT", "manual")
        assert "BTCUSDT" not in operador.book.posicoes
        assert len(operador.book.fechadas) == 1

    def test_fechar_posicao_inexistente_devolve_none(self, operador):
        assert operador.fechar("NAOEXISTE", "manual") is None


class TestRevisao:
    def test_stop_dispara_quando_o_preco_cai(self, operador):
        operador.abrir("BTCUSDT", 1, stop=99.0, alvo=200.0, score=70.0)
        operador.cliente.bid, operador.cliente.ask = 98.0, 98.2
        fechadas = operador.revisar_posicoes()
        assert len(fechadas) == 1
        assert fechadas[0].motivo == "stop"

    def test_alvo_dispara_quando_o_preco_sobe(self, operador):
        operador.abrir("BTCUSDT", 1, stop=50.0, alvo=101.0, score=70.0)
        operador.cliente.bid, operador.cliente.ask = 102.0, 102.2
        fechadas = operador.revisar_posicoes()
        assert len(fechadas) == 1
        assert fechadas[0].motivo == "alvo"

    def test_preco_no_meio_do_caminho_nao_fecha_nada(self, operador):
        operador.abrir("BTCUSDT", 1, stop=50.0, alvo=200.0, score=70.0)
        assert operador.revisar_posicoes() == []


class TestRisco:
    def test_kill_switch_barra_a_ordem(self, operador):
        operador.risco.activate_kill_switch("teste")
        assert operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0) is None
        assert "BTCUSDT" not in operador.book.posicoes

    def test_ordem_grande_demais_e_barrada(self, operador):
        operador.risco.limits.max_position_size_pct = 0.01
        operador.fracao = 0.9
        assert operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0) is None


class TestPersistencia:
    def test_book_sobrevive_a_gravacao_e_leitura(self, operador, tmp_path):
        operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=72.5)
        papel.salvar(operador.book, tmp_path / "book.json")
        lido = papel.carregar(tmp_path / "book.json")
        assert "BTCUSDT" in lido.posicoes
        assert lido.posicoes["BTCUSDT"].score_entrada == pytest.approx(72.5)

    def test_arquivo_inexistente_cria_book_novo(self, tmp_path):
        book = papel.carregar(tmp_path / "nao_existe.json", capital_inicial=500.0)
        assert book.capital_inicial == 500.0
        assert book.caixa == 500.0
        assert book.posicoes == {}

    def test_json_gerado_e_legivel(self, operador, tmp_path):
        operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0)
        caminho = tmp_path / "book.json"
        papel.salvar(operador.book, caminho)
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        assert dados["capital_inicial"] == 1_000.0
        assert "BTCUSDT" in dados["posicoes"]


class TestResumo:
    def test_resumo_mostra_o_custo_pago(self, operador):
        operador.abrir("BTCUSDT", 1, stop=95.0, alvo=110.0, score=70.0)
        operador.fechar("BTCUSDT", "manual")
        r = operador.resumo()
        assert r["custo_total_pago_usd"] > 0
        assert r["operacoes_fechadas"] == 1

    def test_resumo_inclui_o_estado_de_risco(self, operador):
        r = operador.resumo()
        assert "risco" in r
        assert "kill_switch_ativo" in r["risco"]

    def test_marcar_valor_registra_a_serie(self, operador):
        operador.marcar_valor()
        assert len(operador.book.serie_valor) == 1
        assert operador.book.serie_valor[0]["valor_usd"] > 0
