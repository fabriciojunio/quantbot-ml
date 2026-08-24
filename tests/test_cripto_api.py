"""
Testes da API de leitura.

Cobrem o que importa numa API pública: entrada validada, limite por IP,
cabeçalho de segurança e — o mais importante para este projeto — o campo
`sintetico: false`, que é a promessa de que nenhum número da tela foi
inventado.
"""

import pytest
from fastapi.testclient import TestClient

from cripto import api
from cripto.exchanges.base import Book, Negocio, NivelBook


class ClienteFalso:
    nome = "falso"

    def __init__(self):
        self.ultima_fonte = "falso"
        self.clientes = [self]

    def book(self, simbolo, profundidade=20):
        return Book(
            simbolo=simbolo,
            bids=tuple(NivelBook(100.0 - i * 0.1, 2.0 + i) for i in range(profundidade)),
            asks=tuple(NivelBook(100.2 + i * 0.1, 1.0 + i) for i in range(profundidade)),
            timestamp_ms=1_787_508_000_000,
        )

    def negocios(self, simbolo, limite=300):
        return [
            Negocio(1_787_508_000_000 + i, 100.0 + (i % 5) * 0.01, 0.5 + (i % 7), i % 2 == 0)
            for i in range(limite)
        ]


@pytest.fixture
def cliente_http(monkeypatch):
    monkeypatch.setattr(api, "cliente", ClienteFalso())
    api.cache._itens.clear()
    api.limitador._acessos.clear()
    return TestClient(api.app)


class TestSaude:
    def test_responde_ok(self, cliente_http):
        r = cliente_http.get("/api/cripto/saude")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_diz_que_e_somente_leitura(self, cliente_http):
        assert "leitura" in cliente_http.get("/api/cripto/saude").json()["modo"]


class TestCabecalhosDeSeguranca:
    @pytest.mark.parametrize(
        "cabecalho,valor",
        [
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
            ("Referrer-Policy", "no-referrer"),
            ("Cache-Control", "no-store"),
        ],
    )
    def test_cabecalho_presente(self, cliente_http, cabecalho, valor):
        r = cliente_http.get("/api/cripto/saude")
        assert r.headers.get(cabecalho) == valor


class TestValidacaoDeSimbolo:
    @pytest.mark.parametrize("ruim", ["BTC/USDT", "../etc/passwd", "BTC;DROP", "AB"])
    def test_simbolo_invalido_da_400(self, cliente_http, ruim):
        assert cliente_http.get(f"/api/cripto/book/{ruim}").status_code in (400, 404)

    def test_tempo_grafico_invalido_da_400(self, cliente_http):
        r = cliente_http.get("/api/cripto/sinal/BTCUSDT", params={"tempo": "7m"})
        assert r.status_code == 400


class TestPrecos:
    def test_traz_bid_ask_e_spread(self, cliente_http):
        item = cliente_http.get("/api/cripto/precos", params={"simbolos": "BTCUSDT"}).json()["precos"][0]
        assert item["melhor_bid"] == 100.0
        assert item["melhor_ask"] == 100.2
        assert item["spread_pct"] > 0

    def test_marca_o_dado_como_nao_sintetico(self, cliente_http):
        item = cliente_http.get("/api/cripto/precos", params={"simbolos": "BTCUSDT"}).json()["precos"][0]
        assert item["sintetico"] is False

    def test_informa_a_fonte(self, cliente_http):
        item = cliente_http.get("/api/cripto/precos", params={"simbolos": "BTCUSDT"}).json()["precos"][0]
        assert item["fonte"] == "falso"

    def test_simbolo_quebrado_nao_derruba_a_lista_toda(self, cliente_http):
        r = cliente_http.get("/api/cripto/precos", params={"simbolos": "BTCUSDT,XX"})
        assert r.status_code == 400  # símbolo inválido é erro explícito, não silencioso


class TestBook:
    def test_traz_as_maiores_paredes(self, cliente_http):
        d = cliente_http.get("/api/cripto/book/BTCUSDT").json()
        assert d["maior_parede_compra"]["notional_usd"] > 0
        assert d["maior_parede_venda"]["notional_usd"] > 0

    def test_desequilibrio_fica_entre_zero_e_um(self, cliente_http):
        d = cliente_http.get("/api/cripto/book/BTCUSDT").json()
        assert 0.0 <= d["desequilibrio"] <= 1.0

    def test_profundidade_e_limitada(self, cliente_http):
        d = cliente_http.get("/api/cripto/book/BTCUSDT", params={"profundidade": 9999}).json()
        assert len(d["niveis_bid"]) <= 20


class TestTape:
    def test_calcula_delta_de_agressao(self, cliente_http):
        d = cliente_http.get("/api/cripto/tape/BTCUSDT").json()
        assert d["volume_agressao_compra"] > 0
        assert d["volume_agressao_venda"] > 0
        assert d["delta"] == pytest.approx(
            d["volume_agressao_compra"] - d["volume_agressao_venda"], abs=1e-6
        )

    def test_marca_como_nao_sintetico(self, cliente_http):
        assert cliente_http.get("/api/cripto/tape/BTCUSDT").json()["sintetico"] is False


class TestCusto:
    def test_alvo_de_dez_centavos_sai_reprovado(self, cliente_http):
        d = cliente_http.get(
            "/api/cripto/custo", params={"notional_usd": 1000, "alvo_usd": 0.10}
        ).json()
        assert d["teste_do_alvo"]["viavel"] is False

    def test_exchange_muda_a_taxa(self, cliente_http):
        binance = cliente_http.get("/api/cripto/custo", params={"exchange": "binance"}).json()
        gate = cliente_http.get("/api/cripto/custo", params={"exchange": "gateio"}).json()
        assert gate["taxa_taker_pct"] > binance["taxa_taker_pct"]


class TestCompatibilidadeComOPainel:
    def test_mantem_o_formato_que_o_front_espera(self, cliente_http):
        d = cliente_http.get("/api/crypto-live-prices").json()
        assert d["success"] is True
        assert isinstance(d["prices"], list)
        assert "source" in d

    def test_cada_item_declara_que_nao_e_sintetico(self, cliente_http):
        d = cliente_http.get("/api/crypto-live-prices").json()
        assert all(item["sintetico"] is False for item in d["prices"])


class TestLimitePorIP:
    def test_estourar_a_janela_devolve_429(self, cliente_http):
        api.limitador.maximo = 3
        for _ in range(3):
            assert cliente_http.get("/api/cripto/saude").status_code == 200
        assert cliente_http.get("/api/cripto/saude").status_code == 429
        api.limitador.maximo = 120


class TestCache:
    def test_segunda_chamada_usa_o_cache(self, cliente_http, monkeypatch):
        chamadas = {"n": 0}
        original = api.cliente.book

        def contando(*a, **k):
            chamadas["n"] += 1
            return original(*a, **k)

        monkeypatch.setattr(api.cliente, "book", contando)
        cliente_http.get("/api/cripto/precos", params={"simbolos": "BTCUSDT"})
        primeira = chamadas["n"]
        cliente_http.get("/api/cripto/precos", params={"simbolos": "BTCUSDT"})
        assert chamadas["n"] == primeira
