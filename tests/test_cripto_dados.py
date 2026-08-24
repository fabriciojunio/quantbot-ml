"""
Testes da camada de dados: exchanges, cache e validação de série.

Nenhum teste aqui toca a rede. As chamadas HTTP são substituídas por respostas
gravadas com o formato real de cada exchange — assim o parser é exercitado de
verdade sem depender de a Binance estar de pé no momento do CI.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from cripto import mercado
from cripto.exchanges import base as exbase
from cripto.exchanges.base import Candle, ErroExchange, valida_intervalo, valida_simbolo
from cripto.exchanges.binance import Binance
from cripto.exchanges.bybit import Bybit
from cripto.exchanges.gate import GateIO, par_gate
from cripto.exchanges.roteador import Roteador

# ── Respostas gravadas, no formato exato de cada API ─────────────────────────

KLINE_BINANCE = [
    [1787504400000, "77429.87", "77443.95", "77148.20", "77176.41", "396.15",
     1787507999999, "30608324.76", 104111, "178.30", "13776382.43", "0"],
    [1787508000000, "77176.41", "77399.30", "77106.60", "77344.20", "77.33",
     1787511599999, "5975983.11", 50000, "38.10", "2947382.10", "0"],
]

CANDLE_GATE = [
    ["1787504400", "14650974.29", "77177.7", "77446.1", "77152.2", "77416.7", "189.58", "true"],
    ["1787508000", "5975983.11", "77344.2", "77399.3", "77106.6", "77177.7", "77.33", "true"],
]

KLINE_BYBIT = {
    "retCode": 0,
    "retMsg": "OK",
    "result": {
        "category": "spot",
        "symbol": "BTCUSDT",
        # A Bybit devolve do mais recente para o mais antigo.
        "list": [
            ["1787508000000", "77180.9", "77403.6", "77105.2", "77339.9", "61.78", "4773480.95"],
            ["1787504400000", "77429.8", "77443.9", "77148.2", "77180.9", "396.15", "30608324.7"],
        ],
    },
}


@pytest.fixture
def sem_rede(monkeypatch):
    """Intercepta `http_json` e devolve a resposta gravada por rota."""
    chamadas = []

    def falso(url, params=None, timeout=10.0, tentativas=3):
        chamadas.append((url, params))
        if "binance.com" in url and "klines" in url:
            return KLINE_BINANCE
        if "gateio.ws" in url and "candlesticks" in url:
            return CANDLE_GATE
        if "bybit.com" in url and "kline" in url:
            return KLINE_BYBIT
        if "depth" in url:
            return {"bids": [["100.0", "2.0"], ["99.5", "3.0"]],
                    "asks": [["100.5", "1.0"], ["101.0", "4.0"]], "lastUpdateId": 1}
        if "trades" in url:
            return [{"time": 1787508000000, "price": "100.0", "qty": "1.5", "isBuyerMaker": False}]
        raise ErroExchange(f"rota não gravada: {url}")

    monkeypatch.setattr(exbase, "http_json", falso)
    for modulo in ("binance", "gate", "bybit"):
        monkeypatch.setattr(f"cripto.exchanges.{modulo}.http_json", falso)
    return chamadas


# ── Validação de entrada ─────────────────────────────────────────────────────


class TestValidacaoDeEntrada:
    def test_simbolo_valido_e_normalizado(self):
        assert valida_simbolo(" btcusdt ") == "BTCUSDT"

    @pytest.mark.parametrize("ruim", ["BTC/USDT", "BTC USDT", "../etc", "BTC;DROP", "AB", "X" * 30])
    def test_simbolo_malformado_e_recusado(self, ruim):
        with pytest.raises(ValueError):
            valida_simbolo(ruim)

    def test_intervalo_desconhecido_e_recusado(self):
        with pytest.raises(ValueError, match="não suportado"):
            valida_intervalo("7m")

    def test_esquema_de_url_fora_de_http_e_bloqueado(self):
        with pytest.raises(ErroExchange, match="esquema"):
            exbase.http_json("file:///etc/passwd")


class TestCandle:
    def test_candle_com_maxima_abaixo_da_minima_e_recusado(self):
        with pytest.raises(ValueError, match="máxima"):
            Candle(1, 10.0, 8.0, 9.0, 9.5, 100.0)

    def test_volume_negativo_e_recusado(self):
        with pytest.raises(ValueError, match="volume"):
            Candle(1, 10.0, 11.0, 9.0, 9.5, -1.0)


# ── Parsers de cada exchange ─────────────────────────────────────────────────


class TestBinance:
    def test_le_o_formato_de_klines(self, sem_rede):
        candles = Binance().candles("BTCUSDT", "1h", 2)
        assert len(candles) == 2
        assert candles[0].abertura == pytest.approx(77429.87)
        assert candles[0].fechamento == pytest.approx(77176.41)

    def test_book_traz_melhor_bid_e_ask(self, sem_rede):
        b = Binance().book("BTCUSDT", 10)
        assert b.melhor_bid == 100.0
        assert b.melhor_ask == 100.5
        assert b.spread == pytest.approx(0.5)

    def test_is_buyer_maker_falso_significa_comprador_agressor(self, sem_rede):
        assert Binance().negocios("BTCUSDT", 1)[0].comprador_agressor is True


class TestGate:
    def test_converte_o_par_para_o_formato_da_casa(self):
        assert par_gate("BTCUSDT") == "BTC_USDT"
        assert par_gate("ETHBTC") == "ETH_BTC"

    def test_par_sem_quote_conhecida_e_recusado(self):
        with pytest.raises(ValueError, match="base/quote"):
            par_gate("ABCDEF")

    def test_le_a_ordem_estranha_de_colunas_da_gate(self, sem_rede):
        """Na Gate o fechamento vem antes da abertura — se o parser trocar,
        todo indicador fica errado sem dar erro nenhum."""
        candles = GateIO().candles("BTCUSDT", "1h", 2)
        assert candles[0].abertura == pytest.approx(77416.7)
        assert candles[0].fechamento == pytest.approx(77177.7)
        assert candles[0].maxima == pytest.approx(77446.1)
        assert candles[0].minima == pytest.approx(77152.2)


class TestBybit:
    def test_reordena_do_mais_antigo_para_o_mais_novo(self, sem_rede):
        candles = Bybit().candles("BTCUSDT", "1h", 2)
        assert candles[0].abertura_ms < candles[1].abertura_ms

    def test_retcode_de_erro_vira_excecao(self, monkeypatch):
        monkeypatch.setattr(
            "cripto.exchanges.bybit.http_json",
            lambda *a, **k: {"retCode": 10001, "retMsg": "parâmetro inválido"},
        )
        with pytest.raises(ErroExchange, match="10001"):
            Bybit().candles("BTCUSDT", "1h", 2)


class TestRoteador:
    def test_usa_a_primeira_fonte_que_responde(self, sem_rede):
        r = Roteador()
        r.candles("BTCUSDT", "1h", 2)
        assert r.ultima_fonte == "binance"

    def test_cai_para_a_proxima_quando_a_primeira_falha(self, sem_rede, monkeypatch):
        def quebrado(*a, **k):
            raise ErroExchange("binance fora do ar")

        monkeypatch.setattr("cripto.exchanges.binance.http_json", quebrado)
        r = Roteador()
        candles = r.candles("BTCUSDT", "1h", 2)
        assert candles
        assert r.ultima_fonte == "gateio"

    def test_todas_falhando_levanta_erro_em_vez_de_inventar(self, monkeypatch):
        def quebrado(*a, **k):
            raise ErroExchange("fora do ar")

        for modulo in ("binance", "gate", "bybit"):
            monkeypatch.setattr(f"cripto.exchanges.{modulo}.http_json", quebrado)
        with pytest.raises(ErroExchange, match="todas as fontes falharam"):
            Roteador().candles("BTCUSDT", "1h", 2)


# ── Normalização e validação de série ────────────────────────────────────────


class TestDataFrame:
    def test_converte_candles_para_dataframe_ordenado(self):
        candles = [
            Candle(1787508000000, 2, 3, 1, 2.5, 10),
            Candle(1787504400000, 1, 2, 0.5, 1.5, 20),
        ]
        df = mercado.para_dataframe(candles)
        assert df.index.is_monotonic_increasing
        assert list(df.columns) == mercado.COLUNAS

    def test_lista_vazia_vira_dataframe_vazio_com_as_colunas(self):
        df = mercado.para_dataframe([])
        assert df.empty
        assert list(df.columns) == mercado.COLUNAS

    def test_duplicatas_ficam_com_a_ultima_leitura(self):
        candles = [
            Candle(1787504400000, 1, 2, 0.5, 1.5, 20),
            Candle(1787504400000, 1, 2, 0.5, 9.9, 20),
        ]
        assert mercado.para_dataframe(candles)["fechamento"].iloc[0] == pytest.approx(9.9)


class TestValidaSerie:
    def _df(self, **trocas):
        base = pd.DataFrame(
            {
                "abertura": [10.0, 11.0],
                "maxima": [12.0, 12.0],
                "minima": [9.0, 10.0],
                "fechamento": [11.0, 11.5],
                "volume": [100.0, 120.0],
            },
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        )
        for coluna, valores in trocas.items():
            base[coluna] = valores
        return base

    def test_serie_boa_passa(self):
        mercado.valida_serie(self._df())

    def test_serie_vazia_falha(self):
        with pytest.raises(ValueError, match="sem candles"):
            mercado.valida_serie(pd.DataFrame())

    def test_preco_zerado_falha(self):
        with pytest.raises(ValueError, match="zerado"):
            mercado.valida_serie(self._df(fechamento=[0.0, 11.5]))

    def test_ohlc_impossivel_falha(self):
        with pytest.raises(ValueError, match="inconsistente"):
            mercado.valida_serie(self._df(maxima=[1.0, 12.0]))

    def test_coluna_faltando_falha(self):
        df = self._df().drop(columns=["volume"])
        with pytest.raises(ValueError, match="faltam colunas"):
            mercado.valida_serie(df)

    def test_indice_fora_de_ordem_falha(self):
        df = self._df().iloc[::-1]
        with pytest.raises(ValueError, match="fora de ordem"):
            mercado.valida_serie(df)


class TestCache:
    def test_grava_e_le_o_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mercado, "CACHE_DIR", tmp_path)
        df = pd.DataFrame(
            {
                "abertura": [10.0],
                "maxima": [12.0],
                "minima": [9.0],
                "fechamento": [11.0],
                "volume": [100.0],
            },
            index=pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC", name="data"),
        )
        mercado._gravar_cache("BTCUSDT", "1h", df)
        lido = mercado._ler_cache("BTCUSDT", "1h")
        assert lido is not None
        assert lido["fechamento"].iloc[0] == pytest.approx(11.0)
        assert lido.index.tz is not None

    def test_cache_inexistente_devolve_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mercado, "CACHE_DIR", tmp_path)
        assert mercado._ler_cache("NAOEXISTE", "1h") is None

    def test_cache_corrompido_nao_derruba_o_processo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mercado, "CACHE_DIR", tmp_path)
        (tmp_path / "BTCUSDT_1h.csv").write_text("isto não é csv válido\x00", encoding="utf-8")
        assert mercado._ler_cache("BTCUSDT", "1h") is None
