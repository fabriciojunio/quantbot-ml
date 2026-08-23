"""Testes dos indicadores: valor conferido contra a definição, não contra si mesmo."""

import numpy as np
import pandas as pd
import pytest

from cripto import indicadores


@pytest.fixture
def serie_ohlcv():
    """Série determinística com tendência e ruído — sem rede, sem aleatoriedade global."""
    rng = np.random.default_rng(42)
    n = 400
    base = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    alta = base + np.abs(rng.normal(0.5, 0.3, n))
    baixa = base - np.abs(rng.normal(0.5, 0.3, n))
    return pd.DataFrame(
        {
            "abertura": base + rng.normal(0, 0.1, n),
            "maxima": np.maximum(alta, base),
            "minima": np.minimum(baixa, base),
            "fechamento": base,
            "volume": np.abs(rng.normal(1000, 200, n)),
        },
        index=idx,
    )


class TestEMA:
    def test_ema_de_serie_constante_e_a_constante(self):
        s = pd.Series([50.0] * 100)
        assert indicadores.ema(s, 20).iloc[-1] == pytest.approx(50.0)

    def test_aquecimento_fica_nan(self):
        s = pd.Series(range(100), dtype=float)
        e = indicadores.ema(s, 20)
        assert e.iloc[:19].isna().all()
        assert e.iloc[19:].notna().all()

    def test_ema_reage_mais_rapido_que_sma(self):
        s = pd.Series([10.0] * 50 + [20.0] * 10)
        assert indicadores.ema(s, 20).iloc[-1] > indicadores.sma(s, 20).iloc[-1]


class TestRSI:
    def test_alta_ininterrupta_satura_em_100(self):
        s = pd.Series(np.arange(1, 101, dtype=float))
        assert indicadores.rsi(s, 14).iloc[-1] == pytest.approx(100.0)

    def test_queda_ininterrupta_vai_a_zero(self):
        s = pd.Series(np.arange(100, 0, -1, dtype=float))
        assert indicadores.rsi(s, 14).iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_fica_sempre_entre_0_e_100(self, serie_ohlcv):
        r = indicadores.rsi(serie_ohlcv["fechamento"]).dropna()
        assert r.between(0, 100).all()


class TestMACD:
    def test_histograma_e_a_diferenca_entre_linha_e_sinal(self, serie_ohlcv):
        m = indicadores.macd(serie_ohlcv["fechamento"]).dropna()
        assert np.allclose(m["histograma"], m["macd"] - m["sinal"])

    def test_linha_e_a_diferenca_das_emas(self, serie_ohlcv):
        p = serie_ohlcv["fechamento"]
        esperado = indicadores.ema(p, 12) - indicadores.ema(p, 26)
        obtido = indicadores.macd(p)["macd"]
        assert np.allclose(esperado.dropna(), obtido.dropna())


class TestBollinger:
    def test_bandas_cercam_a_media(self, serie_ohlcv):
        b = indicadores.bollinger(serie_ohlcv["fechamento"]).dropna()
        assert (b["superior"] > b["media"]).all()
        assert (b["inferior"] < b["media"]).all()

    def test_pct_b_meio_da_banda_e_meio(self):
        s = pd.Series([100.0] * 30 + [101.0])
        b = indicadores.bollinger(s, periodo=20)
        assert b["pct_b"].iloc[-1] > 0.5


class TestATR:
    def test_atr_e_positivo(self, serie_ohlcv):
        assert (indicadores.atr(serie_ohlcv).dropna() > 0).all()

    def test_true_range_cobre_o_gap(self):
        df = pd.DataFrame(
            {
                "maxima": [10.0, 20.0],
                "minima": [9.0, 19.0],
                "fechamento": [9.5, 19.5],
            }
        )
        # Segundo candle abriu com gap: o TR precisa medir do fechamento anterior.
        assert indicadores.true_range(df).iloc[1] == pytest.approx(20.0 - 9.5)


class TestADX:
    def test_adx_entre_0_e_100(self, serie_ohlcv):
        a = indicadores.adx(serie_ohlcv).dropna()
        assert a["adx"].between(0, 100).all()

    def test_tendencia_forte_eleva_o_adx(self):
        n = 200
        base = np.arange(n, dtype=float) * 2 + 100
        idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "abertura": base,
                "maxima": base + 1,
                "minima": base - 1,
                "fechamento": base,
                "volume": np.ones(n) * 100,
            },
            index=idx,
        )
        assert indicadores.adx(df)["adx"].iloc[-1] > 40


class TestSuperTrend:
    def test_tendencia_e_sempre_mais_ou_menos_um(self, serie_ohlcv):
        st = indicadores.supertrend(serie_ohlcv)
        assert set(st["tendencia"].unique()) <= {-1, 1}

    def test_alta_persistente_mantem_tendencia_positiva(self):
        n = 200
        base = np.arange(n, dtype=float) + 100
        idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "abertura": base,
                "maxima": base + 0.5,
                "minima": base - 0.5,
                "fechamento": base,
                "volume": np.ones(n) * 100,
            },
            index=idx,
        )
        assert indicadores.supertrend(df)["tendencia"].iloc[-1] == 1


class TestOBV:
    def test_obv_sobe_quando_preco_sobe(self):
        df = pd.DataFrame(
            {
                "fechamento": [10.0, 11.0, 12.0, 13.0],
                "volume": [100.0, 100.0, 100.0, 100.0],
            }
        )
        o = indicadores.obv(df)
        assert o.iloc[-1] == pytest.approx(300.0)

    def test_obv_cai_quando_preco_cai(self):
        df = pd.DataFrame(
            {"fechamento": [13.0, 12.0, 11.0], "volume": [100.0, 100.0, 100.0]}
        )
        assert indicadores.obv(df).iloc[-1] == pytest.approx(-200.0)


class TestTabelaCompleta:
    def test_traz_todas_as_colunas_esperadas(self, serie_ohlcv):
        t = indicadores.tabela_completa(serie_ohlcv)
        for coluna in ("rsi_14", "macd_hist", "adx", "atr_14", "supertrend_tendencia", "obv"):
            assert coluna in t.columns

    def test_nao_inventa_valor_no_aquecimento(self, serie_ohlcv):
        t = indicadores.tabela_completa(serie_ohlcv)
        # A EMA 200 não pode existir antes de 200 barras — preencher aqui seria
        # vazamento de informação.
        assert t["ema_200"].iloc[:199].isna().all()

    def test_nao_altera_o_dataframe_original(self, serie_ohlcv):
        antes = serie_ohlcv.copy()
        indicadores.tabela_completa(serie_ohlcv)
        pd.testing.assert_frame_equal(serie_ohlcv, antes)
