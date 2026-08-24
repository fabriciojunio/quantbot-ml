"""Testes do score de confluência e da geração de sinais."""

import numpy as np
import pandas as pd
import pytest

from cripto import indicadores, sinais
from cripto.sinais import ConfigSinal, PesosConfluencia


def serie(n=500, deriva=0.0, semente=11):
    rng = np.random.default_rng(semente)
    base = np.maximum(100 + np.cumsum(rng.normal(deriva, 0.7, n)), 1.0)
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    abertura = base + rng.normal(0, 0.1, n)
    return pd.DataFrame(
        {
            "abertura": abertura,
            "maxima": np.maximum(base + np.abs(rng.normal(0.4, 0.2, n)), abertura),
            "minima": np.minimum(base - np.abs(rng.normal(0.4, 0.2, n)), abertura),
            "fechamento": base,
            "volume": np.abs(rng.normal(1_000, 100, n)),
        },
        index=idx,
    )


def rampa(n=400, inclinacao=0.5):
    """Rampa monotônica. O piso existe para o preço nunca chegar a zero —
    preço negativo quebraria o retorno logarítmico e o teste mediria lixo."""
    base = np.maximum(500 + np.arange(n, dtype=float) * inclinacao, 10.0)
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "abertura": base,
            "maxima": base + abs(inclinacao) * 0.5 + 0.1,
            "minima": base - abs(inclinacao) * 0.5 - 0.1,
            "fechamento": base,
            "volume": np.full(n, 1_000.0),
        },
        index=idx,
    )


class TestPesos:
    def test_pesos_normalizados_somam_um(self):
        assert sum(PesosConfluencia().normalizados().values()) == pytest.approx(1.0)

    def test_peso_negativo_ou_zerado_e_recusado(self):
        with pytest.raises(ValueError, match="positiva"):
            PesosConfluencia(
                tendencia_ema=0, supertrend=0, macd=0, rsi=0,
                adx=0, estocastico=0, obv=0, bollinger=0,
            ).normalizados()

    def test_peso_customizado_muda_o_score(self):
        df = indicadores.tabela_completa(serie(deriva=0.4))
        padrao = sinais.score_confluencia(df).dropna()
        so_rsi = sinais.score_confluencia(
            df,
            PesosConfluencia(
                tendencia_ema=0, supertrend=0, macd=0, rsi=100,
                adx=0, estocastico=0, obv=0, bollinger=0,
            ),
        ).dropna()
        assert not np.allclose(padrao.tail(50), so_rsi.tail(50))


class TestVotos:
    def test_todo_voto_fica_entre_menos_um_e_um(self):
        v = sinais.votos(indicadores.tabela_completa(serie())).dropna()
        assert ((v >= -1.0001) & (v <= 1.0001)).all().all()

    def test_alta_continua_deixa_a_tendencia_no_maximo(self):
        v = sinais.votos(indicadores.tabela_completa(rampa(inclinacao=0.5)))
        assert v["tendencia_ema"].iloc[-1] == pytest.approx(1.0)

    def test_queda_continua_deixa_a_tendencia_no_minimo(self):
        v = sinais.votos(indicadores.tabela_completa(rampa(inclinacao=-0.3)))
        assert v["tendencia_ema"].iloc[-1] == pytest.approx(-1.0)

    def test_rsi_esticado_recua_em_vez_de_saturar(self):
        """RSI acima de 80 é alerta de esticamento, não convicção máxima."""
        df = indicadores.tabela_completa(rampa(inclinacao=1.0))
        v = sinais.votos(df)
        esticado = df["rsi_14"] > 80
        assert esticado.any()
        assert np.allclose(v.loc[esticado, "rsi"], 0.3)


class TestScore:
    def test_score_fica_entre_zero_e_cem(self):
        s = sinais.score_confluencia(indicadores.tabela_completa(serie())).dropna()
        assert s.between(0, 100).all()

    def test_alta_forte_produz_score_alto(self):
        s = sinais.score_confluencia(indicadores.tabela_completa(rampa(inclinacao=0.5)))
        assert s.iloc[-1] > 60

    def test_queda_forte_produz_score_baixo(self):
        s = sinais.score_confluencia(indicadores.tabela_completa(rampa(inclinacao=-0.5)))
        assert s.iloc[-1] < 40

    def test_aquecimento_nao_vira_score_neutro_falso(self):
        s = sinais.score_confluencia(indicadores.tabela_completa(serie()))
        assert s.iloc[:20].isna().all()


class TestGerarSinais:
    def test_posicao_alvo_so_assume_valores_validos(self):
        d = sinais.gerar_sinais(serie(), ConfigSinal(permitir_venda=True))
        assert set(d["posicao_alvo"].unique()) <= {-1, 0, 1}

    def test_sem_permitir_venda_nunca_aparece_posicao_vendida(self):
        d = sinais.gerar_sinais(serie(deriva=-0.3), ConfigSinal(permitir_venda=False))
        assert (d["posicao_alvo"] >= 0).all()

    def test_filtro_de_adx_reduz_o_numero_de_sinais(self):
        sem_filtro = sinais.gerar_sinais(serie(), ConfigSinal(adx_minimo=0.0))
        com_filtro = sinais.gerar_sinais(serie(), ConfigSinal(adx_minimo=30.0))
        assert (com_filtro["posicao_alvo"] != 0).sum() <= (sem_filtro["posicao_alvo"] != 0).sum()

    def test_stop_de_compra_fica_abaixo_e_alvo_acima(self):
        d = sinais.gerar_sinais(serie()).dropna(subset=["atr_14"])
        assert (d["stop_long"] < d["fechamento"]).all()
        assert (d["alvo_long"] > d["fechamento"]).all()

    def test_limite_impossivel_zera_os_sinais(self):
        d = sinais.gerar_sinais(serie(), ConfigSinal(limite_compra=101.0))
        assert (d["posicao_alvo"] == 0).all()


class TestExplicacao:
    def test_explicacao_soma_ao_score_da_barra(self):
        d = sinais.gerar_sinais(serie(deriva=0.3))
        quando = d.index[-1]
        e = sinais.explica_barra(d, quando)
        soma = sum(
            i["contribuicao"] for i in e["indicadores"].values() if i["contribuicao"] is not None
        )
        assert soma == pytest.approx(e["score"], abs=0.5)

    def test_explicacao_lista_todos_os_indicadores(self):
        d = sinais.gerar_sinais(serie())
        e = sinais.explica_barra(d, d.index[-1])
        assert len(e["indicadores"]) == 8

    def test_peso_declarado_bate_com_a_configuracao(self):
        d = sinais.gerar_sinais(serie())
        e = sinais.explica_barra(d, d.index[-1])
        total = sum(i["peso_pct"] for i in e["indicadores"].values())
        assert total == pytest.approx(100.0, abs=0.5)
