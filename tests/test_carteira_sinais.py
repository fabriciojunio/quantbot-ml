"""
Testes dos sinais de alocação.

Todos usam séries construídas à mão, com resposta conhecida: rampa de alta,
rampa de queda, preço parado, e um caso com ruído determinístico. Nenhum toca
a rede. O objetivo é garantir que o sinal diz o que promete — e, principalmente,
que ele nunca enxerga o futuro.
"""

import numpy as np
import pandas as pd
import pytest

from carteira import sinais

TICKERS = ["A", "B", "C", "D"]


def _idx(n):
    return pd.bdate_range("2020-01-01", periods=n)


def rampa(n=400, inclinacoes=(0.5, 0.2, -0.2, 0.0), base=100.0):
    """Séries monotônicas com inclinações diferentes por ativo."""
    dados = {}
    for t, inc in zip(TICKERS, inclinacoes):
        dados[t] = np.maximum(base + np.arange(n, dtype=float) * inc, 1.0)
    return pd.DataFrame(dados, index=_idx(n))


def ruidosa(n=500, semente=7):
    rng = np.random.default_rng(semente)
    dados = {}
    for i, t in enumerate(TICKERS):
        deriva = 0.0004 * (i - 1)
        dados[t] = 100 * np.exp(np.cumsum(rng.normal(deriva, 0.012, n)))
    return pd.DataFrame(dados, index=_idx(n))


def parada(n=400, preco=100.0):
    return pd.DataFrame({t: np.full(n, preco) for t in TICKERS}, index=_idx(n))


class TestRetornoPeriodo:
    @pytest.mark.parametrize("dias", [1, 5, 21, 63, 126, 252])
    def test_janelas_validas(self, dias):
        r = sinais.retorno_periodo(rampa(), dias)
        assert r.iloc[:dias].isna().all().all()
        assert r.iloc[dias:].notna().all().all()

    @pytest.mark.parametrize("dias", [0, -1, -21])
    def test_janela_invalida_e_recusada(self, dias):
        with pytest.raises(ValueError, match="positiva"):
            sinais.retorno_periodo(rampa(), dias)

    def test_preco_parado_da_retorno_zero(self):
        r = sinais.retorno_periodo(parada(), 21).dropna()
        assert np.allclose(r, 0.0)

    def test_alta_da_retorno_positivo(self):
        r = sinais.retorno_periodo(rampa(), 21)
        assert float(r["A"].iloc[-1]) > 0

    def test_queda_da_retorno_negativo(self):
        r = sinais.retorno_periodo(rampa(), 21)
        assert float(r["C"].iloc[-1]) < 0


class TestMomentumAbsoluto:
    def test_alta_forte_da_momentum_positivo(self):
        m = sinais.momentum_absoluto(rampa())
        assert float(m["A"].iloc[-1]) > 0

    def test_queda_da_momentum_negativo(self):
        m = sinais.momentum_absoluto(rampa())
        assert float(m["C"].iloc[-1]) < 0

    def test_preco_parado_da_momentum_zero(self):
        m = sinais.momentum_absoluto(parada()).dropna()
        assert np.allclose(m, 0.0)

    def test_ordena_pela_inclinacao(self):
        m = sinais.momentum_absoluto(rampa()).iloc[-1]
        assert m["A"] > m["B"] > m["D"] > m["C"]

    @pytest.mark.parametrize(
        "janelas",
        [("1m",), ("3m",), ("12m",), ("1m", "3m"), ("1m", "3m", "12m"), ("6m", "12m")],
    )
    def test_combinacoes_de_janela(self, janelas):
        m = sinais.momentum_absoluto(rampa(), janelas)
        assert m.shape == (400, 4)

    def test_janela_desconhecida_e_recusada(self):
        with pytest.raises(ValueError, match="desconhecida"):
            sinais.momentum_absoluto(rampa(), ("7m",))

    def test_aquecimento_da_janela_mais_longa_fica_nan(self):
        m = sinais.momentum_absoluto(rampa(), ("12m",))
        assert m.iloc[:251].isna().all().all()

    def test_nao_altera_a_entrada(self):
        px = rampa()
        antes = px.copy()
        sinais.momentum_absoluto(px)
        pd.testing.assert_frame_equal(px, antes)


class TestMomentumRelativo:
    def test_ranking_fica_entre_zero_e_um(self):
        r = sinais.momentum_relativo(ruidosa()).dropna()
        assert ((r >= 0) & (r <= 1)).all().all()

    def test_melhor_ativo_fica_no_topo(self):
        r = sinais.momentum_relativo(rampa(), dias=252, pular_ultimo_mes=False)
        assert float(r["A"].iloc[-1]) == pytest.approx(1.0)

    def test_pior_ativo_fica_na_base(self):
        r = sinais.momentum_relativo(rampa(), dias=252, pular_ultimo_mes=False)
        assert float(r["C"].iloc[-1]) == pytest.approx(0.25)

    def test_pular_o_ultimo_mes_muda_o_sinal(self):
        com = sinais.momentum_relativo(ruidosa(), pular_ultimo_mes=True)
        sem = sinais.momentum_relativo(ruidosa(), pular_ultimo_mes=False)
        assert not np.allclose(com.tail(20).fillna(0), sem.tail(20).fillna(0))

    @pytest.mark.parametrize("dias", [21, 63, 126, 252])
    def test_varias_janelas(self, dias):
        r = sinais.momentum_relativo(ruidosa(), dias=dias)
        assert r.shape[1] == 4


class TestVolatilidade:
    def test_preco_parado_tem_vol_zero(self):
        v = sinais.volatilidade(parada()).dropna()
        assert np.allclose(v, 0.0)

    def test_vol_e_sempre_nao_negativa(self):
        v = sinais.volatilidade(ruidosa()).dropna()
        assert (v >= 0).all().all()

    def test_anualizar_multiplica_por_raiz_de_252(self):
        px = ruidosa()
        diaria = sinais.volatilidade(px, anualizar=False)
        anual = sinais.volatilidade(px, anualizar=True)
        assert np.allclose(anual.dropna(), diaria.dropna() * np.sqrt(252))

    @pytest.mark.parametrize("dias", [21, 42, 63, 126, 252])
    def test_varias_janelas(self, dias):
        v = sinais.volatilidade(ruidosa(), dias=dias)
        assert v.shape[1] == 4

    def test_serie_mais_agitada_tem_vol_maior(self):
        rng = np.random.default_rng(3)
        n = 400
        px = pd.DataFrame(
            {
                "calmo": 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n))),
                "agitado": 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))),
            },
            index=_idx(n),
        )
        v = sinais.volatilidade(px).iloc[-1]
        assert v["agitado"] > v["calmo"]


class TestFiltroDeTendencia:
    def test_alta_continua_fica_acima_da_media(self):
        f = sinais.filtro_de_tendencia(rampa(n=400), media_dias=200)
        assert bool(f["A"].iloc[-1]) is True

    def test_queda_continua_fica_abaixo_da_media(self):
        f = sinais.filtro_de_tendencia(rampa(n=400), media_dias=200)
        assert bool(f["C"].iloc[-1]) is False

    @pytest.mark.parametrize("media", [50, 100, 200])
    def test_aquecimento_nao_dispara_sinal(self, media):
        f = sinais.filtro_de_tendencia(rampa(n=400), media_dias=media)
        # Antes do aquecimento a média é NaN e a comparação vira False.
        assert not f.iloc[: media - 1].any().any()

    def test_devolve_apenas_booleanos(self):
        f = sinais.filtro_de_tendencia(ruidosa())
        assert set(np.unique(f.to_numpy())) <= {True, False}


class TestCorrelacaoMedia:
    def test_fica_entre_menos_um_e_um(self):
        c = sinais.correlacao_media(ruidosa()).dropna()
        assert ((c >= -1.001) & (c <= 1.001)).all()

    def test_series_identicas_tem_correlacao_um(self):
        n = 300
        rng = np.random.default_rng(5)
        base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        px = pd.DataFrame({"A": base, "B": base}, index=_idx(n))
        assert float(sinais.correlacao_media(px).iloc[-1]) == pytest.approx(1.0)

    def test_serie_curta_devolve_vazio(self):
        assert sinais.correlacao_media(ruidosa(n=50), dias=126).empty


class TestSinalComposto:
    def test_score_fica_entre_zero_e_um(self):
        s = sinais.sinal_composto(ruidosa()).dropna()
        assert ((s >= 0) & (s <= 1)).all().all()

    def test_melhor_ativo_tem_score_maior(self):
        s = sinais.sinal_composto(rampa()).iloc[-1]
        assert s["A"] > s["C"]

    def test_descontar_cdi_reduz_o_score_de_todo_mundo(self):
        px = rampa(inclinacoes=(0.05, 0.04, 0.03, 0.02))
        cdi = pd.Series(
            np.exp(np.arange(len(px)) * 0.0005), index=px.index
        )  # CDI subindo forte
        sem = sinais.sinal_composto(px).iloc[-1].mean()
        com = sinais.sinal_composto(px, cdi).iloc[-1].mean()
        assert com <= sem


class TestDiagnostico:
    def test_lista_todos_os_ativos(self):
        d = sinais.diagnostico(ruidosa())
        assert set(d["ativos"]) == set(TICKERS)

    def test_traz_os_quatro_campos_por_ativo(self):
        d = sinais.diagnostico(ruidosa())
        for info in d["ativos"].values():
            assert set(info) == {
                "momentum_absoluto_pct",
                "posicao_no_ranking",
                "volatilidade_anual_pct",
                "acima_da_media_200",
            }

    def test_aceita_data_explicita(self):
        px = ruidosa()
        quando = px.index[-10]
        assert sinais.diagnostico(px, quando)["quando"] == str(quando)[:10]


class TestSemOlharOFuturo:
    """A garantia estrutural: truncar a série não pode mudar o passado."""

    @pytest.mark.parametrize(
        "funcao",
        [
            sinais.momentum_absoluto,
            sinais.momentum_relativo,
            sinais.volatilidade,
            sinais.filtro_de_tendencia,
            sinais.sinal_composto,
        ],
    )
    def test_truncar_nao_muda_o_que_ja_passou(self, funcao):
        completa = ruidosa(n=500)
        curta = completa.iloc[:400]

        a = funcao(completa).iloc[:400]
        b = funcao(curta)

        pd.testing.assert_frame_equal(
            a.astype(float).fillna(-999), b.astype(float).fillna(-999), check_freq=False
        )

    @pytest.mark.parametrize("dias", [21, 63, 126, 252])
    def test_retorno_periodo_nao_olha_a_frente(self, dias):
        completa = ruidosa(n=500)
        a = sinais.retorno_periodo(completa, dias).iloc[:400]
        b = sinais.retorno_periodo(completa.iloc[:400], dias)
        pd.testing.assert_frame_equal(a.fillna(-999), b.fillna(-999), check_freq=False)
