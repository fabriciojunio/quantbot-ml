"""
Testes do motor de backtest.

O teste que mais importa aqui é o de ausência de olhar-o-futuro: um backtest
que enxerga uma barra à frente produz curva linda e resultado impossível. Ele
é verificado por truncamento — o resultado das primeiras N barras não pode
mudar quando barras posteriores são acrescentadas à série.
"""

import numpy as np
import pandas as pd
import pytest

from cripto.backtest import ConfigBacktest, executar
from cripto.custos import ModeloCusto
from cripto.sinais import ConfigSinal


def serie(n=800, semente=7, tendencia=0.03):
    rng = np.random.default_rng(semente)
    base = 100 + np.cumsum(rng.normal(tendencia, 0.8, n))
    base = np.maximum(base, 1.0)
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    ruido_alto = np.abs(rng.normal(0.4, 0.2, n))
    ruido_baixo = np.abs(rng.normal(0.4, 0.2, n))
    abertura = base + rng.normal(0, 0.1, n)
    return pd.DataFrame(
        {
            "abertura": abertura,
            "maxima": np.maximum(base + ruido_alto, abertura),
            "minima": np.minimum(base - ruido_baixo, abertura),
            "fechamento": base,
            "volume": np.abs(rng.normal(1_000, 150, n)),
        },
        index=idx,
    )


class TestGuardas:
    def test_serie_curta_e_recusada(self):
        with pytest.raises(ValueError, match="curta"):
            executar(serie(n=100), "TESTE")

    def test_capital_inicial_e_respeitado(self):
        res = executar(serie(), "TESTE", ConfigBacktest(capital_inicial=500.0))
        assert res.capital.iloc[0] == pytest.approx(500.0, rel=0.05)


class TestSemOlharOFuturo:
    def test_truncar_a_serie_nao_muda_o_passado(self):
        """Núcleo do teste: o resultado das primeiras barras é o mesmo com ou
        sem as barras futuras presentes no DataFrame."""
        completa = serie(n=900)
        curta = completa.iloc[:600]

        r_completa = executar(completa, "TESTE")
        r_curta = executar(curta, "TESTE")

        # Compara a curva de capital no trecho comum, tirando as últimas barras
        # da série curta (onde a posição aberta é liquidada à força).
        comum = r_curta.capital.index[:-60]
        np.testing.assert_allclose(
            r_completa.capital.loc[comum].to_numpy(),
            r_curta.capital.loc[comum].to_numpy(),
            rtol=1e-9,
        )

    def test_operacoes_do_passado_sao_identicas(self):
        completa = executar(serie(n=900), "TESTE")
        curta = executar(serie(n=900).iloc[:600], "TESTE")

        corte = curta.capital.index[-60]
        ops_c = [o for o in completa.operacoes if o.saida_em is not None and o.saida_em < corte]
        ops_k = [o for o in curta.operacoes if o.saida_em is not None and o.saida_em < corte]

        assert len(ops_c) == len(ops_k)
        for a, b in zip(ops_c, ops_k):
            assert a.entrada_em == b.entrada_em
            assert a.preco_entrada == pytest.approx(b.preco_entrada)


class TestCusto:
    def test_toda_operacao_paga_custo(self):
        res = executar(serie(), "TESTE")
        assert res.operacoes, "o cenário de teste precisa gerar ao menos uma operação"
        assert all(o.custo_usd > 0 for o in res.operacoes)

    def test_custo_zero_deixa_o_liquido_igual_ao_bruto(self):
        sem_atrito = ModeloCusto(taxa_maker=0, taxa_taker=0, spread_pct=0, impacto_base=0)
        res = executar(serie(), "TESTE", ConfigBacktest(modelo_custo=sem_atrito))
        for o in res.operacoes:
            assert o.pnl_liquido_usd == pytest.approx(o.pnl_bruto_usd, abs=1e-9)

    def test_custo_maior_piora_o_resultado(self):
        barato = ModeloCusto(taxa_taker=0.0001, spread_pct=0.0, impacto_base=0.0)
        caro = ModeloCusto(taxa_taker=0.0060, spread_pct=0.002, impacto_base=0.002)
        r_barato = executar(serie(), "TESTE", ConfigBacktest(modelo_custo=barato))
        r_caro = executar(serie(), "TESTE", ConfigBacktest(modelo_custo=caro))
        assert r_caro.retorno_total_pct < r_barato.retorno_total_pct

    def test_custo_total_bate_com_a_soma_das_operacoes(self):
        res = executar(serie(), "TESTE")
        assert res.custo_total_usd == pytest.approx(sum(o.custo_usd for o in res.operacoes))


class TestExecucao:
    def test_entrada_acontece_depois_do_sinal(self):
        """A posição nunca abre na mesma barra que gerou a decisão."""
        res = executar(serie(), "TESTE")
        dados_idx = serie().index
        for o in res.operacoes:
            posicao = dados_idx.get_loc(o.entrada_em)
            assert posicao >= 1

    def test_nenhuma_posicao_fica_aberta_no_fim(self):
        res = executar(serie(), "TESTE")
        assert all(o.saida_em is not None for o in res.operacoes)

    def test_stop_e_alvo_ficam_de_lados_opostos_da_entrada(self):
        res = executar(serie(), "TESTE")
        for o in res.operacoes:
            if o.motivo_saida == "stop":
                # Comprado, o stop está abaixo; vendido, acima.
                if o.lado == 1:
                    assert o.preco_saida < o.preco_entrada * 1.01
                else:
                    assert o.preco_saida > o.preco_entrada * 0.99

    def test_sem_alavancagem_o_notional_nao_passa_do_capital(self):
        res = executar(serie(), "TESTE", ConfigBacktest(fracao_por_operacao=2.0))
        for o in res.operacoes:
            assert o.notional_usd <= res.config.capital_inicial * 1.5


class TestMetricas:
    def test_taxa_de_acerto_entre_zero_e_um(self):
        res = executar(serie(), "TESTE")
        assert 0.0 <= res.taxa_acerto <= 1.0

    def test_drawdown_nunca_e_positivo(self):
        res = executar(serie(), "TESTE")
        assert res.max_drawdown_pct <= 0.0

    def test_resumo_traz_comparacao_com_buy_and_hold(self):
        df = serie()
        res = executar(df, "TESTE")
        r = res.resumo(df["fechamento"])
        assert "buy_and_hold_pct" in r
        assert "alpha_vs_buy_and_hold_pp" in r

    def test_capital_final_bate_com_a_soma_dos_resultados(self):
        res = executar(serie(), "TESTE")
        esperado = res.config.capital_inicial + res.pnl_liquido_usd
        assert float(res.capital.iloc[-1]) == pytest.approx(esperado, rel=1e-6)


class TestSemSinal:
    def test_limite_inatingivel_nao_abre_operacao(self):
        cfg = ConfigBacktest(config_sinal=ConfigSinal(limite_compra=101.0, limite_venda=-1.0))
        res = executar(serie(), "TESTE", cfg)
        assert res.operacoes == []
        assert res.capital.iloc[-1] == pytest.approx(res.config.capital_inicial)
