"""
Testes de sanidade do motor contra resultados conhecidos.

Os testes de unidade dizem que cada peça faz o que promete. Estes dizem que a
montagem toda produz o número certo em cenários onde a resposta pode ser
calculada à mão. É a defesa contra o erro mais perigoso de um backtest: rodar
sem falhar e devolver um valor errado.
"""

import numpy as np
import pandas as pd
import pytest

from cripto.backtest import ConfigBacktest, executar
from cripto.custos import ModeloCusto
from cripto.sinais import ConfigSinal


def rampa(n=600, taxa=0.001, inicio=100.0):
    """Alta composta constante, sem ruído: o retorno é conhecido de antemão."""
    base = inicio * (1 + taxa) ** np.arange(n)
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "abertura": base,
            "maxima": base * 1.0005,
            "minima": base * 0.9995,
            "fechamento": base,
            "volume": np.full(n, 1_000.0),
        },
        index=idx,
    )


def plana(n=600, preco=100.0):
    """Preço absolutamente parado. Qualquer resultado diferente de -custo é bug."""
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "abertura": np.full(n, preco),
            "maxima": np.full(n, preco),
            "minima": np.full(n, preco),
            "fechamento": np.full(n, preco),
            "volume": np.full(n, 1_000.0),
        },
        index=idx,
    )


class TestMercadoParado:
    """Com preço congelado, a estratégia não pode ganhar dinheiro."""

    def test_nunca_termina_com_lucro(self):
        res = executar(plana(), "PARADO", ConfigBacktest(config_sinal=ConfigSinal(adx_minimo=0.0)))
        assert float(res.capital.iloc[-1]) <= res.config.capital_inicial + 1e-6

    def test_todo_resultado_e_negativo_por_causa_do_custo(self):
        res = executar(plana(), "PARADO", ConfigBacktest(config_sinal=ConfigSinal(adx_minimo=0.0)))
        for o in res.operacoes:
            assert o.pnl_liquido_usd < 0

    def test_sem_custo_o_resultado_e_exatamente_zero(self):
        sem_atrito = ModeloCusto(taxa_maker=0, taxa_taker=0, spread_pct=0, impacto_base=0)
        res = executar(
            plana(),
            "PARADO",
            ConfigBacktest(modelo_custo=sem_atrito, config_sinal=ConfigSinal(adx_minimo=0.0)),
        )
        assert float(res.capital.iloc[-1]) == pytest.approx(res.config.capital_inicial, abs=1e-6)


class TestAltaConstante:
    """Numa alta ininterrupta, comprar e segurar é o teto do que dá para ganhar."""

    def test_estrategia_nao_supera_o_buy_and_hold(self):
        df = rampa(taxa=0.001)
        res = executar(df, "SUBINDO", ConfigBacktest(fracao_por_operacao=1.0))
        bh = res.buy_and_hold_pct(df["fechamento"])
        assert res.retorno_total_pct <= bh + 1e-6

    def test_com_stop_largo_a_operacao_fecha_no_alvo(self):
        df = rampa(taxa=0.002)
        res = executar(df, "SUBINDO", ConfigBacktest())
        motivos = {o.motivo_saida for o in res.operacoes}
        assert motivos & {"alvo", "tempo", "fim_da_serie"}
        assert "stop" not in motivos


class TestConservacaoDeCapital:
    def test_capital_final_e_a_soma_de_tudo_que_aconteceu(self):
        df = rampa(taxa=0.0008)
        res = executar(df, "SUBINDO", ConfigBacktest(capital_inicial=2_500.0))
        esperado = 2_500.0 + sum(o.pnl_liquido_usd for o in res.operacoes)
        assert float(res.capital.iloc[-1]) == pytest.approx(esperado, rel=1e-9)

    def test_bruto_menos_custo_e_o_liquido_em_toda_operacao(self):
        df = rampa(taxa=0.0008)
        res = executar(df, "SUBINDO")
        for o in res.operacoes:
            assert o.pnl_bruto_usd - o.custo_usd == pytest.approx(o.pnl_liquido_usd, abs=1e-9)

    def test_capital_nunca_fica_negativo(self):
        """Sem alavancagem, perder mais que o capital é impossível."""
        rng = np.random.default_rng(3)
        n = 800
        base = np.maximum(100 * np.exp(np.cumsum(rng.normal(-0.002, 0.03, n))), 0.5)
        idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "abertura": base,
                "maxima": base * 1.02,
                "minima": base * 0.98,
                "fechamento": base,
                "volume": np.full(n, 1_000.0),
            },
            index=idx,
        )
        res = executar(df, "CAINDO", ConfigBacktest(config_sinal=ConfigSinal(adx_minimo=0.0)))
        assert (res.capital > 0).all()


class TestCustoProporcional:
    def test_dobrar_a_taxa_dobra_o_custo_de_taxa(self):
        df = rampa(taxa=0.0008)
        cfg_a = ConfigBacktest(
            modelo_custo=ModeloCusto(taxa_taker=0.0010, spread_pct=0.0, impacto_base=0.0)
        )
        cfg_b = ConfigBacktest(
            modelo_custo=ModeloCusto(taxa_taker=0.0020, spread_pct=0.0, impacto_base=0.0)
        )
        a = executar(df, "SUBINDO", cfg_a)
        b = executar(df, "SUBINDO", cfg_b)
        assert len(a.operacoes) == len(b.operacoes)

        # Fica logo abaixo de 2x, e isso é o comportamento certo: a taxa maior
        # corrói o capital, o capital menor gera posição menor, e a posição
        # menor paga menos taxa em dólar. Exigir exatamente 2x seria pedir que
        # o motor ignorasse a composição.
        razao = b.custo_total_usd / a.custo_total_usd
        assert 1.9 < razao <= 2.0

    def test_posicao_maior_paga_mais_slippage(self):
        df = rampa(taxa=0.0008)
        pequena = executar(df, "SUBINDO", ConfigBacktest(fracao_por_operacao=0.05))
        grande = executar(df, "SUBINDO", ConfigBacktest(fracao_por_operacao=0.50))
        custo_relativo_p = pequena.custo_total_usd / sum(o.notional_usd for o in pequena.operacoes)
        custo_relativo_g = grande.custo_total_usd / sum(o.notional_usd for o in grande.operacoes)
        assert custo_relativo_g > custo_relativo_p


class TestDeterminismo:
    def test_rodar_duas_vezes_da_o_mesmo_resultado(self):
        df = rampa(taxa=0.0008)
        a = executar(df, "SUBINDO")
        b = executar(df, "SUBINDO")
        assert a.retorno_total_pct == pytest.approx(b.retorno_total_pct)
        assert len(a.operacoes) == len(b.operacoes)

    def test_o_motor_nao_altera_a_serie_recebida(self):
        df = rampa()
        antes = df.copy()
        executar(df, "SUBINDO")
        pd.testing.assert_frame_equal(df, antes)
