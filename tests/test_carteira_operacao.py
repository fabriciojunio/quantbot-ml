"""
Testes da operação diária.

Três coisas precisam ser verdade todo dia, e são o que este arquivo protege:

* o aporte do mês entra **uma vez**, mesmo que o dia seja reprocessado
* o caixa nunca fica negativo, porque caixa negativo é alavancagem
* nenhum caminho do código liga dinheiro real

O primeiro é o mais fácil de quebrar sem perceber: reexecutar um dia é normal
quando a fonte corrige um preço, e um aporte não idempotente transformaria cada
reexecução em dinheiro que nunca existiu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auditoria import TipoEvento, abrir, verificar
from carteira import operacao
from carteira.conta import Conta
from carteira.multimoeda import TETO_DE_CRIPTO


@pytest.fixture
def livro(tmp_path):
    lv = abrir(tmp_path / "op.sqlite3")
    yield lv
    lv.fechar()


def mercado(dias=800, semente=4, inicio="2022-01-03"):
    idx = pd.bdate_range(inicio, periods=dias)
    rng = np.random.default_rng(semente)
    tickers = ["BOVA11.SA", "SMAL11.SA", "IVVB11.SA", "IMAB11.SA",
               "B5P211.SA", "GOLD11.SA", "XFIX11.SA"]
    precos = pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, dias))) for t in tickers},
        index=idx,
    )
    cdi = pd.Series(np.full(dias, 0.00045), index=idx)
    return precos, cdi


CONFIG = operacao.Configuracao(aporte_mensal=500.0, risco_pct=80.0, fatia_cripto=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

class TesteConfiguracao:
    def test_valores_padrao_sao_os_combinados(self):
        c = operacao.Configuracao()
        assert c.aporte_mensal == 500.0
        assert c.risco_pct == 80.0

    def test_risco_fora_da_faixa_e_recusado(self):
        with pytest.raises(ValueError, match="0 a 100"):
            operacao.Configuracao(risco_pct=150)

    def test_aporte_negativo_e_recusado(self):
        with pytest.raises(ValueError, match="negativo"):
            operacao.Configuracao(aporte_mensal=-1)

    def test_cripto_acima_do_teto_e_recusada(self):
        with pytest.raises(ValueError, match="cripto"):
            operacao.Configuracao(fatia_cripto=0.5)

    def test_cripto_no_teto_e_aceita(self):
        operacao.Configuracao(fatia_cripto=TETO_DE_CRIPTO)

    def test_rebalanceamento_zero_e_recusado(self):
        with pytest.raises(ValueError, match="ao menos um mês"):
            operacao.Configuracao(rebalancear_a_cada_meses=0)

    def test_dicionario_traz_o_que_a_tela_precisa(self):
        d = operacao.Configuracao().para_dict()
        assert {"aporte_mensal", "risco_pct", "fatia_cripto_pct"} <= set(d)


# ─────────────────────────────────────────────────────────────────────────────
# Trava de dinheiro real
# ─────────────────────────────────────────────────────────────────────────────

class TesteTrava:
    def test_dinheiro_real_esta_desligado(self):
        assert operacao.DINHEIRO_REAL is False

    def test_a_trava_nao_e_configuravel(self):
        """Ligar dinheiro real tem que ser alteração de código, não flag."""
        campos = set(operacao.Configuracao().__dataclass_fields__)
        assert not any("real" in c for c in campos)

    def test_toda_ordem_registrada_diz_que_e_simulada(self, livro):
        precos, cdi = mercado(120)
        conta, decisao = operacao.rodar_dia(livro, CONFIG, precos, cdi, hoje=precos.index[0])
        operacao.registrar_dia(
            livro, conta, decisao,
            {t: float(v) for t, v in precos.iloc[0].items()}, 1.0, 1.0,
        )
        for e in livro.eventos(TipoEvento.ORDEM):
            assert e.dados.get("simulado") is True
        for e in livro.eventos(TipoEvento.APORTE):
            assert e.dados.get("simulado") is True


# ─────────────────────────────────────────────────────────────────────────────
# Dia de aporte
# ─────────────────────────────────────────────────────────────────────────────

class TesteDiaDeAporte:
    def test_primeiro_pregao_do_mes_e_reconhecido(self):
        idx = pd.bdate_range("2024-01-01", periods=200)
        assert operacao.e_primeiro_pregao_do_mes(idx[0], idx)

    def test_segundo_pregao_nao_e(self):
        idx = pd.bdate_range("2024-01-01", periods=200)
        assert not operacao.e_primeiro_pregao_do_mes(idx[1], idx)

    def test_funciona_quando_o_dia_um_nao_e_pregao(self):
        """Setembro de 2024 começa num domingo: o primeiro pregão é dia 2."""
        idx = pd.bdate_range("2024-08-01", periods=60)
        setembro = idx[(idx.month == 9)]
        assert operacao.e_primeiro_pregao_do_mes(setembro[0], idx)
        assert setembro[0].day != 1

    def test_dia_fora_do_calendario_nao_e_primeiro(self):
        idx = pd.bdate_range("2024-01-01", periods=60)
        assert not operacao.e_primeiro_pregao_do_mes(pd.Timestamp("2030-01-01"), idx)


# ─────────────────────────────────────────────────────────────────────────────
# Idempotência
# ─────────────────────────────────────────────────────────────────────────────

class TesteIdempotencia:
    def test_reprocessar_o_dia_nao_aporta_duas_vezes(self, livro):
        precos, cdi = mercado(60)
        dia = precos.index[0]

        conta, d1 = operacao.rodar_dia(livro, CONFIG, precos, cdi, hoje=dia)
        assert d1.dia_de_aporte
        assert conta.total_aportado == pytest.approx(500.0)

        conta2, d2 = operacao.rodar_dia(livro, CONFIG, precos, cdi, hoje=dia, conta=conta)
        assert not d2.dia_de_aporte
        assert conta2.total_aportado == pytest.approx(500.0)
        assert any("já registrado" in a for a in d2.avisos)

    def test_meses_diferentes_aportam_cada_um(self, livro):
        precos, cdi = mercado(300)
        conta = None
        aportes = 0
        for dia in precos.index[:70]:
            conta, d = operacao.rodar_dia(livro, CONFIG, precos, cdi, hoje=dia, conta=conta)
            aportes += int(d.dia_de_aporte)
        assert aportes >= 3
        assert conta.total_aportado == pytest.approx(500.0 * aportes)


# ─────────────────────────────────────────────────────────────────────────────
# Alvo do dia
# ─────────────────────────────────────────────────────────────────────────────

class TesteAlvo:
    def test_pesos_somam_um(self):
        alvo, _, _ = operacao.alvo_do_dia(CONFIG, favoravel=True, satelite_liberado=False)
        assert sum(alvo.values()) == pytest.approx(1.0)

    def test_regime_ruim_reduz_o_risco_sem_zerar(self):
        _, risco, avisos = operacao.alvo_do_dia(CONFIG, favoravel=False, satelite_liberado=False)
        assert 0 < risco < CONFIG.risco_pct
        assert any("não zerado" in a for a in avisos)

    def test_regime_bom_mantem_o_risco(self):
        _, risco, avisos = operacao.alvo_do_dia(CONFIG, favoravel=True, satelite_liberado=False)
        assert risco == pytest.approx(CONFIG.risco_pct)
        assert avisos == []

    def test_perfil_conservador_nao_e_reduzido(self):
        """Abaixo de 40% de risco não há o que cortar."""
        conservador = operacao.Configuracao(risco_pct=30.0, fatia_cripto=0.0)
        _, risco, _ = operacao.alvo_do_dia(conservador, favoravel=False, satelite_liberado=False)
        assert risco == pytest.approx(30.0)

    def test_cripto_entra_no_alvo(self):
        com = operacao.Configuracao(fatia_cripto=0.02)
        alvo, _, _ = operacao.alvo_do_dia(com, favoravel=True, satelite_liberado=False)
        assert alvo["BTC-BRL"] == pytest.approx(0.02)
        assert sum(alvo.values()) == pytest.approx(1.0)

    def test_cripto_nao_toca_no_caixa(self):
        sem = operacao.Configuracao(fatia_cripto=0.0)
        com = operacao.Configuracao(fatia_cripto=0.05)
        alvo_sem, _, _ = operacao.alvo_do_dia(sem, True, False)
        alvo_com, _, _ = operacao.alvo_do_dia(com, True, False)
        assert alvo_com["CDI"] == pytest.approx(alvo_sem["CDI"])


# ─────────────────────────────────────────────────────────────────────────────
# Regime
# ─────────────────────────────────────────────────────────────────────────────

class TesteRegime:
    def test_historico_curto_devolve_incerto(self):
        precos, cdi = mercado(50)
        acumulado = (1 + cdi).cumprod()
        favoravel, leitura = operacao.ler_regime(precos, acumulado)
        assert favoravel
        assert leitura.get("incerto")

    def test_mercado_em_alta_e_favoravel(self):
        idx = pd.bdate_range("2020-01-01", periods=600)
        precos = pd.DataFrame({"BOVA11.SA": 100 * 1.0008 ** np.arange(600)}, index=idx)
        acumulado = pd.Series(np.full(600, 1.0), index=idx).cumsum() / np.arange(1, 601)
        acumulado = pd.Series((1.00005) ** np.arange(600), index=idx)
        favoravel, _ = operacao.ler_regime(precos, acumulado)
        assert favoravel

    def test_mercado_em_queda_nao_e_favoravel(self):
        idx = pd.bdate_range("2020-01-01", periods=600)
        precos = pd.DataFrame({"BOVA11.SA": 100 * 0.999 ** np.arange(600)}, index=idx)
        acumulado = pd.Series((1.0004) ** np.arange(600), index=idx)
        favoravel, _ = operacao.ler_regime(precos, acumulado)
        assert not favoravel

    def test_sem_a_referencia_devolve_incerto(self):
        idx = pd.bdate_range("2020-01-01", periods=600)
        precos = pd.DataFrame({"OUTRO11.SA": np.ones(600)}, index=idx)
        favoravel, leitura = operacao.ler_regime(precos, pd.Series(np.ones(600), index=idx))
        assert leitura.get("incerto")


# ─────────────────────────────────────────────────────────────────────────────
# Fim a fim
# ─────────────────────────────────────────────────────────────────────────────

class TesteFimAFim:
    def _rodar(self, livro, dias=400, config=CONFIG):
        precos, cdi = mercado(dias)
        conta = None
        for dia in precos.index:
            conta, decisao = operacao.rodar_dia(livro, config, precos, cdi, hoje=dia, conta=conta)
            operacao.registrar_dia(
                livro, conta, decisao,
                {t: float(v) for t, v in precos.loc[dia].items()}, 1.0, 1.0,
            )
        return conta, precos

    def test_roda_o_periodo_inteiro_sem_quebrar(self, livro):
        conta, precos = self._rodar(livro, 300)
        assert conta.total_aportado > 0
        assert conta.patrimonio({t: float(v) for t, v in precos.iloc[-1].items()}) > 0

    def test_a_cadeia_fica_integra(self, livro):
        self._rodar(livro, 200)
        assert verificar(livro).intacta

    def test_o_caixa_nunca_fica_negativo(self, livro):
        conta, _ = self._rodar(livro, 400)
        assert conta.caixa >= -0.01

    def test_o_estado_e_retomado_do_livro(self, livro):
        precos, cdi = mercado(120)
        conta = None
        for dia in precos.index[:60]:
            conta, d = operacao.rodar_dia(livro, CONFIG, precos, cdi, hoje=dia, conta=conta)
            operacao.registrar_dia(
                livro, conta, d, {t: float(v) for t, v in precos.loc[dia].items()}, 1.0, 1.0)

        # Sem passar a conta: tem que reconstruir do último fechamento.
        retomada, _ = operacao.rodar_dia(livro, CONFIG, precos, cdi, hoje=precos.index[60])
        assert retomada.total_aportado == pytest.approx(conta.total_aportado)
        assert set(retomada.posicoes) == set(conta.posicoes)

    def test_o_patrimonio_aparece_no_livro(self, livro):
        self._rodar(livro, 100)
        historico = livro.patrimonio()
        assert len(historico) > 50
        assert all(h["total"] >= 0 for h in historico)

    def test_toda_decisao_tem_regime_escrito(self, livro):
        self._rodar(livro, 80)
        for e in livro.eventos(TipoEvento.DECISAO):
            assert e.dados.get("regime")

    def test_sem_cotacao_o_peso_vai_para_o_caixa(self, livro):
        precos, cdi = mercado(60)
        parcial = precos.drop(columns=["GOLD11.SA", "XFIX11.SA"])
        conta, decisao = operacao.rodar_dia(livro, CONFIG, parcial, cdi, hoje=parcial.index[0])
        assert any("sem cotação" in a for a in decisao.avisos)
        assert sum(decisao.alvo.values()) == pytest.approx(1.0)

    def test_serie_vazia_e_recusada(self, livro):
        with pytest.raises(ValueError, match="sem preços"):
            operacao.rodar_dia(livro, CONFIG, pd.DataFrame(), pd.Series(dtype=float))

    def test_aporte_zerado_nao_gera_ordem(self, livro):
        sem_aporte = operacao.Configuracao(aporte_mensal=0.0, fatia_cripto=0.0)
        conta, _ = self._rodar(livro, 100, sem_aporte)
        assert conta.total_aportado == 0.0
        assert not conta.posicoes

    def test_ordem_abaixo_do_minimo_fica_no_caixa(self, livro):
        migalha = operacao.Configuracao(aporte_mensal=10.0, ordem_minima=1000.0, fatia_cripto=0.0)
        conta, _ = self._rodar(livro, 200, migalha)
        assert not conta.posicoes
        assert conta.caixa > 0

    def test_universo_da_operacao_inclui_a_cripto(self):
        u = operacao.universo_da_operacao()
        assert "BTC-BRL" in u
        assert u["BTC-BRL"].regime.value == "cripto"
