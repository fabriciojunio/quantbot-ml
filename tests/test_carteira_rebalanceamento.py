"""
Testes da política de rebalanceamento.

O que decide o resultado de uma carteira pequena não é o peso alvo, é quantas
vezes por ano ela é mexida. Estes testes garantem que a banda realmente corta
operação inútil e que a política com custo só executa quando o desvio paga a
conta de corrigir.
"""

import numpy as np
import pandas as pd
import pytest

from carteira.rebalanceamento import (
    ConfigRebalanceamento,
    Politica,
    contar_eventos,
    datas_de_calendario,
    marcar_datas,
    precisa_rebalancear,
)

TICKERS = ["A", "B", "C", "D"]
CALENDARIO = [
    Politica.DIARIO, Politica.SEMANAL, Politica.MENSAL,
    Politica.TRIMESTRAL, Politica.SEMESTRAL, Politica.ANUAL,
]


def _idx(n=504):
    return pd.bdate_range("2020-01-01", periods=n)


def pesos_alvo(n=504, alvo=None):
    alvo = alvo or {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    return pd.DataFrame({t: alvo.get(t, 0.0) for t in TICKERS}, index=_idx(n))


def retornos(n=504, semente=9, escala=0.01):
    rng = np.random.default_rng(semente)
    return pd.DataFrame(
        rng.normal(0.0003, escala, (n, len(TICKERS))), index=_idx(n), columns=TICKERS
    )


class TestCalendario:
    @pytest.mark.parametrize("politica", CALENDARIO)
    def test_gera_datas(self, politica):
        assert len(datas_de_calendario(_idx(), politica)) > 0

    def test_diario_marca_todo_pregao(self):
        idx = _idx()
        assert len(datas_de_calendario(idx, Politica.DIARIO)) == len(idx)

    @pytest.mark.parametrize(
        "politica,passo", [(Politica.SEMANAL, 5), (Politica.MENSAL, 21),
                           (Politica.TRIMESTRAL, 63), (Politica.ANUAL, 252)]
    )
    def test_espacamento_correto(self, politica, passo):
        datas = datas_de_calendario(_idx(504), politica)
        assert len(datas) == pytest.approx(504 / passo, abs=1)

    @pytest.mark.parametrize("politica", [Politica.BANDA, Politica.BANDA_COM_CUSTO])
    def test_banda_nao_tem_calendario(self, politica):
        with pytest.raises(ValueError, match="não tem calendário"):
            datas_de_calendario(_idx(), politica)

    def test_frequencia_maior_gera_mais_datas(self):
        idx = _idx()
        anterior = 0
        for politica in reversed(CALENDARIO):
            atual = len(datas_de_calendario(idx, politica))
            assert atual >= anterior
            anterior = atual


class TestPrecisaRebalancear:
    def test_carteira_no_alvo_nao_precisa(self):
        atuais = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        decidiu, diag = precisa_rebalancear(atuais, atuais, ConfigRebalanceamento())
        assert not decidiu
        assert "dentro da banda" in diag["motivo"]

    def test_desvio_grande_dispara(self):
        alvo = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        atuais = pd.Series({"A": 0.45, "B": 0.20, "C": 0.20, "D": 0.15})
        decidiu, diag = precisa_rebalancear(atuais, alvo, ConfigRebalanceamento())
        assert decidiu
        assert "A" in diag["ativos_fora_da_banda"]

    def test_calendario_sempre_dispara(self):
        atuais = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        cfg = ConfigRebalanceamento(politica=Politica.MENSAL)
        decidiu, diag = precisa_rebalancear(atuais, atuais, cfg)
        assert decidiu
        assert "calendário" in diag["motivo"]

    @pytest.mark.parametrize("banda", [1.0, 3.0, 5.0, 10.0, 20.0])
    def test_banda_maior_dispara_menos(self, banda):
        alvo = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        atuais = pd.Series({"A": 0.32, "B": 0.23, "C": 0.23, "D": 0.22})
        cfg = ConfigRebalanceamento(banda_pp=banda, banda_relativa=0.0)
        decidiu, _ = precisa_rebalancear(atuais, alvo, cfg)
        assert decidiu == (banda < 7.0)

    def test_diagnostico_traz_o_giro_necessario(self):
        alvo = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        atuais = pd.Series({"A": 0.45, "B": 0.20, "C": 0.20, "D": 0.15})
        _, diag = precisa_rebalancear(atuais, alvo, ConfigRebalanceamento())
        assert diag["giro_necessario_pct"] > 0
        assert diag["custo_estimado_pct"] > 0

    def test_diagnostico_identifica_o_mais_desviado(self):
        alvo = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        atuais = pd.Series({"A": 0.45, "B": 0.20, "C": 0.20, "D": 0.15})
        _, diag = precisa_rebalancear(atuais, alvo, ConfigRebalanceamento())
        assert diag["ativo_mais_desviado"] == "A"

    def test_banda_relativa_protege_posicao_pequena(self):
        """Um ativo com alvo de 5% não deveria disparar por desvio de 1 p.p."""
        alvo = pd.Series({"A": 0.05, "B": 0.95})
        atuais = pd.Series({"A": 0.06, "B": 0.94})
        cfg = ConfigRebalanceamento(banda_pp=5.0, banda_relativa=0.25)
        decidiu, _ = precisa_rebalancear(atuais, alvo, cfg)
        assert not decidiu

    def test_com_custo_recusa_desvio_que_nao_paga(self):
        alvo = pd.Series({"A": 0.50, "B": 0.50})
        atuais = pd.Series({"A": 0.56, "B": 0.44})
        cfg = ConfigRebalanceamento(
            politica=Politica.BANDA_COM_CUSTO,
            banda_pp=5.0, banda_relativa=0.0,
            custo_ida_volta_pct=0.50,           # custo absurdo de propósito
            ganho_minimo_vs_custo=1.5,
        )
        decidiu, diag = precisa_rebalancear(atuais, alvo, cfg)
        assert not decidiu
        assert "não paga o custo" in diag["motivo"]

    def test_com_custo_aceita_desvio_que_compensa(self):
        alvo = pd.Series({"A": 0.50, "B": 0.50})
        atuais = pd.Series({"A": 0.70, "B": 0.30})
        cfg = ConfigRebalanceamento(
            politica=Politica.BANDA_COM_CUSTO,
            banda_pp=5.0, banda_relativa=0.0, custo_ida_volta_pct=0.004,
        )
        decidiu, diag = precisa_rebalancear(atuais, alvo, cfg)
        assert decidiu
        assert "supera o limiar" in diag["motivo"]

    def test_ativo_ausente_no_alvo_conta_como_zero(self):
        atuais = pd.Series({"A": 0.50, "B": 0.50})
        alvo = pd.Series({"A": 1.0})
        decidiu, _ = precisa_rebalancear(atuais, alvo, ConfigRebalanceamento())
        assert decidiu


class TestMarcarDatas:
    @pytest.mark.parametrize("politica", CALENDARIO)
    def test_calendario_marca_pelo_passo(self, politica):
        marcas = marcar_datas(pesos_alvo(), retornos(), ConfigRebalanceamento(politica=politica))
        assert marcas.sum() > 0
        assert len(marcas) == 504

    def test_banda_marca_menos_que_mensal(self):
        pa, ret = pesos_alvo(), retornos()
        mensal = marcar_datas(pa, ret, ConfigRebalanceamento(politica=Politica.MENSAL))
        banda = marcar_datas(pa, ret, ConfigRebalanceamento(politica=Politica.BANDA))
        assert banda.sum() < mensal.sum()

    def test_primeira_data_sempre_marca(self):
        marcas = marcar_datas(pesos_alvo(), retornos(), ConfigRebalanceamento())
        assert bool(marcas.iloc[0])

    def test_mercado_parado_quase_nao_rebalanceia(self):
        """Sem movimento de preço, a carteira não sai do alvo e a banda não dispara."""
        ret = pd.DataFrame(0.0, index=_idx(), columns=TICKERS)
        marcas = marcar_datas(pesos_alvo(), ret, ConfigRebalanceamento(politica=Politica.BANDA))
        assert marcas.sum() <= 2

    def test_mercado_agitado_rebalanceia_mais(self):
        pa = pesos_alvo()
        calmo = marcar_datas(pa, retornos(escala=0.002), ConfigRebalanceamento(politica=Politica.BANDA))
        agitado = marcar_datas(pa, retornos(escala=0.05), ConfigRebalanceamento(politica=Politica.BANDA))
        assert agitado.sum() >= calmo.sum()

    def test_devolve_serie_booleana(self):
        marcas = marcar_datas(pesos_alvo(), retornos(), ConfigRebalanceamento())
        assert marcas.dtype == bool


class TestContarEventos:
    def test_conta_certo(self):
        marcas = pd.Series([True, False, True, False, True], index=_idx(5))
        assert contar_eventos(marcas)["eventos"] == 3

    def test_calcula_por_ano(self):
        marcas = marcar_datas(
            pesos_alvo(), retornos(), ConfigRebalanceamento(politica=Politica.MENSAL)
        )
        r = contar_eventos(marcas)
        assert 10 <= r["por_ano"] <= 14

    def test_serie_de_um_ponto_nao_quebra(self):
        r = contar_eventos(pd.Series([True], index=_idx(1)))
        assert r["por_ano"] == 0.0

    @pytest.mark.parametrize(
        "politica,minimo,maximo",
        [
            (Politica.DIARIO, 240, 265),
            (Politica.SEMANAL, 48, 55),
            (Politica.MENSAL, 10, 14),
            (Politica.TRIMESTRAL, 3, 5),
            (Politica.ANUAL, 0.8, 1.5),
        ],
    )
    def test_frequencia_bate_com_o_esperado(self, politica, minimo, maximo):
        marcas = marcar_datas(pesos_alvo(), retornos(), ConfigRebalanceamento(politica=politica))
        por_ano = contar_eventos(marcas)["por_ano"]
        assert minimo <= por_ano <= maximo
