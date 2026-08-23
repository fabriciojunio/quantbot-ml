"""
Testes do robô.

Sem rede: os preços e o CDI são substituídos por séries construídas à mão. O
que se verifica aqui é a cadeia de decisão — reserva antes de tudo, satélite
travado na validação, guarda de regime reduzindo (e não zerando) a exposição,
e rodízio quando o aporte é pequeno demais para fatiar.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.robo as robo  # noqa: E402
from carteira.plano_aporte import Situacao  # noqa: E402

UNIVERSO = robo.UNIVERSO


def _idx(n):
    return pd.bdate_range("2022-01-03", periods=n)


def precos_falsos(n=800, subindo=True):
    passo = 0.0006 if subindo else -0.0006
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(passo, 0.008, n))) for t in UNIVERSO},
        index=_idx(n),
    )


@pytest.fixture
def sem_rede(monkeypatch, tmp_path):
    px = precos_falsos()
    monkeypatch.setattr(robo.dados, "precos", lambda *a, **k: px)
    monkeypatch.setattr(robo.dados, "alinhar", lambda d, **k: d)
    monkeypatch.setattr(robo.dados, "valida_precos", lambda *a, **k: None)
    monkeypatch.setattr(
        robo.dados, "cdi_diario", lambda *a, **k: pd.Series(0.0004, index=px.index)
    )
    monkeypatch.setattr(robo, "REGISTRO", tmp_path / "decisoes.jsonl")
    monkeypatch.setattr(robo, "ESTADO", tmp_path / "carteira.json")
    return px


def sit(reserva=10**9, aporte=500.0, despesa=2_000.0):
    return Situacao(aporte_mensal=aporte, despesa_mensal=despesa, reserva_atual=reserva)


class TestReservaPrimeiro:
    def test_sem_reserva_tudo_vai_para_renda_fixa(self, sem_rede):
        d = robo.decidir(500, 80, sit(reserva=0))
        assert "reserva" in d["decisao"]
        assert len(d["ordens"]) == 1
        assert d["ordens"][0]["valor"] == 500.0

    def test_sem_reserva_o_satelite_fica_fechado(self, sem_rede):
        d = robo.decidir(500, 100, sit(reserva=0))
        assert d["satelite"]["liberado"] is False

    def test_sem_reserva_nao_compra_bolsa(self, sem_rede):
        d = robo.decidir(500, 100, sit(reserva=0))
        assert all(o["onde"] not in UNIVERSO for o in d["ordens"])

    def test_com_reserva_pronta_vai_para_o_nucleo(self, sem_rede):
        d = robo.decidir(500, 60, sit())
        assert "núcleo" in d["decisao"]

    @pytest.mark.parametrize("reserva", [0, 5_000, 20_000, 23_999])
    def test_reserva_incompleta_bloqueia_o_nucleo(self, sem_rede, reserva):
        d = robo.decidir(500, 80, sit(reserva=reserva))
        assert "reserva" in d["decisao"]


class TestOrcamentoDeRisco:
    @pytest.mark.parametrize("risco", [0, 20, 40, 60, 80, 100])
    def test_pesos_sempre_fecham(self, sem_rede, risco):
        d = robo.decidir(500, risco, sit())
        assert sum(d["alvo_por_ativo_pct"].values()) == pytest.approx(100.0, abs=0.5)

    def test_mais_risco_reduz_o_caixa(self, sem_rede):
        baixo = robo.decidir(500, 20, sit())["alvo_por_ativo_pct"].get("CDI", 0)
        alto = robo.decidir(500, 100, sit())["alvo_por_ativo_pct"].get("CDI", 0)
        assert alto < baixo

    def test_risco_cem_nao_deixa_caixa(self, sem_rede):
        d = robo.decidir(500, 100, sit())
        assert d["alvo_por_ativo_pct"].get("CDI", 0) == pytest.approx(0.0, abs=0.1)

    @pytest.mark.parametrize("risco", [20, 40, 60, 80, 100])
    def test_sempre_tem_exposicao_a_dolar(self, sem_rede, risco):
        d = robo.decidir(500, risco, sit())
        assert d["exposicao_a_dolar_pct"] > 0

    def test_dolar_cresce_com_o_risco(self, sem_rede):
        baixo = robo.decidir(500, 20, sit())["exposicao_a_dolar_pct"]
        alto = robo.decidir(500, 100, sit())["exposicao_a_dolar_pct"]
        assert alto > baixo


class TestGuardaDeRegime:
    def test_regime_ruim_reduz_mas_nao_zera(self, monkeypatch, sem_rede):
        monkeypatch.setattr(robo, "guarda_de_regime", lambda *a, **k: (False, {"leitura": "ruim"}))
        d = robo.decidir(500, 80, sit())
        assert d["risco_aplicado_pct"] < d["risco_pedido_pct"]
        assert d["risco_aplicado_pct"] > 0
        assert d["ajuste_de_regime"] is not None

    def test_regime_bom_mantem_o_risco_pedido(self, monkeypatch, sem_rede):
        monkeypatch.setattr(robo, "guarda_de_regime", lambda *a, **k: (True, {"leitura": "bom"}))
        d = robo.decidir(500, 80, sit())
        assert d["risco_aplicado_pct"] == 80
        assert d["ajuste_de_regime"] is None

    def test_risco_baixo_nao_sofre_ajuste(self, monkeypatch, sem_rede):
        monkeypatch.setattr(robo, "guarda_de_regime", lambda *a, **k: (False, {"leitura": "ruim"}))
        d = robo.decidir(500, 30, sit())
        assert d["risco_aplicado_pct"] == 30

    def test_reducao_e_de_um_quarto(self, monkeypatch, sem_rede):
        monkeypatch.setattr(robo, "guarda_de_regime", lambda *a, **k: (False, {"leitura": "ruim"}))
        d = robo.decidir(500, 80, sit())
        assert d["risco_aplicado_pct"] == pytest.approx(60.0)


class TestSatelite:
    def test_sem_historico_fica_fechado(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        liberado, motivo = robo.satelite_esta_liberado()
        assert not liberado
        assert "sem histórico" in motivo

    def test_validacao_reprovada_mantem_fechado(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        h = tmp_path / "data/carteira/monitor/historico.jsonl"
        h.parent.mkdir(parents=True, exist_ok=True)
        h.write_text(
            json.dumps({"validacao": {"veredito": {
                "aprovado_para_papel": False, "motivos": ["Deflated Sharpe baixo"]
            }}}) + "\n",
            encoding="utf-8",
        )
        liberado, motivo = robo.satelite_esta_liberado()
        assert not liberado
        assert "reprovou" in motivo

    def test_validacao_aprovada_libera(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        h = tmp_path / "data/carteira/monitor/historico.jsonl"
        h.parent.mkdir(parents=True, exist_ok=True)
        h.write_text(
            json.dumps({"validacao": {"veredito": {"aprovado_para_papel": True, "motivos": []}}}) + "\n",
            encoding="utf-8",
        )
        liberado, _ = robo.satelite_esta_liberado()
        assert liberado

    def test_usa_o_veredito_mais_recente(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        h = tmp_path / "data/carteira/monitor/historico.jsonl"
        h.parent.mkdir(parents=True, exist_ok=True)
        h.write_text(
            json.dumps({"validacao": {"veredito": {"aprovado_para_papel": True, "motivos": []}}}) + "\n"
            + json.dumps({"validacao": {"veredito": {"aprovado_para_papel": False, "motivos": ["piorou"]}}}) + "\n",
            encoding="utf-8",
        )
        liberado, _ = robo.satelite_esta_liberado()
        assert not liberado

    def test_linha_corrompida_nao_derruba(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        h = tmp_path / "data/carteira/monitor/historico.jsonl"
        h.parent.mkdir(parents=True, exist_ok=True)
        h.write_text("nao e json\n", encoding="utf-8")
        liberado, _ = robo.satelite_esta_liberado()
        assert not liberado


class TestRodizio:
    def test_aporte_pequeno_vira_uma_ordem_so(self, sem_rede):
        d = robo.decidir(500, 60, sit())
        assert len(d["ordens"]) == 1
        assert "rodízio" in d["ordens"][0]["motivo"]

    def test_aporte_grande_divide_entre_ativos(self, sem_rede, monkeypatch, tmp_path):
        monkeypatch.setattr(robo, "ESTADO", tmp_path / "c.json")
        robo.salvar_carteira({"IVVB11.SA": 5_000, "BOVA11.SA": 3_000})
        d = robo.decidir(10_000, 60, sit())
        assert len(d["ordens"]) > 1

    def test_ordem_nunca_fica_abaixo_do_minimo(self, sem_rede, monkeypatch, tmp_path):
        monkeypatch.setattr(robo, "ESTADO", tmp_path / "c.json")
        robo.salvar_carteira({"IVVB11.SA": 5_000})
        d = robo.decidir(10_000, 60, sit())
        assert all(o["valor"] >= robo.ORDEM_MINIMA for o in d["ordens"])

    def test_a_soma_das_ordens_nao_passa_do_aporte(self, sem_rede):
        d = robo.decidir(500, 60, sit())
        assert sum(o["valor"] for o in d["ordens"]) <= 500.01

    def test_ordem_traz_custo_estimado(self, sem_rede):
        d = robo.decidir(500, 60, sit())
        assert "custo_estimado" in d["ordens"][0]


class TestExplicabilidade:
    @pytest.mark.parametrize("risco", [0, 30, 60, 100])
    def test_toda_ordem_tem_motivo(self, sem_rede, risco):
        d = robo.decidir(500, risco, sit())
        assert all(o["motivo"] for o in d["ordens"])

    def test_decisao_registra_o_regime(self, sem_rede):
        assert "leitura" in robo.decidir(500, 60, sit())["regime"]

    def test_decisao_registra_o_satelite_e_o_porque(self, sem_rede):
        d = robo.decidir(500, 60, sit())
        assert "liberado" in d["satelite"]
        assert d["satelite"]["motivo"]

    def test_decisao_tem_carimbo_de_tempo(self, sem_rede):
        assert robo.decidir(500, 60, sit())["quando"]


class TestEstado:
    def test_carteira_vazia_quando_nao_existe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(robo, "ESTADO", tmp_path / "nao_existe.json")
        assert robo.carregar_carteira() == {}

    def test_grava_e_le(self, tmp_path, monkeypatch):
        monkeypatch.setattr(robo, "ESTADO", tmp_path / "c.json")
        robo.salvar_carteira({"BOVA11.SA": 1_000.0})
        assert robo.carregar_carteira()["BOVA11.SA"] == 1_000.0

    def test_estado_corrompido_nao_derruba(self, tmp_path, monkeypatch):
        caminho = tmp_path / "c.json"
        caminho.write_text("{quebrado", encoding="utf-8")
        monkeypatch.setattr(robo, "ESTADO", caminho)
        assert robo.carregar_carteira() == {}


class TestCasosLimite:
    @pytest.mark.parametrize("valor", [50, 100, 500, 1_000, 5_000, 20_000])
    def test_varios_tamanhos_de_aporte(self, sem_rede, valor):
        d = robo.decidir(valor, 60, sit())
        assert d["ordens"]

    def test_risco_zero_nao_compra_bolsa(self, sem_rede):
        d = robo.decidir(500, 0, sit())
        assert d["alvo_por_ativo_pct"].get("BOVA11.SA", 0) == pytest.approx(0.0, abs=0.1)

    @pytest.mark.parametrize("risco", [-1, 101, 200])
    def test_risco_invalido_e_recusado(self, sem_rede, risco):
        with pytest.raises(ValueError):
            robo.decidir(500, risco, sit())
