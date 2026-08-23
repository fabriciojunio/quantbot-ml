"""
Testes do carrego de funding.

Sem rede: o histórico de funding é substituído por séries construídas à mão,
para checar a aritmética. O teste que mais importa é o de custo — se o módulo
esquecer as quatro pernas de montagem, ele passa a prometer um retorno que
não existe, que é exatamente o defeito que este projeto está corrigindo.
"""

import pytest

from cripto import carrego
from cripto.carrego import CustoMontagem


def _resposta(taxas, inicio_ms=1_780_000_000_000, passo_ms=8 * 3_600_000):
    return [
        {"symbol": "BTCUSDT", "fundingTime": inicio_ms + i * passo_ms,
         "fundingRate": f"{t:.8f}", "markPrice": "77000.0"}
        for i, t in enumerate(taxas)
    ]


@pytest.fixture
def funding_positivo(monkeypatch):
    """Taxa constante de 0,01% a cada 8h — o padrão histórico da Binance."""
    monkeypatch.setattr(
        carrego, "http_json", lambda url, params=None, **k: _resposta([0.0001] * 1095)
    )


@pytest.fixture
def funding_negativo(monkeypatch):
    monkeypatch.setattr(
        carrego, "http_json", lambda url, params=None, **k: _resposta([-0.0002] * 1095)
    )


class TestCustoMontagem:
    def test_cobra_as_quatro_pernas(self):
        """Montar e desmontar = 2 pernas à vista + 2 no perpétuo."""
        c = CustoMontagem(taxa_spot_taker=0.0010, taxa_perp_taker=0.0005, spread_pct=0.0)
        assert c.total_pct() == pytest.approx(2 * (0.0010 + 0.0005))

    def test_spread_entra_pela_metade_em_cada_perna(self):
        sem = CustoMontagem(spread_pct=0.0).total_pct()
        com = CustoMontagem(spread_pct=0.0004).total_pct()
        assert com - sem == pytest.approx(2 * 0.0004)

    def test_custo_e_sempre_positivo(self):
        assert CustoMontagem().total_pct() > 0


class TestAnalise:
    def test_taxa_constante_produz_carrego_bruto_positivo(self, funding_positivo):
        r = carrego.analisar("BTCUSDT")
        assert r["carrego_bruto_no_periodo_pct"] > 0

    def test_liquido_e_sempre_menor_que_o_bruto(self, funding_positivo):
        r = carrego.analisar("BTCUSDT")
        assert r["carrego_liquido_no_periodo_pct"] < r["carrego_bruto_no_periodo_pct"]

    def test_desconto_bate_com_o_custo_declarado(self, funding_positivo):
        r = carrego.analisar("BTCUSDT")
        diferenca = r["carrego_bruto_no_periodo_pct"] - r["carrego_liquido_no_periodo_pct"]
        assert diferenca == pytest.approx(r["custo_de_montagem_pct"], abs=1e-6)

    def test_funding_negativo_da_carrego_negativo(self, funding_negativo):
        r = carrego.analisar("BTCUSDT")
        assert r["carrego_liquido_anualizado_pct"] < 0
        assert r["fatia_negativa_pct"] == 100.0

    def test_funding_negativo_nunca_paga_a_montagem(self, funding_negativo):
        assert carrego.analisar("BTCUSDT")["dias_para_pagar_a_montagem"] is None

    def test_conta_as_liquidacoes_negativas(self, monkeypatch):
        monkeypatch.setattr(
            carrego,
            "http_json",
            lambda url, params=None, **k: _resposta([0.0001] * 70 + [-0.0001] * 30),
        )
        r = carrego.analisar("BTCUSDT")
        assert r["liquidacoes_negativas"] == 30
        assert r["fatia_negativa_pct"] == pytest.approx(30.0)


class TestCapital:
    def test_notional_e_metade_do_capital(self, funding_positivo):
        """Metade do dinheiro fica na perna à vista, metade na margem."""
        r = carrego.analisar("BTCUSDT", capital_usd=1_000.0)
        assert r["com_capital_de"]["notional_do_carrego_usd"] == pytest.approx(500.0)

    def test_ganho_escala_com_o_capital(self, funding_positivo):
        pequeno = carrego.analisar("BTCUSDT", capital_usd=1_000.0)
        grande = carrego.analisar("BTCUSDT", capital_usd=10_000.0)
        assert (
            grande["com_capital_de"]["ganho_liquido_estimado_por_ano_usd"]
            > pequeno["com_capital_de"]["ganho_liquido_estimado_por_ano_usd"]
        )

    def test_capital_pequeno_gera_ganho_pequeno(self, funding_positivo):
        """O número tem que deixar óbvio que R$ 500 não sustentam ninguém."""
        r = carrego.analisar("BTCUSDT", capital_usd=100.0)
        assert r["com_capital_de"]["ganho_liquido_estimado_por_ano_usd"] < 20


class TestAvisos:
    def test_sempre_lista_os_riscos(self, funding_positivo):
        avisos = carrego.analisar("BTCUSDT")["avisos"]
        assert len(avisos) >= 4
        texto = " ".join(avisos).lower()
        assert "liquidad" in texto
        assert "contraparte" in texto


class TestErros:
    def test_historico_vazio_levanta(self, monkeypatch):
        monkeypatch.setattr(carrego, "http_json", lambda *a, **k: [])
        with pytest.raises(ValueError, match="sem histórico"):
            carrego.historico_funding("BTCUSDT")

    def test_simbolo_invalido_e_recusado(self, monkeypatch):
        monkeypatch.setattr(carrego, "http_json", lambda *a, **k: [])
        with pytest.raises(ValueError):
            carrego.historico_funding("BTC/USDT")
