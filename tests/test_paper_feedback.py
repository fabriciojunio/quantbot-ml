"""Testes da carteira paper e do aprendizado por feedback."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from datetime import date, timedelta

from income.paper_book import PaperBook
from income.feedback import Feedback


@dataclass
class FakePick:
    ticker: str
    score: float
    dy_12m: float
    pvp: float = 0.9


# ── PaperBook ───────────────────────────────────────────────────────────

class TestPaperBook:
    def test_compra_top_picks(self, tmp_path):
        book = PaperBook(arquivo=str(tmp_path / "pb.json"),
                         capital_inicial=10_000, max_posicoes=5)
        dec = book.decidir([FakePick("AAA11", 80, 12), FakePick("BBB11", 79, 11)],
                           {"AAA11": 100, "BBB11": 50})
        assert sum(1 for d in dec if d["acao"] == "COMPRAR") == 2
        assert book.caixa < 10_000
        assert len(book.estado["posicoes"]) == 2

    def test_vende_quando_sai_da_lista(self, tmp_path):
        book = PaperBook(arquivo=str(tmp_path / "pb.json"), capital_inicial=10_000)
        book.decidir([FakePick("AAA11", 80, 12)], {"AAA11": 100})
        # AAA11 não está mais entre os aprovados → deve vender
        dec = book.decidir([FakePick("BBB11", 80, 12)], {"AAA11": 110, "BBB11": 50})
        assert any(d["acao"] == "VENDER" and d["ticker"] == "AAA11" for d in dec)
        assert len(book.estado["fechadas"]) == 1
        assert book.estado["fechadas"][0]["retorno_total_pct"] > 0  # vendeu a 110 (comprou a 100)

    def test_marca_mercado_acumula_dividendos(self, tmp_path):
        book = PaperBook(arquivo=str(tmp_path / "pb.json"), capital_inicial=10_000)
        book.decidir([FakePick("AAA11", 80, 12)], {"AAA11": 100})
        p = book.estado["posicoes"]["AAA11"]
        p["ultima_avaliacao"] = (date.today() - timedelta(days=365)).isoformat()
        caixa_antes = book.caixa
        book.marcar_mercado({"AAA11": 100})
        assert book.caixa > caixa_antes          # proventos entraram no caixa
        assert book.estado["posicoes"]["AAA11"]["dividendos"] > 0

    def test_avaliar_gera_acerto_ou_erro(self, tmp_path):
        book = PaperBook(arquivo=str(tmp_path / "pb.json"), capital_inicial=10_000)
        book.decidir([FakePick("AAA11", 81, 12)], {"AAA11": 100})
        antiga = (date.today() - timedelta(days=60)).isoformat()
        p = book.estado["posicoes"]["AAA11"]
        p["data_entrada"] = antiga
        p["ultima_avaliacao"] = antiga
        res = book.avaliar({"AAA11": 130}, cdi_anual=14.0)   # +30%
        assert len(res) == 1
        assert res[0]["acerto"] is True
        assert res[0]["excedente_cdi_pct"] > 0


# ── Feedback ────────────────────────────────────────────────────────────

class TestFeedback:
    def test_ajuste_score_precisa_de_amostra(self, tmp_path):
        fb = Feedback(arquivo=str(tmp_path / "fb.json"))
        assert fb.ajuste_score(81, 12, "neutro") == 0.0  # sem dados ainda

    def test_aprende_faixa_boa(self, tmp_path):
        fb = Feedback(arquivo=str(tmp_path / "fb.json"))
        outs = [{"score_entrada": 81, "dy_entrada": 12, "sentimento_entrada": "neutro",
                 "excedente_cdi_pct": 10.0, "acerto": True} for _ in range(6)]
        fb.atualizar_decisoes(outs)
        aj = fb.ajuste_score(81, 12, "neutro")
        assert 0 < aj <= 3.0          # nudge positivo e limitado

    def test_aprende_sentimento_preditivo(self, tmp_path):
        fb = Feedback(arquivo=str(tmp_path / "fb.json"))
        base = date(2026, 1, 1)
        sent, valor = [], []
        for i in range(6):
            d0 = base + timedelta(days=i * 10)
            d1 = d0 + timedelta(days=7)
            sent.append({"data": d0.isoformat(), "indice_sentimento": -0.4})  # medo
            valor.append({"data": d0.isoformat(), "valor_total": 1000})
            valor.append({"data": d1.isoformat(), "valor_total": 1050})       # +5%
        fb.aprender_sentimento(sent, valor, horizonte_dias=7)
        # medo precedeu alta → viés positivo, limitado
        assert fb.vies_de_mercado(-0.4) > 0
        assert fb.vies_de_mercado(-0.4) <= 2.0

    def test_resumo_texto_nao_quebra(self, tmp_path):
        fb = Feedback(arquivo=str(tmp_path / "fb.json"))
        txt = fb.resumo_texto()
        assert "aprendeu" in txt.lower()
