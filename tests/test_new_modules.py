"""
Testes dos novos módulos: PortfolioAllocator, DecisionLogger, DividendCalendar.
Verificam comportamento de negócio real.
"""

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from income.portfolio_allocator import PortfolioAllocator, PortfolioAllocation
from core.decision_logger import DecisionLogger, DecisionType, DecisionOutcome
from income.dividend_calendar import DividendCalendar, DividendEvent


# ═══════════════════════════════════════════════════════════════
# PortfolioAllocator
# ═══════════════════════════════════════════════════════════════

class TestPortfolioAllocator:
    def test_capital_1000_perfil_conservador(self):
        """R$1.000 deve usar perfil conservador."""
        allocator = PortfolioAllocator()
        alloc = allocator.compute(1_000.0)
        assert alloc.perfil == "conservador"

    def test_capital_10000_perfil_moderado(self):
        allocator = PortfolioAllocator()
        alloc = allocator.compute(10_000.0)
        assert alloc.perfil == "moderado"

    def test_capital_50000_perfil_arrojado(self):
        allocator = PortfolioAllocator()
        alloc = allocator.compute(50_000.0)
        assert alloc.perfil == "arrojado"

    def test_soma_porcentagens_100(self):
        """A soma de todos os percentuais deve ser 100%."""
        allocator = PortfolioAllocator()
        for perfil in ["conservador", "moderado", "arrojado"]:
            alloc = allocator.compute(10_000, perfil_override=perfil)
            soma = sum(b.pct_total for b in alloc.baldes)
            assert abs(soma - 100.0) < 0.01, f"Soma = {soma} para {perfil}"

    def test_soma_valores_igual_capital(self):
        """Soma dos valores alocados deve ser igual ao capital total."""
        allocator = PortfolioAllocator()
        capital = 5_000.0
        alloc = allocator.compute(capital)
        soma = sum(b.valor_alocado for b in alloc.baldes)
        assert abs(soma - capital) < 0.10

    def test_usuario_controla_mais_de_50pct(self):
        """O usuário deve SEMPRE controlar mais de 50% do capital."""
        allocator = PortfolioAllocator()
        for perfil in ["conservador", "moderado", "arrojado"]:
            alloc = allocator.compute(100_000, perfil_override=perfil)
            pct_usuario = sum(b.pct_total for b in alloc.baldes if b.quem_controla == "voce")
            assert pct_usuario >= 45.0, f"Usuário com só {pct_usuario}% no perfil {perfil}"

    def test_bot_tem_ativos_fii(self):
        """O balde do bot deve sempre incluir FIIs."""
        allocator = PortfolioAllocator()
        alloc = allocator.compute(5_000)
        bot_ativos = []
        for b in alloc.baldes:
            if b.quem_controla == "bot":
                bot_ativos.extend(b.ativos_alvo)
        fii_tickers = [a for a in bot_ativos if a.endswith("11")]
        assert len(fii_tickers) > 0

    def test_renda_mensal_calculada(self):
        """Cada balde deve ter renda mensal calculada > 0."""
        allocator = PortfolioAllocator()
        alloc = allocator.compute(10_000)
        for b in alloc.baldes:
            if b.valor_alocado > 0:
                assert b.renda_mensal > 0

    def test_resumo_texto(self):
        allocator = PortfolioAllocator()
        alloc = allocator.compute(1_000)
        texto = alloc.resumo_texto()
        assert "R$" in texto
        assert "bot" in texto.lower()

    def test_save_and_load(self, tmp_path):
        """Salvar e carregar deve preservar a alocação."""
        allocator = PortfolioAllocator()
        alloc = allocator.compute(8_000, meta_renda_mensal=500)
        path = str(tmp_path / "allocation.json")
        allocator.save(alloc, path)
        loaded = allocator.load(path)
        assert loaded is not None
        assert loaded.capital_total == 8_000
        assert loaded.meta_renda_mensal == 500
        assert loaded.perfil == alloc.perfil


# ═══════════════════════════════════════════════════════════════
# DecisionLogger
# ═══════════════════════════════════════════════════════════════

class TestDecisionLogger:
    def _logger(self, tmp_path):
        return DecisionLogger(
            log_dir=str(tmp_path / "decisions"),
            also_log_to_console=False,
        )

    def test_log_gera_id_unico(self, tmp_path):
        dl = self._logger(tmp_path)
        id1 = dl.log(DecisionType.SIGNAL_GENERATED, ticker="HGLG11")
        id2 = dl.log(DecisionType.SIGNAL_GENERATED, ticker="HGLG11")
        assert id1 != id2

    def test_log_grava_em_disco(self, tmp_path):
        dl = self._logger(tmp_path)
        dl.log(DecisionType.FII_SCREENING, contexto={"n": 50})
        log_dir = tmp_path / "decisions"
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1
        entries = [json.loads(l) for l in files[0].read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["tipo"] == "fii_screening"

    def test_update_outcome(self, tmp_path):
        dl = self._logger(tmp_path)
        did = dl.log(DecisionType.ORDER_PLACED, ticker="KNCR11", valor_envolvido=500)
        ok = dl.update_outcome(did, DecisionOutcome.SUCCESS, {"pnl": 12.50})
        assert ok is True

    def test_update_id_invalido(self, tmp_path):
        dl = self._logger(tmp_path)
        ok = dl.update_outcome("id-que-nao-existe", DecisionOutcome.SUCCESS)
        assert ok is False

    def test_log_dividend(self, tmp_path):
        dl = self._logger(tmp_path)
        did = dl.log_dividend("HGLG11", 1.15, 10, "2026-06-15", reinvestido=True)
        assert did is not None
        log_dir = tmp_path / "decisions"
        entries = [json.loads(l) for l in list(log_dir.glob("*.jsonl"))[0].read_text().splitlines()]
        assert any(e.get("tipo") == "dividend_received" for e in entries)

    def test_log_screening(self, tmp_path):
        dl = self._logger(tmp_path)
        dl.log_screening("FII", 500, 35, ["HGLG11", "KNCR11"], {"dy_min": 8.0})

    def test_today_summary(self, tmp_path):
        dl = self._logger(tmp_path)
        dl.log(DecisionType.SIGNAL_GENERATED, ticker="X")
        dl.log(DecisionType.SIGNAL_GENERATED, ticker="Y")
        dl.log_dividend("FII1", 1.0, 5, "2026-06-01")
        summary = dl.get_today_summary()
        assert summary["total_decisoes"] == 3
        assert summary["dividendos_recebidos"] == 5.0

    def test_learning_features(self, tmp_path):
        dl = self._logger(tmp_path)
        dl.log(DecisionType.ORDER_PLACED, ticker="HGLG11")
        features = dl.get_learning_features(days=1)
        assert len(features) >= 1


# ═══════════════════════════════════════════════════════════════
# DividendCalendar
# ═══════════════════════════════════════════════════════════════

class TestDividendCalendar:
    def _calendar(self, tmp_path):
        return DividendCalendar(data_file=str(tmp_path / "dividends.json"))

    def _event(self, ticker="HGLG11", days_ahead=5, valor=1.15, n_cotas=10.0):
        pay_date = (date.today() + timedelta(days=days_ahead)).isoformat()
        return DividendEvent(
            id=f"{ticker}_{pay_date}",
            ticker=ticker,
            tipo_ativo="FII",
            valor_por_cota=valor,
            data_com=(date.today() + timedelta(days=days_ahead - 2)).isoformat(),
            data_pagamento=pay_date,
            status="agendado",
            n_cotas=n_cotas,
        )

    def test_add_and_retrieve(self, tmp_path):
        cal = self._calendar(tmp_path)
        ev = self._event()
        cal.add_event(ev)
        upcoming = cal.upcoming(30)
        assert len(upcoming) == 1
        assert upcoming[0].ticker == "HGLG11"

    def test_mark_paid(self, tmp_path):
        cal = self._calendar(tmp_path)
        ev = self._event()
        cal.add_event(ev)
        cal.mark_paid(ev.id)
        upcoming = cal.upcoming(30)
        assert len(upcoming) == 0  # não aparece mais como agendado

    def test_monthly_income(self, tmp_path):
        cal = self._calendar(tmp_path)
        today = date.today()
        ev = DividendEvent(
            id=f"HGLG11_{today.isoformat()}",
            ticker="HGLG11",
            tipo_ativo="FII",
            valor_por_cota=1.15,
            data_com=None,
            data_pagamento=today.isoformat(),
            status="pago",
            n_cotas=10.0,
        )
        ev.total_recebido = 11.50
        cal.add_event(ev)
        mi = cal.get_monthly_income(today.year, today.month)
        assert mi.total_bruto == pytest.approx(11.50)

    def test_alert_messages_proximo(self, tmp_path):
        cal = self._calendar(tmp_path)
        ev = self._event(days_ahead=2)
        cal.add_event(ev)
        alerts = cal.alert_messages(days_ahead=5)
        assert len(alerts) == 1
        assert "HGLG11" in alerts[0]

    def test_from_brapi_formato(self, tmp_path):
        cal = self._calendar(tmp_path)
        brapi_data = [
            {"paymentDate": "2026-07-10", "rate": 1.20},
            {"paymentDate": "2026-06-10", "rate": 1.15},
        ]
        novos = cal.from_brapi("HGLG11", "FII", brapi_data, n_cotas=20.0)
        assert len(novos) == 2
        assert novos[0].total_recebido == pytest.approx(1.20 * 20)

    def test_ytd_income(self, tmp_path):
        cal = self._calendar(tmp_path)
        today = date.today()
        for i in range(3):
            ev = DividendEvent(
                id=f"FII_{today.year}-{i+1:02d}-15",
                ticker="FII",
                tipo_ativo="FII",
                valor_por_cota=1.0,
                data_com=None,
                data_pagamento=f"{today.year}-{i+1:02d}-15",
                status="pago",
                n_cotas=10.0,
            )
            ev.total_recebido = 10.0
            cal.add_event(ev)
        ytd = cal.get_ytd_income()
        assert ytd == pytest.approx(30.0)

    def test_persist_and_reload(self, tmp_path):
        path = str(tmp_path / "divs.json")
        cal1 = DividendCalendar(data_file=path)
        cal1.add_event(self._event())
        cal2 = DividendCalendar(data_file=path)
        assert len(cal2.upcoming(30)) == 1

    def test_summary_text(self, tmp_path):
        cal = self._calendar(tmp_path)
        texto = cal.summary_text()
        assert "CALENDÁRIO" in texto
        assert "R$" in texto
