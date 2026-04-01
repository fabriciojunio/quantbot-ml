"""Testes para o rastreador de acurácia."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.accuracy import AccuracyTracker, Prediction


class TestAccuracyTracker:

    def _make_tracker_with_data(self):
        """Cria tracker com predições simuladas e resolvidas."""
        tracker = AccuracyTracker()

        # 10 predições de COMPRA
        for i in range(10):
            tracker.record_prediction(
                symbol="PETR4.SA", signal="COMPRA",
                score=60 + i, confidence=65 + i * 2,
                price=38.0, sentiment="positivo",
                model_agreement=70 + i,
            )

        # 5 predições de VENDA
        for i in range(5):
            tracker.record_prediction(
                symbol="AAPL", signal="VENDA",
                score=35 - i, confidence=60 + i * 3,
                price=195.0, sentiment="negativo",
            )

        # 3 predições NEUTRO
        for i in range(3):
            tracker.record_prediction(
                symbol="BTC-USD", signal="NEUTRO",
                score=50, confidence=50,
                price=67000.0,
            )

        # Resolve: 6/10 compras acertaram (preço subiu)
        for i in range(6):
            tracker.resolve_prediction(i + 1, actual_price=39.5)
        for i in range(4):
            tracker.resolve_prediction(7 + i, actual_price=37.0)

        # Resolve: 3/5 vendas acertaram (preço caiu)
        for i in range(3):
            tracker.resolve_prediction(11 + i, actual_price=190.0)
        for i in range(2):
            tracker.resolve_prediction(14 + i, actual_price=200.0)

        # Resolve: 2/3 neutros acertaram (preço ficou parecido)
        tracker.resolve_prediction(16, actual_price=67200.0)  # +0.3% ~ neutro ✓
        tracker.resolve_prediction(17, actual_price=66800.0)  # -0.3% ~ neutro ✓
        tracker.resolve_prediction(18, actual_price=72000.0)  # +7.5% ~ errou

        return tracker

    def test_record_prediction(self):
        tracker = AccuracyTracker()
        pred = tracker.record_prediction("PETR4.SA", "COMPRA", 72.5, 85.0, 38.50)
        assert pred.id == 1
        assert pred.symbol == "PETR4.SA"
        assert pred.signal == "COMPRA"
        assert pred.resolved == False
        assert len(tracker.predictions) == 1

    def test_resolve_correct_buy(self):
        tracker = AccuracyTracker()
        tracker.record_prediction("AAPL", "COMPRA", 70, 80, 100.0)
        pred = tracker.resolve_prediction(1, actual_price=105.0)
        assert pred.resolved == True
        assert pred.was_correct == True
        assert pred.actual_return == pytest.approx(5.0)

    def test_resolve_incorrect_buy(self):
        tracker = AccuracyTracker()
        tracker.record_prediction("AAPL", "COMPRA", 70, 80, 100.0)
        pred = tracker.resolve_prediction(1, actual_price=95.0)
        assert pred.was_correct == False
        assert pred.actual_return == pytest.approx(-5.0)

    def test_resolve_correct_sell(self):
        tracker = AccuracyTracker()
        tracker.record_prediction("AAPL", "VENDA", 30, 75, 100.0)
        pred = tracker.resolve_prediction(1, actual_price=92.0)
        assert pred.was_correct == True

    def test_resolve_incorrect_sell(self):
        tracker = AccuracyTracker()
        tracker.record_prediction("AAPL", "VENDA", 25, 80, 100.0)
        pred = tracker.resolve_prediction(1, actual_price=108.0)
        assert pred.was_correct == False

    def test_metrics_calculation(self):
        tracker = self._make_tracker_with_data()
        m = tracker.get_metrics()

        assert m.total_predictions == 18
        assert m.resolved_predictions == 18
        assert m.correct_predictions == 11  # 6 + 3 + 2
        assert m.hit_rate == pytest.approx(61.1, abs=0.1)
        assert m.hit_rate_buys == pytest.approx(60.0)
        assert m.hit_rate_sells == pytest.approx(60.0)
        assert m.profit_factor > 0

    def test_metrics_empty(self):
        tracker = AccuracyTracker()
        m = tracker.get_metrics()
        assert m.total_predictions == 0
        assert m.hit_rate == 0

    def test_by_symbol(self):
        tracker = self._make_tracker_with_data()
        m = tracker.get_metrics()
        assert "PETR4.SA" in m.by_symbol
        assert "AAPL" in m.by_symbol

    def test_by_market(self):
        tracker = self._make_tracker_with_data()
        m = tracker.get_metrics()
        assert "B3" in m.by_market
        assert "US" in m.by_market

    def test_confusion_matrix(self):
        tracker = self._make_tracker_with_data()
        m = tracker.get_metrics()
        cm = m.confusion_matrix
        assert cm["true_buy"] == 6
        assert cm["false_buy"] == 4
        assert cm["true_sell"] == 3
        assert cm["false_sell"] == 2

    def test_export_and_load(self, tmp_path):
        tracker = self._make_tracker_with_data()
        filepath = str(tmp_path / "test_accuracy.json")
        tracker.export(filepath)

        loaded = AccuracyTracker.load(filepath)
        assert len(loaded.predictions) == 18
        m = loaded.get_metrics()
        assert m.hit_rate == pytest.approx(61.1, abs=0.1)

    def test_resolve_nonexistent(self):
        tracker = AccuracyTracker()
        result = tracker.resolve_prediction(999, 100.0)
        assert result is None

    def test_high_confidence_accuracy(self):
        tracker = AccuracyTracker()
        # Alta confiança + acerto
        tracker.record_prediction("TEST", "COMPRA", 80, 85, 100.0)
        tracker.resolve_prediction(1, 110.0)
        # Alta confiança + erro
        tracker.record_prediction("TEST", "COMPRA", 75, 90, 100.0)
        tracker.resolve_prediction(2, 95.0)
        # Baixa confiança + acerto
        tracker.record_prediction("TEST", "COMPRA", 55, 55, 100.0)
        tracker.resolve_prediction(3, 105.0)

        m = tracker.get_metrics()
        assert m.hit_rate_high_confidence == pytest.approx(50.0)  # 1/2
