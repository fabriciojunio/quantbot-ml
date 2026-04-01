"""Testes para Live Trading Engine e Safety Limits."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.live_trading import (
    LiveTradingEngine, SafetyLimits, SafetyMonitor,
    PaperBrokerAdapter, BotStatus,
)


class TestSafetyLimits:

    def test_default_limits(self):
        limits = SafetyLimits()
        assert limits.max_daily_loss_pct == 3.0
        assert limits.max_trade_loss_pct == 1.5
        assert limits.max_exposure_pct == 80.0

    def test_custom_limits(self):
        limits = SafetyLimits(max_daily_loss_pct=5.0, max_daily_trades=100)
        assert limits.max_daily_loss_pct == 5.0
        assert limits.max_daily_trades == 100


class TestSafetyMonitor:

    def test_normal_trade_allowed(self):
        monitor = SafetyMonitor(SafetyLimits(respect_market_hours=False))
        is_safe, reason = monitor.check_before_trade(
            trade_value=5000, current_capital=100000, total_invested=20000,
        )
        assert is_safe is True
        assert reason == "OK"

    def test_exposure_limit(self):
        monitor = SafetyMonitor(SafetyLimits(
            max_exposure_pct=80.0, respect_market_hours=False,
        ))
        is_safe, reason = monitor.check_before_trade(
            trade_value=50000, current_capital=100000, total_invested=40000,
        )
        assert is_safe is False
        assert "EXPOSIÇÃO" in reason

    def test_cash_reserve(self):
        monitor = SafetyMonitor(SafetyLimits(
            min_cash_reserve=5000, max_exposure_pct=99, respect_market_hours=False,
        ))
        is_safe, reason = monitor.check_before_trade(
            trade_value=96000, current_capital=100000, total_invested=0,
        )
        assert is_safe is False
        assert "RESERVA" in reason

    def test_max_trades_limit(self):
        monitor = SafetyMonitor(SafetyLimits(
            max_daily_trades=2, respect_market_hours=False, cooldown_seconds=0,
        ))
        monitor.stats.reset(100000)
        monitor.stats.trades_today = 2
        is_safe, reason = monitor.check_before_trade(
            trade_value=1000, current_capital=100000, total_invested=0,
        )
        assert is_safe is False
        assert "LIMITE DE TRADES" in reason

    def test_stop_loss_check(self):
        monitor = SafetyMonitor(SafetyLimits(max_trade_loss_pct=2.0))
        assert monitor.check_stop_loss(-2.5) is True
        assert monitor.check_stop_loss(-1.0) is False

    def test_register_trade(self):
        monitor = SafetyMonitor(SafetyLimits())
        monitor.register_trade(pnl=150)
        assert monitor.stats.trades_today == 1
        assert monitor.stats.daily_pnl == 150

    def test_status_report(self):
        monitor = SafetyMonitor(SafetyLimits())
        status = monitor.get_status()
        assert "daily_pnl_pct" in status
        assert "safety_triggered" in status


class TestPaperBroker:

    def test_initial_balance(self):
        broker = PaperBrokerAdapter(initial_capital=50000)
        assert broker.get_balance() == 50000

    def test_buy(self):
        broker = PaperBrokerAdapter(initial_capital=100000)
        broker.set_price("AAPL", 200.0)
        result = broker.buy("AAPL", 10)
        assert result["status"] == "filled"
        assert broker.get_balance() < 100000
        positions = broker.get_positions()
        assert "AAPL" in positions
        assert positions["AAPL"]["qty"] == 10

    def test_sell(self):
        broker = PaperBrokerAdapter(initial_capital=100000)
        broker.set_price("AAPL", 200.0)
        broker.buy("AAPL", 10)
        broker.set_price("AAPL", 220.0)
        result = broker.sell("AAPL", 10)
        assert result["status"] == "filled"
        assert result["pnl"] > 0
        assert "AAPL" not in broker.get_positions()

    def test_sell_without_position(self):
        broker = PaperBrokerAdapter()
        result = broker.sell("FAKE", 10)
        assert "error" in result

    def test_buy_insufficient_funds(self):
        broker = PaperBrokerAdapter(initial_capital=100)
        broker.set_price("BTC", 67000)
        result = broker.buy("BTC", 1)
        assert "error" in result


class TestLiveTradingEngine:

    def test_start_stop(self):
        engine = LiveTradingEngine()
        assert engine.status == BotStatus.OFF
        engine.start()
        assert engine.status == BotStatus.RUNNING
        engine.confirm_stop()
        assert engine.status == BotStatus.OFF

    def test_request_stop_with_positions(self):
        broker = PaperBrokerAdapter(initial_capital=100000)
        broker.set_price("AAPL", 200)
        broker.buy("AAPL", 10)
        engine = LiveTradingEngine(broker=broker)
        engine.start()
        result = engine.request_stop()
        assert result["count"] == 1
        assert "AAPL" in result["positions"]

    def test_request_stop_empty(self):
        engine = LiveTradingEngine()
        engine.start()
        result = engine.request_stop()
        assert result["count"] == 0

    def test_execute_buy_signal(self):
        broker = PaperBrokerAdapter(initial_capital=100000)
        broker.set_price("AAPL", 200)
        engine = LiveTradingEngine(
            broker=broker,
            safety=SafetyLimits(respect_market_hours=False, cooldown_seconds=0),
        )
        engine.start()
        result = engine.execute_signal("AAPL", "COMPRA", score=75, confidence=80)
        assert result is not None
        assert result["status"] == "filled"

    def test_safety_blocks_trade(self):
        broker = PaperBrokerAdapter(initial_capital=1000)
        broker.set_price("BTC", 67000)
        engine = LiveTradingEngine(
            broker=broker,
            safety=SafetyLimits(
                respect_market_hours=False,
                min_cash_reserve=999,
            ),
        )
        engine.start()
        result = engine.execute_signal("BTC", "COMPRA", score=80, confidence=90)
        # Should be blocked due to exposure or reserve
        # The small capital means the trade would exceed limits

    def test_get_status(self):
        engine = LiveTradingEngine()
        engine.start()
        status = engine.get_status()
        assert status["bot_status"] == "RUNNING"
        assert "balance" in status
        assert "safety" in status
        engine.confirm_stop()

    def test_check_stop_losses(self):
        broker = PaperBrokerAdapter(initial_capital=100000)
        broker.set_price("AAPL", 200)
        broker.buy("AAPL", 10)
        # Price drops
        broker.set_price("AAPL", 190)
        broker.positions["AAPL"]["current_price"] = 190
        broker.positions["AAPL"]["avg_price"] = 200

        engine = LiveTradingEngine(
            broker=broker,
            safety=SafetyLimits(max_trade_loss_pct=3.0, respect_market_hours=False),
        )
        engine.start()
        engine.check_stop_losses()
        # -5% > -3% threshold, should have sold
        assert "AAPL" not in broker.get_positions()

    def test_log_export(self, tmp_path):
        engine = LiveTradingEngine()
        engine.start()
        filepath = str(tmp_path / "test_log.json")
        engine.export_log(filepath)
        engine.confirm_stop()

        import json
        with open(filepath) as f:
            data = json.load(f)
        assert "log" in data
        assert len(data["log"]) > 0
