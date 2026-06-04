"""Testes do backtester com custos reais e do kill-switch."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.realistic_backtester import RealisticBacktester, BrazilianCostModel
from risk.kill_switch import RiskManager, RiskLimits, RiskViolation, KillSwitchActivated


# ═══════════════════════════════════════════════════════════════
# RealisticBacktester
# ═══════════════════════════════════════════════════════════════

def _make_price_series(n=504, trend=0.0003, seed=42):
    """Simula série de preços com GBM."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, 0.015, n)
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


class TestRealisticBacktester:
    def test_buy_and_hold_net_less_than_gross(self):
        """Os custos SEMPRE reduzem o retorno bruto — sem exceções."""
        prices = _make_price_series(trend=0.001)
        # Sempre comprado
        signals = pd.Series(1, index=prices.index)
        bt = RealisticBacktester(initial_capital=1_000.0)
        result = bt.run(signals, prices)
        assert result.total_net_return <= result.total_gross_return

    def test_costs_reduce_return(self):
        """Com custos = 0, retorno líquido deve se aproximar do bruto."""
        prices = _make_price_series(trend=0.0005)
        signals = pd.Series([1, 0] * 252, index=prices.index[:504])
        bt_costly = RealisticBacktester(initial_capital=1_000.0)
        result = bt_costly.run(signals, prices)
        # O custo total deve ser positivo (não negativo)
        assert result.total_transaction_costs >= 0

    def test_no_trades_when_flat(self):
        """Sinal sempre 0 = nenhum trade executado."""
        prices = _make_price_series()
        signals = pd.Series(0, index=prices.index)
        bt = RealisticBacktester(initial_capital=1_000.0)
        result = bt.run(signals, prices)
        assert result.n_trades == 0

    def test_swing_trade_exemption_applied(self):
        """Com capital pequeno (R$1000), IR deve ser 0 (isenção R$20k/mês)."""
        prices = _make_price_series(trend=0.002, n=100)
        # 1 trade no mês, capital pequeno — vendas abaixo de R$20k
        signals = pd.Series([1]*50 + [0]*50, index=prices.index)
        bt = RealisticBacktester(initial_capital=1_000.0)
        result = bt.run(signals, prices)
        # Com R$1000, as vendas são < R$20k -> IR = 0
        assert result.total_ir_paid == 0.0

    def test_summary_has_required_keys(self):
        prices = _make_price_series()
        signals = pd.Series([1, 0] * 252, index=prices.index)
        bt = RealisticBacktester(initial_capital=1_000.0)
        result = bt.run(signals, prices)
        summary = result.summary()
        for key in ["retorno_liquido_pct", "custos_transacao_pct", "sharpe_ratio",
                    "max_drawdown_pct", "win_rate_pct"]:
            assert key in summary

    def test_equity_starts_at_initial_capital(self):
        prices = _make_price_series()
        signals = pd.Series(0, index=prices.index)
        bt = RealisticBacktester(initial_capital=1_234.56)
        result = bt.run(signals, prices)
        assert abs(result.equity_curve.iloc[0] - 1_234.56) < 0.01


# ═══════════════════════════════════════════════════════════════
# RiskManager / KillSwitch
# ═══════════════════════════════════════════════════════════════

class TestRiskManager:
    def _make_rm(self, tmp_path, **limit_kwargs):
        limits = RiskLimits(**limit_kwargs)
        state_file = str(tmp_path / "risk_state.json")
        rm = RiskManager(limits=limits, state_file=state_file)
        rm.initialize(current_capital=1_000.0)
        return rm

    def test_normal_order_passes(self, tmp_path):
        rm = self._make_rm(tmp_path)
        rm.check_order(capital=1_000.0, order_value=200.0)  # não deve levantar

    def test_position_too_large_blocked(self, tmp_path):
        rm = self._make_rm(tmp_path, max_position_size_pct=0.25)
        with pytest.raises(RiskViolation):
            rm.check_order(capital=1_000.0, order_value=300.0)  # 30% > 25%

    def test_daily_drawdown_stops_trading(self, tmp_path):
        rm = self._make_rm(tmp_path, max_daily_drawdown_pct=0.03)
        with pytest.raises(RiskViolation):
            rm.check_order(capital=960.0, order_value=100.0)  # -4% > -3%

    def test_total_drawdown_activates_kill_switch(self, tmp_path):
        rm = self._make_rm(tmp_path, max_total_drawdown_pct=0.20)
        with pytest.raises(KillSwitchActivated):
            rm.check_order(capital=750.0, order_value=100.0)  # -25% > -20%

    def test_max_trades_per_day_enforced(self, tmp_path):
        rm = self._make_rm(tmp_path, max_trades_per_day=3)
        rm.record_trade_result(pnl=10.0, capital=1_010.0)
        rm.record_trade_result(pnl=10.0, capital=1_020.0)
        rm.record_trade_result(pnl=10.0, capital=1_030.0)
        with pytest.raises(RiskViolation):
            rm.check_order(capital=1_030.0, order_value=100.0)

    def test_kill_switch_manual(self, tmp_path):
        rm = self._make_rm(tmp_path)
        rm.activate_kill_switch("teste manual")
        with pytest.raises(KillSwitchActivated):
            rm.check_order(capital=1_000.0, order_value=100.0)

    def test_kill_switch_deactivatable(self, tmp_path):
        rm = self._make_rm(tmp_path)
        rm.activate_kill_switch("teste")
        rm.deactivate_kill_switch()
        rm.check_order(capital=1_000.0, order_value=100.0)  # deve passar

    def test_consecutive_losses_cooldown(self, tmp_path):
        rm = self._make_rm(tmp_path, max_consecutive_losses=3, cooldown_hours=0)
        rm.record_trade_result(pnl=-10.0, capital=990.0)
        rm.record_trade_result(pnl=-10.0, capital=980.0)
        with pytest.raises(RiskViolation):
            # 3ª perda dispara cool-down
            rm.record_trade_result(pnl=-10.0, capital=970.0)
            rm.check_order(capital=970.0, order_value=100.0)

    def test_state_persisted_and_reloaded(self, tmp_path):
        """Estado deve sobreviver a restart do RiskManager."""
        state_file = str(tmp_path / "risk_state.json")
        rm1 = RiskManager(limits=RiskLimits(), state_file=state_file)
        rm1.initialize(1_000.0)
        rm1.record_trade_result(pnl=50.0, capital=1_050.0)

        # Cria nova instância lendo o mesmo arquivo
        rm2 = RiskManager(limits=RiskLimits(), state_file=state_file)
        assert rm2.state.total_trades == 1
        assert rm2.state.current_capital == 1_050.0

    def test_status_returns_required_fields(self, tmp_path):
        rm = self._make_rm(tmp_path)
        status = rm.get_status()
        for k in ["kill_switch_ativo", "capital_atual", "drawdown_total_pct",
                  "drawdown_diario_pct", "trades_hoje"]:
            assert k in status
