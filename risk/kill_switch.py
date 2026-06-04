"""
Kill-Switch e Camada de Risco Independente.

Princípio de design
-------------------
O risco é uma camada SEPARADA das estratégias. A estratégia pode dizer
"compre" e o risk layer pode silenciosamente bloquear a ordem. Esse
isolamento garante que as travas funcionem mesmo quando a estratégia
apresenta comportamento inesperado — corrupção de dados, bug de feature,
regime não visto antes.

Esta é a arquitetura que fundos profissionais usam. O risk manager nunca
é um "if" dentro da estratégia — é um portão independente que toda ordem
deve cruzar.

Limites monitorados:
    - Drawdown diário máximo (para tudo no dia se perder > X%)
    - Drawdown total máximo (encerra completamente se perder > Y%)
    - Nº máximo de trades por dia (evita loops de ordens)
    - Exposição máxima por posição (% do capital)
    - Perda consecutiva máxima (cool-down após N perdas seguidas)
    - Kill-switch manual (flag que para tudo imediatamente)
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("quantbot.risk.killswitch")


@dataclass
class RiskLimits:
    """
    Parâmetros de risco configuráveis.

    Todos têm defaults conservadores adequados para uma conta de R$1.000.
    """
    max_daily_drawdown_pct: float = 0.03       # Para no dia se perder 3%
    max_total_drawdown_pct: float = 0.20       # Encerra tudo se perder 20%
    max_trades_per_day: int = 5                # Máximo de trades por dia
    max_position_size_pct: float = 0.25        # Máx 25% do capital por trade
    max_consecutive_losses: int = 5            # Cool-down após 5 perdas seguidas
    cooldown_hours: int = 24                   # Horas de pausa após cool-down
    kill_switch_file: str = "killswitch.flag"  # Arquivo que para tudo manualmente


@dataclass
class RiskState:
    """Estado persistente do risk manager (salvo em disco a cada ciclo)."""
    peak_capital: float = 0.0
    current_capital: float = 0.0
    daily_start_capital: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    last_trade_date: str = ""
    cooldown_until: str = ""
    kill_switch_active: bool = False
    total_trades: int = 0
    last_updated: str = ""


class RiskViolation(Exception):
    """Levantada quando uma ordem viola os limites de risco."""
    pass


class KillSwitchActivated(Exception):
    """Levantada quando o kill-switch está ativo."""
    pass


class RiskManager:
    """
    Portão de risco independente. Toda ordem passa por aqui antes de executar.

    Usage:
        rm = RiskManager(limits=RiskLimits(), state_file="risk_state.json")
        rm.initialize(current_capital=1000.0)

        # Antes de cada trade:
        rm.check_order(capital=980.0, order_value=200.0)  # levanta se violação

        # Após cada trade:
        rm.record_trade_result(pnl=-15.0, capital=965.0)
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        state_file: str = "data/risk_state.json",
    ):
        self.limits = limits or RiskLimits()
        self.state_file = Path(state_file)
        self.state = self._load_state()

    # ── Inicialização ────────────────────────────────────────────────────────

    def initialize(self, current_capital: float) -> None:
        """Deve ser chamado uma vez no início de cada sessão de trading."""
        today = str(date.today())
        if self.state.last_trade_date != today:
            self.state.daily_start_capital = current_capital
            self.state.trades_today = 0

        if self.state.peak_capital == 0:
            self.state.peak_capital = current_capital

        self.state.current_capital = current_capital
        self.state.last_trade_date = today
        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        self._save_state()
        logger.info(f"RiskManager inicializado. Capital: R${current_capital:.2f}")

    # ── Verificação de ordem ─────────────────────────────────────────────────

    def check_order(self, capital: float, order_value: float) -> None:
        """
        Verifica se a ordem pode ser executada.

        Levanta RiskViolation ou KillSwitchActivated se algum limite for
        violado. NÃO retorna bool — falha explicitamente (fail-loud).

        Args:
            capital: capital atual em R$.
            order_value: valor da ordem em R$.
        """
        self._check_kill_switch()
        self._check_cooldown()
        self._check_total_drawdown(capital)
        self._check_daily_drawdown(capital)
        self._check_trades_per_day()
        self._check_position_size(capital, order_value)
        self._check_consecutive_losses()

        logger.debug(
            f"Ordem aprovada: valor=R${order_value:.2f}, "
            f"capital=R${capital:.2f}, trades_hoje={self.state.trades_today}"
        )

    def record_trade_result(self, pnl: float, capital: float) -> None:
        """Registra o resultado de um trade e atualiza o estado."""
        self.state.trades_today += 1
        self.state.total_trades += 1
        self.state.current_capital = capital

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        if capital > self.state.peak_capital:
            self.state.peak_capital = capital

        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        self._save_state()
        logger.info(
            f"Trade registrado: PnL=R${pnl:.2f}, capital=R${capital:.2f}, "
            f"losses_consecutivas={self.state.consecutive_losses}"
        )

    def activate_kill_switch(self, reason: str = "manual") -> None:
        """Para tudo. Cria arquivo de flag e atualiza estado."""
        self.state.kill_switch_active = True
        self._save_state()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        flag = self.state_file.parent / self.limits.kill_switch_file
        flag.write_text(f"KILL SWITCH ATIVO: {reason}\nTimestamp: {datetime.now(timezone.utc).isoformat()}")
        logger.critical(f"KILL SWITCH ATIVADO: {reason}")

    def deactivate_kill_switch(self) -> None:
        """Reativa o sistema manualmente após revisão humana."""
        self.state.kill_switch_active = False
        flag = self.state_file.parent / self.limits.kill_switch_file
        if flag.exists():
            flag.unlink()
        self._save_state()
        logger.warning("Kill switch desativado. Sistema reativado manualmente.")

    def get_status(self) -> dict:
        """Retorna status atual do risk manager para o dashboard."""
        capital = self.state.current_capital
        peak = self.state.peak_capital
        daily_start = self.state.daily_start_capital

        total_dd = ((capital - peak) / peak) if peak > 0 else 0
        daily_dd = ((capital - daily_start) / daily_start) if daily_start > 0 else 0

        return {
            "kill_switch_ativo": self.state.kill_switch_active,
            "capital_atual": round(capital, 2),
            "capital_pico": round(peak, 2),
            "drawdown_total_pct": round(total_dd * 100, 2),
            "drawdown_diario_pct": round(daily_dd * 100, 2),
            "trades_hoje": self.state.trades_today,
            "losses_consecutivas": self.state.consecutive_losses,
            "limite_drawdown_total_pct": self.limits.max_total_drawdown_pct * 100,
            "limite_drawdown_diario_pct": self.limits.max_daily_drawdown_pct * 100,
        }

    # ── Verificações individuais ─────────────────────────────────────────────

    def _check_kill_switch(self) -> None:
        flag = self.state_file.parent / self.limits.kill_switch_file
        if self.state.kill_switch_active or flag.exists():
            raise KillSwitchActivated(
                "Kill switch ativo. Revise o estado do sistema antes de continuar."
            )

    def _check_cooldown(self) -> None:
        if not self.state.cooldown_until:
            return
        cooldown_end = datetime.fromisoformat(self.state.cooldown_until)
        if datetime.now(timezone.utc) < cooldown_end:
            raise RiskViolation(
                f"Sistema em cool-down até {cooldown_end.isoformat()}. "
                f"Aguarde após {self.limits.max_consecutive_losses} perdas consecutivas."
            )
        self.state.cooldown_until = ""

    def _check_total_drawdown(self, capital: float) -> None:
        if self.state.peak_capital <= 0:
            return
        drawdown = (capital - self.state.peak_capital) / self.state.peak_capital
        if drawdown <= -self.limits.max_total_drawdown_pct:
            self.activate_kill_switch(
                f"Drawdown total de {drawdown*100:.1f}% atingiu o limite de "
                f"{self.limits.max_total_drawdown_pct*100:.0f}%"
            )
            raise KillSwitchActivated(
                f"Drawdown máximo total atingido: {drawdown*100:.1f}%. Sistema parado."
            )

    def _check_daily_drawdown(self, capital: float) -> None:
        if self.state.daily_start_capital <= 0:
            return
        daily_dd = (capital - self.state.daily_start_capital) / self.state.daily_start_capital
        if daily_dd <= -self.limits.max_daily_drawdown_pct:
            raise RiskViolation(
                f"Drawdown diário de {daily_dd*100:.1f}% atingiu o limite de "
                f"{self.limits.max_daily_drawdown_pct*100:.0f}%. Parado pelo resto do dia."
            )

    def _check_trades_per_day(self) -> None:
        if self.state.trades_today >= self.limits.max_trades_per_day:
            raise RiskViolation(
                f"Limite de {self.limits.max_trades_per_day} trades/dia atingido."
            )

    def _check_position_size(self, capital: float, order_value: float) -> None:
        if capital <= 0:
            raise RiskViolation("Capital zerado ou negativo.")
        position_pct = order_value / capital
        if position_pct > self.limits.max_position_size_pct:
            raise RiskViolation(
                f"Tamanho da posição ({position_pct*100:.1f}%) excede o limite "
                f"de {self.limits.max_position_size_pct*100:.0f}% do capital."
            )

    def _check_consecutive_losses(self) -> None:
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            from datetime import timedelta
            cooldown_end = datetime.now(timezone.utc) + timedelta(hours=self.limits.cooldown_hours)
            self.state.cooldown_until = cooldown_end.isoformat()
            self.state.consecutive_losses = 0
            self._save_state()
            raise RiskViolation(
                f"{self.limits.max_consecutive_losses} perdas consecutivas. "
                f"Cool-down de {self.limits.cooldown_hours}h ativado."
            )

    # ── Persistência ─────────────────────────────────────────────────────────

    def _load_state(self) -> RiskState:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                return RiskState(**data)
            except Exception as e:
                logger.warning(f"Erro ao carregar estado de risco: {e}. Iniciando do zero.")
        return RiskState()

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(asdict(self.state), indent=2))
