"""
Stop-Loss Dinâmico — Trailing Stop + Volatilidade Adaptativa.

O stop-loss fixo do Luan (5%) funciona, mas tem uma fraqueza:
ele não se adapta à volatilidade do ativo. BTC com vol de 4%
precisa de stop diferente de AAPL com vol de 1.5%.

Melhorias implementadas:
    1. ATR Trailing Stop: stop sobe conforme preço sobe (nunca desce)
    2. Volatility-Adjusted Stop: stop baseado em múltiplos do ATR
    3. Time-Based Exit: fecha posição se não atingir TP em N dias
    4. Chandelier Exit: clássico trailing stop baseado em máxima

Resultado esperado:
    - Protege lucros que o stop fixo deixaria escapar
    - Reduz drawdown em ativos voláteis (cripto)
    - Mantém posições vencedoras por mais tempo

Referências:
    - Wilder (1978) — ATR para sizing de stop
    - Chandelier Exit — Chuck LeBeau
    - QuantifiedStrategies (2026) — volatility-adjusted positioning
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

logger = logging.getLogger("quantbot.risk.dynamic_stop")


@dataclass
class TrailingStopState:
    """Estado do trailing stop para uma posição."""
    symbol: str
    entry_price: float
    entry_date: str
    highest_price: float  # Máxima desde a entrada
    current_stop: float   # Nível atual do stop
    atr_at_entry: float   # ATR no momento da entrada
    days_held: int = 0

    @property
    def profit_locked_pct(self) -> float:
        """% de lucro protegido pelo trailing stop."""
        if self.entry_price <= 0:
            return 0
        return ((self.current_stop - self.entry_price) / self.entry_price) * 100


class DynamicStopManager:
    """
    Gerencia stops dinâmicos para todas as posições abertas.

    Métodos de stop:
        1. ATR Trailing: stop = highest_price - (atr_mult * ATR)
        2. Percent Trailing: stop sobe com o preço (nunca desce)
        3. Time Exit: fecha após max_holding dias sem TP
    """

    def __init__(
        self,
        atr_mult: float = 2.5,       # Stop a 2.5x ATR da máxima
        min_stop_pct: float = 0.03,   # Stop mínimo de 3%
        max_stop_pct: float = 0.10,   # Stop máximo de 10%
        max_holding_days: int = 30,   # Saída temporal máxima
        breakeven_trigger: float = 0.02,  # Move stop para breakeven após +2%
    ):
        self.atr_mult = atr_mult
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
        self.max_holding_days = max_holding_days
        self.breakeven_trigger = breakeven_trigger

        self.positions: Dict[str, TrailingStopState] = {}

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        entry_date: str,
        atr: float,
    ):
        """Registra nova posição com stop inicial."""
        # Stop inicial baseado no ATR
        atr_stop = entry_price - (self.atr_mult * atr)

        # Garante que está dentro dos limites min/max
        min_stop = entry_price * (1 - self.max_stop_pct)
        max_stop = entry_price * (1 - self.min_stop_pct)
        initial_stop = np.clip(atr_stop, min_stop, max_stop)

        self.positions[symbol] = TrailingStopState(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=entry_date,
            highest_price=entry_price,
            current_stop=initial_stop,
            atr_at_entry=atr,
        )

        logger.info(
            f"Stop aberto: {symbol} @ ${entry_price:.2f}, "
            f"stop=${initial_stop:.2f} ({((entry_price-initial_stop)/entry_price)*100:.1f}%)"
        )

    def update(self, symbol: str, current_price: float, current_atr: float) -> str:
        """
        Atualiza stop de uma posição e retorna ação.

        Returns:
            "hold" — manter posição
            "stop_triggered" — stop atingido, fechar
            "time_exit" — tempo máximo atingido
            "breakeven_set" — stop movido para breakeven
        """
        if symbol not in self.positions:
            return "hold"

        pos = self.positions[symbol]
        pos.days_held += 1

        # ── Time Exit ──
        if pos.days_held >= self.max_holding_days:
            logger.info(f"Time exit: {symbol} após {pos.days_held} dias")
            return "time_exit"

        # ── Verifica stop ──
        if current_price <= pos.current_stop:
            pnl = ((current_price - pos.entry_price) / pos.entry_price) * 100
            logger.info(
                f"Stop triggered: {symbol} @ ${current_price:.2f} "
                f"(stop=${pos.current_stop:.2f}, P&L={pnl:+.1f}%)"
            )
            return "stop_triggered"

        # ── Atualiza highest price ──
        action = "hold"
        if current_price > pos.highest_price:
            pos.highest_price = current_price

            # Novo trailing stop baseado no ATR atual
            new_stop = current_price - (self.atr_mult * current_atr)

            # Garante que stop só sobe, nunca desce
            if new_stop > pos.current_stop:
                pos.current_stop = new_stop
                logger.debug(
                    f"Trailing stop atualizado: {symbol} "
                    f"stop=${new_stop:.2f} (locked: {pos.profit_locked_pct:+.1f}%)"
                )

        # ── Breakeven trigger ──
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        if pnl_pct >= self.breakeven_trigger and pos.current_stop < pos.entry_price:
            pos.current_stop = pos.entry_price * 1.001  # Breakeven + 0.1%
            action = "breakeven_set"
            logger.info(f"Breakeven set: {symbol} @ ${pos.entry_price:.2f}")

        return action

    def close_position(self, symbol: str) -> Optional[TrailingStopState]:
        """Remove posição do tracking."""
        return self.positions.pop(symbol, None)

    def get_all_stops(self) -> Dict[str, dict]:
        """Retorna estado de todos os stops ativos."""
        return {
            sym: {
                "entry": pos.entry_price,
                "highest": pos.highest_price,
                "current_stop": pos.current_stop,
                "days_held": pos.days_held,
                "profit_locked": pos.profit_locked_pct,
            }
            for sym, pos in self.positions.items()
        }
