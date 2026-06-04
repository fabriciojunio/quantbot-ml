"""Trailing stop dinâmico baseado em ATR, com breakeven e saída por tempo.

Regras:
- Stop inicial = entrada - atr_mult * ATR.
- O stop só sobe (nunca desce) conforme o preço avança (trailing).
- Ao atingir o gatilho de breakeven, o stop é movido para o preço de entrada.
- Saída por tempo após max_holding_days barras.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class StopPosition:
    symbol: str
    entry_price: float
    entry_date: str
    atr: float
    current_stop: float
    days_held: int = 0
    breakeven_set: bool = False


class DynamicStopManager:
    """Gerencia trailing stops de múltiplas posições."""

    def __init__(
        self,
        atr_mult: float = 1.5,
        max_holding_days: int = 20,
        breakeven_trigger: float = 0.03,
    ):
        self.atr_mult = atr_mult
        self.max_holding_days = max_holding_days
        self.breakeven_trigger = breakeven_trigger
        self.positions: Dict[str, StopPosition] = {}

    def open_position(
        self, symbol: str, entry_price: float, entry_date: str, atr: float
    ) -> StopPosition:
        stop = entry_price - self.atr_mult * atr
        pos = StopPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=entry_date,
            atr=atr,
            current_stop=stop,
        )
        self.positions[symbol] = pos
        return pos

    def update(self, symbol: str, price: float, atr: float) -> Optional[str]:
        """Atualiza a posição e retorna um evento de saída, se houver."""
        pos = self.positions.get(symbol)
        if pos is None:
            return None

        pos.days_held += 1
        pos.atr = atr

        # 1. Stop atingido tem prioridade máxima.
        if price <= pos.current_stop:
            return "stop_triggered"

        # 2. Trailing: o stop só sobe.
        trailing_stop = price - self.atr_mult * atr
        if trailing_stop > pos.current_stop:
            pos.current_stop = trailing_stop

        # 3. Breakeven: move o stop para a entrada ao atingir o gatilho.
        profit_pct = (price - pos.entry_price) / pos.entry_price
        if not pos.breakeven_set and profit_pct >= self.breakeven_trigger:
            pos.current_stop = max(pos.current_stop, pos.entry_price)
            pos.breakeven_set = True
            return "breakeven_set"

        # 4. Saída por tempo.
        if pos.days_held >= self.max_holding_days:
            return "time_exit"

        return None

    def close_position(self, symbol: str) -> Optional[StopPosition]:
        return self.positions.pop(symbol, None)

    def get_all_stops(self) -> Dict[str, float]:
        return {sym: pos.current_stop for sym, pos in self.positions.items()}
