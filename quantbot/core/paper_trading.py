"""
Paper Trading Engine — Modo Teste com Dinheiro Simulado.

Opera com dados REAIS de mercado mas com capital fictício,
permitindo testar estratégias sem risco financeiro.

Funcionalidades:
- Carteira virtual com saldo inicial configurável
- Execução de ordens (compra/venda) com preços reais
- Histórico completo de transações
- P&L em tempo real
- Exportação de relatórios
- Persistência do estado em JSON

Uso:
    engine = PaperTradingEngine(initial_capital=100000)
    engine.execute_buy("PETR4.SA", quantity=100)
    engine.execute_sell("PETR4.SA", quantity=50)
    engine.print_portfolio()
    engine.export_state("meu_teste.json")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import Signal, Market, UNIVERSE, BACKTEST_CONFIG
from utils.logger import get_logger
from utils.security import InputValidator
from utils.formatters import fmt_currency, fmt_pct

logger = get_logger("quantbot.paper_trading")


# ═══════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════

class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


@dataclass
class Order:
    """Representa uma ordem executada ou rejeitada."""
    id: int
    timestamp: str
    symbol: str
    order_type: str  # "BUY" or "SELL"
    quantity: float
    price: float
    total_value: float
    commission: float
    status: str  # "FILLED" or "REJECTED"
    reason: str = ""
    balance_after: float = 0.0
    pnl: float = 0.0  # P&L realizado (só para SELL)


@dataclass
class HoldingInfo:
    """Informações de uma posição mantida."""
    symbol: str
    name: str
    market: str
    quantity: float
    avg_price: float
    total_cost: float
    current_price: float = 0.0

    @property
    def current_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_value(self) -> float:
        return self.current_value - self.total_cost

    @property
    def pnl_pct(self) -> float:
        if self.total_cost == 0:
            return 0.0
        return (self.pnl_value / self.total_cost) * 100


@dataclass
class PortfolioSnapshot:
    """Snapshot do estado do portfólio em um momento."""
    timestamp: str
    cash: float
    holdings_value: float
    total_value: float
    pnl_total: float
    pnl_pct: float
    num_positions: int


# ═══════════════════════════════════════════════════════════════
# PAPER TRADING ENGINE
# ═══════════════════════════════════════════════════════════════

class PaperTradingEngine:
    """
    Motor de Paper Trading com dados reais e dinheiro simulado.

    Simula uma conta de corretora completa com:
    - Saldo em caixa (cash)
    - Posições abertas (holdings)
    - Histórico de ordens
    - Snapshots periódicos do portfólio
    - Métricas de performance

    Exemplo de uso:
        >>> engine = PaperTradingEngine(initial_capital=100000)
        >>> engine.execute_buy("PETR4.SA", quantity=200)
        >>> engine.execute_buy("AAPL", quantity=10)
        >>> engine.update_prices()
        >>> engine.print_portfolio()
        >>> engine.execute_sell("PETR4.SA", quantity=100)
        >>> engine.print_orders()
        >>> engine.export_state("paper_trading_state.json")
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.001,
        currency: str = "USD",
    ):
        """
        Inicializa o Paper Trading Engine.

        Args:
            initial_capital: Capital inicial simulado
            commission_rate: Taxa de corretagem (0.001 = 0.1%)
            currency: Moeda base (USD)
        """
        self.initial_capital = InputValidator.validate_capital(initial_capital)
        self.commission_rate = commission_rate
        self.currency = currency

        # Estado da conta
        self.cash: float = initial_capital
        self.holdings: Dict[str, HoldingInfo] = {}
        self.orders: List[Order] = []
        self.snapshots: List[PortfolioSnapshot] = []

        # Contadores
        self._order_counter = 0
        self._created_at = datetime.now().isoformat()

        # Cache de preços
        self._price_cache: Dict[str, float] = {}

        logger.info(
            f"💰 Paper Trading iniciado: "
            f"capital={fmt_currency(initial_capital)}, "
            f"comissão={commission_rate:.2%}"
        )

        # Snapshot inicial
        self._take_snapshot()

    # ─── PREÇOS ───────────────────────────────────────────

    def get_current_price(self, symbol: str) -> float:
        """
        Obtém preço atual real do ativo via yfinance.

        Args:
            symbol: Símbolo do ativo

        Returns:
            Preço atual

        Raises:
            ValueError: Se não conseguir obter o preço
        """
        symbol = InputValidator.validate_symbol(symbol)

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")

            if hist.empty:
                raise ValueError(f"Sem dados para {symbol}")

            price = float(hist["Close"].iloc[-1])

            if price <= 0:
                raise ValueError(f"Preço inválido para {symbol}: {price}")

            self._price_cache[symbol] = price
            return price

        except Exception as e:
            # Tenta cache
            if symbol in self._price_cache:
                logger.warning(
                    f"{symbol}: usando preço em cache "
                    f"({fmt_currency(self._price_cache[symbol])})"
                )
                return self._price_cache[symbol]
            raise ValueError(f"Não foi possível obter preço de {symbol}: {e}")

    def update_prices(self) -> Dict[str, float]:
        """
        Atualiza preços de todos os ativos em carteira.

        Returns:
            Dicionário {símbolo: preço_atualizado}
        """
        updated = {}

        for symbol, holding in self.holdings.items():
            try:
                price = self.get_current_price(symbol)
                holding.current_price = price
                updated[symbol] = price
            except ValueError as e:
                logger.warning(f"Erro atualizando {symbol}: {e}")

        if updated:
            logger.info(f"📊 Preços atualizados: {len(updated)} ativos")
            self._take_snapshot()

        return updated

    # ─── ORDENS ───────────────────────────────────────────

    def execute_buy(
        self,
        symbol: str,
        quantity: Optional[float] = None,
        amount: Optional[float] = None,
    ) -> Order:
        """
        Executa ordem de COMPRA.

        Pode especificar quantidade (quantity) OU valor (amount).
        Se amount for fornecido, calcula a quantidade automaticamente.

        Args:
            symbol: Símbolo do ativo
            quantity: Quantidade de ações/cotas
            amount: Valor em $ para investir

        Returns:
            Order com detalhes da execução

        Exemplo:
            >>> engine.execute_buy("PETR4.SA", quantity=100)
            >>> engine.execute_buy("AAPL", amount=5000)  # compra ~$5000
        """
        symbol = InputValidator.validate_symbol(symbol)

        # Obtém preço real
        try:
            price = self.get_current_price(symbol)
        except ValueError as e:
            return self._reject_order(symbol, "BUY", 0, 0, str(e))

        # Calcula quantidade
        if amount is not None:
            if amount <= 0:
                return self._reject_order(symbol, "BUY", 0, price, "Valor inválido")
            quantity = amount / price

        if quantity is None or quantity <= 0:
            return self._reject_order(
                symbol, "BUY", 0, price, "Quantidade inválida"
            )

        # Calcula custo total
        gross_value = quantity * price
        commission = gross_value * self.commission_rate
        total_cost = gross_value + commission

        # Verifica saldo
        if total_cost > self.cash:
            return self._reject_order(
                symbol, "BUY", quantity, price,
                f"Saldo insuficiente: precisa {fmt_currency(total_cost)}, "
                f"tem {fmt_currency(self.cash)}"
            )

        # Executa
        self.cash -= total_cost

        # Atualiza ou cria posição
        if symbol in self.holdings:
            holding = self.holdings[symbol]
            new_total_cost = holding.total_cost + gross_value
            new_quantity = holding.quantity + quantity
            holding.avg_price = new_total_cost / new_quantity
            holding.quantity = new_quantity
            holding.total_cost = new_total_cost
            holding.current_price = price
        else:
            # Busca nome do ativo
            name = symbol
            market = "Unknown"
            for mkt_assets in UNIVERSE.values():
                for asset in mkt_assets:
                    if asset.symbol == symbol:
                        name = asset.name
                        market = asset.market.value
                        break

            self.holdings[symbol] = HoldingInfo(
                symbol=symbol,
                name=name,
                market=market,
                quantity=quantity,
                avg_price=price,
                total_cost=gross_value,
                current_price=price,
            )

        order = self._create_order(
            symbol=symbol,
            order_type="BUY",
            quantity=quantity,
            price=price,
            total_value=gross_value,
            commission=commission,
            status="FILLED",
        )

        logger.info(
            f"✅ COMPRA: {quantity:.4f} x {symbol} @ {fmt_currency(price)} "
            f"= {fmt_currency(gross_value)} (taxa: {fmt_currency(commission)})"
        )

        self._take_snapshot()
        return order

    def execute_sell(
        self,
        symbol: str,
        quantity: Optional[float] = None,
        sell_all: bool = False,
    ) -> Order:
        """
        Executa ordem de VENDA.

        Args:
            symbol: Símbolo do ativo
            quantity: Quantidade a vender (None + sell_all=True para vender tudo)
            sell_all: Se True, vende toda a posição

        Returns:
            Order com detalhes da execução
        """
        symbol = InputValidator.validate_symbol(symbol)

        # Verifica se tem posição
        if symbol not in self.holdings:
            return self._reject_order(
                symbol, "SELL", 0, 0,
                f"Sem posição em {symbol}"
            )

        holding = self.holdings[symbol]

        # Determina quantidade
        if sell_all:
            quantity = holding.quantity
        elif quantity is None or quantity <= 0:
            return self._reject_order(
                symbol, "SELL", 0, 0, "Quantidade inválida"
            )

        if quantity > holding.quantity:
            return self._reject_order(
                symbol, "SELL", quantity, 0,
                f"Quantidade excede posição: {quantity} > {holding.quantity}"
            )

        # Obtém preço real
        try:
            price = self.get_current_price(symbol)
        except ValueError as e:
            return self._reject_order(symbol, "SELL", quantity, 0, str(e))

        # Calcula receita
        gross_value = quantity * price
        commission = gross_value * self.commission_rate
        net_value = gross_value - commission

        # P&L realizado
        cost_basis = quantity * holding.avg_price
        realized_pnl = net_value - cost_basis

        # Executa
        self.cash += net_value

        # Atualiza posição
        if quantity >= holding.quantity:
            del self.holdings[symbol]
        else:
            holding.quantity -= quantity
            holding.total_cost = holding.quantity * holding.avg_price
            holding.current_price = price

        order = self._create_order(
            symbol=symbol,
            order_type="SELL",
            quantity=quantity,
            price=price,
            total_value=gross_value,
            commission=commission,
            status="FILLED",
            pnl=realized_pnl,
        )

        pnl_str = fmt_currency(realized_pnl)
        pnl_emoji = "📈" if realized_pnl >= 0 else "📉"

        logger.info(
            f"✅ VENDA: {quantity:.4f} x {symbol} @ {fmt_currency(price)} "
            f"= {fmt_currency(gross_value)} | P&L: {pnl_str} {pnl_emoji}"
        )

        self._take_snapshot()
        return order

    def execute_signal(
        self,
        symbol: str,
        signal: Signal,
        score: float,
        portfolio_pct: float = 0.05,
    ) -> Optional[Order]:
        """
        Executa ordem baseada em sinal ML.

        Converte sinais de ML em ações de trading:
        - STRONG_BUY/BUY: compra se não tem posição
        - STRONG_SELL/SELL: vende se tem posição
        - HOLD: não faz nada

        Args:
            symbol: Símbolo do ativo
            signal: Sinal ML
            score: Score do modelo (0-100)
            portfolio_pct: % do portfólio para alocar

        Returns:
            Order executada ou None se HOLD
        """
        total_value = self.get_total_value()

        if signal in (Signal.STRONG_BUY, Signal.BUY):
            if symbol not in self.holdings:
                # Calcula quanto investir baseado no score
                base_amount = total_value * portfolio_pct
                score_multiplier = score / 100
                amount = base_amount * score_multiplier
                return self.execute_buy(symbol, amount=amount)

        elif signal in (Signal.STRONG_SELL, Signal.SELL):
            if symbol in self.holdings:
                if signal == Signal.STRONG_SELL:
                    return self.execute_sell(symbol, sell_all=True)
                else:
                    # Vende metade
                    qty = self.holdings[symbol].quantity / 2
                    return self.execute_sell(symbol, quantity=qty)

        return None  # HOLD

    # ─── CONSULTAS ────────────────────────────────────────

    def get_total_value(self) -> float:
        """Retorna valor total do portfólio (cash + holdings)."""
        holdings_value = sum(h.current_value for h in self.holdings.values())
        return self.cash + holdings_value

    def get_total_pnl(self) -> Tuple[float, float]:
        """Retorna P&L total (valor, percentual)."""
        total = self.get_total_value()
        pnl_value = total - self.initial_capital
        pnl_pct = (pnl_value / self.initial_capital) * 100
        return pnl_value, pnl_pct

    def get_holdings_summary(self) -> pd.DataFrame:
        """Retorna DataFrame com resumo das posições."""
        if not self.holdings:
            return pd.DataFrame()

        rows = []
        total_value = self.get_total_value()

        for symbol, h in self.holdings.items():
            weight = (h.current_value / total_value * 100) if total_value > 0 else 0
            rows.append({
                "symbol": h.symbol,
                "name": h.name,
                "market": h.market,
                "quantity": h.quantity,
                "avg_price": h.avg_price,
                "current_price": h.current_price,
                "cost": h.total_cost,
                "value": h.current_value,
                "pnl_value": h.pnl_value,
                "pnl_pct": h.pnl_pct,
                "weight_pct": weight,
            })

        df = pd.DataFrame(rows)
        return df.sort_values("value", ascending=False).reset_index(drop=True)

    def get_orders_history(self) -> pd.DataFrame:
        """Retorna DataFrame com histórico de ordens."""
        if not self.orders:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "id": o.id,
                "timestamp": o.timestamp,
                "symbol": o.symbol,
                "type": o.order_type,
                "qty": o.quantity,
                "price": o.price,
                "value": o.total_value,
                "commission": o.commission,
                "status": o.status,
                "pnl": o.pnl,
                "balance_after": o.balance_after,
                "reason": o.reason,
            }
            for o in self.orders
        ])

    def get_realized_pnl(self) -> float:
        """Retorna P&L realizado total (de ordens de venda fechadas)."""
        return sum(
            o.pnl for o in self.orders
            if o.order_type == "SELL" and o.status == "FILLED"
        )

    def get_unrealized_pnl(self) -> float:
        """Retorna P&L não realizado (posições abertas)."""
        return sum(h.pnl_value for h in self.holdings.values())

    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance do paper trading."""
        total_value = self.get_total_value()
        pnl_value, pnl_pct = self.get_total_pnl()

        filled_orders = [o for o in self.orders if o.status == "FILLED"]
        buy_orders = [o for o in filled_orders if o.order_type == "BUY"]
        sell_orders = [o for o in filled_orders if o.order_type == "SELL"]
        winning_sells = [o for o in sell_orders if o.pnl > 0]
        losing_sells = [o for o in sell_orders if o.pnl <= 0]

        total_commission = sum(o.commission for o in filled_orders)
        win_rate = (len(winning_sells) / len(sell_orders) * 100) if sell_orders else 0

        return {
            "initial_capital": self.initial_capital,
            "current_value": total_value,
            "cash": self.cash,
            "holdings_value": total_value - self.cash,
            "pnl_total": pnl_value,
            "pnl_pct": pnl_pct,
            "pnl_realized": self.get_realized_pnl(),
            "pnl_unrealized": self.get_unrealized_pnl(),
            "total_orders": len(filled_orders),
            "buy_orders": len(buy_orders),
            "sell_orders": len(sell_orders),
            "win_rate": win_rate,
            "total_commission": total_commission,
            "num_positions": len(self.holdings),
            "created_at": self._created_at,
            "last_update": datetime.now().isoformat(),
        }

    # ─── DISPLAY ──────────────────────────────────────────

    def print_portfolio(self):
        """Imprime resumo do portfólio no terminal."""
        self.update_prices()

        total_value = self.get_total_value()
        pnl_value, pnl_pct = self.get_total_pnl()

        print("\n" + "═" * 95)
        print("  💰 PAPER TRADING — PORTFÓLIO SIMULADO")
        print("═" * 95)
        print(f"\n  Capital Inicial: {fmt_currency(self.initial_capital)}")
        print(f"  Valor Atual:     {fmt_currency(total_value)}")
        print(f"  Caixa (Cash):    {fmt_currency(self.cash)}")
        print(f"  Em Posições:     {fmt_currency(total_value - self.cash)}")
        print(f"  P&L Total:       {fmt_currency(pnl_value)} ({fmt_pct(pnl_pct)})")
        print(f"  P&L Realizado:   {fmt_currency(self.get_realized_pnl())}")
        print(f"  P&L Não Realiz.: {fmt_currency(self.get_unrealized_pnl())}")

        if self.holdings:
            print(f"\n  {'Ativo':<12} {'Nome':<16} {'Merc.':<7} {'Qtd':>8} "
                  f"{'P.Médio':>10} {'P.Atual':>10} {'P&L':>8} "
                  f"{'Valor':>12} {'Peso':>6}")
            print("  " + "─" * 93)

            df = self.get_holdings_summary()
            for _, row in df.iterrows():
                pnl_str = f"{row['pnl_pct']:+.1f}%"
                print(
                    f"  {row['symbol']:<12} {row['name'][:15]:<16} "
                    f"{row['market']:<7} {row['quantity']:>8.2f} "
                    f"{fmt_currency(row['avg_price']):>10} "
                    f"{fmt_currency(row['current_price']):>10} "
                    f"{pnl_str:>8} {fmt_currency(row['value']):>12} "
                    f"{row['weight_pct']:>5.1f}%"
                )

            print("  " + "─" * 93)
        else:
            print("\n  (Nenhuma posição aberta)")

        print(f"\n  Ordens executadas: {len([o for o in self.orders if o.status == 'FILLED'])}")
        print(f"  Comissões pagas:   {fmt_currency(sum(o.commission for o in self.orders if o.status == 'FILLED'))}")
        print("═" * 95 + "\n")

    def print_orders(self, last_n: int = 20):
        """Imprime histórico de ordens."""
        print("\n" + "═" * 95)
        print("  📋 HISTÓRICO DE ORDENS")
        print("═" * 95)

        orders = self.orders[-last_n:]

        if not orders:
            print("  (Nenhuma ordem registrada)")
            return

        print(f"\n  {'ID':>4} {'Data/Hora':<20} {'Tipo':<5} {'Ativo':<12} "
              f"{'Qtd':>8} {'Preço':>10} {'Total':>12} {'P&L':>10} {'Status':<8}")
        print("  " + "─" * 93)

        for o in orders:
            pnl_str = fmt_currency(o.pnl) if o.pnl != 0 else "-"
            status_emoji = "✅" if o.status == "FILLED" else "❌"

            print(
                f"  {o.id:>4} {o.timestamp[:19]:<20} {o.order_type:<5} "
                f"{o.symbol:<12} {o.quantity:>8.2f} "
                f"{fmt_currency(o.price):>10} {fmt_currency(o.total_value):>12} "
                f"{pnl_str:>10} {status_emoji} {o.status:<8}"
            )

            if o.reason:
                print(f"       └─ {o.reason}")

        print("═" * 95 + "\n")

    # ─── PERSISTÊNCIA ─────────────────────────────────────

    def export_state(self, filepath: str = "paper_trading_state.json"):
        """
        Exporta estado completo para JSON.

        Permite salvar e retomar a simulação depois.
        """
        state = {
            "metadata": {
                "version": "1.0",
                "created_at": self._created_at,
                "exported_at": datetime.now().isoformat(),
                "engine": "QuantBot Paper Trading",
            },
            "config": {
                "initial_capital": self.initial_capital,
                "commission_rate": self.commission_rate,
                "currency": self.currency,
            },
            "account": {
                "cash": self.cash,
                "order_counter": self._order_counter,
            },
            "holdings": {
                symbol: {
                    "symbol": h.symbol,
                    "name": h.name,
                    "market": h.market,
                    "quantity": h.quantity,
                    "avg_price": h.avg_price,
                    "total_cost": h.total_cost,
                    "current_price": h.current_price,
                }
                for symbol, h in self.holdings.items()
            },
            "orders": [
                {
                    "id": o.id,
                    "timestamp": o.timestamp,
                    "symbol": o.symbol,
                    "order_type": o.order_type,
                    "quantity": o.quantity,
                    "price": o.price,
                    "total_value": o.total_value,
                    "commission": o.commission,
                    "status": o.status,
                    "reason": o.reason,
                    "balance_after": o.balance_after,
                    "pnl": o.pnl,
                }
                for o in self.orders
            ],
            "performance": self.get_performance_metrics(),
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "cash": s.cash,
                    "holdings_value": s.holdings_value,
                    "total_value": s.total_value,
                    "pnl_total": s.pnl_total,
                    "pnl_pct": s.pnl_pct,
                    "num_positions": s.num_positions,
                }
                for s in self.snapshots
            ],
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Estado exportado: {filepath}")
        print(f"\n  💾 Estado salvo em: {filepath}")

    @classmethod
    def load_state(cls, filepath: str) -> "PaperTradingEngine":
        """
        Carrega estado salvo de um JSON.

        Args:
            filepath: Caminho do arquivo JSON

        Returns:
            PaperTradingEngine com estado restaurado
        """
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        config = state["config"]
        engine = cls(
            initial_capital=config["initial_capital"],
            commission_rate=config["commission_rate"],
            currency=config["currency"],
        )

        # Restaura conta
        engine.cash = state["account"]["cash"]
        engine._order_counter = state["account"]["order_counter"]
        engine._created_at = state["metadata"]["created_at"]

        # Restaura holdings
        for symbol, h_data in state["holdings"].items():
            engine.holdings[symbol] = HoldingInfo(**h_data)

        # Restaura ordens
        for o_data in state["orders"]:
            engine.orders.append(Order(**o_data))

        # Restaura snapshots
        for s_data in state["snapshots"]:
            engine.snapshots.append(PortfolioSnapshot(**s_data))

        logger.info(
            f"💾 Estado carregado: {filepath} "
            f"({len(engine.holdings)} posições, {len(engine.orders)} ordens)"
        )

        return engine

    # ─── INTERNOS ─────────────────────────────────────────

    def _create_order(self, **kwargs) -> Order:
        """Cria e registra uma ordem."""
        self._order_counter += 1
        order = Order(
            id=self._order_counter,
            timestamp=datetime.now().isoformat(),
            balance_after=self.cash,
            **kwargs,
        )
        self.orders.append(order)
        return order

    def _reject_order(
        self, symbol: str, order_type: str,
        quantity: float, price: float, reason: str
    ) -> Order:
        """Rejeita uma ordem e registra o motivo."""
        self._order_counter += 1
        order = Order(
            id=self._order_counter,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            price=price,
            total_value=0,
            commission=0,
            status="REJECTED",
            reason=reason,
            balance_after=self.cash,
        )
        self.orders.append(order)

        logger.warning(f"❌ Ordem rejeitada: {order_type} {symbol} — {reason}")
        return order

    def _take_snapshot(self):
        """Registra snapshot do portfólio."""
        total_value = self.get_total_value()
        pnl_value, pnl_pct = self.get_total_pnl()
        holdings_value = sum(h.current_value for h in self.holdings.values())

        self.snapshots.append(PortfolioSnapshot(
            timestamp=datetime.now().isoformat(),
            cash=self.cash,
            holdings_value=holdings_value,
            total_value=total_value,
            pnl_total=pnl_value,
            pnl_pct=pnl_pct,
            num_positions=len(self.holdings),
        ))
