"""
Live Trading Engine — Base para Operação com Dinheiro Real.

Arquitetura preparada para conectar com qualquer corretora:
- Binance (Crypto)
- Alpaca (Ações US)
- MetaTrader 5 (B3 / Forex)
- Interactive Brokers
- Qualquer outra via adaptador

Segurança:
- Limite máximo de perda diária (auto desliga o bot)
- Limite máximo de perda por operação
- Limite de exposição máxima
- Cooldown entre operações
- Kill switch manual e automático
- Logging completo de todas as operações

IMPORTANTE: Para operar de verdade, você precisa:
1. Criar conta na corretora
2. Gerar API Key e Secret
3. Configurar no arquivo .env (NUNCA no código)
4. Testar no modo Paper da corretora primeiro
5. Só depois ativar modo real

Uso:
    # 1. Escolha o adaptador da corretora
    broker = BinanceAdapter(api_key="...", api_secret="...")
    # broker = AlpacaAdapter(api_key="...", api_secret="...")

    # 2. Configure os limites de segurança
    safety = SafetyLimits(
        max_daily_loss_pct=3.0,      # para se perder 3% no dia
        max_trade_loss_pct=1.5,       # máximo 1.5% por trade
        max_exposure_pct=80.0,        # máximo 80% do capital investido
        cooldown_seconds=60,          # 60s entre operações
    )

    # 3. Crie o engine
    engine = LiveTradingEngine(broker, safety)

    # 4. Liga o bot (mesmo botão do dashboard)
    engine.start()

    # 5. Para o bot
    engine.stop()
"""

import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum

from utils.logger import get_logger

logger = get_logger("quantbot.live")


# ═══════════════════════════════════════════════════════════════
# SAFETY LIMITS — TRAVA DE SEGURANÇA
# ═══════════════════════════════════════════════════════════════

class BotStatus(Enum):
    OFF = "OFF"
    RUNNING = "RUNNING"
    PAUSED_BY_SAFETY = "PAUSED_BY_SAFETY"
    ERROR = "ERROR"


@dataclass
class SafetyLimits:
    """
    Limites de segurança para operação com dinheiro real.

    Quando qualquer limite é atingido, o bot PARA automaticamente
    e registra o motivo. Você precisa religar manualmente.
    """
    # Perda máxima no dia (% do capital). Bot desliga se atingir.
    max_daily_loss_pct: float = 3.0

    # Perda máxima por operação individual (%)
    max_trade_loss_pct: float = 1.5

    # Máximo do capital que pode estar investido ao mesmo tempo (%)
    max_exposure_pct: float = 80.0

    # Tempo mínimo entre operações (segundos)
    cooldown_seconds: int = 60

    # Máximo de operações por dia
    max_daily_trades: int = 50

    # Valor mínimo que deve permanecer em caixa ($)
    min_cash_reserve: float = 1000.0

    # Se True, só opera em horário de mercado
    respect_market_hours: bool = True

    # Horário de operação (UTC) — B3: 13:00-21:00 / US: 13:30-20:00
    market_open_hour: int = 13
    market_close_hour: int = 21


@dataclass
class DailyStats:
    """Estatísticas do dia para controle de limites."""
    date: str = ""
    starting_capital: float = 0.0
    current_capital: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    trades_today: int = 0
    last_trade_time: float = 0.0
    safety_triggered: bool = False
    safety_reason: str = ""

    def reset(self, capital: float):
        today = date.today().isoformat()
        if self.date != today:
            self.date = today
            self.starting_capital = capital
            self.current_capital = capital
            self.daily_pnl = 0.0
            self.daily_pnl_pct = 0.0
            self.trades_today = 0
            self.safety_triggered = False
            self.safety_reason = ""


class SafetyMonitor:
    """
    Monitor de segurança que verifica limites antes de cada operação.

    Se qualquer limite é violado, bloqueia a operação e pode
    desligar o bot automaticamente.
    """

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self.stats = DailyStats()
        self.violations: List[dict] = []

    def check_before_trade(
        self,
        trade_value: float,
        current_capital: float,
        total_invested: float,
    ) -> Tuple[bool, str]:
        """
        Verifica se é seguro executar a operação.

        Returns:
            (is_safe, reason) — True se pode operar, False + motivo se não
        """
        self.stats.reset(current_capital)
        now = time.time()

        # 1. Verificar perda diária
        self.stats.current_capital = current_capital
        self.stats.daily_pnl = current_capital - self.stats.starting_capital
        self.stats.daily_pnl_pct = (
            (self.stats.daily_pnl / self.stats.starting_capital) * 100
            if self.stats.starting_capital > 0 else 0
        )

        if self.stats.daily_pnl_pct <= -self.limits.max_daily_loss_pct:
            reason = (
                f"LIMITE DIÁRIO ATINGIDO: perda de {self.stats.daily_pnl_pct:.2f}% "
                f"excede limite de -{self.limits.max_daily_loss_pct}%"
            )
            self._trigger_safety(reason)
            return False, reason

        # 2. Verificar exposição máxima
        exposure_pct = (total_invested / current_capital * 100) if current_capital > 0 else 100
        new_exposure = ((total_invested + trade_value) / current_capital * 100)

        if new_exposure > self.limits.max_exposure_pct:
            reason = (
                f"EXPOSIÇÃO MÁXIMA: {new_exposure:.1f}% excede "
                f"limite de {self.limits.max_exposure_pct}%"
            )
            return False, reason

        # 3. Verificar reserva mínima de caixa
        cash_after = current_capital - total_invested - trade_value
        if cash_after < self.limits.min_cash_reserve:
            reason = (
                f"RESERVA MÍNIMA: caixa ficaria em ${cash_after:,.2f}, "
                f"abaixo do mínimo de ${self.limits.min_cash_reserve:,.2f}"
            )
            return False, reason

        # 4. Verificar cooldown
        elapsed = now - self.stats.last_trade_time
        if elapsed < self.limits.cooldown_seconds and self.stats.trades_today > 0:
            reason = (
                f"COOLDOWN: {self.limits.cooldown_seconds - elapsed:.0f}s "
                f"restantes entre operações"
            )
            return False, reason

        # 5. Verificar limite diário de trades
        if self.stats.trades_today >= self.limits.max_daily_trades:
            reason = (
                f"LIMITE DE TRADES: {self.stats.trades_today} trades hoje, "
                f"máximo é {self.limits.max_daily_trades}"
            )
            return False, reason

        # 6. Verificar horário de mercado
        if self.limits.respect_market_hours:
            current_hour = datetime.utcnow().hour
            if (current_hour < self.limits.market_open_hour or
                    current_hour >= self.limits.market_close_hour):
                reason = (
                    f"FORA DO HORÁRIO: mercado opera das "
                    f"{self.limits.market_open_hour}h às "
                    f"{self.limits.market_close_hour}h UTC"
                )
                return False, reason

        return True, "OK"

    def register_trade(self, pnl: float = 0.0):
        """Registra que uma operação foi executada."""
        self.stats.trades_today += 1
        self.stats.last_trade_time = time.time()
        self.stats.daily_pnl += pnl

    def check_stop_loss(self, position_pnl_pct: float) -> bool:
        """Verifica se a posição deve ser encerrada por stop-loss."""
        return position_pnl_pct <= -self.limits.max_trade_loss_pct

    def _trigger_safety(self, reason: str):
        """Aciona a trava de segurança."""
        self.stats.safety_triggered = True
        self.stats.safety_reason = reason
        self.violations.append({
            "time": datetime.now().isoformat(),
            "reason": reason,
            "daily_pnl": self.stats.daily_pnl_pct,
            "trades_today": self.stats.trades_today,
        })
        logger.critical(f"🛑 SEGURANÇA: {reason}")

    def get_status(self) -> dict:
        """Retorna status atual dos limites."""
        return {
            "daily_pnl_pct": round(self.stats.daily_pnl_pct, 2),
            "daily_limit": self.limits.max_daily_loss_pct,
            "trades_today": self.stats.trades_today,
            "max_trades": self.limits.max_daily_trades,
            "safety_triggered": self.stats.safety_triggered,
            "safety_reason": self.stats.safety_reason,
            "violations_total": len(self.violations),
        }


# ═══════════════════════════════════════════════════════════════
# BROKER ADAPTER — INTERFACE PARA CORRETORAS
# ═══════════════════════════════════════════════════════════════

class BrokerAdapter(ABC):
    """
    Interface abstrata para conectar qualquer corretora.

    Para adicionar uma nova corretora, crie uma classe que
    herda de BrokerAdapter e implemente todos os métodos.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Conecta com a corretora. Retorna True se sucesso."""
        pass

    @abstractmethod
    def disconnect(self):
        """Desconecta da corretora."""
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """Retorna saldo disponível em $."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, dict]:
        """Retorna posições abertas. {symbol: {qty, avg_price, current_price}}"""
        pass

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Retorna preço atual de um ativo."""
        pass

    @abstractmethod
    def buy(self, symbol: str, quantity: float, price: float = None) -> dict:
        """Executa ordem de compra. Retorna detalhes da execução."""
        pass

    @abstractmethod
    def sell(self, symbol: str, quantity: float, price: float = None) -> dict:
        """Executa ordem de venda. Retorna detalhes da execução."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Nome da corretora."""
        pass


# ═══════════════════════════════════════════════════════════════
# ADAPTADORES PRONTOS (descomente e configure)
# ═══════════════════════════════════════════════════════════════

class BinanceAdapter(BrokerAdapter):
    """
    Adaptador para Binance (Crypto).

    Instalação: pip install python-binance
    Docs: https://python-binance.readthedocs.io/

    Configuração:
        1. Crie conta em binance.com
        2. Vá em API Management e crie uma API Key
        3. Passe api_key e api_secret ao criar o adaptador
        4. NUNCA coloque as keys no código — use variáveis de ambiente
    """

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet  # True = dinheiro fake da Binance
        self.client = None

    def connect(self) -> bool:
        try:
            from binance.client import Client
            if self.testnet:
                self.client = Client(self.api_key, self.api_secret, testnet=True)
                logger.info("✅ Binance TESTNET conectada (dinheiro simulado)")
            else:
                self.client = Client(self.api_key, self.api_secret)
                logger.info("✅ Binance REAL conectada ⚠️ DINHEIRO REAL")
            return True
        except ImportError:
            logger.error("❌ pip install python-binance")
            return False
        except Exception as e:
            logger.error(f"❌ Binance: {e}")
            return False

    def disconnect(self):
        self.client = None
        logger.info("Binance desconectada")

    def get_balance(self) -> float:
        if not self.client:
            return 0
        account = self.client.get_account()
        for bal in account["balances"]:
            if bal["asset"] == "USDT":
                return float(bal["free"])
        return 0

    def get_positions(self) -> Dict[str, dict]:
        if not self.client:
            return {}
        account = self.client.get_account()
        positions = {}
        for bal in account["balances"]:
            qty = float(bal["free"]) + float(bal["locked"])
            if qty > 0 and bal["asset"] != "USDT":
                symbol = bal["asset"] + "USDT"
                try:
                    price = float(self.client.get_symbol_ticker(symbol=symbol)["price"])
                    positions[symbol] = {"qty": qty, "current_price": price, "value": qty * price}
                except Exception:
                    pass
        return positions

    def get_price(self, symbol: str) -> float:
        if not self.client:
            return 0
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])

    def buy(self, symbol: str, quantity: float, price: float = None) -> dict:
        if not self.client:
            return {"error": "Não conectado"}
        try:
            order = self.client.create_order(
                symbol=symbol, side="BUY", type="MARKET",
                quantity=round(quantity, 6),
            )
            logger.info(f"📈 COMPRA REAL: {quantity} {symbol}")
            return {"status": "filled", "order": order}
        except Exception as e:
            logger.error(f"❌ Erro na compra: {e}")
            return {"error": str(e)}

    def sell(self, symbol: str, quantity: float, price: float = None) -> dict:
        if not self.client:
            return {"error": "Não conectado"}
        try:
            order = self.client.create_order(
                symbol=symbol, side="SELL", type="MARKET",
                quantity=round(quantity, 6),
            )
            logger.info(f"📉 VENDA REAL: {quantity} {symbol}")
            return {"status": "filled", "order": order}
        except Exception as e:
            logger.error(f"❌ Erro na venda: {e}")
            return {"error": str(e)}

    def get_name(self) -> str:
        return "Binance" + (" TESTNET" if self.testnet else " REAL")


class AlpacaAdapter(BrokerAdapter):
    """
    Adaptador para Alpaca (Ações US — sem comissão).

    Instalação: pip install alpaca-trade-api
    Docs: https://alpaca.markets/docs/

    Configuração:
        1. Crie conta em alpaca.markets
        2. Pegue API Key e Secret no dashboard
        3. Use paper=True para testar com dinheiro simulado
    """

    def __init__(self, api_key: str = "", api_secret: str = "", paper: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.api = None

    def connect(self) -> bool:
        try:
            import alpaca_trade_api as tradeapi
            base_url = (
                "https://paper-api.alpaca.markets" if self.paper
                else "https://api.alpaca.markets"
            )
            self.api = tradeapi.REST(self.api_key, self.api_secret, base_url)
            account = self.api.get_account()
            logger.info(
                f"✅ Alpaca {'PAPER' if self.paper else 'REAL'} conectada | "
                f"Capital: ${float(account.equity):,.2f}"
            )
            return True
        except ImportError:
            logger.error("❌ pip install alpaca-trade-api")
            return False
        except Exception as e:
            logger.error(f"❌ Alpaca: {e}")
            return False

    def disconnect(self):
        self.api = None
        logger.info("Alpaca desconectada")

    def get_balance(self) -> float:
        if not self.api:
            return 0
        account = self.api.get_account()
        return float(account.cash)

    def get_positions(self) -> Dict[str, dict]:
        if not self.api:
            return {}
        positions = {}
        for pos in self.api.list_positions():
            positions[pos.symbol] = {
                "qty": float(pos.qty),
                "avg_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "value": float(pos.market_value),
                "pnl": float(pos.unrealized_pl),
                "pnl_pct": float(pos.unrealized_plpc) * 100,
            }
        return positions

    def get_price(self, symbol: str) -> float:
        if not self.api:
            return 0
        trade = self.api.get_latest_trade(symbol)
        return float(trade.price)

    def buy(self, symbol: str, quantity: float, price: float = None) -> dict:
        if not self.api:
            return {"error": "Não conectado"}
        try:
            order = self.api.submit_order(
                symbol=symbol, qty=int(quantity),
                side="buy", type="market", time_in_force="day",
            )
            logger.info(f"📈 COMPRA REAL: {quantity} {symbol}")
            return {"status": order.status, "id": order.id}
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            return {"error": str(e)}

    def sell(self, symbol: str, quantity: float, price: float = None) -> dict:
        if not self.api:
            return {"error": "Não conectado"}
        try:
            order = self.api.submit_order(
                symbol=symbol, qty=int(quantity),
                side="sell", type="market", time_in_force="day",
            )
            logger.info(f"📉 VENDA REAL: {quantity} {symbol}")
            return {"status": order.status, "id": order.id}
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            return {"error": str(e)}

    def get_name(self) -> str:
        return "Alpaca" + (" PAPER" if self.paper else " REAL")


class PaperBrokerAdapter(BrokerAdapter):
    """
    Adaptador de Paper Trading (simulação local).
    Não precisa de API key, não conecta a nada.
    Usado como padrão quando nenhuma corretora está configurada.
    """

    def __init__(self, initial_capital: float = 100000):
        self.cash = initial_capital
        self.positions: Dict[str, dict] = {}
        self.prices: Dict[str, float] = {}

    def connect(self) -> bool:
        logger.info("✅ Paper Broker conectado (simulação local)")
        return True

    def disconnect(self):
        logger.info("Paper Broker desconectado")

    def get_balance(self) -> float:
        return self.cash

    def get_positions(self) -> Dict[str, dict]:
        return self.positions

    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 0)

    def set_price(self, symbol: str, price: float):
        """Define preço de um ativo (usado na simulação)."""
        self.prices[symbol] = price

    def buy(self, symbol: str, quantity: float, price: float = None) -> dict:
        if price is None:
            price = self.get_price(symbol)
        cost = quantity * price
        if cost > self.cash:
            return {"error": "Saldo insuficiente"}
        self.cash -= cost * 1.001  # com comissão
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos["qty"] + quantity
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + price * quantity) / total_qty
            pos["qty"] = total_qty
        else:
            self.positions[symbol] = {"qty": quantity, "avg_price": price, "current_price": price}
        return {"status": "filled", "price": price, "qty": quantity}

    def sell(self, symbol: str, quantity: float, price: float = None) -> dict:
        if symbol not in self.positions:
            return {"error": "Sem posição"}
        pos = self.positions[symbol]
        if pos["qty"] < quantity:
            return {"error": "Quantidade insuficiente"}
        if price is None:
            price = self.get_price(symbol)
        self.cash += quantity * price * 0.999  # com comissão
        pos["qty"] -= quantity
        if pos["qty"] <= 0:
            del self.positions[symbol]
        return {"status": "filled", "price": price, "qty": quantity, "pnl": (price - pos["avg_price"]) * quantity}

    def get_name(self) -> str:
        return "Paper Trading (Local)"


# ═══════════════════════════════════════════════════════════════
# LIVE TRADING ENGINE — MOTOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════

class LiveTradingEngine:
    """
    Motor de operação real com mesmo padrão liga/desliga do dashboard.

    Uso:
        # Paper (padrão)
        engine = LiveTradingEngine()

        # Binance Testnet
        engine = LiveTradingEngine(
            broker=BinanceAdapter(api_key="...", api_secret="...", testnet=True)
        )

        # Alpaca Paper
        engine = LiveTradingEngine(
            broker=AlpacaAdapter(api_key="...", api_secret="...", paper=True)
        )

        # Ligar
        engine.start()

        # Status
        engine.get_status()

        # Desligar (mostra posições abertas antes)
        engine.request_stop()
        engine.confirm_stop()
    """

    def __init__(
        self,
        broker: BrokerAdapter = None,
        safety: SafetyLimits = None,
    ):
        self.broker = broker or PaperBrokerAdapter()
        self.safety = SafetyMonitor(safety or SafetyLimits())
        self.status = BotStatus.OFF
        self.operation_log: List[dict] = []
        self._running = False

    def start(self) -> bool:
        """
        Liga o bot. Mesmo botão do dashboard.

        Returns:
            True se iniciou com sucesso
        """
        if self.status == BotStatus.RUNNING:
            logger.warning("Bot já está rodando")
            return False

        # Conecta com a corretora
        if not self.broker.connect():
            self.status = BotStatus.ERROR
            return False

        self.status = BotStatus.RUNNING
        self._running = True
        self._log("🤖 Bot LIGADO", "start")
        self._log(f"Corretora: {self.broker.get_name()}", "info")
        self._log(f"Saldo: ${self.broker.get_balance():,.2f}", "info")
        self._log(
            f"Limites: perda diária {self.safety.limits.max_daily_loss_pct}% | "
            f"exposição {self.safety.limits.max_exposure_pct}% | "
            f"max {self.safety.limits.max_daily_trades} trades/dia",
            "info",
        )

        return True

    def request_stop(self) -> dict:
        """
        Solicita desligamento do bot.
        Retorna posições abertas para confirmação.
        """
        positions = self.broker.get_positions()
        if positions:
            self._log(
                f"⚠️ Solicitação de parada com {len(positions)} posições abertas",
                "warn",
            )
        return {
            "positions": positions,
            "count": len(positions),
            "message": (
                f"Você tem {len(positions)} posições abertas. "
                "As posições serão mantidas. O bot apenas parará de operar."
                if positions
                else "Nenhuma posição aberta. Bot será desligado."
            ),
        }

    def confirm_stop(self):
        """Confirma desligamento do bot."""
        self._running = False
        self.status = BotStatus.OFF
        self.broker.disconnect()
        self._log("⛔ Bot DESLIGADO", "stop")

    def execute_signal(
        self,
        symbol: str,
        signal: str,
        score: float,
        confidence: float,
        sentiment: str = "neutro",
    ) -> Optional[dict]:
        """
        Executa uma operação baseada em sinal ML.
        Verifica limites de segurança antes de operar.

        Args:
            symbol: Símbolo do ativo
            signal: "COMPRA_FORTE", "COMPRA", "VENDA", etc.
            score: Score do modelo (0-100)
            confidence: Confiança (0-100)
            sentiment: Sentimento de notícias

        Returns:
            Detalhes da operação ou None se bloqueada
        """
        if self.status != BotStatus.RUNNING:
            return None

        balance = self.broker.get_balance()
        positions = self.broker.get_positions()
        total_invested = sum(p.get("value", 0) for p in positions.values())

        # Calcula tamanho da posição
        alloc_pct = 0.08 if score > 70 else 0.05 if score > 60 else 0.03
        trade_value = balance * alloc_pct

        # Verifica segurança
        is_safe, reason = self.safety.check_before_trade(
            trade_value, balance + total_invested, total_invested
        )

        if not is_safe:
            self._log(f"🛑 Bloqueado: {symbol} — {reason}", "blocked")
            if self.safety.stats.safety_triggered:
                self.status = BotStatus.PAUSED_BY_SAFETY
                self._log("⛔ Bot PAUSADO pela segurança", "safety")
            return None

        # Executa
        price = self.broker.get_price(symbol)
        if price <= 0:
            self._log(f"⚠️ Preço inválido para {symbol}", "warn")
            return None

        result = None

        if signal in ("COMPRA_FORTE", "COMPRA"):
            quantity = trade_value / price
            result = self.broker.buy(symbol, quantity, price)
            if "error" not in result:
                self.safety.register_trade()
                self._log(
                    f"📈 COMPRA: {quantity:.4f}x {symbol} @ ${price:,.2f} | "
                    f"Score: {score} | Sentimento: {sentiment}",
                    "buy",
                )

        elif signal in ("VENDA_FORTE", "VENDA"):
            if symbol in positions:
                quantity = positions[symbol]["qty"]
                result = self.broker.sell(symbol, quantity, price)
                if "error" not in result:
                    pnl = result.get("pnl", 0)
                    self.safety.register_trade(pnl)
                    self._log(
                        f"📉 VENDA: {quantity:.4f}x {symbol} @ ${price:,.2f} | "
                        f"P&L: ${pnl:,.2f} | Score: {score}",
                        "sell",
                    )

        return result

    def check_stop_losses(self):
        """Verifica stop-loss de todas as posições abertas."""
        positions = dict(self.broker.get_positions())
        for symbol, pos in positions.items():
            if "avg_price" in pos and "current_price" in pos:
                pnl_pct = ((pos["current_price"] - pos["avg_price"]) / pos["avg_price"]) * 100
                if self.safety.check_stop_loss(pnl_pct):
                    self._log(
                        f"🛑 STOP-LOSS: {symbol} P&L={pnl_pct:.2f}% | Vendendo...",
                        "stoploss",
                    )
                    self.broker.sell(symbol, pos["qty"])

    def get_status(self) -> dict:
        """Retorna status completo do bot para o dashboard."""
        positions = self.broker.get_positions()
        balance = self.broker.get_balance()
        total_invested = sum(p.get("value", 0) for p in positions.values())

        return {
            "bot_status": self.status.value,
            "broker": self.broker.get_name(),
            "balance": balance,
            "total_invested": total_invested,
            "total_value": balance + total_invested,
            "positions_count": len(positions),
            "positions": positions,
            "safety": self.safety.get_status(),
            "log": self.operation_log[-20:],
        }

    def _log(self, message: str, log_type: str = "info"):
        """Registra operação no log."""
        entry = {
            "time": datetime.now().isoformat(),
            "message": message,
            "type": log_type,
        }
        self.operation_log.append(entry)
        logger.info(f"[{log_type.upper()}] {message}")

    def export_log(self, filepath: str = "live_trading_log.json"):
        """Exporta log completo para JSON."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "broker": self.broker.get_name(),
            "status": self.status.value,
            "safety": self.safety.get_status(),
            "log": self.operation_log,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Log exportado: {filepath}")
