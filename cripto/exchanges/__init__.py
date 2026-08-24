"""Clientes REST públicos de exchange, com fallback entre fontes."""

from .base import Candle, ClienteMercado, ErroExchange
from .binance import Binance
from .bybit import Bybit
from .gate import GateIO
from .roteador import Roteador, cliente_padrao

__all__ = [
    "Candle",
    "ClienteMercado",
    "ErroExchange",
    "Binance",
    "Bybit",
    "GateIO",
    "Roteador",
    "cliente_padrao",
]
