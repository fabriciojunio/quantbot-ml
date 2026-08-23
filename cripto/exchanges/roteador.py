"""
Roteador de exchanges.

Uma fonte de dados que cai leva o sistema inteiro junto. O roteador tenta as
exchanges em ordem e registra qual respondeu, para que o relatório sempre diga
de onde veio o número. Se todas falharem ele levanta erro em vez de devolver
dado inventado — é a diferença entre um sistema honesto e um painel bonito.
"""

from __future__ import annotations

import logging

from .base import Book, Candle, ErroExchange, Negocio
from .binance import Binance
from .bybit import Bybit
from .gate import GateIO

logger = logging.getLogger("quantbot.cripto.roteador")


class Roteador:
    """Tenta cada cliente na ordem informada até um responder."""

    def __init__(self, clientes: list | None = None):
        self.clientes = clientes or [Binance(), GateIO(), Bybit()]
        self.ultima_fonte: str | None = None

    def _tentar(self, metodo: str, *args, **kwargs):
        erros: list[str] = []
        for cliente in self.clientes:
            try:
                resultado = getattr(cliente, metodo)(*args, **kwargs)
                if resultado:
                    self.ultima_fonte = cliente.nome
                    return resultado
                erros.append(f"{cliente.nome}: resposta vazia")
            except Exception as exc:
                erros.append(f"{cliente.nome}: {type(exc).__name__} {exc}")
                logger.warning("Falha em %s.%s — %s", cliente.nome, metodo, exc)
        raise ErroExchange(f"todas as fontes falharam em {metodo}: {'; '.join(erros)}")

    def candles(self, simbolo: str, intervalo: str = "1h", limite: int = 500, **kwargs) -> list[Candle]:
        return self._tentar("candles", simbolo, intervalo, limite, **kwargs)

    def book(self, simbolo: str, profundidade: int = 100) -> Book:
        return self._tentar("book", simbolo, profundidade)

    def negocios(self, simbolo: str, limite: int = 500) -> list[Negocio]:
        return self._tentar("negocios", simbolo, limite)

    @property
    def taxa_taker(self) -> float:
        """Taxa da fonte que respondeu por último, ou a da primeira da fila."""
        for cliente in self.clientes:
            if cliente.nome == self.ultima_fonte:
                return cliente.taxa_taker
        return self.clientes[0].taxa_taker


def cliente_padrao() -> Roteador:
    return Roteador()
