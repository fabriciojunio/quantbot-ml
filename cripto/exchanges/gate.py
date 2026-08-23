"""Cliente REST público da Gate.io spot.

É a fonte que o painel do Sentinela já usava para preço à vista, então fica
como primeiro fallback: se a Binance bloquear a região ou estourar limite de
taxa, a série continua vindo de um lugar que o sistema antigo já validava.
"""

from __future__ import annotations

from .base import (
    Book,
    Candle,
    ErroExchange,
    Negocio,
    NivelBook,
    http_json,
    valida_intervalo,
    valida_simbolo,
)

_BASE = "https://api.gateio.ws/api/v4"

_INTERVALO = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

LIMITE_CANDLES = 1000

# Quotes conhecidas, da mais longa para a mais curta, para fatiar o símbolo certo.
_QUOTES = ("USDT", "USDC", "USD", "BTC", "ETH")


def par_gate(simbolo: str) -> str:
    """Converte BTCUSDT no formato BTC_USDT que a Gate.io usa."""
    s = valida_simbolo(simbolo)
    for quote in _QUOTES:
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[: -len(quote)]}_{quote}"
    raise ValueError(f"não consegui separar base/quote de {simbolo!r}")


class GateIO:
    nome = "gateio"
    taxa_maker = 0.0020
    taxa_taker = 0.0020

    def candles(
        self,
        simbolo: str,
        intervalo: str = "1h",
        limite: int = 500,
        inicio_ms: int | None = None,
        fim_ms: int | None = None,
    ) -> list[Candle]:
        params: dict[str, object] = {
            "currency_pair": par_gate(simbolo),
            "interval": _INTERVALO[valida_intervalo(intervalo)],
            "limit": min(max(1, limite), LIMITE_CANDLES),
        }
        if inicio_ms is not None:
            params["from"] = int(inicio_ms // 1000)
        if fim_ms is not None:
            params["to"] = int(fim_ms // 1000)

        bruto = http_json(f"{_BASE}/spot/candlesticks", params)
        if not isinstance(bruto, list):
            raise ErroExchange(f"resposta inesperada de candlesticks: {type(bruto)}")

        # Formato da Gate: [ts_s, volume_quote, close, high, low, open, volume_base, fechado]
        candles = [
            Candle(
                abertura_ms=int(k[0]) * 1000,
                abertura=float(k[5]),
                maxima=float(k[3]),
                minima=float(k[4]),
                fechamento=float(k[2]),
                volume=float(k[6]),
            )
            for k in bruto
        ]
        candles.sort(key=lambda c: c.abertura_ms)
        return candles

    def book(self, simbolo: str, profundidade: int = 100) -> Book:
        bruto = http_json(
            f"{_BASE}/spot/order_book",
            {"currency_pair": par_gate(simbolo), "limit": min(profundidade, 100)},
        )
        return Book(
            simbolo=simbolo.upper(),
            bids=tuple(NivelBook(float(p), float(q)) for p, q in bruto.get("bids", [])),
            asks=tuple(NivelBook(float(p), float(q)) for p, q in bruto.get("asks", [])),
            timestamp_ms=int(bruto.get("current", 0)),
        )

    def negocios(self, simbolo: str, limite: int = 500) -> list[Negocio]:
        bruto = http_json(
            f"{_BASE}/spot/trades",
            {"currency_pair": par_gate(simbolo), "limit": min(limite, 1000)},
        )
        return [
            Negocio(
                timestamp_ms=int(t.get("create_time_ms", "0").split(".")[0]),
                preco=float(t["price"]),
                quantidade=float(t["amount"]),
                comprador_agressor=t.get("side") == "buy",
            )
            for t in bruto
        ]
