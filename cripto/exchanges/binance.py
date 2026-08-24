"""Cliente REST público da Binance spot.

Fonte primária: tem o histórico mais longo e o book mais fundo dos três.
Só endpoints `/api/v3` públicos, sem assinatura.
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

_BASE = "https://api.binance.com"

# Binance usa exatamente as mesmas strings que o pacote, então o mapa é identidade.
_INTERVALO = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Limites máximos por chamada, conforme documentação pública.
LIMITE_CANDLES = 1000
LIMITE_BOOK = 5000
LIMITE_NEGOCIOS = 1000


class Binance:
    nome = "binance"
    # Taxa spot padrão sem desconto de BNB nem VIP (agosto/2026).
    taxa_maker = 0.0010
    taxa_taker = 0.0010

    def candles(
        self,
        simbolo: str,
        intervalo: str = "1h",
        limite: int = 500,
        inicio_ms: int | None = None,
        fim_ms: int | None = None,
    ) -> list[Candle]:
        params: dict[str, object] = {
            "symbol": valida_simbolo(simbolo),
            "interval": _INTERVALO[valida_intervalo(intervalo)],
            "limit": min(max(1, limite), LIMITE_CANDLES),
        }
        if inicio_ms is not None:
            params["startTime"] = int(inicio_ms)
        if fim_ms is not None:
            params["endTime"] = int(fim_ms)

        bruto = http_json(f"{_BASE}/api/v3/klines", params)
        if not isinstance(bruto, list):
            raise ErroExchange(f"resposta inesperada de klines: {type(bruto)}")

        return [
            Candle(
                abertura_ms=int(k[0]),
                abertura=float(k[1]),
                maxima=float(k[2]),
                minima=float(k[3]),
                fechamento=float(k[4]),
                volume=float(k[5]),
            )
            for k in bruto
        ]

    def book(self, simbolo: str, profundidade: int = 100) -> Book:
        bruto = http_json(
            f"{_BASE}/api/v3/depth",
            {"symbol": valida_simbolo(simbolo), "limit": min(profundidade, LIMITE_BOOK)},
        )
        return Book(
            simbolo=simbolo.upper(),
            bids=tuple(NivelBook(float(p), float(q)) for p, q in bruto.get("bids", [])),
            asks=tuple(NivelBook(float(p), float(q)) for p, q in bruto.get("asks", [])),
            timestamp_ms=int(bruto.get("lastUpdateId", 0)),
        )

    def negocios(self, simbolo: str, limite: int = 500) -> list[Negocio]:
        bruto = http_json(
            f"{_BASE}/api/v3/trades",
            {"symbol": valida_simbolo(simbolo), "limit": min(limite, LIMITE_NEGOCIOS)},
        )
        # `isBuyerMaker=True` significa que quem agrediu foi o vendedor.
        return [
            Negocio(
                timestamp_ms=int(t["time"]),
                preco=float(t["price"]),
                quantidade=float(t["qty"]),
                comprador_agressor=not bool(t["isBuyerMaker"]),
            )
            for t in bruto
        ]
