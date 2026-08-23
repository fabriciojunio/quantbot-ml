"""Cliente REST público da Bybit spot (segundo fallback)."""

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

_BASE = "https://api.bybit.com/v5/market"

# A Bybit conta intervalo em minutos, com D para diário.
_INTERVALO = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
    "1d": "D",
}

LIMITE_CANDLES = 1000


def _resultado(bruto: dict) -> dict:
    if bruto.get("retCode") != 0:
        raise ErroExchange(f"bybit retCode={bruto.get('retCode')}: {bruto.get('retMsg')}")
    return bruto.get("result", {})


class Bybit:
    nome = "bybit"
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
            "category": "spot",
            "symbol": valida_simbolo(simbolo),
            "interval": _INTERVALO[valida_intervalo(intervalo)],
            "limit": min(max(1, limite), LIMITE_CANDLES),
        }
        if inicio_ms is not None:
            params["start"] = int(inicio_ms)
        if fim_ms is not None:
            params["end"] = int(fim_ms)

        lista = _resultado(http_json(f"{_BASE}/kline", params)).get("list", [])
        # A Bybit devolve do mais novo para o mais antigo.
        candles = [
            Candle(
                abertura_ms=int(k[0]),
                abertura=float(k[1]),
                maxima=float(k[2]),
                minima=float(k[3]),
                fechamento=float(k[4]),
                volume=float(k[5]),
            )
            for k in lista
        ]
        candles.sort(key=lambda c: c.abertura_ms)
        return candles

    def book(self, simbolo: str, profundidade: int = 100) -> Book:
        res = _resultado(
            http_json(
                f"{_BASE}/orderbook",
                {
                    "category": "spot",
                    "symbol": valida_simbolo(simbolo),
                    "limit": min(profundidade, 200),
                },
            )
        )
        return Book(
            simbolo=simbolo.upper(),
            bids=tuple(NivelBook(float(p), float(q)) for p, q in res.get("b", [])),
            asks=tuple(NivelBook(float(p), float(q)) for p, q in res.get("a", [])),
            timestamp_ms=int(res.get("ts", 0)),
        )

    def negocios(self, simbolo: str, limite: int = 500) -> list[Negocio]:
        res = _resultado(
            http_json(
                f"{_BASE}/recent-trade",
                {
                    "category": "spot",
                    "symbol": valida_simbolo(simbolo),
                    "limit": min(limite, 1000),
                },
            )
        )
        return [
            Negocio(
                timestamp_ms=int(t["time"]),
                preco=float(t["price"]),
                quantidade=float(t["size"]),
                comprador_agressor=t.get("side") == "Buy",
            )
            for t in res.get("list", [])
        ]
