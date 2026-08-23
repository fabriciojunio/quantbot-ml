"""
Contrato comum dos clientes de mercado.

Nenhum cliente aqui assina requisição nem guarda chave de API. Todos usam
apenas endpoints públicos de leitura: candles, book e negócios recentes.
Ordem não passa por este pacote — quem executa é `cripto.papel` (simulado) ou
`core.live_trading` (real, atrás da trava `QUANTBOT_ALLOW_LIVE`).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger("quantbot.cripto.exchanges")

# Intervalos aceitos em todo o pacote. Cada cliente traduz para o formato dele.
INTERVALOS = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")

_USER_AGENT = "quantbot-cripto/0.1 (+https://github.com/fabriciojunio/quantbot-ml)"


class ErroExchange(RuntimeError):
    """Falha de rede, formato ou limite de taxa em uma exchange."""


@dataclass(frozen=True)
class Candle:
    """Um candle OHLCV normalizado. `abertura_ms` é epoch em milissegundos, UTC."""

    abertura_ms: int
    abertura: float
    maxima: float
    minima: float
    fechamento: float
    volume: float

    def __post_init__(self) -> None:
        if self.maxima < self.minima:
            raise ValueError(f"candle inválido: máxima {self.maxima} < mínima {self.minima}")
        if self.volume < 0:
            raise ValueError(f"candle inválido: volume negativo {self.volume}")


@dataclass(frozen=True)
class NivelBook:
    preco: float
    quantidade: float


@dataclass(frozen=True)
class Book:
    """Snapshot do livro de ofertas. `bids` desc por preço, `asks` asc."""

    simbolo: str
    bids: tuple[NivelBook, ...]
    asks: tuple[NivelBook, ...]
    timestamp_ms: int

    @property
    def melhor_bid(self) -> float:
        return self.bids[0].preco if self.bids else 0.0

    @property
    def melhor_ask(self) -> float:
        return self.asks[0].preco if self.asks else 0.0

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.melhor_ask - self.melhor_bid

    @property
    def spread_pct(self) -> float:
        meio = (self.melhor_ask + self.melhor_bid) / 2
        return (self.spread / meio * 100) if meio > 0 else 0.0

    def desequilibrio(self, niveis: int = 20) -> float:
        """Fração do volume do topo do book que está do lado comprador (0 a 1).

        Acima de 0.5 há mais tamanho apoiando o preço; abaixo, pressionando.
        É uma leitura de microestrutura, não previsão.
        """
        vb = sum(n.quantidade * n.preco for n in self.bids[:niveis])
        va = sum(n.quantidade * n.preco for n in self.asks[:niveis])
        total = vb + va
        return (vb / total) if total > 0 else 0.5


@dataclass(frozen=True)
class Negocio:
    """Uma execução real do tape."""

    timestamp_ms: int
    preco: float
    quantidade: float
    comprador_agressor: bool


class ClienteMercado(Protocol):
    """O que todo cliente de exchange precisa entregar."""

    nome: str

    def candles(self, simbolo: str, intervalo: str, limite: int) -> list[Candle]: ...

    def book(self, simbolo: str, profundidade: int) -> Book: ...

    def negocios(self, simbolo: str, limite: int) -> list[Negocio]: ...


def http_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = 10.0,
    tentativas: int = 3,
) -> Any:
    """GET JSON com timeout explícito e backoff exponencial.

    Só aceita http/https — bloqueia `file://` e outros esquemas caso uma URL
    venha de configuração. Toda chamada tem timeout: nenhuma requisição pode
    pendurar o processo de coleta.
    """
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    esquema = urllib.parse.urlparse(url).scheme
    if esquema not in ("http", "https"):
        raise ErroExchange(f"esquema de URL não permitido: {esquema!r}")

    ultimo_erro: Exception | None = None
    for tentativa in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            # O esquema é validado logo acima: só http/https chegam aqui.
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
                if resp.status != 200:
                    raise ErroExchange(f"HTTP {resp.status} em {url}")
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            ultimo_erro = exc
            # 429/418 são limite de taxa: espera mais antes de insistir.
            espera = 5.0 if exc.code in (418, 429) else 1.0
            if tentativa < tentativas - 1:
                time.sleep(espera * (2**tentativa))
        except Exception as exc:  # rede, DNS, timeout, JSON inválido
            ultimo_erro = exc
            if tentativa < tentativas - 1:
                time.sleep(1.0 * (2**tentativa))

    raise ErroExchange(f"falha ao consultar {url}: {ultimo_erro}")


def valida_intervalo(intervalo: str) -> str:
    if intervalo not in INTERVALOS:
        raise ValueError(f"intervalo {intervalo!r} não suportado. Use um de {INTERVALOS}")
    return intervalo


def valida_simbolo(simbolo: str) -> str:
    """Aceita apenas letras e dígitos (ex.: BTCUSDT). Corta injeção via símbolo."""
    s = simbolo.strip().upper()
    if not s.isalnum() or not (4 <= len(s) <= 20):
        raise ValueError(f"símbolo inválido: {simbolo!r}")
    return s
