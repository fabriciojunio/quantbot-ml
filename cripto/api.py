"""
API de leitura do motor de cripto.

Existe para alimentar o painel do Sentinela com número real. O painel antigo
tinha componentes bonitos ligados a `Math.random()` — book, times & trades,
on-chain e "8 indicadores" eram todos sintéticos. Os endpoints aqui devolvem
os mesmos conceitos, mas cada campo tem procedência: `fonte` diz qual exchange
respondeu e `calculado_de` diz de que série o número saiu.

Regras de segurança herdadas de `api/server.py`:

* só GET — nenhuma rota abre, fecha ou altera posição;
* rate limit por IP, com 429 ao estourar;
* cabeçalhos de segurança em toda resposta;
* CORS restrito à origem do front;
* símbolo validado por regex antes de virar chamada externa.

Subir:  uvicorn cripto.api:app --port 8100
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import mercado, papel, sinais
from .custos import ModeloCusto, resumo_viabilidade
from .exchanges import cliente_padrao
from .sinais import ConfigSinal

_SIMBOLO_RE = re.compile(r"^[A-Z0-9]{4,20}$")

ORIGENS_PERMITIDAS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

_HEADERS_SEGURANCA = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


class _Cache:
    """Cache com TTL. Evita bater na exchange a cada refresh do painel."""

    def __init__(self) -> None:
        self._itens: dict[str, tuple[float, object]] = {}

    def pegar(self, chave: str, ttl: float):
        item = self._itens.get(chave)
        if item and (time.time() - item[0]) <= ttl:
            return item[1]
        return None

    def guardar(self, chave: str, valor) -> None:
        self._itens[chave] = (time.time(), valor)


class _LimitadorIP:
    def __init__(self, maximo: int = 120, janela: float = 60.0):
        self.maximo = maximo
        self.janela = janela
        self._acessos: dict[str, list[float]] = defaultdict(list)

    def permitido(self, ip: str) -> bool:
        agora = time.time()
        self._acessos[ip] = [t for t in self._acessos[ip] if agora - t < self.janela]
        if len(self._acessos[ip]) >= self.maximo:
            return False
        self._acessos[ip].append(agora)
        return True


cache = _Cache()
limitador = _LimitadorIP()
cliente = cliente_padrao()

app = FastAPI(
    title="QuantBot Cripto",
    version="0.1",
    description="Dados reais de cripto com custo de execução explícito. Só leitura.",
    docs_url="/api/cripto/docs",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENS_PERMITIDAS,
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _limite_e_headers(request: Request, call_next):
    ip = request.client.host if request.client else "desconhecido"
    if not limitador.permitido(ip):
        resp = JSONResponse(
            status_code=429,
            content={"detalhe": "Muitas requisições. Tente de novo em instantes."},
        )
    else:
        resp = await call_next(request)
    for chave, valor in _HEADERS_SEGURANCA.items():
        resp.headers[chave] = valor
    return resp


def _valida(simbolo: str) -> str:
    s = simbolo.upper()
    if not _SIMBOLO_RE.match(s):
        raise HTTPException(status_code=400, detail="Símbolo inválido")
    return s


# ── Rotas ────────────────────────────────────────────────────────────────────


@app.get("/api/cripto/saude")
def saude() -> dict:
    return {
        "status": "ok",
        "quando": datetime.now(timezone.utc).isoformat(),
        "fontes": [c.nome for c in cliente.clientes],
        "ultima_fonte_usada": cliente.ultima_fonte,
        "modo": "somente leitura — nenhuma rota executa ordem",
    }


@app.get("/api/cripto/precos")
def precos(simbolos: str = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT") -> dict:
    """Preço de topo de book, spread real e desequilíbrio de liquidez."""
    chave = f"precos:{simbolos}"
    guardado = cache.pegar(chave, ttl=5.0)
    if guardado is not None:
        return guardado

    itens = []
    for bruto in simbolos.split(","):
        try:
            simbolo = _valida(bruto.strip())
            book = cliente.book(simbolo, profundidade=20)
            itens.append(
                {
                    "simbolo": simbolo,
                    "melhor_bid": book.melhor_bid,
                    "melhor_ask": book.melhor_ask,
                    "meio": round((book.melhor_bid + book.melhor_ask) / 2, 8),
                    "spread_pct": round(book.spread_pct, 5),
                    "desequilibrio_book": round(book.desequilibrio(), 4),
                    "fonte": cliente.ultima_fonte,
                    "sintetico": False,
                }
            )
        except HTTPException:
            raise
        except Exception as exc:
            itens.append({"simbolo": bruto.strip().upper(), "erro": str(exc)})

    resposta = {"quando": datetime.now(timezone.utc).isoformat(), "precos": itens}
    cache.guardar(chave, resposta)
    return resposta


@app.get("/api/cripto/sinal/{simbolo}")
def sinal(simbolo: str, tempo: str = "1h") -> dict:
    """Score de confluência da barra mais recente, aberto indicador por indicador."""
    simbolo = _valida(simbolo)
    if tempo not in mercado.DURACAO_MS:
        raise HTTPException(status_code=400, detail="Tempo gráfico inválido")

    chave = f"sinal:{simbolo}:{tempo}"
    guardado = cache.pegar(chave, ttl=60.0)
    if guardado is not None:
        return guardado

    try:
        df = mercado.historico(simbolo, tempo, dias=60)
        dados = sinais.gerar_sinais(df, ConfigSinal())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dados indisponíveis: {exc}") from exc

    ultima = dados.index[-1]
    detalhe = sinais.explica_barra(dados, ultima)
    resposta = {
        **detalhe,
        "simbolo": simbolo,
        "tempo": tempo,
        "fechamento": float(dados["fechamento"].iloc[-1]),
        "posicao_alvo": int(dados["posicao_alvo"].iloc[-1]),
        "stop_sugerido": float(dados["stop_long"].iloc[-1]),
        "alvo_sugerido": float(dados["alvo_long"].iloc[-1]),
        "calculado_de": f"{len(dados)} candles OHLCV reais",
        "fonte": cliente.ultima_fonte,
        "aviso": (
            "Score é leitura de indicador sobre o passado recente, não previsão. "
            "Só vira ordem depois que a validação fora da amostra aprovar."
        ),
    }
    cache.guardar(chave, resposta)
    return resposta


@app.get("/api/cripto/book/{simbolo}")
def book(simbolo: str, profundidade: int = 50) -> dict:
    """Livro de ofertas real, com as maiores paredes de cada lado."""
    simbolo = _valida(simbolo)
    profundidade = max(5, min(profundidade, 200))

    try:
        b = cliente.book(simbolo, profundidade=profundidade)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Book indisponível: {exc}") from exc

    def parede(niveis):
        if not niveis:
            return None
        maior = max(niveis, key=lambda n: n.quantidade * n.preco)
        return {
            "preco": maior.preco,
            "quantidade": maior.quantidade,
            "notional_usd": round(maior.preco * maior.quantidade, 2),
        }

    return {
        "simbolo": simbolo,
        "fonte": cliente.ultima_fonte,
        "melhor_bid": b.melhor_bid,
        "melhor_ask": b.melhor_ask,
        "spread_pct": round(b.spread_pct, 5),
        "desequilibrio": round(b.desequilibrio(profundidade), 4),
        "maior_parede_compra": parede(b.bids),
        "maior_parede_venda": parede(b.asks),
        "niveis_bid": [{"preco": n.preco, "quantidade": n.quantidade} for n in b.bids[:20]],
        "niveis_ask": [{"preco": n.preco, "quantidade": n.quantidade} for n in b.asks[:20]],
        "sintetico": False,
    }


@app.get("/api/cripto/tape/{simbolo}")
def tape(simbolo: str, limite: int = 300) -> dict:
    """Negócios executados de verdade, com delta de volume e blocos grandes."""
    simbolo = _valida(simbolo)
    limite = max(50, min(limite, 1000))

    try:
        negocios = cliente.negocios(simbolo, limite=limite)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Tape indisponível: {exc}") from exc

    if not negocios:
        raise HTTPException(status_code=503, detail="Nenhum negócio retornado")

    volume_compra = sum(n.quantidade for n in negocios if n.comprador_agressor)
    volume_venda = sum(n.quantidade for n in negocios if not n.comprador_agressor)
    total = volume_compra + volume_venda

    notionais = sorted(n.quantidade * n.preco for n in negocios)
    corte = notionais[int(len(notionais) * 0.95)] if len(notionais) > 20 else float("inf")
    blocos = [n for n in negocios if n.quantidade * n.preco >= corte]

    return {
        "simbolo": simbolo,
        "fonte": cliente.ultima_fonte,
        "negocios_analisados": len(negocios),
        "volume_agressao_compra": round(volume_compra, 6),
        "volume_agressao_venda": round(volume_venda, 6),
        "delta": round(volume_compra - volume_venda, 6),
        "delta_pct": round((volume_compra - volume_venda) / total * 100, 2) if total else 0.0,
        "corte_bloco_usd": round(corte, 2) if corte != float("inf") else None,
        "blocos_grandes": [
            {
                "quando_ms": n.timestamp_ms,
                "preco": n.preco,
                "quantidade": n.quantidade,
                "notional_usd": round(n.preco * n.quantidade, 2),
                "lado": "compra" if n.comprador_agressor else "venda",
            }
            for n in blocos[-15:]
        ],
        "sintetico": False,
    }


@app.get("/api/cripto/custo")
def custo(notional_usd: float = 1000.0, alvo_usd: float = 0.10, exchange: str = "binance") -> dict:
    """Quanto custa abrir e fechar, e se um alvo em dólares se paga."""
    modelo = ModeloCusto.da_exchange(exchange)
    return {
        "exchange": exchange,
        "taxa_taker_pct": modelo.taxa_taker * 100,
        "taxa_maker_pct": modelo.taxa_maker * 100,
        "custo_ida_volta_pct": round(modelo.custo_ida_volta_pct(notional_usd) * 100, 5),
        "teste_do_alvo": resumo_viabilidade(alvo_usd, notional_usd, modelo),
    }


@app.get("/api/cripto/papel")
def carteira_papel() -> dict:
    """Estado do book de paper trading, incluindo o que já foi pago em custo."""
    book = papel.carregar()
    fechadas = book.fechadas
    ganhos = [f.pnl_liquido_usd for f in fechadas if f.pnl_liquido_usd > 0]
    valor = book.serie_valor[-1]["valor_usd"] if book.serie_valor else book.caixa

    return {
        "capital_inicial_usd": book.capital_inicial,
        "valor_atual_usd": round(valor, 2),
        "retorno_pct": round((valor / book.capital_inicial - 1) * 100, 2)
        if book.capital_inicial
        else 0.0,
        "caixa_usd": round(book.caixa, 2),
        "posicoes_abertas": [
            {
                "simbolo": p.simbolo,
                "lado": "COMPRA" if p.lado == 1 else "VENDA",
                "preco_entrada": p.preco_entrada,
                "notional_usd": round(p.notional_usd, 2),
                "stop": p.stop,
                "alvo": p.alvo,
                "score_entrada": p.score_entrada,
                "aberta_em": p.aberta_em,
            }
            for p in book.posicoes.values()
        ],
        "operacoes_fechadas": len(fechadas),
        "taxa_acerto_pct": round(len(ganhos) / len(fechadas) * 100, 1) if fechadas else 0.0,
        "custo_total_pago_usd": round(sum(f.custo_total_usd for f in fechadas), 2),
        "pnl_liquido_usd": round(sum(f.pnl_liquido_usd for f in fechadas), 2),
        "modo": "paper — nenhuma ordem real foi enviada",
    }


@app.get("/api/crypto-live-prices")
def compat_precos_do_painel() -> dict:
    """Rota compatível com o painel antigo do Sentinela.

    Mantém o formato que o front já consome (`success`, `prices`, `source`),
    mas o conteúdo agora sai do book real e traz `sintetico: false` em cada
    item, para que a origem do número fique visível na própria resposta.
    """
    dados = precos()
    itens = []
    for item in dados["precos"]:
        if "erro" in item:
            continue
        itens.append(
            {
                "symbol": item["simbolo"].replace("USDT", ""),
                "priceUsd": item["meio"],
                "change24h": None,  # variação vem do endpoint de sinal, com base em candle real
                "isRealMarketLive": True,
                "spreadPct": item["spread_pct"],
                "sintetico": False,
                "source": item["fonte"],
            }
        )

    return {
        "success": True,
        "prices": itens,
        "source": f"QuantBot Cripto ({cliente.ultima_fonte})",
        "timestamp": dados["quando"],
    }
