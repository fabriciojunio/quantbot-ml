"""
Motor de cripto do QuantBot.

Este pacote existe para responder a uma pergunta específica, com dados reais:
*a estratégia de confluência que roda no painel do Sentinela tem vantagem
depois de descontar taxa, spread e slippage?*

Tudo aqui parte de OHLCV real baixado de exchange pública. Não há valor
sintético, não há `random` alimentando indicador, e nenhum número de
performance aparece sem que os custos de execução já tenham sido cobrados.

Módulos:
    exchanges/   clientes REST públicos (Binance, Gate.io, Bybit) com fallback
    mercado      download, cache e normalização de candles
    indicadores  RSI, EMA, MACD, Bollinger, ATR, ADX, OBV, estocástico
    custos       taxa maker/taker, spread e slippage por notional
    sinais       score de confluência ponderado (a ideia do painel, com dado real)
    backtest     motor event-driven que desconta custo em toda entrada e saída
    walkforward  validação purgada + Deflated Sharpe + PBO (usa `validation/`)
    papel        book de paper trading persistente
    relatorio    saída em Markdown do que foi apurado
"""

__all__ = [
    "exchanges",
    "mercado",
    "indicadores",
    "custos",
    "sinais",
    "backtest",
    "walkforward",
    "papel",
    "relatorio",
]

__version__ = "0.1.0"
