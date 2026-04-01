# QuantBot ML — Roadmap & Documentação do Projeto

## O que é o QuantBot ML?

Sistema de trading quantitativo com Machine Learning para análise de mercados financeiros (B3, EUA e Cripto). O sistema combina indicadores técnicos clássicos com modelos de ML para gerar sinais de compra e venda, executar backtests realistas e operar em paper trading ou ao vivo via Alpaca.

---

## Arquitetura Geral

```
quantbot/
├── config/          Configurações centralizadas (enums, perfis de risco, parâmetros ML)
├── data/            Coleta, validação, features e sentimento de notícias
├── models/          Modelos de ML, geração de sinais e validação walk-forward
├── strategies/      Estratégias técnicas modulares (SMA, RSI, MACD, Ensemble)
├── risk/            Gerenciamento de risco, métricas e stop dinâmico
├── backtest/        Motor de backtesting, benchmarks e relatórios
├── core/            Paper trading, live trading (Alpaca), rastreamento de performance
├── visualization/   Gráficos profissionais em tema escuro
├── utils/           Logger, formatadores e segurança (validação de inputs)
└── tests/           206 testes automatizados
```

---

## Fluxo de Dados

```
yfinance / RSS Feeds
        │
        ▼
 MarketDataFetcher          ← coleta OHLCV + cache + rate limiting
        │
        ▼
  FeatureEngineer           ← 35+ indicadores técnicos (RSI, MACD, Bollinger, ATR…)
        │
     ┌──┴──┐
     │     │
     ▼     ▼
EnsembleModel   Estratégias Técnicas
(RF+XGB+GBM)    (SMA / RSI / MACD / Voting)
     │     │
     └──┬──┘
        │
        ▼
  SignalGenerator            ← Score 0–100 + sinal + confiança
        │
        ▼
   RiskManager               ← Tamanho de posição, stop-loss, take-profit
        │
     ┌──┴──────────┐
     ▼             ▼
BacktestEngine   PaperTradingEngine / AlpacaAdapter
     │
     ▼
ReportGenerator + Visualization
```

---

## Módulos em Detalhe

### `config/settings.py`
Fonte única de verdade para todas as configurações:
- **Signal**: enum com `STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL`
- **RiskProfile**: `conservative`, `moderate` (padrão), `aggressive`
- **UNIVERSE**: 25 ativos — B3 (10), EUA (10), Cripto (5)
- **ML_CONFIG**: thresholds de sinal, splits de CV
- **BACKTEST_CONFIG**: capital inicial, comissão, slippage, warmup

---

### `data/`

| Arquivo | Responsabilidade |
|---|---|
| `fetcher.py` | `MarketDataFetcher` — download via yfinance, cache em memória, rate limiting |
| `features.py` | `FeatureEngineer` — 35+ features + colunas raw para estratégias (RSI, MACD, SMA_20/50) |
| `validators.py` | `DataValidator` — checagem de qualidade dos dados |
| `news_fetcher.py` | `NewsFetcher` — 5 feeds RSS (InfoMoney, Reuters, CoinDesk…) |
| `sentiment.py` | `SentimentAnalyzer` — FinBERT (se disponível) + léxico financeiro PT/EN |
| `macro_data.py` | Dados macroeconômicos (BCB), correlações, Monte Carlo, pesos ótimos |
| `cusum_filter.py` | Filtro CUSUM para detecção de eventos relevantes (López de Prado) |

**Features geradas por `FeatureEngineer.compute_all()`:**
- Retornos: `return_1d/5d/10d/20d`, `log_return`
- Tendência: `sma_ratio_5/10/20/50`, `ema_ratio_*`, `sma_cross_5_20`, `sma_cross_10_50`
- Momentum: `RSI`, `MACD`, `momentum_10/20`, `roc_5/10`
- Volatilidade: `bb_width`, `bb_position`, `volatility_10/20`, `volatility_ratio`, `atr_ratio`
- Volume: `volume_ratio`, `volume_change`, `obv_change`
- Osciladores: `stoch_k`, `stoch_d`, `williams_r`
- Temporais: `day_sin/cos`, `month_sin/cos`
- Raw para estratégias: `SMA_20`, `SMA_50`, `RSI`, `MACD`, `MACD_signal`, `MACD_hist`

---

### `models/`

| Arquivo | Responsabilidade |
|---|---|
| `ensemble.py` | `EnsembleModel` — Random Forest + XGBoost + Gradient Boosting (votação por probabilidade) |
| `trainer.py` | `ModelTrainer` — TimeSeriesSplit (5 folds), CV, serialização por ativo |
| `signals.py` | `SignalGenerator` — converte probabilidade em `MLSignal` (score 0–100 + confiança) |
| `regime.py` | Detecção de regime de mercado (bull/bear/lateral) + `AdaptiveSizer` |
| `triple_barrier.py` | Labeling por tripla barreira (López de Prado cap. 3) |
| `walk_forward.py` | `WalkForwardValidator` — retreina periodicamente, detecta overfitting |

---

### `strategies/`

Todas herdam de `BaseStrategy` e recebem/retornam um DataFrame com coluna `signal`.

| Estratégia | Lógica |
|---|---|
| `SMACrossoverStrategy` | Compra quando SMA(20) cruza acima de SMA(50), vende no cruzamento oposto |
| `RSIStrategy` | Compra em sobrevendido (RSI < 30), vende em sobrecomprado (RSI > 70) |
| `MACDStrategy` | Compra no cruzamento MACD acima da linha de sinal, sinal forte quando histograma > 1σ |
| `EnsembleVotingStrategy` | Votação ponderada: SMA (×1.0) + RSI (×1.0) + MACD (×1.2) |

---

### `risk/`

| Arquivo | Responsabilidade |
|---|---|
| `manager.py` | `RiskManager` — tamanho de posição por perfil, `should_stop_loss()`, `should_take_profit()` |
| `metrics.py` | VaR 95/99%, CVaR, Sharpe, Sortino, Beta, Alpha, Max Drawdown |
| `dynamic_stop.py` | Stop dinâmico: trailing stop, breakeven, saída por tempo |

**Perfis de risco:**

| Perfil | Stop-Loss | Take-Profit | Posição Máx |
|---|---|---|---|
| Conservative | 3% | 9% | 5% |
| Moderate | 5% | 15% | 10% |
| Aggressive | 7% | 21% | 20% |

---

### `backtest/`

| Arquivo | Responsabilidade |
|---|---|
| `engine.py` | `BacktestEngine` — simula trades com comissão + slippage, retorna `BacktestResult` |
| `report.py` | `ReportGenerator` — imprime métricas no terminal e gera PNG com 4 subgráficos |
| `benchmarks.py` | `BenchmarkComparator` — compara vs CDI, Ibovespa, S&P 500 (estimativas históricas) |

**`BacktestResult` contém:** `total_return`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `win_rate`, `profit_factor`, `calmar_ratio`, `alpha`, `equity_curve`, `trades`

---

### `core/`

| Arquivo | Responsabilidade |
|---|---|
| `paper_trading.py` | `PaperTradingEngine` — simula ordens com preços reais, `buy/sell/portfolio/export/load` |
| `live_trading.py` | `AlpacaAdapter` + `LiveTradingEngine` + `SafetyMonitor` — trading real via Alpaca |
| `accuracy.py` | `AccuracyTracker` — rastreia acertos/erros de previsão por ativo e mercado |
| `performance.py` | `PerformanceTracker` — breakdown semanal/mensal/anual com benchmark |
| `security.py` | `SecurityManager` — PBKDF2, AES-256, LGPD, audit trail com hash de integridade |

---

### `visualization/`

Todos os gráficos usam tema preto & dourado. Salvos automaticamente em `results/`.

| Função | Saída |
|---|---|
| `plot_backtest_signals()` | Preço + sinais de compra/venda + Volume + RSI + MACD (4 subplots) |
| `plot_equity_curve()` | Curva de equity com fill verde/vermelho vs capital inicial |
| `plot_strategy_comparison()` | Barras de retorno + tabela comparativa das estratégias |
| `plot_feature_importance()` | Feature importance horizontal do modelo ML |

---

## Como Executar

### Análise completa (ML + Backtest + Sinais)
```bash
cd quantbot
python main.py --analyze
# ou interativo:
python main.py
```

### Backtest de estratégias técnicas
```bash
python run_backtest.py                    # tickers padrão
python run_backtest.py AAPL TSLA NVDA     # específicos
python run_backtest.py --all              # todos os mercados
```

### Paper Trading
```bash
python main.py --paper-trade
# Comandos: buy AAPL 10 | sell AAPL all | portfolio | auto | save | load
```

### Trading ao Vivo (Alpaca)
```bash
cp quantbot/.env.example quantbot/.env   # preencha as chaves
python run_live.py --stocks              # ações EUA
python run_live.py --crypto              # criptos 24/7
```

### Testes
```bash
python -m pytest tests/ -v               # 206 testes
python -m pytest tests/ --cov=. -v      # com cobertura
```

---

## Dependências Principais

```
pandas, numpy          → manipulação de dados
yfinance               → dados históricos de mercado
scikit-learn           → Random Forest, GradientBoosting, métricas
xgboost                → XGBoost ensemble
scipy                  → estatística
matplotlib, seaborn    → visualizações
cryptography           → AES-256, PBKDF2 (segurança)
python-dotenv          → variáveis de ambiente
alpaca-trade-api       → broker para live trading (opcional)
transformers, torch    → FinBERT para sentimento (opcional)
```

---

## Roadmap de Evolução

### v1.0 — Concluído
- [x] Pipeline ML completo (coleta → features → treino → sinal)
- [x] 4 estratégias técnicas modulares + ensemble voting
- [x] Motor de backtesting com custos realistas
- [x] Paper trading interativo
- [x] Live trading via Alpaca (paper + real)
- [x] Análise de sentimento (FinBERT + léxico)
- [x] Dados macroeconômicos (BCB)
- [x] Walk-forward validation
- [x] Tripla barreira (labeling avançado)
- [x] Detecção de regime de mercado
- [x] Métricas de risco (VaR, CVaR, Sharpe, Sortino, Beta, Alpha)
- [x] Comparação com benchmarks (CDI, Ibovespa, S&P 500)
- [x] Segurança LGPD + auditoria + AES-256
- [x] 206 testes automatizados

### v1.1 — Próximos Passos
- [ ] Dashboard web React integrado com API Flask/FastAPI
- [ ] Alertas por Telegram/email em sinais relevantes
- [ ] Suporte a Binance para cripto ao vivo
- [ ] Otimização de hiperparâmetros com Optuna
- [ ] Relatório PDF automático por ativo

### v2.0 — Visão de Longo Prazo
- [ ] Modelos de séries temporais (LSTM, Transformer financeiro)
- [ ] Dados alternativos (on-chain cripto, order flow, opções)
- [ ] Multi-asset portfolio optimization (Markowitz + HRP)
- [ ] Deploy em nuvem (AWS/GCP) com scheduler automático
- [ ] API pública com autenticação JWT
