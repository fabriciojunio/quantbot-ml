# QuantBot ML

Sistema de trading quantitativo com ensemble de Machine Learning, análise de sentimento via FinBERT e gestão de risco por regime de mercado. Cobre B3, EUA e criptomoedas.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/testes-233%20passando-brightgreen?style=flat-square)

---

## Arquitetura

```
quantbot-ml/
├── quantbot/
│   ├── api/                       # API REST (FastAPI)
│   │   ├── server.py              # Servidor com endpoints e cache
│   │   ├── market_data.py         # Serviço de dados de mercado
│   │   └── run.py                 # Script de inicialização
│   │
│   ├── backtest/                  # Motor de backtesting
│   │   ├── engine.py              # Backtester com comissão e slippage
│   │   ├── benchmarks.py          # Comparação CDI, Ibovespa, S&P 500, Bitcoin
│   │   └── report.py              # Gerador de relatórios visuais
│   │
│   ├── config/                    # Configuração centralizada
│   │   └── settings.py            # Enums, dataclasses, parâmetros ML/risco
│   │
│   ├── core/                      # Núcleo do sistema
│   │   ├── accuracy.py            # Rastreador de acurácia por símbolo/mercado
│   │   ├── live_trading.py        # Engine de trading ao vivo com safety monitor
│   │   ├── paper_trading.py       # Simulação com dinheiro virtual
│   │   ├── performance.py         # Métricas semanal/mensal/anual
│   │   └── security.py            # SecurityManager (AES-256, LGPD, Audit)
│   │
│   ├── data/                      # Coleta e processamento de dados
│   │   ├── fetcher.py             # Download de dados (yfinance)
│   │   ├── features.py            # Engenharia de 32 features técnicas
│   │   ├── validators.py          # Validação e sanitização de dados
│   │   ├── cusum_filter.py        # CUSUM Filter (López de Prado)
│   │   ├── macro_data.py          # Dados macroeconômicos (BCB/FED)
│   │   ├── news_fetcher.py        # Coleta de notícias (5 fontes)
│   │   └── sentiment.py           # Análise de sentimento (FinBERT/Léxico)
│   │
│   ├── models/                    # Modelos de Machine Learning
│   │   ├── ensemble.py            # Ensemble (RF + XGBoost + GB)
│   │   ├── trainer.py             # Treinamento com cross-validation
│   │   ├── signals.py             # Gerador de sinais calibrados
│   │   ├── regime.py              # Detecção de regime de mercado (4 estados)
│   │   ├── triple_barrier.py      # Triple Barrier Method
│   │   └── walk_forward.py        # Walk-Forward Validation
│   │
│   ├── risk/                      # Módulos de risco
│   │   ├── metrics.py             # VaR, CVaR, Sharpe, Sortino, Beta, Alpha
│   │   ├── manager.py             # Gerenciador de risco com perfis
│   │   └── dynamic_stop.py        # Trailing Stop-Loss dinâmico (ATR)
│   │
│   ├── strategies/                # Estratégias de trading
│   │   ├── base.py                # Classe base (Signal enum, interface)
│   │   ├── sma_crossover.py       # SMA Crossover
│   │   ├── rsi_strategy.py        # RSI
│   │   ├── macd_strategy.py       # MACD
│   │   └── ensemble_voting.py     # Votação entre estratégias
│   │
│   ├── frontend/                  # Dashboard React
│   │   └── src/
│   │       ├── App.jsx            # Dashboard com 8 abas
│   │       └── useMarketData.js   # Hook para conexão com API
│   │
│   ├── tests/                     # 233 testes automatizados
│   │   ├── test_quantbot.py
│   │   ├── test_strategies.py
│   │   ├── test_advanced.py       # CUSUM, Triple Barrier, regime
│   │   ├── test_security.py       # AES, LGPD, audit
│   │   ├── test_live_trading.py
│   │   ├── test_news_sentiment.py
│   │   ├── test_benchmarks.py
│   │   ├── test_accuracy.py
│   │   ├── test_performance.py
│   │   └── test_api.py
│   │
│   ├── main.py                    # Ponto de entrada
│   ├── run_backtest.py
│   ├── run_live.py
│   └── requirements.txt
│
└── README.md
```

---

## Modelos de ML

O sistema usa ensemble com votação ponderada pela probabilidade:

| Modelo | Tipo | Configuração |
|---|---|---|
| Random Forest | Bagging | 200 estimators, max_depth=10, class_weight balanced |
| XGBoost | Boosting | 200 estimators, lr=0.05, subsample=0.8 |
| Gradient Boosting | Boosting | Fallback quando XGBoost não está disponível |

Todos os modelos usam StandardScaler com 32 features técnicas e são treinados com Walk-Forward Validation (`TimeSeriesSplit`).

### FinBERT: Análise de Sentimento

Coleta notícias de 5 fontes (Google News BR, Yahoo Finance, CoinDesk, Investing.com BR) e classifica como positivas, negativas ou neutras. Fallback para análise léxica bilíngue (PT/EN) quando FinBERT não está disponível.

### Sinais de Trading

| Sinal | Threshold |
|---|---|
| Compra forte | Probabilidade >= 75% |
| Compra | Probabilidade >= 65% |
| Dead zone (sem ação) | 35% a 65% |
| Venda | Probabilidade <= 35% |
| Venda forte | Probabilidade <= 25% |

### Técnicas Quantitativas (López de Prado)

- **Triple Barrier Method**: rotulagem de retornos com barreiras de take-profit, stop-loss e tempo máximo
- **CUSUM Filter**: filtro de mudança estrutural para detecção de eventos relevantes no preço
- **Fractional Differentiation**: diferenciação fracionária para manter memória da série temporal enquanto torna os dados estacionários

---

## Gestão de Risco

### Perfis de Risco

| Parâmetro | Conservative | Moderate | Aggressive |
|---|---|---|---|
| Max Posição | 5% | 10% | 20% |
| Stop-Loss | 3% | 5% | 7% |
| Take-Profit | 9% | 15% | 21% |
| Risk/Reward | 1:3 | 1:3 | 1:3 |
| Max Drawdown | 8% | 15% | 25% |
| Max Trades/dia | 5 | 10 | 20 |

### Detecção de Regime de Mercado

| Regime | Ação | Position Scale |
|---|---|---|
| Bull + Low Vol | Operar normalmente | 100% |
| Bull + High Vol | Posições menores | 50% |
| Bear + Low Vol | Reduzir exposição | 0% |
| Bear + High Vol | Não operar | 0% |

---

## API

| Endpoint | Descrição |
|---|---|
| `GET /api/health` | Health check |
| `GET /api/assets` | Todos os ativos com preços e indicadores |
| `GET /api/asset/{symbol}` | Dados detalhados de um ativo |
| `GET /api/selic` | Taxa Selic atual (BCB) |
| `GET /api/cdi` | Histórico CDI |
| `GET /api/crypto` | Dados CoinGecko |
| `GET /api/dashboard` | Dados consolidados para o dashboard |

---

## Dashboard

React com tema black & gold, 8 abas:

1. Visão Geral: equity curve vs benchmark, alocação por mercado
2. Performance: retornos semanal/mensal/anual vs CDI/Ibovespa/S&P 500
3. Notícias: feed com filtro por sentimento
4. Sinais ML: score por ativo, votação dos modelos, feature importance
5. Paper Trading: simulação com capital virtual
6. Acurácia: taxa de acerto, matriz de confusão por mercado
7. Memória: log de eventos que impactaram decisões
8. Risco: VaR, CVaR, Beta, Alpha, Max Drawdown, Sortino

---

## Como Rodar

### Pré-requisitos

- Python 3.10+
- Node.js 18+ (para o dashboard)

### Instalação

```bash
git clone https://github.com/fabriciojunio/quantbot-ml.git
cd quantbot-ml/quantbot
pip install -r requirements.txt
```

### Execução

```bash
# Menu interativo
python main.py

# Análise completa
python main.py --analyze

# Paper Trading
python main.py --paper-trade

# API
python api/run.py

# Testes
python -m pytest tests/ -v

# Dashboard
cd frontend && npm install && npm start
```

---

## Stack

| Categoria | Tecnologias |
|---|---|
| Linguagem | Python 3.10+ |
| ML/AI | scikit-learn, XGBoost, Transformers (FinBERT) |
| API | FastAPI, Uvicorn |
| Dados | yfinance, BCB API, CoinGecko, Google News |
| Frontend | React 18, Node.js |
| Visualização | matplotlib (dark theme) |
| Segurança | AES-256, LGPD compliance |
| Testes | pytest (233 testes) |

---

## Referências

- **López de Prado, M.**: *Advances in Financial Machine Learning*
- **Krauss, C. et al.**: *Statistical Arbitrage on the S&P 500*
- **Markowitz, H.**: *Portfolio Selection*
- **Sharpe, W.**: *Capital Asset Pricing Model*
- **Breiman, L.**: *Random Forests*
- **Chen, T. & Guestrin, C.**: *XGBoost: A Scalable Tree Boosting System*

---

Desenvolvido por **Fabrício Júnio Alves Dias** · [GitHub](https://github.com/fabriciojunio) · [LinkedIn](https://www.linkedin.com/in/fabríciojúnio/)
