<div align="center">

# 🤖 QuantBot ML

### Sistema de Trading Quantitativo com Inteligência Artificial

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6600?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-233%20passing-brightgreen?style=for-the-badge)

**Sistema completo de trading quantitativo que combina Machine Learning, análise de sentimentos com FinBERT e gestão de risco institucional para tomada de decisão em mercados financeiros.**

[Funcionalidades](#-funcionalidades) •
[Arquitetura](#-arquitetura) •
[Modelos de ML](#-modelos-de-machine-learning) •
[API](#-api) •
[Estratégias](#-estratégias-de-trading) •
[Como Rodar](#-como-rodar) •
[Dashboard](#-dashboard)

</div>

---

## 📋 Sobre o Projeto

O QuantBot ML é um sistema de trading quantitativo de nível institucional que utiliza um ensemble de modelos de Machine Learning combinado com análise de sentimento via FinBERT (NLP) para gerar sinais de compra e venda em ativos financeiros.

O sistema cobre três mercados: **B3 (Brasil)**, **mercado americano (EUA)** e **criptomoedas**, aplicando técnicas avançadas de validação, gestão de risco e detecção de regime de mercado para maximizar retornos ajustados ao risco.

### Números do Projeto

| Métrica | Valor |
|---|---|
| Linhas de código | ~13.000 |
| Arquivos de código | 65 |
| Testes automatizados | 233 (todos passando) |
| Modelos de ML | 3 + FinBERT NLP |
| Mercados cobertos | B3, EUA, Cripto |
| Ativos monitorados | 25 |

---

## 🚀 Funcionalidades

### Machine Learning & NLP
- **Ensemble de 3 modelos**: Random Forest, XGBoost e Gradient Boosting com StandardScaler
- **FinBERT Sentiment Analysis**: análise de sentimento de notícias financeiras utilizando o modelo FinBERT (BERT pré-treinado em textos financeiros)
- **Walk-Forward Validation**: validação temporal com `TimeSeriesSplit` para evitar data leakage e overfitting
- **Detecção de Regime de Mercado**: identificação automática de 4 estados de mercado (bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol) com ajuste dinâmico de estratégia e position sizing adaptativo

### Técnicas Quantitativas Avançadas (López de Prado)
- **Triple Barrier Method**: rotulagem de retornos com barreiras de take-profit, stop-loss e tempo máximo
- **CUSUM Filter**: filtro de mudança estrutural para detecção de eventos relevantes no preço
- **Fractional Differentiation**: diferenciação fracionária para manter memória da série temporal enquanto torna os dados estacionários

### Estratégias de Trading
- **SMA Crossover**: cruzamento de médias móveis simples
- **RSI (Relative Strength Index)**: índice de força relativa para identificar sobrecompra/sobrevenda
- **MACD (Moving Average Convergence Divergence)**: convergência e divergência de médias móveis
- **Ensemble Voting**: votação entre múltiplos modelos e estratégias para sinal final
- **Arquitetura modular**: fácil adição de novas estratégias via classe base

### Gestão de Risco
- **3 perfis de risco**: Conservative, Moderate e Aggressive com parâmetros calibrados
- **Dynamic Trailing Stop-Loss**: stop-loss dinâmico baseado em ATR (Average True Range)
- **Simulação de Monte Carlo**: projeção de cenários futuros com milhares de simulações
- **Métricas de risco**: VaR (Value at Risk), CVaR (Conditional VaR), Sharpe Ratio, Sortino Ratio, Beta, Alpha
- **Position sizing adaptativo**: ajusta tamanho da posição automaticamente baseado no regime de mercado
- **Calibração Moderate**: stop-loss a 5%, take-profit a 15% (ratio 1:3), threshold ML de compra a 65%, compra forte a 75%, dead zone entre 35-65%

### API REST (FastAPI)
- **Endpoints de dados**: `/api/assets`, `/api/asset/{symbol}`, `/api/selic`, `/api/cdi`, `/api/crypto`
- **Dashboard endpoint**: `/api/dashboard` com dados consolidados
- **Cache em memória**: thread-safe com TTL para evitar chamadas repetidas
- **CORS configurado**: permite conexão do frontend React
- **Rate limiting**: proteção contra abuso de requisições

### Fontes de Dados
- **yfinance**: dados históricos de ações, ETFs e criptomoedas
- **BCB (Banco Central do Brasil)**: taxa Selic, histórico CDI
- **CoinGecko**: dados de criptomoedas
- **Google News / Yahoo Finance / CoinDesk / Investing.com**: notícias para análise de sentimento
- **Alpaca API**: conexão para trading ao vivo no mercado americano (opcional)

### Segurança & Compliance
- **Criptografia AES-256**: proteção de dados sensíveis (chaves de API, credenciais)
- **Conformidade com a LGPD**: registro de consentimento, exportação e exclusão de dados, relatório de breach
- **Audit Trail**: log de auditoria com verificação de integridade
- **Rate Limiter**: proteção contra abuso
- **Input validation & sanitization**: validação e sanitização de todas as entradas do sistema

---

## 🏗 Arquitetura

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
│   │   ├── test_quantbot.py       # Testes core
│   │   ├── test_strategies.py     # Testes de estratégias
│   │   ├── test_advanced.py       # Testes avançados (CUSUM, Triple Barrier, regime)
│   │   ├── test_security.py       # Testes de segurança (AES, LGPD, audit)
│   │   ├── test_live_trading.py   # Testes de trading ao vivo
│   │   ├── test_news_sentiment.py # Testes de NLP/sentimento
│   │   ├── test_benchmarks.py     # Testes de benchmarks
│   │   ├── test_accuracy.py       # Testes de acurácia
│   │   ├── test_performance.py    # Testes de performance
│   │   └── test_api.py            # Testes da API FastAPI
│   │
│   ├── utils/                     # Utilitários
│   │   ├── logger.py              # Logging configurado
│   │   ├── formatters.py          # Formatação de moeda e percentual
│   │   └── security.py            # Helpers de segurança
│   │
│   ├── dashboard.jsx              # Dashboard React standalone (39KB)
│   ├── main.py                    # Ponto de entrada principal
│   ├── run_backtest.py            # Script de backtest direto
│   ├── run_live.py                # Script de trading ao vivo
│   └── requirements.txt           # Dependências Python
│
└── README.md
```

---

## 🧠 Modelos de Machine Learning

### Ensemble Learning

O sistema utiliza um ensemble de modelos com votação ponderada pela probabilidade:

| Modelo | Tipo | Configuração |
|---|---|---|
| **Random Forest** | Bagging | 200 estimators, max_depth=10, class_weight balanced |
| **XGBoost** | Boosting | 200 estimators, lr=0.05, subsample=0.8 |
| **Gradient Boosting** | Boosting | Fallback quando XGBoost não está disponível |

Todos os modelos utilizam **StandardScaler** para normalização das 32 features técnicas e são treinados com **Walk-Forward Validation** usando `TimeSeriesSplit` do scikit-learn.

### FinBERT — Análise de Sentimento

Coleta notícias de 5 fontes (Google News BR, Yahoo Finance, CoinDesk, Investing.com BR) e classifica como **positivas**, **negativas** ou **neutras**. Inclui fallback para análise léxica bilíngue (PT/EN) quando FinBERT não está disponível.

### Sinais de Trading

| Sinal | Threshold |
|---|---|
| Compra forte | Probabilidade ≥ 75% |
| Compra | Probabilidade ≥ 65% |
| Dead zone (sem ação) | Probabilidade entre 35% e 65% |
| Venda | Probabilidade ≤ 35% |
| Venda forte | Probabilidade ≤ 25% |

---

## 🌐 API

API REST construída com FastAPI que serve dados em tempo real para o dashboard:

| Endpoint | Descrição |
|---|---|
| `GET /api/health` | Health check do servidor |
| `GET /api/assets` | Todos os ativos com preços e indicadores |
| `GET /api/asset/{symbol}` | Dados detalhados de um ativo |
| `GET /api/selic` | Taxa Selic atual (BCB) |
| `GET /api/cdi` | Histórico CDI |
| `GET /api/crypto` | Dados CoinGecko |
| `GET /api/dashboard` | Dados consolidados para o dashboard |

---

## 📈 Gestão de Risco

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
| Bear + High Vol | NÃO operar | 0% |

---

## 🖥 Dashboard

Dashboard interativo em **React** com tema **black & gold**, contendo **8 abas**:

1. **Visão Geral** — equity curve vs benchmark, alocação por mercado, métricas resumo
2. **Performance** — retornos semanal/mensal/anual, comparação com CDI/Ibovespa/S&P 500
3. **Notícias** — feed de notícias com filtro por sentimento
4. **Sinais ML** — score por ativo, votação dos modelos, feature importance
5. **Paper Trading** — simulação com capital virtual, bot automático
6. **Acurácia** — taxa de acerto, matriz de confusão, acurácia por mercado
7. **Memória** — log de eventos que impactaram decisões
8. **Risco** — VaR, CVaR, Beta, Alpha, Max Drawdown, Sortino

---

## ⚙ Como Rodar

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

## 🛠 Stack Tecnológica

| Categoria | Tecnologias |
|---|---|
| **Linguagem** | Python 3.10+ |
| **ML/AI** | scikit-learn, XGBoost, Transformers (FinBERT) |
| **API** | FastAPI, Uvicorn |
| **Dados** | yfinance, BCB API, CoinGecko, Google News |
| **Frontend** | React 18, Node.js |
| **Visualização** | matplotlib (dark theme) |
| **Segurança** | AES-256, LGPD compliance |
| **Testes** | pytest (233 testes) |

---

## 📚 Referências Acadêmicas

- **López de Prado, M.** — *Advances in Financial Machine Learning*
- **Krauss, C. et al.** — *Statistical Arbitrage on the S&P 500*
- **Markowitz, H.** — *Portfolio Selection*
- **Sharpe, W.** — *Capital Asset Pricing Model*
- **Breiman, L.** — *Random Forests*
- **Chen, T. & Guestrin, C.** — *XGBoost: A Scalable Tree Boosting System*

---

<div align="center">

🟢 **Em desenvolvimento ativo**

Desenvolvido por **Fabrício Júnio Almeida Dias**

[![GitHub](https://img.shields.io/badge/GitHub-fabriciojunio-181717?style=for-the-badge&logo=github)](https://github.com/fabriciojunio)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Fabrício%20Júnio-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/fabr%C3%ADcioj%C3%BAnio/)

</div>
