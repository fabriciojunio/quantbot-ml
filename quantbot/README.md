<div align="center">

# 🤖 QuantBot ML

### Sistema de Trading Quantitativo com Inteligência Artificial

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6600?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-206%20passing-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-Private-red?style=for-the-badge)

**Sistema completo de trading quantitativo que combina Machine Learning, análise de sentimentos com FinBERT e gestão de risco institucional para tomada de decisão em mercados financeiros.**

[Funcionalidades](#-funcionalidades) •
[Arquitetura](#-arquitetura) •
[Modelos de ML](#-modelos-de-machine-learning) •
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
| Linhas de código | ~11.400 |
| Arquivos Python | 56 |
| Testes automatizados | 206 (todos passando) |
| Modelos de ML | 3 + FinBERT NLP |
| Mercados cobertos | B3, EUA, Cripto |

---

## 🚀 Funcionalidades

### Machine Learning & NLP
- **Ensemble de 3 modelos**: Random Forest, XGBoost e Gradient Boosting com StandardScaler
- **FinBERT Sentiment Analysis**: análise de sentimento de notícias financeiras utilizando o modelo FinBERT (BERT pré-treinado em textos financeiros)
- **Walk-Forward Validation**: validação temporal com `TimeSeriesSplit` para evitar data leakage e overfitting
- **Detecção de Regime de Mercado**: identificação automática de 4 estados de mercado (alta, baixa, lateralizado, volátil) com ajuste dinâmico de estratégia

### Técnicas Quantitativas Avançadas (López de Prado)
- **Triple Barrier Method**: rotulagem de retornos com barreiras de take-profit, stop-loss e tempo máximo
- **CUSUM Filter**: filtro de mudança estrutural para detecção de eventos relevantes no preço
- **Fractional Differentiation**: diferenciação fracionária para manter memória da série temporal enquanto torna os dados estacionários

### Estratégias de Trading
- **SMA Crossover**: cruzamento de médias móveis simples
- **RSI (Relative Strength Index)**: índice de força relativa para identificar sobrecompra/sobrevenda
- **MACD (Moving Average Convergence Divergence)**: convergência e divergência de médias móveis
- **Ensemble Voting**: votação entre múltiplos modelos e estratégias para sinal final
- **Arquitetura modular**: fácil adição de novas estratégias

### Gestão de Risco
- **Dynamic Trailing Stop-Loss**: stop-loss dinâmico baseado em ATR (Average True Range)
- **Simulação de Monte Carlo**: projeção de cenários futuros com milhares de simulações
- **Magic Formula de Greenblatt**: ranqueamento fundamentalista de ações
- **Métricas de risco**: VaR (Value at Risk), CVaR (Conditional VaR), Sharpe Ratio, Sortino Ratio
- **Calibração**: stop-loss a 5%, take-profit a 15% (ratio 1:3), threshold ML de compra a 65%, compra forte a 75%, dead zone entre 35-65%

### API Backend (FastAPI)
- **Servidor REST**: API FastAPI servindo dados reais para o dashboard
- **Cache inteligente**: cache em memória com TTL configurável por endpoint
- **Rate limiting**: proteção contra abuso com limite por IP
- **CORS seguro**: apenas origens autorizadas (localhost)

### Fontes de Dados
- **yfinance**: dados históricos de ações, ETFs e criptomoedas (gratuito)
- **BCB (Banco Central do Brasil)**: dados macroeconômicos brasileiros — Selic, CDI (gratuito)
- **CoinGecko**: dados complementares de crypto — market cap, volume 24h (gratuito)
- **FED (Federal Reserve)**: dados macroeconômicos americanos
- **Alpaca API**: conexão para trading ao vivo no mercado americano

### Segurança & Compliance
- **Criptografia AES-256**: proteção de dados sensíveis (chaves de API, credenciais)
- **Conformidade com a LGPD**: tratamento adequado de dados pessoais
- **Input validation & sanitization**: validação e sanitização de todas as entradas do sistema

---

## 🏗 Arquitetura

```
quantbot-ml/
├── quantbot/
│   ├── core/                  # Núcleo do sistema
│   │   ├── engine.py          # Motor principal de trading
│   │   ├── portfolio.py       # Gestão de portfólio
│   │   └── risk_manager.py    # Gerenciador de risco
│   │
│   ├── models/                # Modelos de Machine Learning
│   │   ├── ensemble.py        # Ensemble (RF + XGBoost + GB)
│   │   ├── random_forest.py   # Random Forest
│   │   ├── xgboost_model.py   # XGBoost
│   │   ├── gradient_boost.py  # Gradient Boosting
│   │   └── finbert.py         # FinBERT Sentiment Analysis
│   │
│   ├── strategies/            # Estratégias de trading
│   │   ├── sma_crossover.py   # SMA Crossover
│   │   ├── rsi_strategy.py    # RSI
│   │   ├── macd_strategy.py   # MACD
│   │   └── ensemble_voting.py # Votação entre estratégias
│   │
│   ├── data/                  # Coleta e processamento de dados
│   │   ├── market_data.py     # Dados de mercado (yfinance)
│   │   ├── macro_data.py      # Dados macroeconômicos (BCB/FED)
│   │   └── sentiment.py       # Dados de sentimento (notícias)
│   │
│   ├── features/              # Engenharia de features
│   │   ├── technical.py       # Indicadores técnicos
│   │   ├── triple_barrier.py  # Triple Barrier Method
│   │   ├── cusum_filter.py    # CUSUM Filter
│   │   └── frac_diff.py       # Fractional Differentiation
│   │
│   ├── risk/                  # Módulos de risco
│   │   ├── var.py             # Value at Risk
│   │   ├── monte_carlo.py     # Simulação de Monte Carlo
│   │   ├── trailing_stop.py   # Dynamic Trailing Stop-Loss
│   │   └── greenblatt.py      # Magic Formula
│   │
│   ├── security/              # Segurança
│   │   ├── encryption.py      # Criptografia AES-256
│   │   ├── validation.py      # Validação de inputs
│   │   └── lgpd.py            # Compliance LGPD
│   │
│   └── api/                   # API Backend (FastAPI)
│       ├── server.py          # Servidor FastAPI (endpoints REST)
│       ├── market_data.py     # Serviço de dados reais (yfinance, BCB, CoinGecko)
│       └── run.py             # Script de inicialização
│
├── frontend/                  # Dashboard React
│   ├── src/
│   │   ├── App.jsx            # Dashboard principal (8 abas)
│   │   ├── useMarketData.js   # Hook de dados reais com fallback simulado
│   │   └── index.js           # Entry point
│   └── package.json
│
├── tests/                     # 206 testes automatizados
│   ├── test_models/
│   ├── test_strategies/
│   ├── test_risk/
│   ├── test_features/
│   └── test_security/
│
├── requirements.txt
└── README.md
```

---

## 🧠 Modelos de Machine Learning

### Ensemble Learning

O sistema utiliza um ensemble de três modelos com votação ponderada:

| Modelo | Tipo | Uso |
|---|---|---|
| **Random Forest** | Bagging | Robustez e redução de variância |
| **XGBoost** | Boosting | Alta performance e captura de padrões complexos |
| **Gradient Boosting** | Boosting | Complementar ao XGBoost com regularização diferente |

Todos os modelos utilizam **StandardScaler** para normalização das features e são treinados com **Walk-Forward Validation** usando `TimeSeriesSplit` do scikit-learn, garantindo que dados futuros nunca vazam para o treinamento.

### FinBERT — Análise de Sentimento

O FinBERT é um modelo BERT pré-treinado especificamente em textos financeiros. Ele classifica notícias como **positivas**, **negativas** ou **neutras**, e essa informação alimenta o ensemble como uma feature adicional de sentimento de mercado.

### Sinais de Trading

| Sinal | Threshold |
|---|---|
| Compra forte | Probabilidade ≥ 75% |
| Compra | Probabilidade ≥ 65% |
| Dead zone (sem ação) | Probabilidade entre 35% e 65% |
| Venda | Probabilidade ≤ 35% |

---

## 📊 Estratégias de Trading

### SMA Crossover
Cruzamento de médias móveis simples de curto e longo prazo. Quando a média curta cruza a longa para cima, gera sinal de compra. Quando cruza para baixo, sinal de venda.

### RSI (Relative Strength Index)
Mede a força relativa dos movimentos de alta versus baixa. Valores acima de 70 indicam sobrecompra (possível venda), abaixo de 30 indicam sobrevenda (possível compra).

### MACD
Diferença entre duas médias móveis exponenciais. O cruzamento da linha MACD com a linha de sinal gera os sinais de entrada e saída.

### Ensemble Voting
Combina os sinais de todas as estratégias e dos modelos de ML em uma votação ponderada para gerar o sinal final, reduzindo falsos positivos.

---

## 📈 Gestão de Risco

### Parâmetros de Risco

| Parâmetro | Valor | Descrição |
|---|---|---|
| Stop-Loss | 5% | Perda máxima por operação |
| Take-Profit | 15% | Ganho alvo por operação |
| Risk/Reward | 1:3 | Proporção risco/retorno |
| Trailing Stop | ATR-based | Ajuste dinâmico baseado em volatilidade |

### Métricas Calculadas

- **Sharpe Ratio**: retorno ajustado ao risco (benchmark: taxa livre de risco)
- **Sortino Ratio**: como o Sharpe, mas penaliza apenas a volatilidade negativa
- **VaR (Value at Risk)**: perda máxima esperada em um dado nível de confiança
- **CVaR (Conditional VaR)**: perda média esperada além do VaR
- **Monte Carlo**: simulação de milhares de cenários para projeção de retornos futuros

---

## 🖥 Dashboard

Dashboard interativo desenvolvido em **React** com tema visual **black & gold** (#0a0a0a / #D4A843), contendo **8 abas**:

1. **Visão Geral** — resumo do portfólio e performance
2. **Análise de Mercado** — gráficos de candlestick e indicadores técnicos
3. **Sinais de Trading** — sinais gerados pelos modelos de ML
4. **Sentimento** — análise de sentimento FinBERT em tempo real
5. **Risco** — métricas de VaR, CVaR, Sharpe e Sortino
6. **Monte Carlo** — simulações e projeções
7. **Regime de Mercado** — estado atual detectado (alta/baixa/lateral/volátil)
8. **Configurações** — parâmetros do sistema e calibração

---

## ⚙ Como Rodar

### Pré-requisitos

- Python 3.10+
- Node.js 18+ (para o dashboard)
- Chave de API do Alpaca (opcional, para live trading)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/fabriciojunio/quantbot-ml.git
cd quantbot-ml

# Instale as dependências Python
pip install -r requirements.txt

# Instale as dependências do dashboard
cd frontend
npm install
```

### Execução

```bash
# 1. Iniciar o backend (API com dados reais)
cd quantbot
python -m api.run
# API disponível em http://localhost:8000
# Documentação em http://localhost:8000/api/docs

# 2. Iniciar o frontend (em outro terminal)
cd quantbot/frontend
npm start
# Dashboard em http://localhost:3000

# Rodar os testes
cd quantbot
pytest tests/ -v

# Rodar o sistema de trading
python -m quantbot.core.engine
```

> **Nota:** O dashboard funciona mesmo sem o backend — usa dados simulados como fallback. Com o backend rodando, exibe dados reais do mercado.

### Variáveis de Ambiente

```env
ALPACA_API_KEY=sua_chave_aqui
ALPACA_SECRET_KEY=sua_chave_secreta_aqui
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

## 🛠 Stack Tecnológica

| Categoria | Tecnologias |
|---|---|
| **Linguagem** | Python 3.10+ |
| **ML/AI** | scikit-learn, XGBoost, Transformers (FinBERT) |
| **Backend** | FastAPI, Uvicorn |
| **Dados** | yfinance, BCB API, CoinGecko API, Alpaca API |
| **Frontend** | React 18, Node.js |
| **Visualização** | matplotlib (dark theme) |
| **Segurança** | AES-256, LGPD compliance |
| **Testes** | pytest (206 testes) |
| **Versionamento** | Git, GitHub |

---

## 📚 Referências Acadêmicas

O projeto é fundamentado em literatura acadêmica e institucional de referência:

- **López de Prado, M.** — *Advances in Financial Machine Learning* (Triple Barrier, CUSUM, Fractional Differentiation)
- **Krauss, C. et al.** — *Deep Neural Networks, Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the S&P 500*
- **Markowitz, H.** — *Portfolio Selection* (Teoria Moderna de Portfólios)
- **Sharpe, W.** — *Capital Asset Pricing Model* (CAPM e Sharpe Ratio)
- **Breiman, L.** — *Random Forests*
- **Chen, T. & Guestrin, C.** — *XGBoost: A Scalable Tree Boosting System*
- **Greenblatt, J.** — *The Little Book That Beats the Market* (Magic Formula)

---

## 📊 Status do Projeto

🟢 **Em desenvolvimento ativo**

O projeto está em constante evolução com melhorias nos modelos, novas estratégias e otimizações de performance.

---

<div align="center">

Desenvolvido por **Fabrício Júnio Almeida Dias**

[![GitHub](https://img.shields.io/badge/GitHub-fabriciojunio-181717?style=for-the-badge&logo=github)](https://github.com/fabriciojunio)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Fabrício%20Júnio-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/fabr%C3%ADcioj%C3%BAnio/)

</div>
