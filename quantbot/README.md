# QuantBot ML — Sistema de Finanças Quantitativas com Machine Learning

**TCC em Ciência da Computação — UNISAGRADO**

Sistema completo de análise quantitativa de investimentos utilizando Machine Learning
para geração de sinais de compra/venda, análise de sentimento de notícias,
otimização de portfólio, gestão de risco e operação automatizada.

**Mercados:** B3 (10 ativos) | NYSE/NASDAQ (10 ativos) | Crypto (5 ativos)

---

## Números do Projeto

| Métrica | Valor |
|---|---|
| Arquivos Python | 48 |
| Linhas de código | 9.500+ |
| Testes automatizados | 170 |
| Taxa de aprovação | 100% |
| Módulos | 12 |
| Features de ML | 35+ |
| Modelos de ML | 3 (RF, XGBoost, GB) |
| Estratégias técnicas | 4 (SMA, RSI, MACD, Ensemble Votação) |
| Fontes de notícias | 5 RSS feeds |
| Corretoras suportadas | 3 (Binance, Alpaca, Paper) |

---

## Instalação

```bash
git clone https://github.com/seu-usuario/quantbot-ml.git
cd quantbot-ml
python -m venv venv
source venv/bin/activate        # Linux/Mac
pip install -r requirements.txt
python -m pytest tests/ -v      # Valida tudo
python main.py                  # Menu interativo
```

---

## Arquitetura

```
quantbot/
├── config/settings.py          Configurações centralizadas
├── data/
│   ├── fetcher.py              Coleta (Yahoo Finance + cache)
│   ├── features.py             35+ indicadores técnicos
│   ├── validators.py           Validação de dados
│   ├── news_fetcher.py         RSS feeds (5 fontes gratuitas)
│   └── sentiment.py            FinBERT + léxico financeiro
├── models/
│   ├── ensemble.py             RF + XGBoost + GradientBoosting
│   ├── trainer.py              TimeSeriesSplit (5 folds)
│   └── signals.py              Score 0-100 + confiança
├── risk/
│   ├── manager.py              Position sizing + stop-loss
│   └── metrics.py              VaR, CVaR, Sharpe, Sortino, Beta, Alpha
├── backtest/
│   ├── engine.py               Simulação com comissão + slippage
│   ├── benchmarks.py           CDI, Ibovespa, S&P 500
│   └── report.py               Gráficos PNG
├── strategies/
│   ├── base.py                 Classe abstrata + Signal numérico
│   ├── sma_crossover.py        Cruzamento de médias móveis
│   ├── rsi_strategy.py         RSI sobrecompra/sobrevenda
│   ├── macd_strategy.py        MACD cruzamento de sinal
│   └── ensemble_voting.py      Votação ponderada SMA+RSI+MACD
├── visualization/
│   └── __init__.py             Gráficos profissionais (dark theme)
├── execution/                  (para extensões futuras)
├── core/
│   ├── paper_trading.py        Dinheiro fictício + preços reais
│   ├── live_trading.py         Base para corretoras reais
│   ├── accuracy.py             Taxa de acerto + calibração
│   ├── performance.py          Semanal / Mensal / Anual
│   └── security.py             Senha, AES-256, LGPD, Audit Trail
├── utils/
│   ├── logger.py               Logging auditável
│   ├── security.py             Sanitização de inputs
│   └── formatters.py           Formatação de valores
├── tests/                      170 testes automatizados
├── main.py                     Ponto de entrada
├── run_backtest.py             Backtesting comparativo
├── run_live.py                 Trading ao vivo (Alpaca)
├── .env.example                Template de configuração
├── .gitignore                  Proteção de secrets
└── requirements.txt            Dependências
```

---

## Funcionalidades

### Machine Learning
- Ensemble de 3 modelos (Random Forest, XGBoost, Gradient Boosting)
- 35+ features técnicas (RSI, MACD, Bollinger, ATR, momentum, etc.)
- Validação temporal (TimeSeriesSplit) para evitar data leakage
- Score 0-100 com nível de confiança e concordância entre modelos

### Análise de Sentimento
- 5 fontes RSS gratuitas (Google News, Yahoo Finance, CoinDesk, Investing.com)
- FinBERT (deep learning) ou léxico financeiro (fallback leve)
- Detecção automática de ativos mencionados em notícias
- Sentimento como feature do modelo ML

### Gestão de Risco
- Position sizing baseado em volatilidade e confiança
- Stop-loss e take-profit dinâmicos (3 perfis)
- VaR, CVaR, Beta, Alpha, Sharpe, Sortino
- Limite de perda diária (bot desliga automaticamente)

### Estratégias Técnicas Modulares
- SMA Crossover: cruzamento de médias móveis 20/50
- RSI: sobrecomprado (>70) e sobrevendido (<30) com extremos
- MACD: cruzamento com linha de sinal + força do histograma
- Ensemble Votação: combina SMA + RSI + MACD por votação ponderada
- Cada estratégia com explain() para justificar sinais

### Visualização Profissional
- 4 subplots: preço+sinais, volume, RSI, MACD (tema dark)
- Curva de equity com fill verde/vermelho
- Comparativo de estratégias (barras + tabela de métricas)
- Feature importance do modelo ML

### Paper Trading
- Dinheiro fictício com preços reais
- Modo automático (liga/desliga)
- Histórico de ordens e P&L em tempo real
- Persistência em JSON (salvar/retomar)

### Live Trading (base pronta)
- Adaptadores: Binance (crypto), Alpaca (ações US), Paper (simulação)
- Trava de segurança com limite de perda diária
- Autenticação por senha (PBKDF2 600k iterações)
- Criptografia AES-256 para API keys

### Segurança & LGPD
- Autenticação com hash seguro (PBKDF2-HMAC-SHA256)
- Criptografia de dados sensíveis (AES-256/Fernet)
- Audit trail imutável com checksums encadeados
- LGPD: consentimento, portabilidade, eliminação, notificação de breach
- Rate limiting contra abuso
- .gitignore protegendo todos os secrets

### Performance & Benchmarks
- Métricas semanais, mensais e anuais lado a lado
- Comparação com CDI, Ibovespa, S&P 500
- "Quanto $100k virariam em 1 ano"
- Taxa de acerto por ativo, mercado e nível de confiança
- Matriz de confusão dos sinais

---

## Uso

```bash
# Menu interativo
python main.py

# Análise completa
python main.py --analyze

# Análise apenas B3
python main.py --analyze --markets B3

# Paper Trading
python main.py --paper-trade --capital 50000

# Backtesting comparativo (todas as estratégias)
python run_backtest.py                    # Tickers padrão
python run_backtest.py AAPL TSLA NVDA     # Específicos
python run_backtest.py --all              # Todos os mercados

# Trading ao vivo (Alpaca)
cp .env.example .env                      # Preencha as chaves
python run_live.py                        # Menu interativo
python run_live.py --crypto               # Criptos 24/7
python run_live.py AAPL TSLA --stocks     # Ações específicas

# Testes
python main.py --test
python -m pytest tests/ -v --cov
```

---

## Referências Bibliográficas

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Krauss, C., Do, X. A., & Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*.
3. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
4. Sharpe, W. F. (1966). Mutual Fund Performance. *The Journal of Business*, 39(1), 119-138.
5. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
6. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
7. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv:1908.10063*.
8. Brasil (2018). Lei nº 13.709 (LGPD). Lei Geral de Proteção de Dados Pessoais.

---

## Roadmap

- [x] Coleta de dados com cache (yfinance)
- [x] Feature engineering (35+ indicadores)
- [x] Ensemble ML (RF + XGBoost + GBM)
- [x] Estratégias modulares (SMA, RSI, MACD, Ensemble Votação)
- [x] Análise de sentimento (FinBERT + léxico + RSS)
- [x] Gestão de risco (VaR, CVaR, position sizing)
- [x] Motor de backtesting com slippage
- [x] Comparativo automatizado de estratégias
- [x] Integração Alpaca (Paper/Live)
- [x] Visualização profissional (matplotlib dark theme)
- [x] Segurança completa (LGPD, AES-256, audit trail)
- [x] Performance tracker (semanal/mensal/anual)
- [x] Suíte de testes (170 testes)
- [ ] Dashboard React (frontend web)
- [ ] Documentação TCC completa

---

## Disclaimer

Este projeto é para fins acadêmicos e educacionais. Não constitui recomendação
de investimento. Resultados passados não garantem resultados futuros.
