# QuantBot ML — Sistema de Finanças Quantitativas com Machine Learning

**TCC em Ciência da Computação — UNISAGRADO**

Sistema completo de análise quantitativa de investimentos utilizando Machine Learning para geração de sinais de compra/venda, análise de sentimento de notícias, otimização de portfólio, gestão de risco e operação automatizada.

**Mercados:** B3 (10 ativos) | NYSE/NASDAQ (10 ativos) | Cripto (5 ativos)

---

## Números do Projeto

| Métrica | Valor |
|---|---|
| Arquivos Python | 56 |
| Linhas de código | 11.400+ |
| Testes automatizados | 206 |
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
git clone https://github.com/fabriciojunio/quantbot-ml.git
cd quantbot-ml/quantbot
pip install -r requirements.txt
python -m pytest tests/ -v        # Valida tudo (206 testes)
python main.py                    # Menu interativo
```

---

## Arquitetura

```
quantbot/
├── config/
│   └── settings.py             Configurações centralizadas (enums, perfis de risco, parâmetros ML)
├── data/
│   ├── fetcher.py              Coleta (Yahoo Finance + cache + rate limiting)
│   ├── features.py             35+ indicadores técnicos
│   ├── validators.py           Validação de qualidade dos dados
│   ├── news_fetcher.py         RSS feeds (5 fontes gratuitas)
│   ├── sentiment.py            FinBERT + léxico financeiro PT/EN
│   ├── macro_data.py           Dados BCB, Monte Carlo, pesos ótimos
│   └── cusum_filter.py         Filtro CUSUM (López de Prado)
├── models/
│   ├── ensemble.py             RF + XGBoost + GradientBoosting
│   ├── trainer.py              TimeSeriesSplit (5 folds)
│   ├── signals.py              Score 0–100 + confiança + feature importance
│   ├── regime.py               Detecção de regime (bull/bear/lateral)
│   ├── triple_barrier.py       Labeling por tripla barreira
│   └── walk_forward.py         Walk-forward validation
├── strategies/
│   ├── base.py                 Classe abstrata + mapeamento numérico
│   ├── sma_crossover.py        Cruzamento de médias móveis 20/50
│   ├── rsi_strategy.py         RSI sobrecompra/sobrevenda
│   ├── macd_strategy.py        MACD cruzamento de sinal
│   └── ensemble_voting.py      Votação ponderada SMA+RSI+MACD
├── risk/
│   ├── manager.py              Position sizing + stop-loss + take-profit
│   ├── metrics.py              VaR, CVaR, Sharpe, Sortino, Beta, Alpha
│   └── dynamic_stop.py         Stop dinâmico com trailing e breakeven
├── backtest/
│   ├── engine.py               Simulação com comissão + slippage
│   ├── benchmarks.py           CDI, Ibovespa, S&P 500
│   └── report.py               Relatórios visuais PNG
├── core/
│   ├── paper_trading.py        Dinheiro fictício com preços reais
│   ├── live_trading.py         AlpacaAdapter + SafetyMonitor
│   ├── accuracy.py             Taxa de acerto por ativo/mercado
│   ├── performance.py          Semanal / Mensal / Anual
│   └── security.py             AES-256, LGPD, Audit Trail
├── visualization/
│   └── __init__.py             4 tipos de gráficos (tema dark dourado)
├── utils/
│   ├── logger.py               Logging auditável
│   ├── security.py             Sanitização e validação de inputs
│   └── formatters.py           Formatação de moeda e percentual
├── tests/                      206 testes automatizados (100% passing)
├── main.py                     Ponto de entrada principal
├── run_backtest.py             Backtesting comparativo de estratégias
├── run_live.py                 Trading ao vivo via Alpaca
├── dashboard.jsx               Dashboard React
├── .env.example                Template de configuração
└── requirements.txt            Dependências
```

---

## Fluxo de Dados

```
yfinance / RSS Feeds
        │
        ▼
 MarketDataFetcher          coleta OHLCV + cache + rate limiting
        │
        ▼
  FeatureEngineer           35+ indicadores técnicos
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
  SignalGenerator            Score 0–100 + sinal + confiança
        │
        ▼
   RiskManager               Tamanho de posição, stop-loss, take-profit
        │
     ┌──┴──────────────┐
     ▼                 ▼
BacktestEngine    PaperTrading / Alpaca Live
     │
     ▼
ReportGenerator + Visualization
```

---

## Funcionalidades

### Machine Learning
- Ensemble de 3 modelos: Random Forest, XGBoost, Gradient Boosting
- 35+ features técnicas (RSI, MACD, Bollinger, ATR, momentum, OBV, estocástico…)
- Validação temporal com TimeSeriesSplit (5 folds) — sem data leakage
- Walk-forward optimization com detecção automática de overfitting
- Score 0–100 com nível de confiança e concordância entre modelos
- Labeling por tripla barreira (López de Prado, Cap. 3)

### Estratégias Técnicas Modulares
- **SMA Crossover** — compra quando SMA(20) cruza acima de SMA(50)
- **RSI** — compra em sobrevendido (<30), vende em sobrecomprado (>70), sinais fortes nos extremos (<20 / >80)
- **MACD** — cruzamento com linha de sinal + força do histograma
- **Ensemble Votação** — SMA (×1.0) + RSI (×1.0) + MACD (×1.2) com votação ponderada
- Cada estratégia com `explain()` para justificar o sinal gerado

### Análise de Sentimento
- 5 feeds RSS gratuitos (Google News, Yahoo Finance, CoinDesk, Investing.com, InfoMoney)
- FinBERT (deep learning) com fallback para léxico financeiro PT/EN
- Detecção automática de ativos mencionados nas notícias
- Filtro CUSUM para eventos relevantes de mercado

### Gestão de Risco

| Perfil | Stop-Loss | Take-Profit | Posição Máx |
|---|---|---|---|
| Conservative | 3% | 9% | 5% |
| Moderate | 5% | 15% | 10% |
| Aggressive | 7% | 21% | 20% |

- Position sizing baseado em volatilidade e confiança do sinal
- Stop dinâmico com trailing stop, breakeven e saída por tempo
- VaR 95/99%, CVaR, Beta, Alpha, Sharpe, Sortino
- Limite de perda diária — bot desliga automaticamente

### Backtesting
- Simulação realista com comissão e slippage
- Métricas: Sharpe, Sortino, Calmar, Win Rate, Profit Factor, Max Drawdown
- Comparação automática com CDI, Ibovespa e S&P 500
- Gráficos profissionais salvos em `results/`

### Paper Trading
- Dinheiro fictício com preços reais via yfinance
- Comandos interativos: `buy`, `sell`, `portfolio`, `auto`, `save`, `load`
- Modo automático integrado com os sinais de ML
- Persistência em JSON

### Live Trading (Alpaca)
- Suporte a ações US e cripto 24/7
- Stop-loss e take-profit automáticos por posição
- SafetyMonitor: limite de exposição, reserva de caixa, máximo de trades diários
- Verificação de horário de mercado com fallback por horário ET

### Visualização
- Tema dark preto & dourado
- Preço + sinais de compra/venda + Volume + RSI + MACD (4 subplots)
- Curva de equity com fill verde/vermelho vs capital inicial
- Comparativo de estratégias (barras + tabela de métricas)
- Feature importance do modelo ML

### Segurança & LGPD
- Autenticação PBKDF2-HMAC-SHA256 (600k iterações)
- Criptografia AES-256 para API keys e dados sensíveis
- Audit trail imutável com checksums encadeados
- LGPD: consentimento, portabilidade, eliminação de dados, notificação de breach
- Sanitização e validação de todos os inputs externos
- Rate limiting contra abuso

---

## Como Usar

```bash
# Menu interativo
python main.py

# Análise completa (ML + Backtest + Sinais)
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
cp .env.example .env                      # Preencha ALPACA_API_KEY e ALPACA_SECRET_KEY
python run_live.py                        # Menu interativo
python run_live.py --crypto               # Criptos 24/7
python run_live.py --stocks               # Ações EUA

# Testes
python -m pytest tests/ -v               # 206 testes
python -m pytest tests/ -v --cov=.       # Com cobertura
```

---

## Configuração `.env`

```env
ALPACA_API_KEY=sua_chave_aqui
ALPACA_SECRET_KEY=sua_secret_aqui
ALPACA_ENV=paper          # paper ou live — SEMPRE comece com paper
```

---

## Roadmap

### v1.0 — Concluído
- [x] Pipeline ML completo (coleta → features → treino → sinal)
- [x] 4 estratégias técnicas modulares + ensemble voting
- [x] Motor de backtesting com custos realistas
- [x] Paper trading interativo
- [x] Live trading via Alpaca (paper + real)
- [x] Análise de sentimento (FinBERT + léxico)
- [x] Dados macroeconômicos (BCB) e correlações
- [x] Walk-forward validation
- [x] Tripla barreira (labeling avançado)
- [x] Detecção de regime de mercado
- [x] Métricas de risco (VaR, CVaR, Sharpe, Sortino, Beta, Alpha)
- [x] Comparação com benchmarks (CDI, Ibovespa, S&P 500)
- [x] Segurança LGPD + auditoria + AES-256
- [x] 206 testes automatizados (100% passing)

### v1.1 — Próximos Passos
- [ ] Dashboard web React integrado com API Flask/FastAPI
- [ ] Alertas por Telegram/email em sinais relevantes
- [ ] Suporte a Binance para cripto ao vivo
- [ ] Otimização de hiperparâmetros com Optuna

### v2.0 — Visão de Longo Prazo
- [ ] Modelos de séries temporais (LSTM, Transformer financeiro)
- [ ] Multi-asset portfolio optimization (Markowitz + HRP)
- [ ] Deploy em nuvem com scheduler automático
- [ ] API pública com autenticação JWT

---

## Referências Bibliográficas

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Krauss, C., Do, X. A., & Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*.
3. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77–91.
4. Sharpe, W. F. (1966). Mutual Fund Performance. *The Journal of Business*, 39(1), 119–138.
5. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
6. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
7. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv:1908.10063*.
8. Brasil (2018). Lei nº 13.709 (LGPD). Lei Geral de Proteção de Dados Pessoais.

---

> **Disclaimer:** Este projeto é para fins acadêmicos e educacionais. Não constitui recomendação de investimento. Resultados passados não garantem resultados futuros.
