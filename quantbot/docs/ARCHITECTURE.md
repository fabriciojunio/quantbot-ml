# Arquitetura do Sistema QuantBot ML

## Visão Geral

O QuantBot ML é um sistema modular de finanças quantitativas que utiliza
técnicas de Machine Learning para análise de investimentos em múltiplos mercados.

## Diagrama de Fluxo

```
                    ┌─────────────┐
                    │   main.py   │
                    │  (entrada)  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Análise  │ │  Paper   │ │  Testes  │
        │ Completa │ │ Trading  │ │  pytest  │
        └────┬─────┘ └────┬─────┘ └──────────┘
             │             │
             ▼             ▼
    ┌────────────────────────────────┐
    │      data/fetcher.py           │  ← Yahoo Finance API
    │   (coleta + cache + validação) │
    └────────────┬───────────────────┘
                 ▼
    ┌────────────────────────────────┐
    │      data/features.py          │  ← 35+ indicadores técnicos
    │    (feature engineering)       │
    └────────────┬───────────────────┘
                 ▼
    ┌────────────────────────────────┐
    │      models/trainer.py         │  ← TimeSeriesSplit CV
    │   (treino com validação)       │
    └────────────┬───────────────────┘
                 ▼
    ┌────────────────────────────────┐
    │      models/ensemble.py        │  ← RF + XGBoost + GB
    │    (ensemble de modelos)       │
    └────────────┬───────────────────┘
                 ▼
    ┌────────────────────────────────┐
    │      models/signals.py         │  ← Score 0-100 + Sinal
    │    (geração de sinais)         │
    └──────┬─────────────┬───────────┘
           ▼             ▼
    ┌──────────┐  ┌──────────────┐
    │ risk/    │  │  backtest/   │
    │manager.py│  │  engine.py   │
    │metrics.py│  │  report.py   │
    └──────────┘  └──────────────┘
```

## Princípios de Design

1. **Modularidade**: cada módulo tem responsabilidade única
2. **Imutabilidade**: configurações são `frozen=True`
3. **Validação**: inputs são validados na fronteira do sistema
4. **Logging**: todas as operações são registradas
5. **Testabilidade**: cada módulo tem testes independentes

## Decisões Técnicas

### Por que ensemble e não um único modelo?
Modelos individuais têm vieses diferentes. Random Forest é robusto mas
pode não capturar padrões sutis. XGBoost é mais preciso mas pode
overfittar. O ensemble combina os pontos fortes de ambos.

### Por que TimeSeriesSplit?
Em dados financeiros, a ordem temporal importa. Cross-validation
aleatória (K-Fold) causaria data leakage: o modelo veria dados
futuros durante o treino, inflando artificialmente as métricas.

### Por que normalizar features por preço?
Indicadores como MACD e SMA são valores absolutos que dependem
do preço do ativo. Um MACD de 5 em uma ação de $10 é muito
diferente de um MACD de 5 em uma ação de $1000. A normalização
permite comparabilidade entre ativos.

### Por que classificação binária e não regressão?
Prever o retorno exato é extremamente difícil e sujeito a ruído.
Classificação binária (sobe/desce) é mais robusta e suficiente
para gerar sinais acionáveis de trading.
