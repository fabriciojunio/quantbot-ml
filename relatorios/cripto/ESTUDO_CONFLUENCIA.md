# Estudo de viabilidade — motor de cripto

_Gerado em 2026-08-23T18:27:09 UTC, sobre 400 dias de histórico._

## Custo modelado

| Item | Valor |
|---|---|
| Taxa taker | 0.100% |
| Taxa maker | 0.100% |
| Spread médio | 0.020% |
| Custo de ida e volta em US$ 1.000 | 0.226% |

## Conclusão

| Métrica | Valor |
|---|---|
| Casos validados | 24 |
| Com retorno fora da amostra positivo | 2 |
| Aprovados para papel | 0 |
| Retorno fora da amostra mediano | -17.84% |
| Melhor caso | +5.93% |
| Pior caso | -63.29% |

**nenhuma combinação passou nos cortes de validação**

## Caso a caso

| Par | Tempo | Lados | Janelas | Retorno OOS | Sharpe OOS | Janelas + | Aprovado |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 15m | long | 34 | -40.06% | -10.40 | 1/34 | não |
| BTCUSDT | 15m | long+short | 34 | -63.12% | -12.54 | 0/34 | não |
| BTCUSDT | 1h | long | 15 | -18.93% | -4.70 | 1/15 | não |
| BTCUSDT | 1h | long+short | 15 | -26.25% | -4.15 | 3/15 | não |
| BTCUSDT | 4h | long | 6 | -3.54% | -1.35 | 0/6 | não |
| BTCUSDT | 4h | long+short | 6 | -4.07% | -0.81 | 1/6 | não |
| ETHUSDT | 15m | long | 34 | -38.73% | -7.24 | 1/34 | não |
| ETHUSDT | 15m | long+short | 34 | -63.29% | -8.91 | 1/34 | não |
| ETHUSDT | 1h | long | 15 | -12.88% | -2.29 | 3/15 | não |
| ETHUSDT | 1h | long+short | 15 | -16.74% | -1.78 | 4/15 | não |
| ETHUSDT | 4h | long | 6 | -8.43% | -2.13 | 0/6 | não |
| ETHUSDT | 4h | long+short | 6 | -5.96% | -0.89 | 2/6 | não |
| SOLUSDT | 15m | long | 34 | -38.14% | -6.51 | 5/34 | não |
| SOLUSDT | 15m | long+short | 34 | -55.31% | -6.58 | 4/34 | não |
| SOLUSDT | 1h | long | 15 | -16.64% | -2.56 | 2/15 | não |
| SOLUSDT | 1h | long+short | 15 | -32.73% | -3.58 | 2/15 | não |
| SOLUSDT | 4h | long | 6 | -1.78% | -0.43 | 3/6 | não |
| SOLUSDT | 4h | long+short | 6 | +5.93% | 0.86 | 3/6 | não |
| BNBUSDT | 15m | long | 34 | -34.52% | -7.72 | 2/34 | não |
| BNBUSDT | 15m | long+short | 34 | -58.28% | -9.36 | 0/34 | não |
| BNBUSDT | 1h | long | 15 | -11.11% | -2.48 | 2/15 | não |
| BNBUSDT | 1h | long+short | 15 | -24.55% | -3.28 | 2/15 | não |
| BNBUSDT | 4h | long | 6 | -1.93% | -0.75 | 3/6 | não |
| BNBUSDT | 4h | long+short | 6 | +2.35% | 0.52 | 4/6 | não |

## Onde o dinheiro foi, caso a caso

| Configuração | Operações | Bruto US$ | Custo US$ | Líquido US$ | Retorno | Buy and hold |
|---|---|---|---|---|---|---|
| BTC 15m long | 1381 | -29.0 | 527.4 | -556.3 | -55.6% | -34.4% |
| BTC 15m long+short | 2910 | -5.5 | 803.2 | -808.7 | -80.9% | -34.4% |
| BTC 1h long | 356 | -51.5 | 173.3 | -224.7 | -22.5% | -34.5% |
| BTC 1h long+short | 762 | -19.0 | 339.9 | -358.8 | -35.9% | -34.5% |
| BTC 4h long | 86 | -31.7 | 45.5 | -77.2 | -7.7% | -34.4% |
| BTC 4h long+short | 186 | +0.8 | 99.0 | -98.2 | -9.8% | -34.4% |
| BTC 1d long | 10 | -23.0 | 5.4 | -28.4 | -2.8% | -34.0% |
| BTC 1d long+short | 35 | +19.8 | 19.4 | +0.4 | +0.0% | -34.0% |
| ETH 15m long | 1408 | -1.7 | 541.6 | -543.3 | -54.3% | -31.1% |
| ETH 15m long+short | 2880 | +58.8 | 839.6 | -780.8 | -78.1% | -31.1% |
| ETH 1h long | 347 | -0.2 | 174.2 | -174.4 | -17.4% | -31.6% |
| ETH 1h long+short | 716 | +125.3 | 349.8 | -224.5 | -22.4% | -31.6% |
| ETH 4h long | 80 | -20.3 | 43.3 | -63.6 | -6.4% | -31.8% |
| ETH 4h long+short | 185 | +14.6 | 98.6 | -84.1 | -8.4% | -31.8% |
| ETH 1d long | 15 | -86.7 | 7.7 | -94.4 | -9.4% | -34.8% |
| ETH 1d long+short | 30 | -77.1 | 15.1 | -92.2 | -9.2% | -34.8% |
| SOL 15m long | 1397 | -73.1 | 522.0 | -595.1 | -59.5% | -46.1% |
| SOL 15m long+short | 2795 | +22.0 | 819.2 | -797.2 | -79.7% | -46.1% |
| SOL 1h long | 333 | +45.5 | 175.3 | -129.7 | -13.0% | -46.2% |
| SOL 1h long+short | 708 | +96.5 | 357.0 | -260.5 | -26.1% | -46.2% |
| SOL 4h long | 86 | -50.2 | 45.4 | -95.5 | -9.6% | -46.2% |
| SOL 4h long+short | 182 | -70.9 | 91.7 | -162.6 | -16.3% | -46.2% |
| SOL 1d long | 14 | -68.1 | 7.5 | -75.6 | -7.6% | -47.5% |
| SOL 1d long+short | 29 | -18.5 | 15.6 | -34.1 | -3.4% | -47.5% |
| BNB 15m long | 1362 | +1.1 | 532.3 | -531.2 | -53.1% | -4.2% |
| BNB 15m long+short | 2817 | -48.8 | 756.6 | -805.4 | -80.5% | -4.2% |
| BNB 1h long | 384 | -52.9 | 186.7 | -239.6 | -24.0% | -4.7% |
| BNB 1h long+short | 760 | -84.7 | 326.2 | -410.9 | -41.1% | -4.7% |
| BNB 4h long | 95 | -18.9 | 51.8 | -70.7 | -7.1% | -4.6% |
| BNB 4h long+short | 164 | +63.6 | 90.8 | -27.2 | -2.7% | -4.6% |
| BNB 1d long | 20 | -41.2 | 11.0 | -52.3 | -5.2% | -7.6% |
| BNB 1d long+short | 35 | -61.3 | 19.2 | -80.5 | -8.1% | -7.6% |

### Somando tudo

| Item | Valor |
|---|---|
| Capital somado | US$ 32,000 |
| Operações | 22,568 |
| Resultado bruto (antes do atrito) | US$ -486 |
| Custo de execução | US$ -8,091 |
| **Resultado líquido** | **US$ -8,578** |
| Bruto sobre o capital | -1.52% |
| Custo dividido pelo bruto | 16.6x |

O bruto perto de zero diz que a estratégia não tem direção: ela acerta e
erra como cara ou coroa. O custo diz o resto. Cara ou coroa com pedágio em
cada lançamento, repetido milhares de vezes, tem um único desfecho.
