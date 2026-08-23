# Motor de cripto

Pacote `cripto/`. Existe para responder, com dado real e custo real, se uma
estratégia de confluência técnica tem vantagem em cripto — e para servir de
base honesta caso alguma versão futura tenha.

## Por que ele foi escrito

O ponto de partida foi o painel **Sentinela Cripto** (o app em React que veio
do AI Studio). O painel tem uma interface muito boa e uma ideia de produto
clara: vários indicadores votam, o voto vira score, o score vira ordem. O
problema estava embaixo da interface — ver `AUDITORIA_SENTINELA.md`. Resumo do
que faltava e que este pacote entrega:

| O que o painel fazia | O que o `cripto/` faz |
|---|---|
| Book, times & trades e on-chain vindos de `Math.random()` | Book e tape reais, via REST público de Binance/Gate.io/Bybit |
| "8 indicadores técnicos" derivados de `change24h >= 0` | RSI, MACD, EMAs, Bollinger, ATR, ADX, estocástico e OBV calculados de OHLCV |
| Simulador sem taxa, sem spread, sem slippage | Custo cobrado em toda entrada e toda saída (`custos.py`) |
| Alvo de +US$ 0,10 por operação | O mesmo alvo, medido contra o custo: reprovado em todo tamanho de posição |
| Nenhuma validação fora da amostra | Walk-forward + Deflated Sharpe + PBO (`walkforward.py`) |
| Saldo no `localStorage` do navegador | Book de paper trading em disco, com camada de risco e log de decisão |

## Arquitetura

```
exchanges/          REST público, um cliente por corretora
  base.py           contrato, validação de símbolo/URL, http com timeout e backoff
  binance.py        fonte primária (histórico mais longo, book mais fundo)
  gate.py           primeiro fallback (formato de candle com colunas fora de ordem)
  bybit.py          segundo fallback (candles vêm do mais novo para o mais antigo)
  roteador.py       tenta em ordem; se todas falham, LEVANTA em vez de inventar
mercado.py          download paginado, cache em CSV, validação de série
indicadores.py      indicadores de OHLCV, sem preenchimento para trás
custos.py           taxa + meio spread + slippage; o módulo que decide tudo
sinais.py           score de confluência 0–100 com pesos declarados
backtest.py         motor event-driven; sinal em t, execução na abertura de t+1
walkforward.py      validação purgada, Deflated Sharpe, PBO, veredito
estudo.py           bateria em vários pares e tempos gráficos
papel.py            paper trading com book real, custo real e camada de risco
relatorio.py        Markdown com o custo sempre visível
api.py              FastAPI só-leitura para alimentar o painel
cli.py              linha de comando
```

Reaproveita do QuantBot que já existia:

- `validation/` — `PurgedKFold`, `evaluate_strategy` (DSR/PSR), `probability_of_backtest_overfitting`
- `risk/kill_switch.py` — `RiskManager` como portão independente de toda ordem
- `core/decision_logger.py` — trilha auditável de cada decisão, com o score que a motivou

## Como usar

```bash
# a conta do alvo de dez centavos
python -m cripto.cli custo

# preço, spread e desequilíbrio de book agora
python -m cripto.cli precos --simbolos BTCUSDT,ETHUSDT

# backtest com custo, em Markdown
python -m cripto.cli backtest BTCUSDT --tempo 1h --dias 400 --markdown

# validação fora da amostra (é isto que decide se vale operar)
python -m cripto.cli validar BTCUSDT --tempo 1h --dias 400

# bateria completa em vários pares e tempos
python -m cripto.cli estudo --dias 400 --markdown

# score da barra atual, aberto indicador por indicador
python -m cripto.cli sinal BTCUSDT --tempo 4h

# um ciclo de paper trading
python -m cripto.cli papel --simbolos BTCUSDT,ETHUSDT

# API para o painel
uvicorn cripto.api:app --port 8100
```

## As três regras que o motor não quebra

**1. Nada de olhar o futuro.** O score da barra `t` usa só informação até o
fechamento de `t`, e a ordem executa na abertura de `t+1`. O teste
`test_cripto_backtest.py::TestSemOlharOFuturo` verifica isso por truncamento:
acrescentar barras futuras à série não pode mudar em nada o resultado das
barras anteriores. Um backtest que falha nesse teste produz curva perfeita e
prejuízo real.

**2. Custo em toda ponta.** Entrada e saída pagam taxa (`taker` por padrão),
meio spread e slippage proporcional à raiz da participação no book. Quando
stop e alvo caem na mesma barra, o motor assume que o stop veio primeiro. Todo
resultado publicado é líquido.

**3. Número sem procedência não entra.** Se a exchange não responde, o
roteador levanta erro. Não existe valor sintético "para não quebrar o
gráfico" — foi exatamente isso que fez o painel antigo parecer confiável sem
ser.

## O critério de aprovação

`walkforward.veredito()` só libera paper trading quando **tudo** vale:

- pelo menos 3 janelas fora da amostra;
- Sharpe fora da amostra ≥ 1,0;
- retorno fora da amostra positivo;
- PBO ≤ 0,50 (escolher pelo passado precisa prever alguma coisa);
- Deflated Sharpe ≥ 0,95 (o resultado se separa do acaso dado o nº de tentativas);
- pelo menos metade das janelas fora da amostra com lucro.

`aprovado_para_dinheiro_real` é **sempre `False`** no código. Passar da fase de
papel é decisão humana, tomada depois de meses de paper trading e nunca por
uma flag automática.

## O que este motor não é

- Não é previsão. Indicador descreve o passado recente; o futuro do preço
  continua desconhecido.
- Não é garantia. Passar na validação significa "não há evidência de que seja
  ruído" — não significa que vá dar lucro.
- Não é conselho de investimento. É ferramenta de medição.

## Estado atual

Rode `python -m cripto.cli estudo --dias 400 --markdown` para o número do dia.
O resultado da primeira bateria completa está em
`data/cripto/estudos/` e resumido em `AUDITORIA_SENTINELA.md`.
