# Auditoria do painel Sentinela Cripto

Leitura completa do código de `cryptoforum-sentiment` / `FABRICIO-PAULO-ALVES-DIAS-main`
(os dois arquivos `.zip` são o mesmo projeto: 102 arquivos, ~1,5 MB de TypeScript,
gerado no Google AI Studio, front em React 19 + Vite e um `server.ts` de 124 KB
em Express que faz proxy para o Gemini).

Esta auditoria olha o código, não a pessoa. As ideias de mercado que estão ali
— confluência de indicadores, leitura de fluxo, absorção no book, delta de
agressão, trailing por estágios, risco fixo por operação — são ideias de quem
opera de verdade e sabe do que está falando. O problema está inteiro na camada
de implementação: o gerador escreveu a **aparência** de cada uma dessas coisas
em vez da coisa.

---

## 1. O que é real no painel

Vale começar pelo que funciona, porque não é pouco.

**Preço à vista é real.** `server.ts:337` roda um worker a cada 3 segundos que
busca `https://api.gateio.ws/api/v4/spot/tickers` e monta preço, variação de
24 h e volume de ~15 moedas. Tem fallback para Coinbase e Kraken, e a cotação
USD/BRL vem de `open.er-api.com`. Essa parte é bem feita: timeout por
requisição, fallback em cascata, cache em memória, resposta instantânea para o
front. **É a fundação que o motor novo aproveitou.**

**A interface é boa.** Navegação, hierarquia visual, densidade de informação,
PWA com manifest, tema escuro consistente. Como produto, o painel comunica bem.

**As regras de gestão existem e são coerentes.** Máximo de 3 posições
simultâneas, uma por ativo, risco de 2% por operação, teto de 30% do saldo por
posição, filtro de score adaptativo (55% na primeira ordem, 58% na segunda,
62% na terceira), trailing em dois estágios, guarda de reversão do BTC. Isso é
desenho de quem já levou stop na vida.

---

## 2. O que não é real

### 2.1 O livro de ofertas é sorteado

`src/services/orderFlowDataService.ts` gera bid, ask, tamanho de cada nível e
as "paredes" com `Math.random()`:

```ts
const baseAmount = (basePrice > 1000 ? 0.4 : basePrice > 50 ? 12 : 250)
                   * (1 + Math.random() * 1.5) * wallMultiplier;
ordersCount: Math.floor(Math.random() * 18 + 3) * (isBigWall ? 4 : 1),
```

Toda a leitura de absorção, desequilíbrio e parede que aparece no
`OrderBookAndTradesPanel` (49 KB) está lendo esses números sorteados.

### 2.2 O times & trades é sorteado

`src/services/cryptoTapeDbService.ts` inventa cada negócio — inclusive o rótulo
de "baleia" e o tipo de execução:

```ts
const isBuy = Math.random() < buyProbability;
const isWhale = Math.random() < 0.14;              // ~14% de blocos "institucionais"
else if (Math.random() < 0.2) tradeType = 'Varredura de Liquidez';
```

O componente `SingleCryptoTimesAndTrades` tem 98 KB dedicados a exibir esse
fluxo. Nenhum daqueles negócios aconteceu.

### 2.3 Os "8 indicadores técnicos" não são indicadores

Este é o achado mais importante, porque é o que decide as ordens.
`src/services/hftConfluenceService.ts:260`:

```ts
const isBullish = change24h >= 0;
```

E, a partir desse único booleano, saem os oito "indicadores":

```ts
const rsiValue = ... ?? (isBullish ? +(58 + change24h * 1.2) : +(42 + change24h * 1.1));
const macdScore = macdSignal === 'COMPRA' ? (isBullish ? 88 : 72) : 32;
const emaScore  = emaStatus === 'alta_perfeita' ? 95 : ...;
const adxValue  = isBullish ? 32.8 : 22.4;
valueFormatted: macdState === 'positivo' ? '+1.42 (Histograma Expansivo)' : '-0.95 (Histograma Defensivo)',
```

RSI, MACD, EMAs, Bollinger, estocástico, SuperTrend, OBV e ATR são todos
funções da mesma pergunta: *a moeda subiu nas últimas 24 h?* O "+1.42" do
histograma do MACD é uma string fixa. O ADX é 32,8 quando sobe e 22,4 quando
cai, sempre.

**Consequência prática:** o "Score Master Ponderado 14% / 86%" — 14% de
inteligência e sentimento, 86% de análise técnica — colapsa em
`change24h >= 0`. O bot compra o que subiu nas últimas 24 horas e vende o que
caiu, com passos extras que apenas somam ruído. Não existe confluência: existe
um indicador só, disfarçado de oito.

### 2.4 O on-chain é uma tabela escrita à mão

`src/data/onChainHistoryData.ts` tem 83 KB de dados de fluxo de exchange,
carteiras de baleia e reservas — todos literais no arquivo, com
`isBullish = crypto.change24h >= 0` na linha 1505 modulando o resto.

### 2.5 Os fóruns são fixos

`src/data/mockForumsData.ts` (29 KB) são posts inventados. O
`systemAuditService.ts` chega a informar, com número e tudo:

```ts
details: `Varredura de 428 fontes (Binance Square, Reddit, X, 4chan, Telegram).
          ${forumCount} posts em memória verificados.`
sourceConfidence: Math.round(98.5 + Math.random() * 1.4),
```

Nenhuma das 428 fontes é consultada. O módulo que deveria auditar o sistema é
ele próprio uma simulação, com latências fabricadas por
`performance.now() + Math.random() * 6`.

### 2.6 Quando o Gemini falha, o painel inventa e não avisa

`server.ts:90`, `logAndGetFallback`. Se a API do Gemini devolve 429 (cota), 503,
ou se a chave não está configurada, o servidor devolve um texto pré-escrito
**com a mesma forma da resposta real**. Um exemplo literal do arquivo:

```ts
if (lastMsg.includes('sol') || lastMsg.includes('solana')) {
  return "Para a **Solana ($SOL)** ... o livro de ordens apresenta suporte em
          US$ 210,00 e resistência principal na região de US$ 235,00 a US$ 250,00.";
}
```

Preço de suporte e resistência escritos à mão, servidos como análise de IA, sem
nenhuma marca na interface dizendo que aquilo é um texto de reserva. Esse é o
ponto mais grave da auditoria: não é um dado faltando, é um dado **falso
apresentado como verdadeiro**.

---

## 3. O "bot que roda" é o navegador conversando consigo mesmo

`src/services/tradingExecutionService.ts` (48 KB):

```ts
const STORAGE_KEY_POSITIONS = 'hft_demo_positions';
const STORAGE_KEY_ACCOUNT   = 'hft_demo_account';
const defaultAcc = { demoBalanceUsd: 10000, ... }
```

Posições e saldo vivem no `localStorage`. Não existe chave de exchange, não
existe assinatura de requisição, não existe rota de ordem em lugar nenhum do
`server.ts`. **Nenhuma ordem sai do navegador.** É um simulador — e isso, em si,
é a decisão certa. O problema é que o simulador não cobra nada:

- não há taxa de corretora;
- não há spread (entra e sai no mesmo preço de tela);
- não há slippage;
- o resultado é `sizeUsd * variação%`, direto.

E a regra de saída é `unrealizedPnlUsd >= 0.10`: fecha a posição assim que o
lucro não realizado bate dez centavos de dólar.

### A conta que reprova a estratégia

Taxa spot padrão da Binance: 0,10% por ponta. Somando spread e slippage, o
custo de abrir e fechar fica assim (saída de `python -m cripto.cli custo`):

| Posição | Custo ida e volta | Sobra do alvo de US$ 0,10 | Custo / alvo |
|---|---|---|---|
| US$ 50 | US$ 0,1107 | −US$ 0,0107 | 1,1× |
| US$ 100 | US$ 0,2220 | −US$ 0,1220 | 2,2× |
| US$ 200 | US$ 0,4457 | −US$ 0,3457 | 4,5× |
| US$ 500 | US$ 1,1224 | −US$ 1,0224 | 11,2× |
| US$ 1.000 | US$ 2,2632 | −US$ 2,1632 | 22,6× |
| US$ 3.000 | US$ 6,9286 | −US$ 6,8286 | 69,3× |

O tamanho padrão de posição do painel é `(saldo × 2%) / 3,5% ≈ 57% do saldo`,
limitado a 30% — em US$ 10.000, cerca de US$ 3.000. Nessa posição, o alvo de
dez centavos precisaria render **69 vezes mais** só para pagar o custo. Não
existe taxa de acerto que resolva: mesmo acertando 100% das operações, a conta
fecha no vermelho.

Isso não é opinião sobre a estratégia. É aritmética sobre a tabela de preços da
corretora.

### Por que a curva parecia boa

No simulador sem custo, um alvo pequeno tem probabilidade altíssima de ser
tocado antes do stop. O resultado é muitos ganhos minúsculos e perdas raras e
grandes — a curva sobe quase reta e o "win rate" fica bonito. É o formato
clássico de estratégia que só existe porque o atrito foi ignorado. Assim que a
taxa entra, cada uma daquelas vitórias de dez centavos vira prejuízo de vinte a
setenta centavos.

---

## 4. Os R$ 100 → R$ 165

Os R$ 165 vieram da operação **manual** do seu pai, numa plataforma de verdade,
com o dinheiro dele e o julgamento dele. Isso é um resultado real e não tem
relação com o painel: o painel nunca enviou uma ordem.

Duas observações honestas sobre esse número:

1. **+65% é um ótimo resultado e uma amostra pequena.** Com poucas operações,
   sorte e habilidade produzem a mesma curva. O que separa as duas é tempo e
   registro. Vale muito a pena começar a anotar toda operação — data, par,
   entrada, saída, motivo, tamanho — porque em três meses esse caderno vale
   mais que qualquer painel.
2. **Escala muda o jogo.** R$ 100 rendendo 65% dá R$ 65. A mesma habilidade em
   R$ 10.000 encontra spread pior, execução pior e, principalmente, a pressão
   psicológica de perder um salário num dia ruim. A conta não escala linear.

---

## 5. O que foi construído a partir daqui

O pacote `cripto/` (ver `CRIPTO.md`) mantém a ideia do painel e troca a
fundação:

- book, tape e candles reais, com a fonte declarada em cada resposta;
- indicadores calculados de OHLCV de verdade;
- custo cobrado em toda ponta, sempre;
- backtest sem olhar o futuro, verificado por teste de truncamento;
- walk-forward com Deflated Sharpe e PBO antes de qualquer aprovação;
- paper trading em disco, com camada de risco independente e log auditável;
- API só-leitura que devolve `sintetico: false` em cada campo, para que a
  interface possa mostrar a procedência do número.

O resultado da primeira bateria de validação está em `data/cripto/estudos/` e
no relatório final. Adianto a parte que importa: **a estratégia de confluência,
medida com custo real, não mostrou vantagem.** Isso não é um defeito do
trabalho do seu pai — é o que acontece com quase toda estratégia técnica de
curto prazo quando o atrito entra na conta. Saber disso antes de arriscar
dinheiro é exatamente para que a validação existe.
