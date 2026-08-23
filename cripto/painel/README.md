# Ligando o painel Sentinela ao motor real

O painel (React + Vite, pasta `cryptoforum-sentiment`) continua sendo a
interface. O que muda é de onde vêm os números: em vez de `Math.random()` e de
textos de reserva do Gemini, eles passam a vir do motor `cripto/`.

## 1. Subir o motor

```bash
cd quantbot_work
uvicorn cripto.api:app --port 8100
```

Confira: `curl http://localhost:8100/api/cripto/saude`

A API é **só leitura**. Nenhuma rota abre, fecha ou altera posição — ordem só
acontece pelo CLI (`python -m cripto.cli papel`), e ainda assim em papel.

## 2. Copiar a ponte

Copie `quantbotService.ts` para `src/services/` do painel e, no `.env` do
painel, aponte a URL:

```
VITE_QUANTBOT_URL=http://localhost:8100
```

## 3. Trocar as fontes, uma de cada vez

Ordem sugerida — da que mais engana para a que menos engana:

| Passo | Arquivo do painel | O que fazer |
|---|---|---|
| 1 | `services/hftConfluenceService.ts` | Trocar `generateLocalHFTConfluenceAnalysis` por `buscarSinal()`. É a troca mais importante: é ela que tira o `isBullish = change24h >= 0` do caminho da decisão. |
| 2 | `services/orderFlowDataService.ts` | Trocar `generateLiveOrderFlowData` por `buscarBook()`. |
| 3 | `services/cryptoTapeDbService.ts` | Trocar a geração de negócios por `buscarTape()`. |
| 4 | `services/tradingExecutionService.ts` | Trocar a leitura do `localStorage` por `buscarCarteiraPapel()`. O painel passa a mostrar o book de papel do motor, que paga custo. |
| 5 | `components/SystemAuditModule.tsx` | Remover as latências e as "428 fontes" fabricadas; usar `/api/cripto/saude`, que diz quais fontes existem de verdade. |
| 6 | `server.ts` | Nos `generateFallback*`, incluir um campo `fallback: true` na resposta e marcar isso na tela. Texto de reserva pode existir; passar por análise de IA, não. |

## 4. Regras que a interface precisa respeitar

**Dado ausente é dado ausente.** Toda função da ponte devolve `null` quando o
motor não responde. O componente tem que mostrar "sem dados" ou um estado de
carregamento. Preencher com valor plausível foi o erro que fez a versão
anterior parecer confiável sem ser.

**Mostre a procedência.** Todo card que exibe número de mercado deveria trazer
a fonte (`fonte: "binance"`) e o horário. É barato de fazer e muda a relação de
confiança com o painel.

**Mostre o custo junto do alvo.** Onde o painel deixa configurar alvo em
dólares, chame `buscarContaDeCusto()` e mostre `resultado_liquido_usd` ao lado.
Se der negativo, o botão de confirmar deveria ficar desabilitado. É a trava que
teria evitado o alvo de dez centavos.

**Nada de botão de ordem real.** Enquanto o veredito de validação não aprovar
(hoje não aprova), não deve existir caminho na interface que envie ordem para
corretora. No motor, essa trava é dupla: `aprovado_para_dinheiro_real` é sempre
`False` e `core/live_trading.py` exige `QUANTBOT_ALLOW_LIVE=1` no ambiente.

## 5. O que o painel ganha com a troca

- Os oito indicadores viram oito indicadores de verdade, e o card de score pode
  abrir a contribuição de cada um (`indicadores` no retorno de `buscarSinal`).
- O book mostra a parede que existe agora, não uma sorteada.
- O paper trading passa a doer como o real, porque paga taxa e spread — que é a
  única forma de o simulado ter alguma serventia.
- O painel deixa de afirmar coisas que não pode sustentar. Para um produto que
  eventualmente vá para a mão de outra pessoa, isso deixa de ser detalhe
  técnico e passa a ser responsabilidade.
