# Motor de carteira sistemática

Pacote `carteira/`. Foi construído depois de medir dois sistemas que pareciam
funcionar e descobrir que nenhum dos dois funcionava.

## De onde ele veio

**O painel Sentinela** operava com book, tape e indicadores sintéticos, e o
simulador não cobrava taxa. Medido com dado real e custo real: 22.568
operações, resultado bruto de −1,5% e custo de 16,6 vezes o bruto. Ver
`AUDITORIA_SENTINELA.md`.

**O paper trading de FIIs do próprio QuantBot** marcava +17,14% contra CDI de
+2,86%. O accrual de provento era quadrático: a função lia `ultima_avaliacao`
para saber quantos dias tinham passado, mas nunca atualizava esse campo, então
relançava o rendimento de todos os dias desde a entrada a cada marcação. Em n
dias entravam n(n+1)/2 dias de provento. Em 61 dias, 14 vezes o devido.
Corrigido: **−6,32%**. Rode `python scripts/auditar_paper_book.py`.

Os dois erros são diferentes e a doença é a mesma: ninguém estava conferindo o
número contra a realidade.

## A tese

Prever preço no curto prazo não paga. O que paga é o que dá para controlar:

| O que se controla | Onde está |
|---|---|
| Custo de execução | `custos_b3.py` |
| Imposto de renda | `tributos.py` |
| Giro da carteira | `rebalanceamento.py` |
| Quanto se aporta | `aportes.py` |
| Se a estratégia tem evidência | `validacao.py` |

O imposto é o único dos cinco que é vantagem **mecânica**: a alíquota é lei,
não aposta. Duas carteiras com o mesmo retorno bruto e tributação diferente
entregam resultados diferentes todo ano, sem risco nenhum.

## Arquitetura

```
universo.py         ativos, classes e o regime tributário de cada um
dados.py            preços (Yahoo, ajustado) e CDI (BCB, fatiado em blocos)
tributos.py         motor de IR: isenções, tabela regressiva, compensação
custos_b3.py        emolumentos, spread, corretagem, taxa de administração
sinais.py           momentum absoluto e relativo, volatilidade, tendência
alocacao.py         pesos fixos, peso igual, inverso da vol, dual momentum
rebalanceamento.py  calendário, banda de tolerância, banda com custo
regras_pai.py       camada de risco herdada do painel Sentinela
backtest.py         motor com custo E imposto cobrados
validacao.py        walk-forward + Deflated Sharpe + PBO + comparação com CDI
aportes.py          quanto do patrimônio vem de estratégia e quanto de poupança
estudo.py           bateria comparativa
cli.py              linha de comando
```

## O que veio do painel do seu pai

As regras de gestão dele eram a melhor parte do sistema, e foram portadas
inteiras para `regras_pai.py`:

| Regra original | Como ficou |
|---|---|
| Máximo de 3 posições | `LimitesRisco.max_posicoes` |
| Uma posição por ativo | `avaliar_entrada` bloqueia duplicata |
| Risco de 2% por operação | `tamanho_por_risco` dimensiona pela distância até o stop |
| Teto de 30% por posição | `max_por_posicao_pct` |
| Filtro sniper 55/58/62 | `conviccao_exigida` sobe a cada posição aberta |
| Trailing em dois estágios | `atualizar_trailing`, agora medido em ATR |
| Guarda de reversão do BTC | `guarda_de_regime`, agora no índice de referência |
| Prazo máximo de operação | `max_pregoes_na_posicao` |
| Dupla chancela | `exigir_dupla_chancela` |

Duas regras dele foram descartadas de propósito: o **alvo fixo em dólares**
(que ignora tamanho e custo, e foi o que quebrou a conta) e o **fechamento a
qualquer divergência de sinal** (que numa carteira só gera custo e imposto).

A ideia central que ele acertou e que quase nenhum sistema de varejo tem: a
camada de risco é **independente** da estratégia. A estratégia sugere, o risco
decide se passa.

## Como usar

```bash
python -m carteira.cli aportes        # o que mais move o patrimônio
python -m carteira.cli custo          # custo por frequência de giro
python -m carteira.cli imposto        # quanto o giro custa em IR
python -m carteira.cli estudo         # compara todas as carteiras
python -m carteira.cli validar        # walk-forward + DSR + PBO
python -m carteira.cli hoje           # sinal do dia, explicado
python -m carteira.cli meta --meta 1500000
```

O robô diário é `scripts/monitor_carteira.py`, e o
`.github/workflows/monitor-carteira.yml` roda ele na nuvem todo dia útil às
18h30, com validação completa toda segunda-feira.

## O critério de aprovação

`validacao.veredito()` só libera papel quando **tudo** vale:

- pelo menos 3 janelas fora da amostra;
- retorno fora da amostra **acima do CDI** do mesmo período;
- pelo menos metade das janelas batendo o CDI;
- PBO ≤ 0,50;
- Deflated Sharpe ≥ 0,95.

`aprovado_para_dinheiro_real` é sempre `False` no código.

## O que a medição devolveu

Universo de 4 ETFs, 12 anos, 19 janelas fora da amostra (~9,4 anos):

| | |
|---|---|
| Estratégia (seleção por walk-forward) | **+163,4%** |
| CDI no mesmo trecho | **+131,1%** |
| Diferença | **+32,4 p.p. em 9,4 anos** (~3,4 p.p./ano) |
| Janelas que bateram o CDI | 11 de 19 |
| Sharpe fora da amostra | 0,66 |
| PBO | **0,00** (a seleção prevê alguma coisa) |
| Deflated Sharpe | **0,22** (reprovado) |

E o detalhe que mais importa: **entre 2021 e 2023, seis janelas seguidas
perderam para o CDI.** Três anos de decepção. É isso que acontece de verdade
com estratégia de bolsa em país de juro alto.

Veredito atual: **não aprovada**. Não porque seja ruim, mas porque um Sharpe
de 0,66 depois de 11 tentativas não se separa do acaso com a confiança que
justifica arriscar dinheiro de família.

## O que não muda

Nada aqui promete retorno. Um Deflated Sharpe de 0,95 não seria garantia de
lucro — seria apenas evidência de que o resultado não é ruído. E o pacote
inteiro parte de um resultado que já foi medido: **para quem tem pouco
capital, o aporte domina o retorno por uma margem que não é disputa.**
Aportar R$ 300 por mês a 0% ao ano supera 25% ao ano sobre R$ 1.000, em
qualquer horizonte acima de dois anos. Rode `python -m carteira.cli aportes`.
