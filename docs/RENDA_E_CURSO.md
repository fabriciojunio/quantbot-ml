# Dá para viver disso? E dá para vender curso?

Documento de decisão, escrito com os números que o motor `cripto/` mediu e com
o que a pesquisa pública mostra. Nada aqui é conselho de investimento — é a
apresentação honesta das contas para que a decisão seja de vocês.

---

## Parte 1 — A estratégia do painel dá dinheiro?

**Não, e a resposta não depende de opinião.**

### 1.1 O alvo de dez centavos é aritmeticamente impossível

Taxa spot da Binance: 0,10% por ponta. Com spread e slippage, abrir e fechar
uma posição de US$ 1.000 custa **US$ 2,26**. A regra do painel fechava a
operação em **+US$ 0,10**. O custo é 22 vezes o alvo. Numa posição de
US$ 3.000 (o tamanho padrão da configuração), 69 vezes.

Não existe taxa de acerto que resolva isso. Acertando 100% das operações, a
conta ainda fecha no vermelho. Rode `python -m cripto.cli custo` para ver a
tabela.

### 1.2 A confluência de indicadores não mostrou vantagem

A estratégia do painel foi reconstruída com indicadores de verdade (RSI, MACD,
EMAs, Bollinger, ATR, ADX, estocástico, OBV calculados de OHLCV real) e testada
em BTC, ETH, SOL e BNB, em 15m, 1h, 4h e 1d, comprado e comprado+vendido, com
custo cobrado em toda ponta e validação fora da amostra.

Foram 32 configurações e **22.568 operações** ao longo de 400 dias. Somando
tudo:

| | |
|---|---|
| Resultado bruto (antes do atrito) | **−US$ 486** |
| Custo de execução | **−US$ 8.091** |
| **Resultado líquido** | **−US$ 8.578** |

Sobre US$ 32.000 de capital somado, o bruto foi −1,5%. Traduzindo: **a
estratégia não tem direção nenhuma** — acerta e erra como cara ou coroa. E o
custo foi **16,6 vezes** o resultado bruto.

O padrão por tempo gráfico não deixa margem para dúvida:

| Tempo gráfico | Operações | Retorno fora da amostra |
|---|---|---|
| 15m | 1.400 a 2.900 por config | −34% a −63% |
| 1h | 330 a 760 | −11% a −33% |
| 4h | 80 a 190 | −1,8% a +5,9% |

Quanto mais o bot opera, mais ele perde, e a relação é direta. Das 24
configurações que completaram a validação fora da amostra, **zero foram
aprovadas**. O detalhamento está em `relatorios/cripto/ESTUDO_CONFLUENCIA.md`.

### 1.3 Isso não é azar, é a estatística conhecida da atividade

A FGV acompanhou 19.646 pessoas que começaram a fazer day trade de
minicontrato na B3 entre 2013 e 2015: **97% perderam dinheiro**. Entre os 3%
que ganharam, o ganho médio ficou abaixo de R$ 300 por dia. Num estudo maior,
com 98.378 pessoas, **99,4% desistiram antes de 300 pregões** — e o pesquisador
resume a atividade como "muito mais sorte do que técnica".

O ponto relevante do estudo não é o percentual, é a segunda conclusão: os
participantes **não melhoravam com a experiência**. Em quase toda atividade
humana, prática melhora o resultado. Nessa, não melhorou.

### 1.4 A conta que todo bot de scalping tem que fechar

Com taxa taker de 0,10% em spot, uma estratégia de alvo curto precisa acertar
por volta de **92% das operações só para empatar**. Migrando para futuros com
ordem maker (taxa mais baixa), o número cai para perto de 57% — ainda alto, e
agora com alavancagem e risco de liquidação no meio.

É por isso que a curva do simulador subia: sem taxa, um alvo minúsculo é quase
sempre atingido antes do stop. Muitos ganhos de dez centavos, perdas raras e
grandes, curva linda. Com taxa, cada ganho de dez centavos vira perda de vinte
a setenta.

---

## Parte 2 — E a única estratégia com origem mecânica de ganho?

O carrego de funding (*cash and carry*) não adivinha direção: compra à vista,
vende o mesmo tamanho no perpétuo, fica neutro em preço e recebe a taxa de
financiamento que os alavancados comprados pagam a cada 8 horas.

Medi o carrego real dos últimos 166 dias, já descontando as quatro pernas de
montagem e desmontagem (`python -m cripto.cli carrego`):

| Par | Taxa média 8h | Liquidações negativas | **Líquido ao ano** | Em US$ 1.000 |
|---|---|---|---|---|
| BNBUSDT | 0,00305% | 1,2% | **+2,62%** | US$ 13/ano |
| BTCUSDT | 0,00232% | 28,6% | **+1,81%** | US$ 9/ano |
| ETHUSDT | 0,00176% | 30,8% | **+1,19%** | US$ 6/ano |
| SOLUSDT | −0,00041% | 48,0% | **−1,20%** | −US$ 6/ano |

Compare com o **CDI a 13,9% ao ano**, em reais, sem risco de exchange, sem
risco de liquidação e sem trabalho nenhum.

O carrego só fica interessante em mercado eufórico, quando a alavancagem
comprada empurra a taxa para cima — em 2024 e no começo de 2025 ele pagava bem
mais. Hoje, não paga. E, mesmo quando paga 10% a 20% ao ano, ainda exige metade
do capital preso como margem, exposição a liquidação e o dinheiro dentro da
exchange.

**Conclusão da parte 2:** existe uma estratégia com lógica mecânica sólida, e
neste momento ela rende menos que o Tesouro Selic. O monitor diário registra
essa taxa todo dia — se o regime virar, vocês vão saber pelo relatório.

---

## Parte 3 — A conta do tamanho

Esta é a parte que menos se fala e que mais importa.

Os R$ 165 vieram de R$ 100. É um resultado real e é de mérito do seu pai. Mas
percentual não paga conta — reais pagam. Nas melhores hipóteses:

| Capital | 2% ao mês (excelente e improvável de sustentar) | 1% ao mês | CDI (13,9% a.a.) |
|---|---|---|---|
| R$ 500 | R$ 10/mês | R$ 5/mês | R$ 5,80/mês |
| R$ 5.000 | R$ 100/mês | R$ 50/mês | R$ 58/mês |
| R$ 50.000 | R$ 1.000/mês | R$ 500/mês | R$ 579/mês |
| R$ 200.000 | R$ 4.000/mês | R$ 2.000/mês | R$ 2.316/mês |

Duas leituras difíceis mas necessárias:

1. **Para trade virar renda de sustento, o capital precisa ser grande.** Com
   R$ 500, nem 2% ao mês — que é performance de gestor profissional — muda
   alguma coisa. É R$ 10.
2. **O que rende 2% ao mês pode perder 20% num dia.** Retorno alto vem com
   variância alta; não existe a versão só com a parte boa. Com o dinheiro que
   sustenta a casa, a variância é o risco, não o retorno.

---

## Parte 4 — Dá para vender curso sobre isso?

### 4.1 O que é legal

**Vender curso é legal.** Educação financeira não é atividade regulada, não
exige registro na CVM e pode ser vendida por qualquer pessoa. Ensinar análise
técnica, gestão de risco, como funciona uma exchange, como programar um bot —
tudo isso é conteúdo educacional.

### 4.2 O que não é legal

**Dar recomendação de compra e venda para o público exige CNPI** (certificação
da APIMEC) e registro como analista de valores mobiliários, sob a Resolução
CVM 20. Vender sinais, "calls" ou carteiras sem esse registro é infração,
sujeita a multa e processo administrativo.

**Gerir dinheiro de terceiros exige registro de administrador de carteira.**
Não importa se é "só de amigos" ou "só uma sociedade informal": o momento em
que alguém entrega dinheiro para vocês aplicarem, a atividade passa a ser
regulada.

Cripto tem uma diferença jurídica: em regra não é valor mobiliário, então a
competência da CVM é limitada. Mas o Marco Legal (Lei 14.478/2022) e as
Resoluções BCB 519/520/521 — em vigor desde fevereiro de 2026 — colocaram os
prestadores de serviço de ativos virtuais sob o Banco Central, com autorização
obrigatória e prazo de protocolo até 30 de outubro de 2026. Intermediar,
custodiar ou administrar cripto de terceiros hoje é atividade autorizada.
Vender curso e vender software, não.

### 4.3 A pergunta que decide

A pergunta legal tem resposta fácil: pode. A pergunta ética não:

> **Vocês venderiam um método que os próprios números de vocês mostram que não
> tem vantagem?**

Se a resposta for não, e eu acho que é, então a decisão está tomada.

Existe uma versão honesta desse produto, e é uma versão melhor: **ensinar a
medir**. Praticamente todo curso de trade no Brasil ensina a montar setup e
mostra print de resultado. Quase nenhum ensina o que este projeto construiu:

- calcular quanto o atrito come de cada operação;
- fazer backtest sem olhar o futuro (e testar se o backtest olha);
- validar fora da amostra com Deflated Sharpe e PBO;
- reconhecer, no próprio código, quando o número na tela é sintético.

Esse é um conteúdo que quase ninguém tem, que é verdadeiro, e que não depende
de prometer retorno.

### 4.4 O produto que tem mais chance

Ordenado por probabilidade de dar certo:

1. **Software.** O motor `cripto/` é um sistema que mede custo real e valida
   estratégia de forma honesta. Isso é vendável como ferramenta, e vender
   ferramenta não é atividade regulada.
2. **Conteúdo técnico.** "Por que 97% perdem: a matemática do atrito" com o
   código aberto que qualquer um pode rodar. Constrói autoridade sem prometer
   nada.
3. **Curso de método honesto.** Só faz sentido depois de 1 e 2, e a promessa
   tem que ser "você vai saber avaliar uma estratégia", nunca "você vai ganhar
   X%".
4. **Curso prometendo resultado.** Legalmente arriscado e eticamente indefensável
   dado o que os números mostram.

---

## Parte 5 — O que eu recomendaria

**Sobre o dinheiro.** Os R$ 165 não precisam parar. R$ 100 é um valor de
aprendizado, e aprender no simulado custa nada mas também ensina menos que uma
posição pequena de verdade. O que muda a vida financeira da família não é a
estratégia, é a taxa de poupança e o tempo — e essa parte o `main_income.py`
deste mesmo repositório já trata, com FIIs e reinvestimento de dividendos.

**Sobre o painel.** Ele tem valor real como interface. Ligado ao motor de
verdade (`cripto/painel/README.md` explica passo a passo), vira uma ferramenta
que mostra o mercado como ele é. Aí sim é um produto.

**Sobre o conhecimento do seu pai.** A parte insubstituível do painel é
justamente a que veio dele: as regras de gestão, a leitura de fluxo, os
conceitos. O que faltava era a fundação de dados, e essa parte agora existe.
Vale conversar com ele sobre o que o motor mede — quem sabe de mercado costuma
fazer as perguntas certas quando enxerga o número real.

---

## Fontes

- Estudo da FGV EESP sobre day trade no Brasil — 97% de perda, ausência de
  aprendizado com a experiência
- Resolução CVM 20 (analista de valores mobiliários), consolidada
- Lei 14.478/2022 (Marco Legal dos Criptoativos) e Resoluções BCB 519/520/521,
  em vigor desde fevereiro de 2026
- Tabela pública de taxas spot da Binance, Bybit e Gate.io (agosto de 2026)
- Histórico real de funding da Binance Futures, últimos 166 dias, coletado por
  `cripto/carrego.py`
