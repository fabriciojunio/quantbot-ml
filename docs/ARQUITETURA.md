# Arquitetura

Como o sistema é organizado e por que cada decisão foi tomada dessa forma. As
justificativas importam mais que a descrição: daqui a dez anos alguém vai
querer mudar algo, e o valor deste documento é dizer o que se quebra quando
muda.

---

## Camadas

```
      dados brutos            Yahoo Finance · Banco Central · Binance
            │
     carteira/dados.py        cotação, CDI, PTAX, cache, validação
     carteira/serie_longa.py  reconstrução de histórico por índice
            │
     carteira/universo.py     o que existe, com spread e regime tributário
     carteira/tributos.py     IR: regressiva, 15%, isenções, compensação
     carteira/custos_b3.py    emolumento, spread, taxa de administração
            │
     carteira/conta.py        ◄── contabilidade: posição, base, custo, imposto
            │
      ┌─────┴─────┐
      │           │
 aporte_          operacao.py       decisão do dia
 historico.py     (ao vivo)
 (backtest)             │
      │           scripts/rodar_diario.py
      │                 │
 validacao.py     auditoria/livro.py   livro encadeado por hash
 calibracao.py           │
                   relatorios/painel.json
                         │
                     painel/           Next.js estático
```

---

## As decisões que definem o projeto

### 1. Uma contabilidade só, compartilhada entre backtest e operação

`carteira/conta.py` é a única classe que mexe em posição, caixa e imposto. O
backtest histórico e o robô ao vivo passam por ela.

**O que se quebra ao duplicar:** o backtest e a operação viram duas
implementações da mesma regra e divergem devagar — um arredondamento aqui, uma
correção de imposto aplicada só de um lado ali. No fim o backtest promete um
resultado que a operação não entrega, e a diferença é indetectável porque os
dois "funcionam". Era o defeito exato do painel que originou este projeto: o
número da tela e o da corretora não eram a mesma conta.

### 2. Nenhuma série sintética, em lugar nenhum

Se a fonte não responde e não há cache, a função levanta erro. Nunca devolve
dado inventado para "não quebrar o gráfico".

**Por quê:** dado inventado não avisa que é inventado. Ele produz gráfico
bonito, métrica plausível e conclusão errada, e sobrevive a revisão porque
ninguém olha uma série que parece normal.

### 3. Custo e imposto em toda ponta, sempre

Nenhum caminho do código executa uma ordem sem cobrar spread, emolumento e,
quando há ganho, imposto.

**Por quê:** foi o que matou o sistema anterior. Vinte e dois mil operações,
resultado bruto de −US$ 486 e custo de US$ 8.091 — dezesseis vezes o bruto. Um
backtest sem custo teria mostrado uma estratégia quase neutra em vez de uma
que destruiu 86% do capital.

### 4. Validação com trava, não com relatório

`carteira/validacao.py` produz veredito, e o satélite do robô só abre com
veredito favorável. Hoje está fechado: Deflated Sharpe de 0,84 contra mínimo de
0,95, e uma janela fora da amostra contra mínimo de três.

**Por quê:** validação que só informa é ignorada. Toda vez. A trava tem que
estar no caminho do dinheiro.

### 5. Aprendizado com teste estatístico, não com limiar escolhido a dedo

`carteira/calibracao.py` ajusta num bloco, mede no seguinte, e só aceita o
aprendizado se ele passar em teste binomial (p < 0,10) **e** escolher o mesmo
conjunto na maioria dos blocos.

**Por quê:** a primeira versão usava limiar de "60% de vitórias" e aprovava
aprendizado sobre ruído puro em três de cinco sementes. Com seis blocos, ganhar
quatro tem 34% de chance de sair de moeda honesta, e o limiar aprovava
exatamente esse caso.

### 6. Livro append-only encadeado por hash

`auditoria/livro.py`. Só insere. Corrigir é lançar um evento de correção que
aponta para o anterior, e os dois ficam.

**Limite conhecido, registrado em teste:** quem tem o arquivo pode refazer
todos os hashes e a cadeia fecha de novo. Por isso existe `ancora()`, que
produz um carimbo para publicar fora do arquivo. Sem o carimbo externo, a
garantia é contra edição pontual; com ele, é contra reescrita também.

### 7. SQLite, não banco com servidor

**Por quê:** o requisito é durar gerações. Banco que exige servidor exige
alguém pagando a conta do servidor, e esse alguém uma hora para de pagar. O
formato SQLite tem compromisso público de compatibilidade até 2050 e é lido por
qualquer linguagem.

### 8. Painel estático, sem servidor e sem chave

O robô gera um JSON depois de auditar; o build vira HTML.

**Consequência:** não existe credencial para vazar porque não existe chamada a
fazer, não existe conta mensal, e a tela não consegue mostrar número que não
passou pela verificação — é a única fonte que ela tem.

### 9. Trava de dinheiro real como constante, não como configuração

`DINHEIRO_REAL = False` em `carteira/operacao.py`, sem parâmetro que a desligue.

**Por quê:** ligar dinheiro de verdade tem que ser alteração de código,
revisada e versionada. Não uma flag que alguém troca às onze da noite.

### 10. Rodízio em vez de fatiar o aporte

Com R$ 500 e seis ativos, cada ordem sai perto de R$ 80 e paga um spread
inteiro sobre isso. O robô concentra a compra no ativo mais atrasado.

**Efeito medido:** paga um spread em vez de seis e converge para o mesmo peso
alvo em alguns meses.

### 11. Câmbio do Banco Central, não do Yahoo

**Por quê:** `USDBRL=X` não tem um único dia de 2005 e tem um buraco de 575
dias terminando em março de 2006. Depois do alinhamento entre ativos o buraco
sumia, e 26/08/2004 ficava colado em 24/03/2006 — variação de 66% lida como
retorno de um pregão. Daí nasceu `valida_continuidade()`.

---

## O que cada pacote faz

| Pacote | Responsabilidade |
|---|---|
| `carteira/` | Motor de alocação: dados, custo, imposto, decisão, simulação |
| `auditoria/` | Livro encadeado e verificação de integridade |
| `cripto/` | Motor de cripto e as 32 configurações reprovadas |
| `income/` | Carteira de renda e o bug de provento corrigido |
| `validation/` | Sharpe deflacionado, PBO, purged k-fold |
| `painel/` | Tela estática Next.js |
| `scripts/` | Pontos de entrada: robô, simulação, auditoria, monitor |

---

## Como rodar

```bash
python scripts/rodar_diario.py --auditar       # confere a cadeia, não muda nada
python scripts/rodar_diario.py                 # opera o último pregão
python scripts/rodar_diario.py --desde 2021-01-04   # reconstrói do zero
python scripts/simular_aportes.py              # janelas móveis de 5 e 10 anos
python scripts/robo.py --valor 500 --risco 80  # decisão avulsa do mês
python -m pytest                               # a suíte inteira
```

---

## Onde mexer com cuidado

- **`conta.py`** — qualquer mudança altera backtest e operação ao mesmo tempo.
  Rode a suíte inteira e compare a mediana das janelas de 10 anos antes e
  depois; ela é a assinatura do comportamento.
- **`tributos.py`** — alíquota errada muda o resultado de forma crescente com
  o tempo, e passa despercebida em janela curta.
- **`livro.py`** — mudar o formato canônico do JSON muda todos os hashes e
  invalida a cadeia inteira. Se for inevitável, faça migração explícita e
  registre um evento de correção.
- **`perfil.py`** — os pesos precisam somar 1. O código já falha alto se não
  somarem, e essa checagem pegou um bug que fazia 12% da carteira evaporar.
