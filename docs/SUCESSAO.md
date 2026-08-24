# Para quem herdar isto

Se você está lendo este arquivo porque alguém da família morreu, ficou doente
ou simplesmente parou de mexer nisso, este documento é para você. Ele assume
que você **não sabe programar** e **não entende de investimento**. Está escrito
nessa ordem: primeiro o que fazer se você não quiser mexer em nada, depois o que
fazer se quiser continuar.

O sistema foi construído para sobreviver ao próprio autor. Se algo aqui não
funcionar, é falha do projeto, não sua.

---

## Em uma frase

Isto é um programa que decide onde investir um valor mensal, seguindo regras
escritas, e anota **tudo** o que decidiu e por quê num arquivo que não pode ser
alterado sem deixar rastro.

---

## Primeira coisa: nada está em risco agora

Enquanto ninguém tiver mudado o código, **o sistema não movimenta dinheiro de
verdade**. Ele usa preços reais e dinheiro imaginário. Não existe senha de
corretora aqui, não existe conta ligada, não existe ordem sendo enviada para
lugar nenhum.

Para confirmar isso você mesmo, procure no arquivo `carteira/operacao.py` a
linha:

```python
DINHEIRO_REAL = False
```

Se estiver `False`, nada é real. Se alguém tiver mudado para `True`, aí sim há
dinheiro envolvido e você precisa de alguém de confiança para olhar antes de
mexer.

---

## Se você não quer continuar com isto

É uma decisão legítima. Não há penalidade nenhuma.

1. **Não apague nada.** O histórico pode ter valor depois — para você, para
   outra pessoa da família, ou como registro do que foi feito.
2. **Guarde uma cópia** de duas coisas:
   - a pasta inteira do projeto
   - o arquivo `data/auditoria/livro.sqlite3`
3. **Desligue o automático.** No GitHub, em `Actions`, desabilite o fluxo
   chamado `robô diário`. Ele para de rodar e nada mais acontece.
4. Se houver dinheiro de verdade em alguma corretora, ele **não está aqui** —
   está na corretora, no nome de quem abriu a conta, e se resolve por
   inventário como qualquer outro bem.

Acabou. Pode fechar.

---

## Se você quer continuar

### O que você precisa ter

- Um computador com **Python** instalado (site python.org, versão 3.12 ou mais
  nova)
- A pasta do projeto
- Nada mais. Não há senha, não há chave paga, não há assinatura.

### Como ver onde as coisas estão

Abra o terminal na pasta do projeto e rode:

```bash
python scripts/rodar_diario.py --auditar
```

Isso **não muda nada**. Só confere se o histórico está intacto. Se aparecer
`"intacta": true`, está tudo certo.

Depois:

```bash
python scripts/rodar_diario.py
```

Isso processa o pregão mais recente e imprime na tela quanto tem, quanto foi
depositado, e quanto o mesmo dinheiro teria rendido na renda fixa.

### Como ver a tela bonita

```bash
cd painel
npm install
npm run build
```

Abra o arquivo `painel/out/index.html` no navegador.

---

## As três coisas que você precisa entender

### 1. O número que importa é a distância entre duas linhas

No gráfico há uma linha cheia (patrimônio) e uma tracejada (dinheiro
depositado). A cheia sobe quase sempre — porque entra dinheiro novo todo mês, e
isso não é mérito de ninguém.

**O resultado é a distância entre elas.** Se a cheia está acima, o sistema
acrescentou algo. Se está abaixo, tirou.

### 2. Quedas vão acontecer, e não são defeito

O sistema foi medido em 23 anos de histórico. Nos piores períodos, o patrimônio
ficou **até 27% abaixo do que tinha sido depositado**, e demorou meses para
voltar.

Isso é o preço de ter dinheiro em ativo que oscila. Não é bug, não é o sistema
quebrando, e **não é hora de desligar**. A regra que existe aqui é reduzir a
exposição em regime ruim, nunca zerar, porque vender no fundo transforma queda
temporária em prejuízo permanente.

Se você não aguenta ver isso acontecer, a resposta certa é diminuir o risco
(veja abaixo), não desligar.

### 3. Ninguém aqui sabe o futuro

O sistema não prevê nada. Ele decide **quanto** vai para cada tipo de ativo e
rebalanceia. É tudo. Nenhum número deste projeto é promessa.

---

## Como mudar o quanto de risco

O único ajuste que é escolha sua. Rode com o número que quiser, de 0 a 100:

```bash
python scripts/rodar_diario.py --risco 40
```

Medido em histórico real, este é o preço de cada escolha:

| Risco | Retorno ao ano | Pior queda |
|---|---|---|
| 20% | ~10,3% | −2,4% |
| 40% | ~10,9% | −6,1% |
| 60% | ~11,5% | −9,8% |
| 80% | ~11,9% | −13,6% |
| 100% | ~12,2% | −17,4% |

Ir de 20% para 100% aumenta o retorno em menos de 2 pontos e multiplica a queda
por sete. **Se estiver em dúvida, escolha menos risco.** A diferença de retorno
é pequena; a diferença de sofrimento não é.

---

## Como saber se alguém mexeu no histórico

Cada dia registrado carrega uma assinatura do dia anterior. Mudar um dia antigo
quebra todos os seguintes. Para conferir:

```bash
python scripts/rodar_diario.py --auditar
```

Se aparecer `"intacta": false`, alguém alterou alguma coisa, e o resultado
mostra exatamente em qual evento.

Existe também um arquivo `data/auditoria/carimbo.json`, que contém um resumo de
toda a cadeia. Se alguém tiver guardado esse carimbo em outro lugar — um
e-mail, uma mensagem, um papel —, dá para comparar e detectar até uma reescrita
completa.

---

## Perguntas que provavelmente você tem

**Isso rende quanto?**
Nas medições históricas, entre 10% e 13% ao ano dependendo do risco escolhido,
com anos ruins no meio. A renda fixa brasileira (CDI) rendeu entre 7% e 11% nos
mesmos períodos. A vantagem existe, é de 1 a 3 pontos por ano, e some em
qualquer janela curta.

**Posso confiar nesses números?**
Pode conferir. É diferente. Rode `python scripts/simular_aportes.py` e ele
recalcula tudo do zero, baixando os preços de novo.

**Preciso pagar alguma coisa?**
Não. Não há assinatura, servidor ou serviço pago. Preço de mercado vem de fontes
públicas e gratuitas (Yahoo Finance, Banco Central, Binance).

**E se o Yahoo Finance sair do ar?**
O sistema para de atualizar e avisa. Ele nunca inventa preço para "não quebrar o
gráfico" — se a fonte não responde e não há cache, ele levanta erro. Trocar de
fonte é mexer em `carteira/dados.py`, e aí você precisa de alguém que programe.

**Posso colocar dinheiro de verdade?**
Pode, mas o sistema **não faz isso sozinho** e não deve fazer. Ele te diz o que
comprar; você compra na sua corretora. Ligar execução automática exigiria mexer
no código, e a trava existe justamente para que isso seja uma decisão pensada e
não um acidente.

**Quem é o dono disso?**
Ninguém, no sentido de licença. O código é o que está na pasta. O dinheiro, se
houver, está na corretora e segue as regras de herança normais.

---

## Se você quiser ajuda de alguém técnico

Mostre para essa pessoa estes três arquivos, nesta ordem:

1. `docs/ARQUITETURA.md` — como o sistema é organizado
2. `carteira/conta.py` — a contabilidade, que é o coração
3. `auditoria/livro.py` — o registro que não pode ser alterado

Qualquer pessoa que programe em Python entende o sistema em uma tarde. Ele foi
escrito para isso: sem framework exótico, sem dependência obscura, com o motivo
de cada decisão escrito em comentário no próprio código.

---

## Uma última coisa

Se este sistema estiver rodando há anos e o resultado for indistinguível da
renda fixa, **a resposta certa é desligar e pôr o dinheiro na renda fixa**.
Não existe obrigação de continuar por respeito a quem começou. O projeto foi
construído com medição honesta justamente para que essa decisão possa ser
tomada com dado, e não com sentimento.

E se o resultado for bom, ele continua sendo o que sempre foi: alguns pontos ao
ano acima do CDI, com anos ruins no meio. Nunca foi para ser mais que isso.
