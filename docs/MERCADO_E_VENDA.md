# O que existe lá fora, o que a lei exige, e quanto dá para cobrar

Pesquisa feita em agosto de 2026 para responder três perguntas: como o mundo
organiza carteira entre seguro e arriscado, quanto cobram os sistemas que já
fazem isso, e o que é preciso para vender este aqui sem cometer crime.

A terceira pergunta muda o plano inteiro, e por isso vem primeiro.

---

## 1. A parte legal, que decide o resto

No Brasil, **recomendar investimento em valor mobiliário de forma profissional
exige autorização da CVM** como consultor de valores mobiliários (Resolução CVM
19/21). Não é burocracia opcional: atuar sem registro é infração administrativa,
e conforme o formato pode escalar para o artigo 7º da Lei 7.492/86, com pena de
2 a 8 anos.

A linha que separa o que precisa de registro do que não precisa é conhecida:

| Atividade | Precisa de registro? |
|---|---|
| Ensinar como funciona ETF, imposto, custo, rebalanceamento | **Não.** É educação. |
| Publicar o seu próprio resultado, com metodologia aberta | **Não.** É relato. |
| Vender software que **o próprio usuário** configura e opera | **Zona cinzenta.** Depende de quem decide. |
| Dizer a alguém *"compre IVVB11 com 36% do seu dinheiro"* | **Sim.** É recomendação individualizada. |
| Gerir dinheiro de terceiros | **Sim**, e é outro registro (administrador de carteiras). |

A CVM publicou o Ofício Circular CVM/SIN 2/2026 esclarecendo remuneração,
encaminhamento de ordem e certificação para consultores. O documento não trata
especificamente de software, o que mantém a zona cinzenta cinzenta.

**O que isto significa na prática, na ordem do menos para o mais arriscado:**

1. **Curso e conteúdo** — livre. Ensinar método, mostrar código, explicar por
   que custo e imposto matam retorno. Nenhum registro.
2. **Software auto-hospedado** — o usuário instala, configura o próprio
   orçamento de risco e executa as ordens na corretora dele. Você vende a
   ferramenta, não a decisão. É o modelo de risco baixo, e é o que este
   repositório já é.
3. **Serviço hospedado que diz quanto comprar de quê** — aqui você virou
   consultor. Precisa do registro.
4. **Gerir dinheiro dos outros** — não faça sem o registro de administrador de
   carteiras. Sem exceção.

O caminho 1 + 2 não exige registro nenhum e é onde o projeto está hoje. Se um
dia o caminho 3 fizer sentido, o registro de consultor é obtível: exige
certificação e processo na CVM, e leva meses, não anos.

---

## 2. Como o mundo organiza seguro e arriscado

O padrão institucional se chama **core-satellite**, e é literalmente a
arquitetura que este sistema usa — o que é bom sinal, porque não foi inventada
aqui.

- **Núcleo (70% a 90%)**: índices amplos, baratos, girando pouco. A função é
  capturar o mercado com o mínimo de custo e de erro de rastreamento.
- **Satélite (10% a 30%)**: onde entram apostas concentradas, momentum
  tático, setor específico. Aceita concentração e giro maiores.
- **O que liga os dois é orçamento de risco explícito**, não integração de
  sinais. O satélite é dimensionado pela volatilidade dele, pela queda máxima
  esperada e pela correlação com o núcleo — nunca pelo entusiasmo com a ideia.

A literatura institucional é explícita nesse ponto: o núcleo robusto existe
justamente para que o satélite possa operar **sem dominar o risco total**.

**Onde este sistema está:** núcleo 100%, satélite 0%. Não porque o satélite
seja proibido, mas porque a validação reprovou todas as estratégias candidatas
e a trava só abre com aprovação. É a mesma arquitetura da indústria, com o
satélite fechado por falta de evidência — que é o estado correto até haver
evidência.

---

## 3. Quanto cobram os que já fazem isso

### Lá fora

| Plataforma | Taxa anual | Mínimo |
|---|---|---|
| Vanguard Digital Advisor | 0,15% | US$ 3.000 |
| Betterment | 0,25% | sem mínimo |
| Wealthfront | 0,25% | US$ 500 |

### No Brasil

| Plataforma | Taxa anual |
|---|---|
| Warren | ~0,5% |
| Magnetis | 0,6% (devolve cashback de fundos) |
| Vérios | até 0,95% (custo total, tudo incluso) |

A faixa brasileira de carteira automatizada fica entre 0,5% e 1,5% ao ano,
contra 2% a 3% de fundo tradicional.

### Ferramentas que o usuário opera sozinho

| Produto | Preço |
|---|---|
| Kubera | US$ 150/ano (individual), US$ 225/ano (família) |
| Sharesight | camada gratuita, planos pagos acima |

Este é o comparável certo para o caminho 2. **Um sistema que o usuário
configura e opera vale preço de ferramenta, não de gestão.**

---

## 4. O que dá para cobrar por este sistema, com honestidade

Antes do preço, o diagnóstico: **o retorno não é o produto.**

Medido em 23 anos de histórico reconstruído, a mediana das janelas de 10 anos a
80% de risco deu 13,13% ao ano. No backfill real de 2021 até hoje, 13,49% contra
11,10% do CDI. É bom, é honesto, e é indefensável como argumento de venda —
qualquer um pode alegar o mesmo, e a maioria alega mais.

**O produto é a auditoria.** O mercado de "sistema de investimento" é lotado de
resultado que ninguém consegue conferir. Este tem cadeia de hash: cada decisão
carrega o hash da anterior, alterar um dia antigo quebra todos os seguintes, e
existe carimbo publicável fora do arquivo. É a diferença entre "confie em mim" e
"confira você mesmo".

Segundo diferencial: **o sistema documenta os próprios fracassos.** Trinta e
duas configurações de cripto reprovadas. A calibração que aprendeu de ruído
puro e foi corrigida. A auditoria do painel do meu pai que achou custo 16 vezes
maior que o resultado bruto. Isso não se falsifica facilmente e é raríssimo no
mercado.

### Faixas defensáveis

| O que | Preço | Por quê |
|---|---|---|
| **Curso** — método, código, metodologia de validação | **R$ 297 a R$ 597** | O piso de mercado no Hotmart é ~R$ 197. Acima disso precisa de diferencial, e aqui ele existe: código aberto, auditoria verificável, e um estudo de caso real de sistema que não funcionava. |
| **Sistema auto-hospedado**, licença perpétua | **R$ 497 a R$ 997** | Compara com Kubera (US$ 150/ano). Licença única cobra menos suporte e evita a promessa implícita de atualização eterna. |
| **Sistema por assinatura**, com atualização e painel hospedado | **R$ 29 a R$ 49/mês** | Faixa de ferramenta. Acima disso o comprador espera recomendação, e recomendação exige registro. |
| **Gestão de dinheiro de terceiros** | **não vender** | Exige registro de administrador de carteiras. |

### A condição que não dá para pular

**Nada disso se vende antes de ter track record.** Hoje o sistema tem 2.943
eventos e um backfill que mostra o que *teria* acontecido. Isso é evidência de
método, não de resultado.

O que transforma em produto vendável são **dois a três anos rodando ao vivo**,
com a cadeia intacta e o resultado publicado toda semana, ganhando ou perdendo.
Se em três anos o resultado for indistinguível do CDI, não há o que vender — e
descobrir isso terá custado zero, porque o dinheiro era simulado.

Se der certo, aí sim existe algo que quase ninguém tem: um histórico que o
comprador pode verificar linha por linha.

---

## 5. Ordem recomendada

1. **Agora até 2028** — deixar rodando. Sem vender nada. Acumular histórico
   auditável.
2. **A partir de 2027** — publicar o resultado mensal em algum lugar público,
   ganhando ou perdendo. Isso constrói reputação e serve de âncora externa da
   cadeia de hash ao mesmo tempo.
3. **2028, se o resultado se sustentar** — lançar o curso primeiro. É o
   produto sem risco regulatório, e valida se existe público.
4. **Depois do curso** — o sistema auto-hospedado, para quem quer operar.
5. **Só se fizer muito sentido** — tirar o registro de consultor e oferecer
   serviço hospedado.

Vender antes do passo 1 estar cumprido seria vender exatamente o tipo de coisa
que este projeto foi construído para desmontar.

---

## Fontes

- [Best Robo-Advisors in the US (2026) — unbiased.com](https://www.unbiased.com/discover/financial-advice/best-robo-advisors)
- [Betterment vs. Wealthfront: 2026 Comparison — NerdWallet](https://www.nerdwallet.com/investing/learn/betterment-vs-wealthfront)
- [Vanguard Digital Advisor Review 2026 — Wealthvieu](https://wealthvieu.com/investing/robo-advisors/vanguard-digital-advisor-review/)
- [Robôs de Investimento 2026: Warren, Magnetis e CVM — Renova Invest](https://renovainvest.com.br/blog/robos-de-investimento/)
- [Carteira automatizada: como os robo advisors cuidam dos seus investimentos — Exame](https://exame.com/invest/guia/carteira-automatizada-como-os-robo-advisors-cuidam-dos-seus-investimentos/)
- [A Core-Satellite Approach to Portfolio Strategy — CMC Markets](https://www.cmcmarkets.com/en-gb/investing-guides/core-satellite-approach)
- [The rise of the Core-Satellite approach in fixed income — State Street](https://www.ssga.com/us/en/institutional/insights/rise-of-core-satellite-approach-fixed-income)
- [Dynamic Inclusion and Bounded Multi-Factor Tilts for Robust Portfolio Construction — arXiv](https://arxiv.org/pdf/2601.05428)
- [Área técnica da CVM orienta sobre consultoria de valores mobiliários — CVM](https://www.gov.br/cvm/pt-br/assuntos/noticias/2026/area-tecnica-da-cvm-orienta-sobre-atividade-de-consultoria-de-valores-mobiliarios)
- [Consultores de Valores Mobiliários — CVM](https://www.gov.br/cvm/pt-br/assuntos/regulados/consultas-por-participante/consultores-de-valores-mobiliarios)
- [Guia da autorização da CVM para Consultoria de Investimentos — NDM Advogados](https://ndmadvogados.com.br/artigo/autorizacao-da-cvm-para-consultoria-de-investimentos/)
- [Educador Financeiro, Consultor, Planejador e Assessor: a diferença — Grupo Karppa](https://karppa.com.br/blog/diferenca-educador-consultor-planejador-financeiro/)
- [Ofertas e atuações irregulares — CVM](https://conteudo.cvm.gov.br/menu/investidor/alertas/ofertas_atuacoes_irregulares.html)
- [Best Portfolio Tracker 2026: Capitally vs Kubera, Sharesight & Snowball](https://www.mycapitally.com/blog/best-portfolio-tracker-for-the-modern-diy-investor)
- [Kubera Reviews 2026 — G2](https://www.g2.com/products/kubera/reviews)
- [Os 3 Melhores Cursos de Investimentos Online em 2026 — Dinheiro e Vivo](https://dinheiroevivo.com.br/melhores-cursos-de-investimentos/)
