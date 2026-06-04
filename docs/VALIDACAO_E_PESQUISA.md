# QuantBot ML — Validação Estatística, Pesquisa e Roadmap para Produção

> Documento de metodologia e pesquisa. Reúne o que foi implementado na suíte
> de validação anti-overfitting, as fontes acadêmicas e profissionais que a
> fundamentam, e o caminho honesto até (eventual) operação com capital real.
>
> Última atualização: 01/06/2026

---

## 1. Por que esta suíte é o que separa profissionais de amadores

A maioria dos bots de trading "morre" pela mesma causa: **overfitting**. O
modelo decora o passado e parece lucrativo no backtest, mas falha no mercado
real. Os dados confirmam:

- O estudo da **Quantopian** com 888 estratégias reais mostrou que o Sharpe
  ratio do backtest tinha poder preditivo **quase nulo** sobre retornos ao
  vivo. Estratégias super-otimizadas perdem até **80%** do lucro do backtest
  quando vão a produção.
- **70% a 90% dos traders de varejo perdem dinheiro** (múltiplos estudos).
- O melhor trader do fundo de Steve Cohen acerta **63%** das vezes; a maioria
  dos profissionais opera entre **50% e 55%** de win rate.
- Marcos López de Prado: *"você está quase garantidamente fazendo overfitting,
  mesmo sem querer. Um Sharpe alto é inútil a menos que você saiba qual a
  probabilidade de ter sido sorte."*

**Conclusão de design:** o objetivo do projeto não é "prever o mercado" (isso
não existe), e sim **medir honestamente** se há vantagem estatística real
antes de arriscar capital. A suíte de validação faz exatamente isso.

---

## 2. O que foi implementado (módulo `validation/`)

### 2.1 Purged K-Fold + Embargo (`purged_cv.py`)

**Problema que resolve:** o k-fold padrão do scikit-learn permite que o modelo
"veja o futuro". Em finanças, o label de um instante `t` depende de preços em
`[t, t+h]`. Se um ponto de treino cai nessa janela, há vazamento (leakage).

**Solução (López de Prado, *Advances in Financial ML*, cap. 7):**
- **Purging:** remove do treino qualquer amostra cujo label se sobreponha no
  tempo ao intervalo de teste.
- **Embargo:** bloqueia uma fração de amostras logo após o teste, pois a
  autocorrelação serial ainda pode vazar informação.

### 2.2 Combinatorial Purged CV — CPCV (`purged_cv.py`)

**Problema que resolve:** o walk-forward gera **um único** caminho histórico,
fácil de superajustar. O CPCV combina grupos de teste para gerar **múltiplos**
caminhos out-of-sample, produzindo uma *distribuição* de performance — não um
número único que pode ser sorte. (López de Prado, cap. 12.)

### 2.3 Deflated Sharpe Ratio — DSR (`deflated_sharpe.py`)

**Problema que resolve:** o Sharpe "cru" ignora (a) caudas gordas e assimetria
dos retornos e (b) **quantas estratégias você testou**. Quem testa 200
variações e escolhe a melhor quase sempre encontra um falso positivo.

**Solução (Bailey & López de Prado, 2014):**
- **PSR (Probabilistic Sharpe Ratio):** P(Sharpe verdadeiro > benchmark),
  corrigido por skew e curtose.
- **DSR:** usa como benchmark o **Sharpe máximo esperado por puro acaso** dado
  N tentativas. Se o seu Sharpe não supera o que se obteria por sorte testando
  N estratégias, o DSR é baixo.
- **Meta prática:** DSR > 0.95 para considerar que há skill real.

### 2.4 Probability of Backtest Overfitting — PBO (`pbo.py`)

**Pergunta que responde:** "se eu escolho a estratégia que parece melhor no
backtest, qual a chance de ela ficar abaixo da mediana no período real?"
- PBO < 0.20 → baixo risco de overfitting.
- PBO ≈ 0.50 → escolher pelo backtest não tem valor preditivo.

**Método:** Combinatorially Symmetric Cross-Validation — CSCV
(Bailey, Borwein, López de Prado, Zhu, 2016).

### 2.5 Garantia de correção

Os 22 testes em `tests/test_validation.py` verificam **propriedades
estatísticas conhecidas**, não apenas que o código roda:
- Ruído puro → DSR baixo (corretamente marcado como "sem skill").
- Ruído puro → PBO ≈ 0.50 (média em 20 seeds = 0.534).
- Skill genuíno → PBO cai.
- Mais tentativas testadas → DSR menor (penaliza data snooping).

---

## 3. Dossiê de pesquisa (fontes)

| Técnica | Fonte | Achado central |
|---|---|---|
| Purged/Embargo CV | López de Prado (2018), *Advances in Financial ML*, cap. 7 | k-fold padrão vaza o futuro em séries financeiras |
| CPCV | López de Prado (2018), cap. 12 | múltiplos caminhos OOS > um único walk-forward |
| PSR / DSR | Bailey & López de Prado (2012, 2014), *J. of Portfolio Mgmt* | corrige Sharpe por nº de testes e não-normalidade |
| PBO / CSCV | Bailey, Borwein, López de Prado, Zhu (2016), *J. Computational Finance* | mede se a seleção pelo backtest tem valor |
| Falha de fundos de ML | López de Prado, *The 10 Reasons Most ML Funds Fail* (GARP) | walk-forward único é facilmente superajustado |
| Backtest overfitting (era ML) | ScienceDirect (2024) — comparação de métodos OOS | CV convencional infla performance |
| GT-Score / anti-overfitting | arXiv 2602.00080 (2026) | t-stat ≥ 3.0 (não 2.0) no "factor zoo" |
| Reality Check / SPA | White (2000); Hansen (2005) | ajustam significância ao comparar muitos modelos |
| Realidade do varejo | QuantifiedStrategies; Stockio (2026) | 70–90% dos traders de varejo perdem |
| Camadas de risco independentes | Appinventiv (2026) — guia de bots de produção | risco como camada independente das estratégias |

---

## 4. O que "nível profissional" realmente significa aqui

Não é "chegar ao nível do Steve Cohen" — ele tem o Point72 com centenas de
PhDs, dados que custam milhões e infraestrutura institucional. Isso é
inalcançável para um bot de varejo, e tudo bem.

O alvo realista e honesto:
1. **Engenharia de produção** (segurança, observabilidade, testes, kill-switch).
2. **Validação estatística honesta** (esta suíte) antes de qualquer capital.
3. **Aceitar que a vantagem é modesta e incerta** — win rate de 50–55%,
   pensando em probabilidades, nunca em "previsões".

---

## 5. Roadmap honesto até operação real

| Fase | O que acontece | Critério para avançar |
|---|---|---|
| **1. Engenharia (em curso)** | Suíte de validação ✅, depois hardening de segurança/observabilidade | Todos os testes passando |
| **2. Validação** | Rodar CPCV + DSR + PBO nos modelos com dados recentes | DSR alto **e** PBO baixo |
| **3. Paper trading** | 3–6 meses com dinheiro fictício e dados reais | Sharpe e win rate reais se sustentam |
| **4. Capital mínimo** | Só se as fases 2–3 derem certo; valor que não afete sua vida | Decisão sua, idealmente com um profissional |

**Regra inegociável:** testes unitários passando = código correto. Isso **não**
significa estratégia lucrativa. São coisas separadas. Nenhum arquivo pode ser
declarado "pronto para investir" — essa garantia não existe para ninguém.

---

## 6. Como o sistema operaria com capital real (modelo)

Tecnicamente, o fluxo que você descreveu é viável:
1. Conecta-se a uma corretora via API (Alpaca para mercado US é o caminho mais
   direto; B3 tem mais restrições regulatórias para auto-trading de varejo).
2. O bot lê os sinais, dimensiona posições pelo `RiskManager` e envia ordens.
3. Os ganhos são reinvestidos automaticamente (juros compostos) até você sacar
   ou aportar mais.

**Cuidados críticos para contas pequenas (ex.: R$1.000):**
- Custos de corretagem, spread, slippage e impostos pesam
  **desproporcionalmente** em contas pequenas e podem inviabilizar estratégias
  que funcionariam em contas maiores.
- Entre esperando que pode perder parte relevante — trate como custo de
  aprendizado, não como semente que vai crescer com segurança.
- **Kill-switch obrigatório:** o bot para tudo automaticamente se o drawdown
  passar de um limite (ex.: -20%).

---

## 7. Próximos passos de engenharia (pendentes)

- [ ] Integrar a suíte de validação ao pipeline de treino (`models/trainer.py`)
- [ ] Camada de risco independente com kill-switch e limites rígidos
- [ ] Observabilidade: logging estruturado, métricas, health checks
- [ ] Segurança: validação de input estrita, gestão de segredos, audit trail
- [ ] Testes de integração ponta-a-ponta do pipeline
- [ ] Paper trading instrumentado com registro de DSR/PBO ao longo do tempo
