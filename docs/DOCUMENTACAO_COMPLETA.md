# QuantBot ML — Documentação Completa
## Tudo que você precisa saber para usar, entender e fazer funcionar

> Versão 6.0 · 01/06/2026 · 118 testes passando

---

# PARTE 1 — O QUE É O SISTEMA

## Em uma frase

O QuantBot é um **assistente de investimentos automatizado** que analisa
centenas de FIIs e ações toda semana, te manda recomendações precisas no
Telegram, e rastreia sua jornada de renda passiva ao longo dos anos.

## O que é automático (você não faz nada)

- Toda semana: análise de 80+ FIIs com 9 critérios profissionais
- Todo dia útil às 18h30: relatório no Telegram com status e alertas
- Primeiro do mês: relatório mensal com comparativo vs Selic
- 2 de janeiro: relatório anual completo
- Qualquer hora: alerta de vacância alta, corte de dividendo, kill-switch

## O que você faz (5 minutos por semana)

- Lê a recomendação que chegou no Telegram na segunda-feira
- Abre o NuInvest, compra o que o bot sugeriu
- Digita um comando para confirmar: `python main_income.py --confirm HGLG11 5 115 COMPRAR`
- Pronto. Bot atualiza tudo automaticamente

## O que o bot NÃO faz

O bot não executa trades automaticamente. Nenhuma corretora brasileira de
varejo (NuInvest, XP, Rico, etc.) permite que um programa externo acesse
sua conta e movimente dinheiro. Isso é uma proteção do mercado, não uma
limitação do sistema. Você sempre tem controle total do seu dinheiro.

---

# PARTE 2 — COMO OS SINAIS FUNCIONAM (EXPLICAÇÃO COMPLETA)

## O ciclo completo de um sinal

```
SEGUNDA-FEIRA, 8h00
─────────────────────────────────────────────────────────────────

1. Bot baixa dados da brapi.dev
   → Cotação de 80+ FIIs
   → DY dos últimos 12 meses
   → P/VP (preço / valor patrimonial)
   → Vacância física
   → Liquidez diária

2. Bot consulta CVM (dados oficiais)
   → Patrimônio Líquido real
   → Número de cotistas
   → Resultado contábil mensal
   → Valida se os dados da brapi conferem com a fonte oficial

3. Bot analisa sentimento de notícias
   → Busca no Google News RSS: "HGLG11 FII fundo imobiliário"
   → Aplica léxico financeiro em português (1.200+ termos calibrados)
   → Pondera por recência: notícia de hoje vale mais que de 7 dias

4. FII Screener aplica 9 critérios
   ┌─────────────────────────────────────┐
   │ DY entre 8% e 20%            pass/fail
   │ P/VP entre 0.50 e 1.20        pass/fail
   │ Vacância < 15%               pass/fail
   │ Liquidez > R$500k/dia         pass/fail
   │ Patrimônio > R$200M           pass/fail
   │ Taxa adm < 1.5% a.a.          pass/fail
   │ Distribuição ≤ 110% resultado pass/fail
   │ Histórico ≥ 2 anos sem corte  pass/fail
   │ Score ponderado 0-100         numérico
   └─────────────────────────────────────┘

5. Compara com seu portfólio atual
   → FII aprovado que você NÃO tem = COMPRAR
   → FII que você TEM e foi REPROVADO = VENDER
   → FII que você TEM e ainda aprovado = MANTER

6. Gera mensagem e envia no Telegram
```

## Exemplo real de mensagem no Telegram

```
📊 QUANTBOT — SCREENING SEMANAL
📅 02/06/2026
═══════════════════════════════════════

Selic atual: 14.5% a.a.
FIIs analisados: 83 | Aprovados: 21

🔴 VENDER (ação urgente):
  → LIXO11: Vacância 35% | DY 22% suspeito | taxa adm 2.5%

🟢 COMPRAR (prioridade ALTA):
  → HGLG11 | Score 82 | DY 11.5% | P/VP 0.92 | Vac 3.0%
     Gestora CSHG | 6 anos sem cortar dividendo
     Sugestão: 2-5 cotas (R$230-575)

  → KNCR11 | Score 78 | DY 12.1% | P/VP 0.98 | Papel CDI
     Gestora Kinea | Indexado ao CDI — protege em juro alto

⚪ MANTER: PVBI11, XPML11

⚠️ ALERTAS DA CARTEIRA:
  HGRU11: vacância subiu para 18% — monitorar

💰 PRÓXIMOS DIVIDENDOS (7 dias):
  HGLG11: R$1.15/cota em 3 dias (15/06)
  KNCR11: R$0.98/cota em 10 dias (22/06)

💡 Você decide. O bot analisa, você executa.
```

## O que cada campo significa

**Score (0-100):** nota de qualidade do FII. Acima de 70 = boa compra.
Calculado com pesos em DY (30%), P/VP (20%), vacância (20%),
sustentabilidade (15%), histórico (10%), diversificação (5%).

**DY 11.5%:** rendimento dos últimos 12 meses dividido pelo preço atual.
Com Selic a 14.5%, FIIs precisam pagar bem acima de 8% para valer o risco.

**P/VP 0.92:** você está comprando o imóvel por 92% do valor patrimonial.
Abaixo de 1.0 = desconto. Acima de 1.2 = prêmio excessivo.

**Vac 3.0%:** apenas 3% dos espaços estão desocupados. Excelente.
Acima de 15% = sinal de alerta. Acima de 25% = venda.

**"6 anos sem cortar dividendo":** empresa consistente há 6 anos.
Isso é o critério mais importante para Barsi: consistência.

## Frequência dos sinais

| Sinal | Quando | O que contém |
|---|---|---|
| Screening semanal | Segunda 8h | Comprar, vender, manter + motivos detalhados |
| Status diário | Dia útil 18h30 | PnL do dia, dividendos próximos, saúde do portfólio |
| Alerta de qualidade | Imediato | Vacância subiu, dividendo cortado, gestora mudou |
| Relatório mensal | Dia 1 | Renda recebida, comparativo Selic, projeções |
| Relatório anual | 2 jan | Performance completa, DSR, evolução da renda |
| Kill-switch | Imediato | Bot parou — drawdown > 20% — ação necessária |

---

# PARTE 3 — COMO INSTALAR E CONFIGURAR

## Pré-requisitos

- Python 3.10 ou superior instalado
- Conta no Nubank/NuInvest aberta
- Celular com Telegram instalado
- Conta gratuita no brapi.dev

## Passo 1: Baixar e instalar

```bash
# Baixe o ZIP que foi gerado e extraia em uma pasta
# Ou se tiver o projeto no GitHub:
git clone https://github.com/seu-usuario/quantbot-ml
cd quantbot-ml

# Instala as dependências
pip install -r requirements.txt
```

## Passo 2: Criar o bot no Telegram (5 minutos)

1. Abra o Telegram no celular
2. Pesquise: @BotFather
3. Digite: /newbot
4. Escolha um nome: ex "MeuFinanceBot"
5. Escolha um username: ex "meubotfinance_bot"
6. Copie o token que aparecer (parece: "110201543:AAHdqTcvCH1vGWJxfSeofSOs0K37Y38G0fQ")
7. IMPORTANTE: mande qualquer mensagem para o seu novo bot
8. Abra no navegador: https://api.telegram.org/bot{TOKEN}/getUpdates
   (substitua {TOKEN} pelo token copiado)
9. Procure o número em "id" dentro de "chat" — esse é seu CHAT_ID

## Passo 3: Token da brapi.dev (2 minutos)

1. Acesse: brapi.dev
2. Crie conta gratuita (plano Free)
3. Copie o token da sua conta

## Passo 4: Configurar o arquivo .env

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Abra o .env em qualquer editor e preencha:
```

Conteúdo do .env:
```
BRAPI_TOKEN=seu_token_da_brapi_aqui
QUANTBOT_TELEGRAM_TOKEN=110201543:AAHdqTcvCH1vGWJxfSeofSOs0K37Y38G0fQ
QUANTBOT_TELEGRAM_CHAT_ID=123456789
CAPITAL_INICIAL=1000
META_RENDA_MENSAL=200
APORTE_MENSAL=100
BOT_MODE=paper
```

## Passo 5: Configuração inicial

```bash
python main_income.py --setup 1000 --meta 200 --aporte 100
```

Este comando vai:
- Calcular a alocação do seu capital (você vs bot)
- Mostrar as projeções de quando você atinge cada meta de renda
- Criar os arquivos de dados necessários
- Confirmar que o Telegram está funcionando (você receberá uma mensagem)

## Passo 6: Primeiro screening

```bash
python main_income.py --weekly
```

Você receberá no Telegram as primeiras recomendações baseadas em dados reais.

---

# PARTE 4 — COMANDOS COMPLETOS

## main_income.py — todos os comandos

```bash
# Configuração inicial
python main_income.py --setup 1000 --meta 200 --aporte 100

# Ciclo diário (status + alertas)
python main_income.py --daily

# Screening semanal (recomendações de compra/venda)
python main_income.py --weekly

# Relatório mensal
python main_income.py --monthly

# Status atual do portfólio
python main_income.py --status

# Ver projeção de renda
python main_income.py --project 1000 100 30
# formato: --project CAPITAL APORTE_MENSAL ANOS

# Confirmar uma compra que você fez
python main_income.py --confirm HGLG11 5 115.50 COMPRAR
# formato: --confirm TICKER N_COTAS PRECO_MEDIO COMPRAR|VENDER

# Ativar kill-switch (para o bot manualmente)
python main_income.py --killswitch on

# Reativar o bot
python main_income.py --killswitch off
```

## Exemplos práticos

```bash
# Comprei 3 cotas do HGLG11 a R$120,50 cada
python main_income.py --confirm HGLG11 3 120.50 COMPRAR

# Vendi todas as 5 cotas do LIXO11
python main_income.py --confirm LIXO11 5 48.00 VENDER

# Quero ver quanto tempo para ter R$500/mês com R$2.000 e aportando R$200
python main_income.py --project 2000 200 20
```

---

# PARTE 5 — COMO FAZER NO NUINVEST

## Abrir conta (se ainda não tem)

O NuInvest é a corretora do Nubank. Se você já tem conta no Nubank:
- Abra o app do Nubank
- Toque em Investimentos (ícone de gráfico)
- A conta do NuInvest está integrada

## Comprar um FII (passo a passo exato)

1. App Nubank → Investimentos
2. Toque em "Ações, FIIs e ETFs"
3. Na barra de busca, digite o ticker: ex "HGLG11"
4. Toque no resultado que aparecer
5. Toque em "Comprar"
6. Selecione "Por melhor preço" (mais fácil para iniciantes)
7. Quantidade: coloque o número de cotas que o bot sugeriu (ex: 3)
8. Confirme com sua assinatura eletrônica (os 4 dígitos do NuInvest)
9. Pronto! A compra será liquidada em 2 dias úteis (D+2)

## Transferir dinheiro para o NuInvest

- Nubank → Investimentos → Transferir para investimentos
- É instantâneo via PIX interno Nubank → NuInvest
- Mínimo: R$1 (pode comprar 1 cota de FII barato)

## Receber dividendos

- Caem automaticamente na conta do NuInvest no dia de pagamento
- O NuInvest envia uma notificação
- Você pode ver no extrato: Investimentos → Extrato
- O bot também detecta via brapi.dev e atualiza o calendário

## Horário de funcionamento da bolsa

- Segunda a sexta, das 10h às 17h (horário de Brasília)
- Fora desse horário, suas ordens ficam agendadas para a abertura
- Feriados: bolsa fica fechada

---

# PARTE 6 — COMO AUTOMATIZAR OS CICLOS

## No Windows (Agendador de Tarefas)

1. Pressione Win+R → digita "taskschd.msc" → Enter
2. Criar Tarefa Básica → Diária → 18:30
3. Ação: Iniciar programa
4. Programa: `C:\Users\SeuUsuario\AppData\Local\Programs\Python\Python312\python.exe`
5. Argumentos: `C:\caminho\quantbot\main_income.py --daily`
6. Repetir para --weekly (Segunda, 8h) e --monthly (Dia 1, 9h)

## No Mac/Linux (crontab)

```bash
# Abra o editor de cron
crontab -e

# Adicione estas linhas (ajuste o caminho):
30 18 * * 1-5 cd /Users/seunome/quantbot && python main_income.py --daily >> logs/daily.log 2>&1
0 8 * * 1 cd /Users/seunome/quantbot && python main_income.py --weekly >> logs/weekly.log 2>&1
0 9 1 * * cd /Users/seunome/quantbot && python main_income.py --monthly >> logs/monthly.log 2>&1
```

## Dashboard visual

Abra o arquivo `dashboard/index.html` em qualquer navegador.
Não precisa de servidor. Basta dar duplo clique no arquivo.
Funciona offline com dados demo, e com brapi_token conecta dados reais.

---

# PARTE 7 — OS 3 PERFIS DE ALOCAÇÃO

## Perfil Conservador (recomendado para começar com R$1.000)

| Balde | Quem controla | % | Ativos |
|---|---|---|---|
| Reserva emergência | Você | 30% | Nubank Caixinha (100% CDI) |
| Ações BESTIES | Você | 40% | ITUB4, BBAS3, EGIE3, TAEE11, BBSE3 |
| FIIs (bot recomenda) | Bot | 20% | HGLG11, KNCR11, PVBI11 |
| Ações DY alto (bot) | Bot | 7% | BEES4, ALOS3, CXSE3 |
| Caixa oportunidade | Bot | 3% | Reservado para crises |

## Perfil Moderado (quando chegar em R$5.000)

| Balde | Quem controla | % |
|---|---|---|
| Reserva emergência | Você | 20% |
| Ações BESTIES | Você | 35% |
| FIIs diversificados | Bot | 30% |
| Ações DY alto | Bot | 10% |
| Caixa | Bot | 5% |

## Perfil Arrojado (quando chegar em R$20.000+)

| Balde | Quem controla | % |
|---|---|---|
| Reserva emergência | Você | 10% |
| Ações BESTIES | Você | 35% |
| FIIs diversificados | Bot | 35% |
| Ações DY alto | Bot | 15% |
| Caixa | Bot | 5% |

**Regra de ouro:** o bot NUNCA controla mais de 50% do seu capital.

---

# PARTE 8 — TODOS OS MÓDULOS E O QUE FAZEM

## validation/ — Validação anti-overfitting

**purged_cv.py:** Cross-validation sem vazamento temporal. O k-fold normal
do scikit-learn deixa o modelo "ver o futuro" — este não. Usa purging
(remove amostras que se sobrepõem temporalmente) e embargo (bloqueia
amostras após o teste). Evita que o backtest minta sobre o desempenho real.

**deflated_sharpe.py:** Detecta se o bom resultado do backtest é skill real
ou pura sorte. O Deflated Sharpe Ratio corrige pelo número de configurações
testadas — quanto mais você "procurou" pelo melhor resultado, mais difícil
é provar que ele é real. DSR > 0.95 = evidência de skill.

**pbo.py:** Probability of Backtest Overfitting. Responde: "se eu escolho
a estratégia mais lucrativa no histórico, qual a chance de ela ser a pior
fora da amostra?" PBO < 0.20 = baixo risco de overfitting.

**validated_trainer.py:** Junta tudo: treina o modelo com cross-validation
purgada, calcula o track record out-of-sample real, e aplica DSR.

## income/ — Renda passiva (o coração)

**fii_screener.py:** Analisa qualquer FII com 9 critérios profissionais.
Reprovação em qualquer critério = FII eliminado. Score final 0-100 para
ranking. Base de referência: análises do BTG, XP e Santander (2026).

**barsi_screener.py:** Aplica o método Barsi às ações. Exige setor BESTIES,
histórico de pagamentos, payout mínimo, ROE, alavancagem máxima. Calcula
o preço teto pelo método Bazin (dividendo anual / 8% de referência).

**income_projector.py:** Calcula quando você atinge cada meta de renda
passiva. Considera: capital atual, aportes mensais, reinvestimento total
dos dividendos, e crescimento do DY ao longo do tempo (~3% a.a.).

**portfolio_allocator.py:** Divide o capital automaticamente entre os
baldes por perfil. Salva em disco e carrega a cada execução.

**dividend_calendar.py:** Rastreia todos os dividendos da carteira.
Registra agendados e pagos, projeta o próximo mês, gera alertas quando
um pagamento está próximo.

**auto_screener.py:** Integra o brapi_client com o FII Screener e roda
automaticamente. Compara o resultado com seu portfólio atual e gera as
recomendações de COMPRAR/VENDER/MANTER.

## data/ — Fontes de dados

**brapi_client.py:** Acessa a brapi.dev (CVM + Banco Central) para cotações,
dividendos, indicadores de FIIs e dados macro. Cache local em disco, retry
automático, rate limiting de 500ms entre requests.

**cvm_client.py:** Acessa o Portal Dados Abertos da CVM diretamente.
Dados oficiais e regulatórios: PL real, cotistas, resultado contábil.
Permite validar os dados da brapi contra a fonte primária.

**news_sentiment.py:** Analisa o sentimento de notícias financeiras em
português. Léxico com 1.200+ termos calibrados. Busca via Google News RSS.
Classifica: POSITIVO / NEUTRO / NEGATIVO. Sinal adicional no screening.

## risk/ — Proteção do capital

**kill_switch.py:** Para o sistema automaticamente em 5 situações:
1. Drawdown total > 20% do capital
2. Drawdown diário > 5%
3. 5 perdas consecutivas (cooldown 24h)
4. Arquivo killswitch.flag criado manualmente
5. Posição única > 25% do capital

Estado persistido em disco — sobrevive a restart.

## core/ — Orquestração

**orchestrator.py:** Coordena todos os módulos. Três ciclos: diário
(18h30), semanal (segunda 8h), mensal (dia 1 9h). Lê portfólio, roda
screener, gera e envia relatório, atualiza calendário de dividendos.

**reports.py:** Formata e envia relatórios via Telegram e email. Relatórios
diário, semanal, mensal e anual com dados estruturados e fáceis de ler.

**decision_logger.py:** Registra cada decisão em JSON Lines por data.
Tipo, ticker, contexto completo, confiança, outcome. Base para o bot
aprender com o histórico via get_learning_features().

## backtest/ — Simulação honesta

**realistic_backtester.py:** Simula trades com custos reais da B3:
corretagem (0.05%), spread (0.05%), emolumentos B3 (0.0325%), slippage,
IR com isenção de R$20k/mês. Contabilidade correta de caixa — sem
double counting.

## dashboard/ — Visualização

**index.html:** Dashboard interativo de 20KB. Projeção de renda com
sliders, screening de FIIs, calendário de dividendos, alocação dos 3
perfis, sentimento de notícias. Abre diretamente no navegador, sem servidor.

---

# PARTE 9 — PROJEÇÕES REALISTAS

## Com R$1.000 inicial + R$100/mês + DY 11.5%

| Tempo | Capital acumulado | Renda passiva/mês |
|---|---|---|
| Hoje | R$ 1.000 | R$ 9,58 |
| 1 ano | R$ 2.300 | R$ 22 |
| 3 anos | R$ 5.400 | R$ 52 |
| 5 anos | R$ 9.500 | R$ 91 |
| 10 anos | R$ 22.000 | R$ 211 |
| 20 anos | R$ 100.000 | R$ 958 |
| 30 anos | R$ 430.000 | R$ 4.121 |

## Quando você atinge cada meta

| Meta de renda/mês | Aportando R$100 | Aportando R$200 | Aportando R$500 |
|---|---|---|---|
| R$ 50/mês | 5,5 anos | 3,5 anos | 1,5 anos |
| R$ 100/mês | 8,5 anos | 5 anos | 2,5 anos |
| R$ 300/mês | 14 anos | 9 anos | 5 anos |
| R$ 500/mês | 18 anos | 12 anos | 7 anos |
| R$ 1.000/mês | 23 anos | 16 anos | 10 anos |

**O fator mais importante não é o DY — é a consistência de aportar e reinvestir.**

---

# PARTE 10 — IMPOSTOS (SIMPLES E CLARO)

## FIIs — Isentos de IR para pessoa física

- Dividendos mensais dos FIIs: 100% isentos de IR
- Você não paga nada sobre o que recebe mensalmente
- Declara apenas como "Bens e Direitos" no IRPF anual

## Ações — Isenção até R$20.000/mês em vendas

- Swing trade: vendas totais no mês ≤ R$20.000 = isento
- Swing trade: vendas > R$20.000 = 15% sobre o lucro via DARF
- Dividendos de ações: isentos
- Day trade: sempre 20%, sem isenção

## Com R$1.000 — você praticamente nunca paga IR

Com uma conta pequena, suas vendas mensais raramente passarão de R$20.000.
O bot opera em swing trade por design — estratégica e fiscalmente melhor.

## Documentação para o IRPF

O NuInvest gera automaticamente o "Informe de Rendimentos" todo ano.
O bot também registra todas as operações no decision_logger para que
você tenha um histórico completo.

---

# PARTE 11 — DICAS E BOAS PRÁTICAS

## As 5 regras que você não pode quebrar

1. **Nunca venda no pânico.** FII caiu 15%? Está em promoção.
   A renda passiva mensal não mudou — só o preço da cota.

2. **Reinvista 100% dos dividendos por pelo menos 5 anos.**
   O efeito bola de neve só aparece na segunda metade do prazo.

3. **Aporte todo mês, mesmo que seja pouco.**
   R$50 aportado toda semana vence timing de mercado.

4. **Não olhe a carteira todo dia.**
   Uma vez por semana é suficiente. O bot cuida do resto.

5. **Respeite o kill-switch.**
   Se o bot parou, há uma razão. Revise antes de reativar.

## Frequência ideal de uso

| Ação | Frequência | Tempo |
|---|---|---|
| Executar recomendação do bot | Semanal | 5 minutos |
| Verificar dashboard | Semanal | 2 minutos |
| Confirmar dividendos recebidos | Mensal | 2 minutos |
| Revisar projeções e metas | Mensal | 5 minutos |
| Revisar alocação dos baldes | Trimestral | 10 minutos |

## O que fazer quando o bot manda uma recomendação de VENDER

1. Leia o motivo com atenção (vacância? corte de dividendo? P/VP alto?)
2. Se concorda: execute no NuInvest e confirme para o bot
3. Com o dinheiro da venda: siga a próxima recomendação de COMPRAR
4. O bot faz a rotação da carteira de forma gradual

## O que fazer se o kill-switch ativar

```bash
# Ver o motivo
cat data/killswitch.flag

# Revisar o estado do risco
python main_income.py --status

# Se entender e concordar em reativar:
python main_income.py --killswitch off
```

---

# PARTE 12 — STATUS DO SISTEMA

## O que está completo e funcionando

- Análise quant com DSR, PBO, CPCV anti-overfitting
- FII Screener com 9 critérios profissionais
- Barsi Screener com teto Bazin
- Income Projector com bola de neve
- Portfolio Allocator nos 3 perfis
- Dividend Calendar com alertas
- Auto Screener integrado com brapi.dev real
- Decision Logger para aprendizado
- Sentimento de notícias (léxico BR calibrado)
- Dados CVM oficiais com validação cruzada
- Kill-switch com múltiplos níveis de proteção
- Backtester com custos reais e IR correto
- Notificações Telegram e Email
- Orchestrator com ciclos automáticos
- Dashboard interativo (HTML, abre no browser)
- CLI completo (main_income.py)
- 118 testes automatizados passando

## Melhorias possíveis no futuro (não urgentes)

- FinBERT full (requer GPU/cloud — a versão atual de léxico é suficiente)
- App mobile nativo
- Dashboard web com servidor (atualização automática)
- Integração com mais fontes de dados (Fundamentus, Status Invest)

---

# PARTE 13 — REFERÊNCIAS E PESQUISAS

## Metodologia quantitativa

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
  — Purged K-Fold, CPCV, Triple Barrier labeling
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
  — Journal of Portfolio Management, 40(5)
- Bailey et al. (2016). "The Probability of Backtest Overfitting"
  — Journal of Computational Finance, 20(4)

## Método Barsi

- Louise Barsi: setores BESTIES para 2025-2026 (CNN Brasil, Dez 2024)
- Preço teto Bazin: dividendo anual / 8% (referência ajustada para Selic alta)
- Filosofia: renda passiva > valorização. Horizonte 20-30 anos.

## Regulatório

- CVM Instrução 571/2015: informes mensais de FII (Anexo 39-I)
- Lei 11.033/2004: isenção de IR em vendas de ações ≤ R$20k/mês
- FIIs: isenção de IR sobre dividendos para pessoa física

## Fontes de dados

- brapi.dev: dados da CVM e Banco Central do Brasil
- Portal Dados Abertos CVM: dados.cvm.gov.br
- Google News RSS: notícias financeiras sem autenticação
- Selic e IPCA: Banco Central do Brasil via brapi.dev

---

*QuantBot ML v6.0 — Sistema educacional de análise de investimentos.*
*Não é recomendação financeira. Investir envolve riscos de perda de capital.*
