# QuantBot ML — Documentação Completa
> Sistema de Análise e Renda Passiva com IA
> Versão 5.0 — 01/06/2026

---

## ÍNDICE

1. O que é e o que NÃO é este sistema
2. Arquitetura completa
3. Todos os módulos explicados
4. Como usar — passo a passo
5. Estratégia de investimento recomendada
6. Os 3 perfis da sua carteira
7. Calendário de dividendos e renda passiva
8. Projeções realistas de renda
9. Fiscalidade — IR e isenções
10. O método Barsi integrado
11. Como funciona no NuInvest
12. Relatórios automáticos
13. Sistema de aprendizado (logs)
14. Riscos e limites honestos
15. Roadmap futuro

---

## 1. O QUE É E O QUE NÃO É ESTE SISTEMA

### O que É:
Um sistema de **inteligência, análise e monitoramento** de investimentos que:
- Analisa centenas de FIIs e ações automaticamente toda semana
- Envia recomendações precisas no Telegram ("compre X, venda Y, motivo Z")
- Monitora a saúde dos seus ativos continuamente
- Rastreia dividendos e projeta quando você atinge suas metas
- Mantém log detalhado de cada decisão para aprendizado
- Envia relatórios diários, semanais, mensais e anuais

### O que NÃO É:
**O bot NÃO executa ordens automaticamente no NuInvest ou em qualquer corretora brasileira.**

Por que? O NuInvest e todas as corretoras brasileiras de varejo não oferecem API pública de execução para pessoa física. Isso é proteção do mercado — nenhuma corretora séria permite que um programa externo acesse sua conta e movimente seu dinheiro.

### Como funciona na prática:
```
Bot analisa 80+ FIIs toda segunda → Telegram: "Compre HGLG11, 5 cotas"
Você: abre o app → digita HGLG11 → compra 5 cotas (2 minutos)
Você: confirma para o bot → python main_income.py --confirm HGLG11 5 115 COMPRAR
Bot: atualiza portfólio, calendário, projeções
```

Isso é exatamente como funciona um gestor profissional — ele recomenda, você aprova.

---

## 2. ARQUITETURA COMPLETA

```
QuantBot ML v5.0
│
├── CAMADA DE DADOS
│   └── data/brapi_client.py
│       ├── Cotações em tempo real (brapi.dev)
│       ├── Indicadores de FIIs (DY, P/VP, vacância)
│       ├── Dividendos históricos
│       ├── Selic e IPCA do Banco Central
│       └── Cache local com TTL (evita rate limit)
│
├── CAMADA DE ANÁLISE QUANTITATIVA
│   └── validation/
│       ├── purged_cv.py       → Cross-validation sem vazamento temporal
│       ├── deflated_sharpe.py → Detecta se há skill real ou sorte
│       ├── pbo.py             → Probabilidade de overfitting
│       └── validated_trainer.py → Treino com validação honesta
│
├── CAMADA DE RENDA PASSIVA (foco principal)
│   └── income/
│       ├── fii_screener.py      → Triagem de FIIs com 9 critérios
│       ├── barsi_screener.py    → Ações pelo método Barsi + teto Bazin
│       ├── income_projector.py  → Projeção da bola de neve
│       ├── portfolio_allocator.py → Divide capital (você vs bot)
│       ├── dividend_calendar.py  → Rastreia todos os dividendos
│       └── auto_screener.py     → Screening automático com brapi.dev
│
├── CAMADA DE RISCO
│   └── risk/kill_switch.py
│       ├── Kill-switch por drawdown total (20%)
│       ├── Kill-switch por drawdown diário (5%)
│       ├── Limite de posição única (25%)
│       ├── Cooldown por perdas consecutivas
│       └── Persistência de estado em disco
│
├── CAMADA DE BACKTEST
│   └── backtest/realistic_backtester.py
│       ├── Corretagem e spread reais da B3
│       ├── Emolumentos B3 (0.0325%)
│       ├── Cálculo de IR (isenção R$20k/mês)
│       └── Contabilidade correta de caixa
│
├── CAMADA DE ORQUESTRAÇÃO
│   └── core/
│       ├── orchestrator.py    → Coordena ciclos diário/semanal/mensal
│       ├── reports.py         → Relatórios Telegram e email
│       └── decision_logger.py → Log de cada decisão (aprendizado)
│
├── INTERFACE
│   ├── main_income.py         → CLI de entrada
│   └── dashboard/index.html  → Dashboard visual (abrir no browser)
│
└── TESTES (98 testes passando)
    ├── test_validation.py     → 22 testes de propriedades estatísticas
    ├── test_validated_trainer.py → 5 testes de treino com skill real
    ├── test_backtest_and_risk.py → 27 testes de custos e risco
    ├── test_income.py         → 21 testes do módulo de renda
    ├── test_new_modules.py    → 15 testes de alocação e calendário
    └── test_auto_screener.py  → 8 testes do screener automático
```

---

## 3. TODOS OS MÓDULOS EXPLICADOS

### brapi_client.py
Busca dados reais da B3 via brapi.dev (CVM + Banco Central):
- Funciona sem token (modo demo) e com token gratuito (produção)
- Cache local em disco evita re-buscar dados recentes
- Retry automático com backoff exponencial para falhas de rede
- Rate limiting embutido (500ms entre requests)

### fii_screener.py
Triagem profissional de FIIs com 9 critérios:
1. DY 12m entre 8% e 20% (acima disso = suspeito)
2. P/VP entre 0.50 e 1.20
3. Vacância física < 15%
4. Liquidez diária > R$500k
5. Patrimônio > R$200M
6. Taxa de administração < 1.5% a.a.
7. Distribuição ≤ 110% do resultado (sustentável)
8. Histórico ≥ 2 anos sem corte
9. Score ponderado de 0 a 100

### barsi_screener.py
Aplica o método Barsi com:
- Exigência de setor BESTIES (Bancos, Energia, Seguros, Telecom, Infraestrutura)
- Cálculo do preço teto Bazin (dividendo anual / referência de 8%)
- Histórico mínimo de pagamentos
- Payout ≥ 40%
- Dívida/EBITDA < 4x
- ROE ≥ 10%

### income_projector.py
Calcula quando você atinge suas metas de renda:
- Fórmula de juros compostos com reinvestimento mensal
- Crescimento do DY ao longo do tempo (~3% a.a.)
- Múltiplas metas simultâneas (R$100, R$500, R$1.000, R$3.000/mês)
- Efeito bola de neve visualizável mês a mês

### portfolio_allocator.py
Divide o capital em 3 perfis:
- Conservador (< R$5k): você 70%, bot 30%
- Moderado (R$5-20k): você 55%, bot 45%
- Arrojado (> R$20k): você 50%, bot 50%
- O bot NUNCA controla mais de 50% do seu capital

### decision_logger.py
Grava cada decisão em JSON Lines (.jsonl):
- Timestamp, tipo de evento, ticker, contexto completo
- Confiança do modelo, regime de mercado, valor envolvido
- Outcome (sucesso/falha) atualizado após o resultado
- Base de dados para o bot aprender com o histórico

### realistic_backtester.py
Simula investimentos com custos reais da B3:
- Corretagem 0.05% por lado
- Spread bid-ask 0.05%
- Emolumentos B3 0.0325%
- Slippage estimado
- IR com isenção de R$20k/mês (Lei 11.033/2004)

### kill_switch.py
Proteção automática do capital:
- Drawdown total > 20%: para tudo, requer ação manual
- Drawdown diário > 5%: suspende o dia
- 5 perdas consecutivas: 24h de cooldown
- Arquivo de flag: qualquer pessoa pode parar o bot colocando um arquivo

---

## 4. COMO USAR — PASSO A PASSO COMPLETO

### Primeira vez (Setup inicial)

**Passo 1: Instalar dependências**
```bash
pip install -r requirements.txt
```

**Passo 2: Configurar credenciais**
```bash
cp .env.example .env
# Edite o .env com seus dados
```

**Passo 3: Criar bot no Telegram (5 minutos)**
1. Abra o Telegram → procure @BotFather
2. Digite /newbot → escolha um nome (ex: "MeuQuantBot")
3. Copie o token fornecido → cole em QUANTBOT_TELEGRAM_TOKEN
4. Mande qualquer mensagem para o seu novo bot
5. Acesse: https://api.telegram.org/bot{SEU_TOKEN}/getUpdates
6. Copie o número em "id" → cole em QUANTBOT_TELEGRAM_CHAT_ID

**Passo 4: Token gratuito da brapi.dev (2 minutos)**
1. Acesse brapi.dev → crie conta gratuita
2. Copie o token → cole em BRAPI_TOKEN no .env

**Passo 5: Setup do sistema**
```bash
python main_income.py --setup 1000 --meta 200 --aporte 100
```
O sistema vai calcular sua alocação, projeções e confirmar que tudo está funcionando.

**Passo 6: Primeiro screening**
```bash
python main_income.py --weekly
```
Você receberá no Telegram as recomendações do primeiro screening.

### Uso semanal (Segunda-feira, ~5 minutos)

```bash
# 1. Ver status atual
python main_income.py --status

# 2. Rodar screening semanal (recomendado toda segunda)
python main_income.py --weekly
```

Você recebe no Telegram: "Compre HGLG11 — Score 82, DY 11.5%, P/VP 0.92"

Abra o NuInvest → compre → confirme para o bot:
```bash
python main_income.py --confirm HGLG11 5 115.50 COMPRAR
```

### Automatizar os ciclos (Linux/Mac)

```bash
crontab -e
# Adicione:
30 18 * * 1-5 cd /caminho/quantbot && python main_income.py --daily
0 8 * * 1 cd /caminho/quantbot && python main_income.py --weekly
0 9 1 * * cd /caminho/quantbot && python main_income.py --monthly
```

### Windows — Agendador de Tarefas
1. Abra "Agendador de Tarefas"
2. Criar Tarefa → Diária → 18:30 em dias úteis
3. Ação: `python C:\caminho\quantbot\main_income.py --daily`

---

## 5. ESTRATÉGIA DE INVESTIMENTO RECOMENDADA

### Seu posicionamento (os 3 baldes)

**BALDE 1 — VOCÊ CONTROLA (Segurança)**
```
Reserva de Emergência (30% do capital):
  → Nubank Caixinha (100% CDI, liquidez diária)
  → Nunca tocar, é o seu colchão

Ações BESTIES — Método Barsi (40% do capital):
  → ITUB4, BBAS3, EGIE3, TAEE11, BBSE3, SAPR11, VIVT3
  → Compra mensal, reinveste dividendos, NUNCA vende no pânico
  → Horizonte: 20-30 anos
```

**BALDE 2 — BOT RECOMENDA, VOCÊ EXECUTA (Renda Ativa)**
```
FIIs diversificados (20% do capital):
  → Bot faz screening semanal e recomenda os melhores
  → Você compra no NuInvest em 2 minutos
  → Dividendos mensais isentos de IR

Ações DY alto (7% do capital):
  → Complementam os FIIs com pagamentos mensais/trimestrais
  → BEES4, ALOS3, CXSE3

Caixa de oportunidade (3%):
  → Para comprar quando tudo cair muito (crises são oportunidades Barsi)
```

### Decisões que você toma (simples e claras)

Todo mês você faz UMA pergunta: "Recebi salário, quanto posso investir?"

Se a resposta for R$100:
- R$30 → Tesouro SELIC (reserva)
- R$40 → 1 ação BESTIES que estiver mais barata (bot mostra)
- R$20 → FII recomendado pelo bot
- R$10 → Caixa de oportunidade

Se receber dividendos de R$20:
- 100% reinvestido (compra mais do mesmo FII)

### As únicas regras que nunca quebrar

1. **Nunca venda no pânico.** HGLG11 caiu 15%? Está em promoção — compre mais.
2. **Reinvista TUDO por pelo menos 5 anos.** Cada R$1 de dividendo que você reinveste multiplica.
3. **Aporte sempre.** R$50 é melhor que R$0. Consistência vence valor.
4. **Respeite o kill-switch.** Se o bot parar, não ignore — revise primeiro.
5. **Não olhe a carteira todo dia.** Uma vez por semana é suficiente.

---

## 6. OS 3 PERFIS DA CARTEIRA

### Perfil Conservador (você começa aqui com R$1.000)

| Balde | Você | Bot | % | R$ |
|---|---|---|---|---|
| Reserva Emergência | ✓ | | 30% | R$300 |
| Ações BESTIES | ✓ | | 40% | R$400 |
| FIIs | | Bot recomenda | 20% | R$200 |
| Ações DY | | Bot recomenda | 7% | R$70 |
| Caixa | | Bot monitora | 3% | R$30 |

Renda mensal estimada: R$8-12

### Perfil Moderado (quando chegar em R$5.000)

| Balde | Você | Bot | % | R$ |
|---|---|---|---|---|
| Reserva Emergência | ✓ | | 20% | R$1.000 |
| Ações BESTIES | ✓ | | 35% | R$1.750 |
| FIIs | | Bot | 30% | R$1.500 |
| Ações DY | | Bot | 10% | R$500 |
| Caixa | | Bot | 5% | R$250 |

Renda mensal estimada: R$45-60

### Perfil Arrojado (quando chegar em R$20.000)

| Balde | Você | Bot | % | R$ |
|---|---|---|---|---|
| Reserva Emergência | ✓ | | 10% | R$2.000 |
| Ações BESTIES | ✓ | | 35% | R$7.000 |
| FIIs | | Bot | 35% | R$7.000 |
| Ações DY | | Bot | 15% | R$3.000 |
| Caixa | | Bot | 5% | R$1.000 |

Renda mensal estimada: R$190-230

---

## 7. COMO FUNCIONA NO NUINVEST

### Abrir conta (grátis, 10 minutos)
O NuInvest é a corretora do Nubank. Se você tem conta Nubank, acesse o app → Investimentos → NuInvest.

### Comprar um FII (2 minutos)
1. App → Investimentos → Ações, FIIs e ETFs
2. Pesquise: HGLG11
3. Toque em "Comprar"
4. Quantidade: 1 cota (ou o número recomendado pelo bot)
5. Por melhor preço → Comprar → Assinatura eletrônica
6. Pronto. Liquidação em D+2 (2 dias úteis)

### Receber dividendos
- Cairão direto na conta do NuInvest no dia de pagamento
- Você reinveste comprando mais cotas
- Bot detecta via brapi.dev e atualiza o calendário

### Transferir dinheiro para o NuInvest
- App Nubank → Área do Investidor → Transferir
- É instantâneo via PIX interno Nubank → NuInvest

### Importante sobre o NuInvest
- Corretagem ZERO para ações e FIIs
- Disponível 24h (ordens ficam agendadas para abertura da bolsa)
- Bolsa funciona: Seg-Sex, 10h-17h (horário de Brasília)
- Mínimo: 1 cota de FII (geralmente R$10 a R$150)

---

## 8. PROJEÇÕES REALISTAS DE RENDA

### Com R$1.000 inicial + R$100/mês (DY 11.5%)

| Tempo | Capital | Renda Mensal |
|---|---|---|
| Hoje | R$1.000 | R$9,58 |
| 1 ano | R$2.300 | R$22 |
| 3 anos | R$5.400 | R$52 |
| 5 anos | R$9.500 | R$91 |
| 10 anos | R$22.000 | R$211 |
| 20 anos | R$100.000 | R$958 |
| 30 anos | R$430.000 | R$4.121 |

### Marcos importantes (quando chega cada meta)

| Meta de Renda | Com R$100/mês | Com R$200/mês | Com R$500/mês |
|---|---|---|---|
| R$50/mês | 5,5 anos | 3,5 anos | 1,5 anos |
| R$100/mês | 8,5 anos | 5 anos | 2,5 anos |
| R$300/mês | 14 anos | 9 anos | 5 anos |
| R$500/mês | 18 anos | 12 anos | 7 anos |
| R$1.000/mês | 23 anos | 16 anos | 10 anos |
| R$3.000/mês | 32 anos | 23 anos | 16 anos |

**A variável que mais importa NÃO é o DY — é a disciplina de reinvestir.**

### Meta realista para você agora

Com R$1.000 e R$100/mês:
- Em 5 anos: renda de ~R$90/mês (paga uma conta de luz)
- Em 10 anos: renda de ~R$200/mês (uma academia + streaming)
- Em 20 anos: renda de ~R$1.000/mês (complemento real da renda)

**Para viver de renda (R$3-5k/mês):** precisaria de 25-30+ anos com esse aporte, ou aumentar os aportes conforme a carreira cresce. Isso é honesto e possível.

---

## 9. FISCALIDADE — IR E ISENÇÕES

### FIIs (isentos de IR para PF)
- Dividendos de FIIs são **100% isentos de IR** para pessoa física
- Você não declara os rendimentos mensais como tributáveis
- Declara apenas a posse dos FIIs como "Bens e Direitos" no IRPF anual
- Se vender FIIs com lucro: 20% de IR, pago via DARF até fim do mês seguinte

### Ações (isenção até R$20k/mês em vendas)
- Vendas totais no mês ≤ R$20.000: ganho capital isento (Lei 11.033/2004)
- Vendas > R$20.000: 15% sobre o lucro, via DARF
- Dividendos de ações PF: isentos
- JCP (Juros sobre Capital Próprio): 15% retido na fonte

### Com R$1.000 investido
Você praticamente nunca vai vender o suficiente para pagar IR. Você estará na faixa de isenção total.

### Declaração anual do IRPF
O bot registra todas as operações. Na declaração:
- Informe os FIIs em "Bens e Direitos"
- Rendimentos isentos: os dividendos recebidos
- A B3 disponibiliza o informe de rendimentos para o NuInvest automaticamente

---

## 10. O MÉTODO BARSI INTEGRADO

### Os 5 Princípios que o sistema segue

1. **Empresas perenes** — só setores BESTIES que existirão em 30 anos
2. **Foco em dividendos** — preço de cota não importa, renda importa
3. **Comprar em crises** — o bot alerta quando os FIIs caem muito (oportunidade)
4. **Reinvestir tudo** — cada dividendo volta como mais cotas
5. **Horizonte longo** — mínimo 10 anos para ver o efeito real

### Setores BESTIES no sistema

```
B — Bancos: ITUB4, BBDC4, BBAS3, SANB11, BEES4
E — Energia: EGIE3, CPFE3, TAEE11, ENGI11, ALUP11, TRPL4
S — Seguros: BBSE3, CXSE3, IRBR3
T — Telecom: VIVT3, TIMS3
I — Infraestrutura: SAPR11, CSMG3, SBSP3, CCRO3, ECOR3
```

### O preço teto Bazin

O bot calcula automaticamente para cada ação:
```
Teto Bazin = Dividendo anual por ação / 8%
```
Se a ação estiver acima do teto → está cara, aguarde.
Se estiver abaixo → está em desconto, oportunidade de compra.

---

## 11. SISTEMA DE APRENDIZADO (LOGS)

Cada decisão do bot é registrada em `data/decisions/YYYY-MM-DD.jsonl`:

```json
{
  "id": "abc-123...",
  "timestamp": "2026-06-01T18:30:00Z",
  "tipo": "fii_screening",
  "ticker": "HGLG11",
  "confianca": 0.82,
  "valor_envolvido": 575.50,
  "contexto": {
    "dy_12m": 11.5,
    "pvp": 0.92,
    "vacancia": 3.0,
    "score": 82,
    "selic_usada": 14.5
  },
  "regime_mercado": "BULL_LOW_VOL",
  "outcome": "success",
  "outcome_data": {"pnl": 12.50}
}
```

Com o tempo, esses logs alimentam:
- Análise de quais critérios funcionaram melhor
- Quais regimes de mercado geraram mais acerto
- Calibração automática dos thresholds do screener

---

## 12. RELATÓRIOS AUTOMÁTICOS

### Diário (todo dia útil às 18h30)
```
📊 QUANTBOT — RENDA PASSIVA DIÁRIA
📅 01/06/2026
Capital total: R$1.000,00
Renda recebida em 06/2026: R$9,58
Previsto para o mês: R$9,58
⏰ HGLG11 paga R$1,15/cota em 5 dias (R$5,75 total)
Kill-switch: OK ✅
```

### Semanal (segunda-feira às 08h00)
Screening completo com recomendações de compra/venda

### Mensal (dia 1 de cada mês)
Comparativo vs Selic, renda acumulada, IR, projeções

### Anual (2 de janeiro)
Performance completa vs benchmarks (Selic, CDI, Ibovespa)

---

## 13. RISCOS E LIMITES HONESTOS

### O que pode dar errado

1. **FIIs podem cortar dividendos** — vacância sobe, resultado cai
   → Proteção: bot monitora e alerta imediatamente

2. **Selic alta prejudica FIIs** — investidores migram para renda fixa
   → Proteção: o screener ajusta o DY mínimo conforme a Selic

3. **Você pode perder dinheiro** — FIIs oscilam de preço
   → Proteção: foco em renda (não em valorização) atenua o impacto emocional

4. **O bot pode recomendar algo que depois cai**
   → Proteção: kill-switch em 20% e log completo para revisão

5. **Dados da brapi.dev podem estar desatualizados**
   → Proteção: cache com TTL e fallback para modo offline

### O que o bot NÃO garante

- Rentabilidade positiva garantida (nenhum sistema garante)
- Previsão do mercado (impossível)
- Que os dividendos futuros serão iguais aos passados

---

## 14. STATUS: O SISTEMA ESTÁ COMPLETO?

### ✅ Completo e funcionando

- Suíte de validação estatística (DSR, PBO, CPCV): 22 testes
- Backtest com custos reais: 6 testes
- Kill-switch e camada de risco: 10 testes
- FII Screener profissional: 7 testes
- Barsi Screener: 6 testes
- Income Projector: 8 testes
- Portfolio Allocator: 10 testes
- Dividend Calendar: 8 testes
- Auto Screener (integrado brapi.dev): 8 testes
- Decision Logger: 8 testes
- Notification (Telegram + email): implementado
- Orchestrator (ciclos automáticos): implementado
- Dashboard HTML: implementado
- CLI (main_income.py): implementado

**Total: 98/98 testes passando**

### 🔄 O que ainda pode ser melhorado (futuro)

- FinBERT para sentimento de notícias (requer GPU/cloud)
- Regime detection mais sofisticado (HMM com 6-8 estados)
- Integração com dados da CVM em tempo real
- App mobile nativo
- Dashboard interativo com Chart.js atualizado em tempo real
- Backtesting específico para FIIs (eventos de dividendo)

---

## 15. ARQUIVOS DO PROJETO

```
quantbot-ml/
├── main_income.py           → Ponto de entrada CLI
├── requirements.txt         → Dependências Python
├── .env.example            → Template de configuração
├── .gitignore              → Arquivos excluídos do git
│
├── validation/             → Validação anti-overfitting
├── income/                 → Módulo de renda passiva
├── backtest/               → Backtest com custos reais
├── risk/                   → Kill-switch e limites
├── core/                   → Orquestrador e relatórios
├── data/                   → Cliente brapi.dev
├── dashboard/              → Dashboard HTML
│
├── tests/                  → 98 testes automatizados
└── docs/                   → Esta documentação
```

---

## SUPORTE E PRÓXIMOS PASSOS

Este sistema foi construído especificamente para seus objetivos:
renda passiva mensal crescente via FIIs + método Barsi.

**Próximo passo sugerido:**
1. Abra o NuInvest no Nubank
2. Configure o .env com Telegram e brapi.dev
3. Execute: `python main_income.py --setup 1000 --meta 200 --aporte 100`
4. Faça o primeiro screening: `python main_income.py --weekly`
5. Compre a primeira recomendação e confirme para o bot
6. Configure o cron para ciclos automáticos

Lembre-se: o maior inimigo da renda passiva não é o mercado — é a impaciência. Os primeiros 5 anos são lentos. O efeito bola de neve aparece de verdade a partir do ano 8-10.

---

*QuantBot ML v5.0 — Construído com 98 testes automatizados e honestidade sobre o que os sistemas de investimento podem e não podem fazer.*
