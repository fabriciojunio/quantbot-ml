# QuantBot ML — Sistema Quantitativo de Trading

> Versão: 2.0 | Atualizado: 01/06/2026  
> Desenvolvido com padrões de engenharia de produção

---

## O que é este sistema

QuantBot ML é um sistema de trading algorítmico com validação estatística
anti-overfitting. Ele **não prevê o mercado com certeza** — isso é impossível.
O que ele faz é medir honestamente se há vantagem estatística real antes de
qualquer capital ser arriscado.

**Meta realista:** superar Selic + 5-10% com Sharpe > 1.0 e win rate 50-55%.

---

## Arquitetura

```
quantbot/
├── validation/          # Validação anti-overfitting (DSR, PBO, PurgedKFold)
├── backtest/            # Backtest com custos reais (corretagem, IR, spread)
├── risk/                # Kill-switch e camada de risco independente
├── models/              # Ensemble ML (RF + XGBoost + GB) + regime detection
├── data/                # Pipeline de dados (features, sentimento, macro)
├── core/                # Paper trading, relatórios, performance
├── config/              # Configurações e variáveis de ambiente
├── tests/               # 43+ testes automatizados
└── docs/                # Documentação e ADRs
```

---

## Como começar — passo a passo

### 1. Instalação

```bash
git clone https://github.com/seu-usuario/quantbot-ml
cd quantbot-ml
pip install -r requirements.txt
cp .env.example .env   # edite com suas configurações
```

### 2. Configurar variáveis de ambiente (.env)

```env
# Notificações (pelo menos um é obrigatório)
QUANTBOT_TELEGRAM_TOKEN=seu_token_aqui
QUANTBOT_TELEGRAM_CHAT_ID=seu_chat_id
QUANTBOT_EMAIL_FROM=seu@email.com
QUANTBOT_EMAIL_TO=destino@email.com
QUANTBOT_EMAIL_PASSWORD=sua_senha_app

# Corretora (só para live trading — depois do paper trading validado)
ALPACA_API_KEY=sua_chave
ALPACA_SECRET_KEY=sua_chave_secreta
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # paper trading
```

**Como obter Token do Telegram:**
1. Acesse @BotFather no Telegram
2. `/newbot` → siga as instruções → copie o token
3. Para o chat_id: mande uma mensagem pro seu bot, acesse
   `https://api.telegram.org/bot{TOKEN}/getUpdates` e veja o "id"

### 3. Configurar Telegram para receber relatórios

```bash
python scripts/setup_telegram.py
```

### 4. Rodar validação estatística nos modelos

```bash
# PRIMEIRO PASSO SEMPRE — antes de qualquer dinheiro
python scripts/run_validation.py --symbols PETR4 VALE3 ITUB4 BBDC4 --years 3
```

Interprete o resultado:
- **DSR > 0.90**: evidência de edge. Pode prosseguir para paper trading.
- **DSR < 0.90**: sem evidência. Não arrisque dinheiro. Revise as features.
- **PBO < 0.20**: baixo risco de overfitting. Bom sinal.
- **PBO > 0.50**: escolher pela performance histórica não tem valor.

### 5. Paper trading (dinheiro fictício, dados reais) — OBRIGATÓRIO

```bash
# Inicia paper trading — mínimo 3-6 meses antes de qualquer real
python main.py --mode paper --capital 10000 --symbols PETR4 VALE3
```

O sistema vai:
- Monitorar sinais em tempo real
- Simular trades com custos reais
- Te mandar relatório diário no Telegram
- Guardar todo o histórico em `data/paper_trading_state.json`

### 6. Verificar resultados do paper trading

```bash
python scripts/analyze_paper_trading.py
```

**Critérios para avançar para dinheiro real:**
- [ ] Pelo menos 3-6 meses de paper trading
- [ ] Sharpe ratio OOS > 0.8
- [ ] Win rate > 50%
- [ ] DSR > 0.90 nos dados do período
- [ ] Drawdown máximo dentro do limite esperado

### 7. Live trading (só se o paper trading validou)

```bash
# Mude ALPACA_BASE_URL para https://api.alpaca.markets
# E coloque seu capital real na Alpaca
python main.py --mode live --capital 1000
```

---

## Relatórios automáticos

O sistema envia automaticamente:

| Relatório | Quando | O que contém |
|-----------|--------|--------------|
| Diário | Fim de cada dia de mercado | PnL do dia, sinais, status do risco |
| Semanal | Toda segunda-feira | Resumo da semana, win rate, melhores/piores dias |
| Mensal | Dia 1 de cada mês | Comparativo vs Selic, IR pago, Sharpe, DSR atualizado |
| Anual | 2 de janeiro | Performance completa, validação estatística do ano |
| Alerta | Imediato | Kill-switch ativado, drawdown crítico, falha de dados |

---

## Kill-switch — proteção do capital

O sistema para automaticamente se:
- Perda diária > 3% do capital
- Perda total > 20% do pico (kill-switch permanente, requer ação manual)
- 5 trades perdedores consecutivos (cool-down 24h)
- 5+ trades por dia

Para ativar manualmente (para tudo imediatamente):
```bash
python scripts/killswitch.py --activate --reason "volatilidade extrema"
```

Para reativar após revisão:
```bash
python scripts/killswitch.py --deactivate
```

---

## Fiscalidade — Imposto de Renda

O sistema opera em **swing trade** (não day trade) por design:

- **Vendas ≤ R$20.000/mês:** ISENTO de IR. Você declara como rendimento isento.
- **Vendas > R$20.000/mês:** IR de 15% sobre o lucro, pago via DARF até o fim do mês seguinte.
- **Day trade:** sempre 20%, sem isenção — por isso o bot evita.

Com R$1.000, você ficará na faixa de isenção. O sistema calcula e registra tudo automaticamente. **Consulte um contador para sua situação específica.**

---

## Dividendos — Método Barsi (complementar)

Enquanto o bot opera ativamente, o método Barsi é uma estratégia paralela e de longo prazo:

**Setores BESTIES** (Barsi + Louise Barsi, 2024-2026):
- **B**ancos: ITUB4, BBDC4, SANB11
- **E**nergia: EGIE3, CPFE3, ENGI11
- **S**eguros: BBSE3, IRBR3
- **T**elecomunicações: VIVT3, TIMS3
- **I**nfraestrutura: SAPR11, CSMG3, TRPL4

**Estratégia Barsi básica:**
1. Compre ações dos setores BESTIES
2. Reinvista 100% dos dividendos por 10-30 anos
3. Nunca venda em pânico — as oscilações não importam, a renda importa
4. Dividendos são ISENTOS de IR na fonte para PF

Isso é **complementar** ao bot, não substituto. O bot tenta gerar alpha de curto prazo; o método Barsi constrói renda passiva permanente.

---

## Como investir em dólar

Opções disponíveis para brasileiros:

1. **BDRs na B3** (mais simples): Brazilian Depositary Receipts de ações americanas (AAPL34, MSFT34, AMZO34). Negociados em reais, mas expostos ao dólar e à empresa estrangeira.

2. **ETFs de dólar na B3** (IVVB11, BOVA11 com hedge cambial): fácil, sem conta externa.

3. **Alpaca (mercado americano):** o próprio sistema já suporta Alpaca, que opera ações americanas. Você envia dólares via câmbio, opera no S&P 500. Tributação: mesmo que B3 (15% acima de R$20k/mês).

4. **Avenue / XP International:** corretoras brasileiras com acesso ao mercado americano.

**Sugestão de alocação** (não é recomendação financeira, é framework geral):
- 60-70% Brasil (renda fixa + ações dividendos estilo Barsi)
- 20-30% dólar (BDRs ou ETFs americanos)
- 5-10% bot quant (capital que tolera perder)

---

## Divisão dos R$1.000 (framework para pensar)

Para um capital inicial de R$1.000, uma divisão conservadora seria:

| Destino | Valor | Objetivo |
|---------|-------|----------|
| Reserva de emergência (primeiro) | Fora disso | 6 meses de despesas em Tesouro Selic |
| Ações dividendos (Barsi) | R$500-700 | Renda passiva de longo prazo |
| ETF dólar (IVVB11) | R$200-300 | Proteção cambial |
| Bot quant (paper trading primeiro) | R$100-200 | Só depois de validado |

**Importante:** não invista o que vai fazer falta. O bot pode perder.

---

## Análise de hedge fund — O que um comitê diria

Avaliação como se fosse um comitê de investimentos profissional:

| Critério | Nota | Observação |
|----------|------|------------|
| Metodologia ML | 7/10 | Triple Barrier correto, walk-forward presente |
| Validação estatística | 8/10 | DSR + PBO agora implementados corretamente |
| Gestão de risco | 6/10 | Kill-switch existe, mas não testado em produção |
| Custos reais | 7/10 | Modelo de custos BR implementado |
| Track record | 0/10 | Zero — precisa de paper trading |
| **Nota geral** | **4.5/10** | Sólido para um projeto pessoal, longe de profissional |

**Chance de atingir 25% ao ano consistentemente por 10 anos: ~2-4%.**  
**Retorno realista em produção: 8-18% em anos bons, negativo possível.**  
**Isso é para uma conta pequena — custos de transação são proporcionalmente maiores.**

---

## Manutenção e melhoria contínua

O sistema aprende do histórico:
- Re-treino automático mensal com dados recentes
- Walk-forward contínuo com detecção de overfitting
- Os relatórios alimentam o aprendizado: padrões de erro são registrados

Para melhorar no futuro:
1. Adicionar mais dados macroeconômicos
2. FinBERT com fontes brasileiras reais (InfoMoney, Valor Econômico)
3. Mais features de order flow e microestrutura
4. Regime detection com 6-8 estados (HMM mais sofisticado)
5. Depois de 2+ anos de track record real, calibrar parâmetros com os dados reais

---

## Suporte e dúvidas

Este sistema é um projeto pessoal de aprendizado e investimento. Não é gerido
por nenhuma instituição financeira. Toda decisão de investimento é de sua
responsabilidade exclusiva.

Consulte sempre um profissional financeiro antes de alocar capital significativo.
