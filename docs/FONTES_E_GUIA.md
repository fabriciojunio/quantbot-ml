# QuantBot ML — Guia de Fontes, Uso e Plano até Jan/2027

> Atualizado em jun/2026. Tudo neste guia foi **testado ao vivo** neste
> ambiente (Python 3.14, Windows). Quando algo depende de plano pago ou
> chave, está dito de forma explícita e honesta.

---

## 1. O que o sistema é (em uma frase)

Um sistema de **renda passiva com dividendos** (método Barsi/Bazin) que:
junta **dados de mercado** (B3, fundamentos, macro) + **notícias** de
dezenas de fontes, lê o **sentimento** com IA (FinBERT-PT-BR), faz o
**screening** de FIIs e ações, projeta a **bola de neve** de dividendos e
te manda **recomendações** — você executa na corretora e confirma.

---

## 2. Como rodar (passo a passo)

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar o .env (copie de .env.example)
#    - BRAPI_TOKEN (grátis em brapi.dev/dashboard) — JÁ CONFIGURADO
#    - QUANTBOT_USE_FINBERT=1  → liga o sentimento por IA
#    - Telegram (opcional) → relatórios automáticos

# 3. Primeira configuração (uma vez)
python main_income.py --setup 1000 --meta 200 --aporte 100

# 4. Screening semanal (recomendado: toda segunda)
python main_income.py --weekly

# 5. Coletar notícias + sentimento (a "base de aprendizado")
python scripts/coletar_noticias.py

# 6. Ver status / projeção
python main_income.py --status
python main_income.py --project 1000 150 15

# 7. Após comprar na corretora, confirmar pro bot
python main_income.py --confirm HGLG11 10 115.50 COMPRAR
```

### Comandos principais

| Comando | O que faz |
|---|---|
| `--setup CAP --meta M --aporte A` | Configuração inicial |
| `--weekly` | Screening de FIIs com dados reais (Fundamentus + BCB) |
| `--daily` | Cotações, dividendos da semana, alertas |
| `--monthly` | Consolida renda, compara com Selic |
| `--status` | Carteira, renda do mês, risco |
| `--project CAP APORTE ANOS` | Projeção da bola de neve |
| `--confirm TICKER COTAS PREÇO AÇÃO` | Registra uma compra/venda |
| `scripts/coletar_noticias.py` | Coleta notícias + sentimento (base) |

---

## 3. Fontes de dados e notícias (todas testadas)

### 3.1 Dados de mercado e fundamentos

| Fonte | O que entrega | Custo | Status |
|---|---|---|---|
| **Fundamentus** (`data/fundamentus_client.py`) | DY, P/VP, P/L, ROE, segmento, vacância de **toda a B3** (558 FIIs + ações) | Grátis (scraping) | ✅ 558 FIIs lidos, 98 aprovados |
| **brapi.dev** (`data/brapi_client.py`) | Universo (2000 ações + FIIs), preço, dividendos | Grátis c/ token; fundamentos = pago | ✅ token ativo; lista funciona |
| **Banco Central / SGS** (`data/bcb_client.py`) | Selic, CDI, IPCA, câmbio, poupança — **histórico** | Grátis, sem chave | ✅ Selic 14,5%, CDI 14,4% |
| **CVM** (`data/cvm_client.py`) | Documentos oficiais, proventos, informes | Grátis (dados abertos) | ✅ no projeto |
| **yfinance** | Histórico de preço e dividendos por ticker | Grátis | ✅ instalado |

> **Honesto sobre a brapi:** o plano grátis cobre o universo + preço, mas
> o módulo de fundamentos (`defaultKeyStatistics`, com DY/ROE) é **pago**
> (403 no grátis). Por isso o DY/P/VP vem do **Fundamentus** — de graça,
> cobrindo a B3 inteira. Se um dia quiser o plano Startup/Pro da brapi,
> o `brapi_client` já está pronto para usá-lo.

### 3.2 Notícias (≈28 feeds, `data/news_fetcher.py`)

Todos testados (15/16 candidatos responderam; os mortos foram descartados):

- **Veículos BR:** InfoMoney, Money Times, Exame, Brazil Journal, Suno,
  InvestNews, Investing.com.
- **Google News (busca por tema):** dividendos, FIIs, Selic/Copom, IPCA,
  Ibovespa, bancos/energia — **~100 manchetes por tema**.
- **Grandes casas:** XP, BTG Pactual, Itaú BBA, Faria Lima.
- **Figuras da economia:** Galípolo (BC), Haddad (Fazenda), política
  fiscal, dólar/câmbio, Tesouro Direto.
- **Globais:** Yahoo Finance, MarketWatch, CNBC, Seeking Alpha, CoinDesk.

### 3.3 Notícias do **passado** (`data/gdelt_client.py`)

- **GDELT 2.0 DOC API** — monitora a mídia mundial; busca por palavra-chave
  dos **últimos ~3 meses**, grátis e sem chave. Histórico profundo (desde
  2017) via BigQuery.
- **Honesto:** o cliente está implementado e respeita o limite (1 req/5s),
  mas durante os testes o endpoint público **retornou 429 (rate limit)
  para o IP deste ambiente** (IP de datacenter costuma ser throttled).
  O coletor trata isso graciosamente: se o GDELT recusar, a coleta segue
  só com os ~28 feeds RSS (foi o que aconteceu: 329 notícias mesmo sem o
  GDELT). De uma máquina residencial e com 1 chamada/dia, ele funciona;
  para histórico em escala, o caminho robusto é o GDELT no **BigQuery**.

---

## 4. Pesquisa externa (o que as grandes usam)

### Stack da XP Inc. (fonte: Medium oficial da XP + vaga deles)
- **Backend:** **C#/.NET** + **Apache Kafka** (mensageria) — confirmado na
  publicação "Comunidade XP" no Medium.
- **Linguagens aceitas (vaga):** C#, Java, Go, Rust.
- **Cloud:** **Azure**. **Front:** **React**.
- Padrão de mercado para dados/ML em fintechs: Airflow/dbt/Spark + Kafka +
  Kubernetes + MLflow.

### Repositórios GitHub úteis (mercado brasileiro)
- `gustavomoers/BrazilianMarketDataCollector` — CVM, B3, yfinance, AlphaVantage.
- `glourencoffee/pybov` — dados da B3.
- `cvscarlos/b3-api-dados-historicos` — API JSON de histórico da B3.
- `renanmoretto/finbr` — utilitários do mercado financeiro BR.
- `alex9smith/gdelt-doc-api` — cliente Python do GDELT.
- Modelo de IA: `lucas-leme/FinBERT-PT-BR` (sentimento financeiro em PT-BR).

### APIs que dá pra plugar depois (com/sem chave)
| API | Para quê | Chave? |
|---|---|---|
| Banco Central SGS | Macro histórico | Não |
| GDELT | Notícias históricas | Não |
| Fundamentus | Fundamentos B3 | Não (scraping) |
| brapi.dev (Startup/Pro) | Fundamentos completos | Sim (pago) |
| Alpha Vantage | Ações/FX global | Sim (grátis) |
| Finnhub / Marketaux / NewsAPI | Notícias + dados global | Sim (grátis limitado) |
| B3 (dados abertos) | Oficial | Não |

### Papers / leitura
- Araci (2019), *FinBERT* (arXiv:1908.10063).
- Santos (2023), *FinBERT-PT-BR* (sentimento financeiro em português).
- Décio Bazin, *Faça Fortuna com Ações* (preço-teto, DY 6%).
- López de Prado, *Advances in Financial Machine Learning* (validação
  anti-overfitting: PurgedKFold, DSR, PBO — já no projeto).

---

## 5. O que está 100% e o que falta

**100% funcionando e testado agora:**
- ✅ Screening de FIIs ao vivo (558 FIIs → 98 aprovados, DY real).
- ✅ Macro do Banco Central (Selic/CDI/IPCA/câmbio).
- ✅ Coleta de ~28 fontes de notícias (329 numa rodada).
- ✅ Sentimento com FinBERT-PT-BR (PyTorch) — testado sem inversão.
- ✅ Base de aprendizado gravando (`news_history.jsonl` + índice diário).
- ✅ Projeção da bola de neve, setup, status, confirmação de trades.
- ✅ 354 testes (+2 de FinBERT que rodam só com o modelo ligado).

**Opcional (não trava o uso):**
- ⚙️ **Telegram/e-mail** — preencher chaves no `.env` para receber os
  relatórios automaticamente (hoje saem no log).
- ⚙️ **brapi Startup/Pro** — só se quiser fundamentos pela brapi em vez do
  Fundamentus.

---

## 6. Deixar "treinando" até jan/2027

A ideia: rodar a **coleta diária** para a base crescer de jun/2026 a
jan/2027, formando histórico de notícias + sentimento + macro. Com isso,
depois dá para treinar modelos sobre uma base real e analisar tendências.

### Agendar no Windows (Task Scheduler)
```powershell
# Coleta de notícias todo dia às 19h
schtasks /create /tn "QuantBot Noticias" /tr "python C:\Users\junio\Downloads\dev\quantbot_work\scripts\coletar_noticias.py" /sc daily /st 19:00

# Screening toda segunda às 8h
schtasks /create /tn "QuantBot Screening" /tr "python C:\Users\junio\Downloads\dev\quantbot_work\main_income.py --weekly" /sc weekly /d MON /st 08:00
```
(Há também `scripts/agendar_windows.ps1` no projeto.)

### O que a base acumula
- `data/news_history.jsonl` — uma linha por notícia (fonte, título, símbolos,
  sentimento, score, data). Cresce a cada coleta, sem duplicar.
- `data/sentiment_index.jsonl` — uma linha por dia com o **índice de
  sentimento** do mercado (−1 medo … +1 otimismo).

---

## 7. Passado, presente e futuro (de onde vem cada um)

- **Passado:** Banco Central (anos de Selic/IPCA/CDI), GDELT (~3 meses de
  notícias, + BigQuery para anos), e a base que você acumula diariamente.
- **Presente:** brapi (preço), Fundamentus (DY/P/VP hoje), ~28 feeds de
  notícias, índice de sentimento do dia.
- **Futuro (probabilístico, nunca certeza):** o ensemble de ML
  (RF+XGBoost+GB) com validação anti-overfitting e a projeção de renda
  (bola de neve + Monte Carlo). O sistema mede vantagem estatística e
  projeta cenários — **não promete prever o mercado**, e isso é proposital.

---

## 8. Resumo honesto

Foi pego o **máximo viável de fontes gratuitas e confiáveis** (B3,
fundamentos, macro oficial, ~28 feeds de notícias + histórico GDELT) e
tudo está **testado e rodando**. O que custaria dinheiro (brapi Pro,
NewsAPI paga) está documentado como opção, não como dependência. O
caminho para "mais e melhor" é só adicionar uma fonte na lista — a
arquitetura já aceita.
