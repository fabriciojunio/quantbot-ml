# QuantBot ML — Verificação Geral, Tutorial e Estado Real

> Atualizado: 01/06/2026
> Este documento registra o que foi verificado, o que foi reconstruído para
> o sistema voltar a rodar, como usar na prática e — com honestidade — o que
> "rodando" significa e o que ele **não** significa.

---

## 1. Resultado da verificação

| Item | Resultado |
|---|---|
| Testes automatizados | **351 passando, 0 falhas** |
| Importação de todos os pacotes | OK (config, utils, data, core, models, strategies, risk, backtest, api, visualization) |
| Entrypoint `main_income.py` | Roda (`--status`, `--project`, `--weekly`, etc.) |
| API FastAPI (`api/server.py`) | Sobe e responde (`/api/health` → 200) |
| Rodar sem configurar `PYTHONPATH` | OK (via `conftest.py` + `pytest.ini`) |

### O que estava quebrado (e foi corrigido)

Na verificação inicial, **apenas 64 de ~351 testes passavam**. A pasta estava
faltando os pacotes-base que o resto do código importa. Sem eles, `core/` e
`data/` nem importavam (efeito cascata), e a aplicação não subia.

Foram reconstruídos, guiados pelos contratos dos próprios testes:

- `config/` — enums de domínio (`Signal`, `Market`, `RiskProfile`), universo
  de ativos e parâmetros de dados/ML/backtest/risco.
- `utils/` — `logger`, `formatters`, `security` (validação de input e
  sanitização de dados).
- `models/` — `signals`, `ensemble` (RF+GB+XGBoost), `triple_barrier`,
  `regime`, `walk_forward`.
- `risk/` — `metrics` (VaR/CVaR/Sharpe/Sortino/beta/alpha), `manager`
  (sizing/stop/take-profit), `dynamic_stop` (trailing ATR).
- `backtest/` — `engine` (long/flat vetorizado), `benchmarks` (vs CDI/Ibov/S&P/BTC).
- `strategies/` — `base`, `sma_crossover`, `rsi_strategy`, `macd_strategy`,
  `ensemble_voting`.
- `api/` — `market_data` (indicadores/sinais), `server` (FastAPI + cache + rate limit).
- `visualization.py` — gráficos de sinais, equity e comparação.
- `requirements.txt`, `pytest.ini`, `conftest.py` — para rodar out of the box.
- Correção de encoding UTF-8 no `main_income.py` (consoles Windows cp1252).

---

## 2. Tutorial de uso

### 2.1 Instalação

```powershell
cd quantbot_work
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # edite com seus tokens
```

### 2.2 Rodar os testes

```powershell
python -m pytest          # 351 testes
```

### 2.3 Comandos do bot (renda passiva / FIIs e dividendos)

```powershell
python main_income.py --status                 # situação atual
python main_income.py --project 1000 200 10    # projeção: capital, aporte/mês, anos
python main_income.py --weekly                 # recomendações da semana
python main_income.py --setup 1000             # configuração inicial com R$1.000
python main_income.py --killswitch on          # para tudo (emergência)
```

### 2.4 Subir a API (para o dashboard)

```powershell
uvicorn api.server:app --reload --port 8000
# Health: http://localhost:8000/api/health
# Docs:   http://localhost:8000/api/docs
```

### 2.5 Validação estatística (passo obrigatório antes de qualquer capital)

A suíte `validation/` (DSR, PBO, Purged CV, CPCV) é o coração da
credibilidade. Rode-a nos seus modelos com dados reais antes de pensar em
dinheiro. Interpretação:

- **DSR > 0.90** e **PBO < 0.20** → evidência de edge; pode ir a paper trading.
- **DSR baixo** ou **PBO ~0.50** → sem evidência; não arrisque capital.

---

## 3. O que "rodando" significa — e o que NÃO significa

**Significa:** o código está correto, importa, os 351 testes de comportamento
passam, a aplicação sobe e os cálculos (indicadores, risco, projeções,
validação) funcionam.

**NÃO significa:** que o sistema é lucrativo. Testes verdes provam engenharia
correta, **não** vantagem de mercado. São coisas separadas. Nenhum arquivo
aqui pode ser declarado "pronto para investir" — essa garantia não existe
para ninguém.

> ⚠️ Dado importante: parte do universo de ativos e dos benchmarks estimados
> usa valores-padrão/sintéticos plausíveis (marcados no código). Antes de
> operar, troque por dados reais e revise os parâmetros em `config/settings.py`.

---

## 4. Caminho responsável até capital real

1. **Validação** — rodar CPCV + DSR + PBO nos modelos com dados recentes.
2. **Paper trading** — 3 a 6 meses com dinheiro fictício e dados reais.
3. **Capital mínimo** — só se 1 e 2 se sustentarem; valor que não afete sua vida.
4. **Kill-switch sempre ligado** — para tudo se o drawdown passar do limite.

Regra inegociável: **teste unitário passando ≠ estratégia lucrativa.**
