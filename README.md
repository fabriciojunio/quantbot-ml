# QuantBot ML

Bot de renda passiva e sinais quantitativos para a B3, ações US e cripto.
Combina indicadores técnicos, um pipeline de ML com validação walk-forward
(purge + embargo, no estilo López de Prado) e análise de sentimento de notícias
em português (léxico ou FinBERT-PT-BR). Opera em modo **paper** por padrão.

## Requisitos

- Python 3.12+
- Dependências em `requirements.txt` (produção) ou `requirements-ci.txt` (leve,
  sem torch/transformers)

## Instalação

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # preencha os tokens que for usar
```

Sem nenhum token o sistema funciona em modo demo (poucos ativos e sentimento
léxico). Veja `.env.example` para BRAPI, FinBERT, Telegram e e-mail.

## Uso

```bash
python main_income.py --setup 1000 --meta 200 --aporte 100   # configuração inicial
python main_income.py --daily        # rotina diária
python main_income.py --status       # situação da carteira
python scripts/operar_paper.py       # operação simulada (paper)
python scripts/gerar_relatorio.py    # relatório do dia
```

A API REST (só leitura, para o dashboard) sobe com:

```bash
uvicorn api.server:app --reload
```

## Motor de cripto

O pacote `cripto/` é a parte de criptomoedas, com dado de exchange real e custo
de execução cobrado em toda ponta:

```bash
python -m cripto.cli custo                      # o que custa abrir e fechar
python -m cripto.cli backtest BTCUSDT --markdown
python -m cripto.cli validar BTCUSDT            # walk-forward + DSR + PBO
python -m cripto.cli carrego                    # funding, líquido de custo
uvicorn cripto.api:app --port 8100              # API para o painel
```

Leia [`docs/CRIPTO.md`](docs/CRIPTO.md) antes de usar, e
[`docs/AUDITORIA_SENTINELA.md`](docs/AUDITORIA_SENTINELA.md) para entender de
onde ele veio e por que cada decisão de projeto está ali.

## Testes

```bash
pytest
```

## Segurança e operação real

O padrão é sempre paper. Conectar a uma corretora real (Binance/Alpaca com
dinheiro de verdade) exige, de propósito, `QUANTBOT_ALLOW_LIVE=1` no ambiente —
caso contrário a conexão é bloqueada antes de qualquer ordem. Detalhes em
[SECURITY.md](SECURITY.md).

## Documentação

Guias completos, arquitetura e validação estão em [`docs/`](docs/):
arquitetura (`ARCHITECTURE.md`), guia de uso (`GUIA_COMPLETO.md`), fontes de
dados (`FONTES_E_GUIA.md`) e metodologia de validação (`VALIDACAO_E_PESQUISA.md`).
