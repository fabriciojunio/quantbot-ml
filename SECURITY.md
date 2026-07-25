# Política de Segurança

## Modelo de operação: paper por padrão

O QuantBot roda em modo **paper** (simulado) por padrão. Nenhuma ordem toca
uma conta real sem duas condições simultâneas:

1. Um adaptador de corretora real construído com `testnet=False` / `paper=False`.
2. A variável de ambiente `QUANTBOT_ALLOW_LIVE=1` explicitamente definida.

Sem a variável, qualquer tentativa de conectar em modo real é abortada antes de
enviar qualquer ordem (`core/live_trading.py`). É proposital: evita operar
dinheiro de verdade por engano.

## Segredos

- Nunca comite `.env`, chaves (`*.key`, `*.pem`) ou tokens. O `.gitignore` já
  bloqueia esses arquivos.
- Todos os segredos (BRAPI, Hugging Face, Telegram, e-mail, chaves de corretora)
  vêm de variáveis de ambiente. Veja `.env.example`.
- Chaves de corretora devem ter permissão mínima (sem saque) e, quando a
  corretora oferecer, IP allowlist.
- Se uma chave for exposta em commit, **rotacione imediatamente** na corretora e
  reescreva o histórico (`git filter-repo`) antes de tornar o repositório público.

## Superfície de rede

- Feeds RSS e APIs externas são conteúdo não confiável. O XML é parseado com
  `defusedxml` (resistente a XXE / billion laughs) e as URLs de feed são
  validadas para aceitar apenas `http`/`https` (barra `file://` e SSRF por
  esquema).
- Toda chamada de rede usa `timeout` explícito.
- O modelo FinBERT do Hugging Face é baixado com `revision` fixada
  (`QUANTBOT_FINBERT_REVISION`), reduzindo risco de cadeia de suprimentos.

## API REST (`api/server.py`)

- Apenas leitura (GET). Não executa ordens.
- CORS restrito às origens do front-end.
- Rate limiting por IP (429 ao estourar a janela).
- Cabeçalhos de segurança em toda resposta (`X-Content-Type-Options`,
  `X-Frame-Options`, `Referrer-Policy`, `Cache-Control`).
- Validação de símbolo por regex (bloqueia tentativas de injeção).

## Verificação contínua

O CI (`.github/workflows/ci.yml`) roda a cada push/PR:

- `pytest` (bloqueia em falha)
- `bandit` bloqueando em severidade alta, com relatório completo informativo
- `pip-audit` sobre as dependências (bloqueia em vulnerabilidade conhecida)
- `ruff` (informativo)

## Como reportar uma vulnerabilidade

Abra uma issue privada ou entre em contato com o mantenedor. Descreva o impacto e
um passo a passo de reprodução. Não divulgue publicamente antes da correção.
