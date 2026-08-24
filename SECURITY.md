# Segurança

Este repositório lida com decisão sobre dinheiro. O modelo de ameaça é
específico e vale escrever, porque quase tudo que costuma preocupar em software
não se aplica aqui, e o que sobra é o que importa.

## O que este sistema não faz

- **Não movimenta dinheiro real.** `DINHEIRO_REAL = False` em
  `carteira/operacao.py`, como constante, sem parâmetro que a desligue.
- **Não guarda credencial de corretora.** Não existe integração de execução.
- **Não recebe dado de terceiros.** Não há usuário, login, formulário ou API
  pública.
- **Não expõe serviço.** O painel é HTML estático.

Isso elimina, por construção, a maior parte da superfície de ataque de um
sistema financeiro. O que sobra são três coisas.

## Ameaça 1 — dado de mercado errado ou envenenado

A fonte de preço é pública e gratuita, e pode estar errada sem avisar.

**Mitigações no código:**

- `valida_precos()` reprova preço zerado, negativo, índice fora de ordem, data
  duplicada e variação diária acima de 60%
- `valida_continuidade()` reprova buraco de calendário acima de 12 dias — foi
  o que pegou a série de câmbio do Yahoo sem 2005
- Nenhuma função devolve série sintética: sem fonte e sem cache, levanta erro
- Câmbio vem da PTAX do Banco Central, fonte oficial
- Cache em disco com validade, para que uma fonte fora do ar não interrompa a
  operação nem force dado inventado

## Ameaça 2 — alteração do histórico

O risco real não é alguém de fora invadir: é alguém de dentro (inclusive o
autor, anos depois) maquiar um resultado ruim.

**Mitigações:**

- Livro append-only: sem UPDATE nem DELETE no caminho normal
- Cada evento carrega o hash do anterior; a verificação recalcula e propaga o
  hash **recalculado**, então uma edição no meio derruba tudo dali para frente
- `ancora()` produz um carimbo publicável fora do arquivo, que detecta até a
  reescrita completa da cadeia
- O workflow diário audita a cadeia **antes e depois** de escrever
- O carimbo de cada execução é guardado como artefato do GitHub por 90 dias

**Limite conhecido e registrado em teste:** quem controla o arquivo pode
recalcular todos os hashes e a cadeia interna volta a fechar. É por isso que o
carimbo externo existe, e por isso o `README` pede para guardá-lo em outro
lugar.

## Ameaça 3 — engano por número bonito

A ameaça mais provável, e a que já se materializou duas vezes neste projeto:
um número correto do ponto de vista computacional e errado do ponto de vista
da realidade.

O painel de origem exibia +US$ 65 de lucro sobre um prejuízo real de
US$ 8.578. A carteira de renda deste repositório exibia +17,14% sobre um
resultado real de −6,32%, por um bug de accrual de provento.

**Mitigações:**

- Custo e imposto cobrados em toda ponta, sem caminho alternativo
- Referência de CDI calculada aporte a aporte, com alíquota regressiva por
  prazo — nunca aplicando o crescimento do período sobre o total depositado
- Curva de cota separada da curva de patrimônio, para que o aporte mensal não
  seja lido como desempenho
- Resultado apresentado como distribuição de janelas móveis, não como número
  único
- Validação com trava no caminho do dinheiro, não relatório informativo
- `conta.conferir()` falha alto em caixa negativo, posição negativa ou base de
  cálculo negativa

## Painel publicado

O painel deve ficar **privado**. Um extrato de patrimônio não tem por que estar
em internet aberta.

- Vercel → Settings → Deployment Protection → **Vercel Authentication ligada**
- `X-Robots-Tag: noindex, nofollow, noarchive`
- CSP restritiva com `frame-ancestors 'none'`, `base-uri 'none'`,
  `form-action 'none'`
- HSTS com `preload`
- Sem `connect-src` externo: a página não faz nenhuma requisição

## Segredos

O único segredo do projeto é `VERCEL_DEPLOY_HOOK`, guardado em GitHub Secrets.
Se ele não existir, o robô continua rodando e gravando; só a publicação da tela
é pulada.

Nenhuma chave de API é necessária: Yahoo Finance, Banco Central e Binance são
consultados por endpoint público.

## Dependências

- `pip-audit` e `bandit` rodam no CI
- `npm audit` no painel
- O Next.js foi atualizado de 15.1.6 para 16.3.2 por vulnerabilidade conhecida
  reportada no install (CVE-2025-66478)

## Como relatar um problema

Este é um repositório privado de uso familiar. Se você tem acesso a ele e
encontrou uma falha, abra uma issue descrevendo o que observou. Se a falha
permite alterar histórico sem detecção, ela é a mais grave possível neste
projeto e deve vir primeiro.
