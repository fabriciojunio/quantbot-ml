# Próximos passos — janeiro de 2027

> Escrito em jun/2026. Quando você voltar, comece por aqui.

## O que vai ter acontecido (jun/2026 → jan/2027)

O bot rodou sozinho na nuvem **todo dia** (GitHub Actions), sem seu PC:

- Coletou notícias (~28 fontes) + leu o sentimento com **FinBERT-PT-BR**.
- **Operou com dinheiro simulado** (carteira paper de R$10.000), comprando
  os FIIs aprovados e vendendo os que saíam da lista.
- **Aprendeu** com (1) os próprios acertos/erros vs CDI e (2) o sentimento
  das notícias vs o retorno seguinte.
- Gerou um **relatório por dia** em `relatorios/`.

## O que conferir primeiro

```bash
git pull
```

1. `relatorios/ULTIMO.md` — foto do dia (macro, sentimento, carteira, picks).
2. `data/paper_book.json` — track record da carteira simulada: retorno,
   posições, operações fechadas (quantas deram lucro).
3. `data/feedback.json` — o que o bot aprendeu (faixas de pick que batem o
   CDI, relação sentimento → retorno).
4. `data/sentiment_index.jsonl` — série de ~210 dias do humor do mercado.

## Decisões que valem a pena nesta fase

- **Avaliar o track record:** a carteira paper bateu o CDI? Os picks de
  score alto se saíram melhor? Isso diz se a estratégia presta.
- **Fechar o ciclo de ML (eu ajudo):** com a base cheia, dá para treinar de
  verdade um modelo sobre os dados rotulados (pick → resultado) e validar
  com `validation/` (PurgedKFold, DSR, PBO). É o "aprender de verdade".
- **Começar pequeno com dinheiro real**, só nos de maior convicção, e
  **reinvestir os dividendos** (a bola de neve é o que importa no longo prazo).
- **Opcional:** ligar Telegram (`.env`) para receber o relatório no celular.

## Lembretes honestos

- 6 meses é curto para dividendos (estratégia de anos). O sinal de
  aprendizado ainda será fraco — não superinterprete.
- O bot informa; **você decide**. Ele nunca operou com dinheiro real.
- Mercado não se prevê com certeza. O valor é disciplina + dados + método.
