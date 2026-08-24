"""
Livro-razão auditável do robô.

Um sistema que decide onde vai dinheiro precisa poder responder, anos depois,
"por que você comprou isto naquele dia". Sem registro, a resposta é uma
reconstrução — e reconstrução é onde a memória conserta o passado a favor de
quem lembra. O painel do meu pai tinha 22.568 operações e nenhum registro do
motivo de nenhuma delas; foi por isso que levou anos para alguém perceber que
o custo era 16 vezes maior que o resultado bruto.

As decisões de projeto vêm todas do requisito de durar gerações:

* **SQLite**, arquivo único, sem servidor. O formato tem compromisso público de
  compatibilidade até 2050 e é lido por qualquer linguagem. Um banco que exige
  servidor rodando exige alguém pagando a conta do servidor, e esse alguém
  eventualmente para de pagar.
* **Só insere, nunca altera.** Não existe UPDATE nem DELETE no caminho normal.
  Corrigir um erro é lançar um evento de correção que aponta para o anterior,
  como em contabilidade de partida dobrada.
* **Encadeamento por hash.** Cada evento carrega o hash do anterior. Alterar
  uma linha antiga quebra a cadeia de todas as seguintes, e `verificar()`
  acusa exatamente onde. Não impede a alteração — torna a alteração visível.
* **Payload em JSON.** O que se registra hoje não é o que se vai querer
  registrar em dez anos. Colunas fixas para o que se consulta, JSON para o
  resto.
"""

from .livro import (
    Evento,
    Livro,
    TipoEvento,
    abrir,
)
from .verificacao import Quebra, verificar

__all__ = ["Evento", "Livro", "Quebra", "TipoEvento", "abrir", "verificar"]
