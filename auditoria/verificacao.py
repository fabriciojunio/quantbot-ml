"""
Verificação de integridade da cadeia.

Recalcula o hash de cada evento e confere se ele bate com o gravado e se o
elo com o anterior está intacto. Não protege contra alteração — protege contra
alteração silenciosa, que é o que dá para garantir num arquivo que o dono
controla.

A distinção importa e vale dizer em voz alta: quem tem o arquivo pode reescrever
a cadeia inteira recalculando todos os hashes. O que este módulo garante é que
uma edição pontual — mudar um preço, apagar uma ordem que deu errado — quebra
tudo o que veio depois e aparece na hora. Para garantia contra o próprio dono
seria preciso um carimbo externo, e é isso que `ancora()` prepara.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from .livro import GENESE, Livro, calcular_hash


@dataclass(frozen=True)
class Quebra:
    """Um ponto onde a cadeia não fecha."""

    evento_id: int
    motivo: str
    esperado: str
    encontrado: str

    def __str__(self) -> str:
        return f"evento {self.evento_id}: {self.motivo}"


@dataclass(frozen=True)
class Resultado:
    intacta: bool
    eventos_conferidos: int
    quebras: tuple[Quebra, ...]
    primeiro_evento: str | None
    ultimo_evento: str | None
    hash_final: str | None

    def resumo(self) -> dict:
        return {
            "intacta": self.intacta,
            "eventos_conferidos": self.eventos_conferidos,
            "quebras": [
                {"evento_id": q.evento_id, "motivo": q.motivo,
                 "esperado": q.esperado, "encontrado": q.encontrado}
                for q in self.quebras
            ],
            "primeiro_evento": self.primeiro_evento,
            "ultimo_evento": self.ultimo_evento,
            "hash_final": self.hash_final,
        }


def verificar(livro: Livro, parar_na_primeira: bool = False) -> Resultado:
    """Percorre a cadeia inteira conferindo hash e elo.

    Continua depois de achar uma quebra, de propósito: quando alguém edita uma
    linha antiga, todas as seguintes quebram o elo. Parar na primeira esconderia
    o tamanho do estrago; ver todas mostra se foi uma linha ou o arquivo todo.
    """
    quebras: list[Quebra] = []
    anterior_esperado = GENESE
    conferidos = 0
    primeiro = ultimo = None
    hash_final = None

    for evento in livro.percorrer():
        conferidos += 1
        if primeiro is None:
            primeiro = evento.quando
        ultimo = evento.quando

        if evento.hash_anterior != anterior_esperado:
            quebras.append(Quebra(
                evento.id,
                "o elo com o evento anterior não fecha",
                anterior_esperado,
                evento.hash_anterior,
            ))
            if parar_na_primeira:
                break

        recalculado = calcular_hash(
            evento.quando, evento.tipo.value, evento.origem,
            evento.resumo, evento.dados, evento.hash_anterior,
        )
        if recalculado != evento.hash:
            quebras.append(Quebra(
                evento.id,
                "o conteúdo do evento não corresponde ao hash gravado",
                recalculado,
                evento.hash,
            ))
            if parar_na_primeira:
                break

        # Propaga o hash RECALCULADO, não o gravado. É o que faz a adulteração
        # cascatear: se alguém muda o conteúdo do evento 5 sem refazer a coluna
        # de hash, o elo do evento 6 passa a apontar para um valor que já não
        # corresponde ao conteúdo do 5, e a quebra segue até o fim do livro.
        # Propagando o valor gravado, só o evento 5 acusava, e o estrago
        # parecia menor do que é.
        anterior_esperado = recalculado
        hash_final = recalculado

    return Resultado(
        intacta=not quebras,
        eventos_conferidos=conferidos,
        quebras=tuple(quebras),
        primeiro_evento=primeiro,
        ultimo_evento=ultimo,
        hash_final=hash_final,
    )


def ancora(livro: Livro) -> dict:
    """Produz o carimbo que pode ser publicado fora do arquivo.

    O hash final resume a cadeia inteira: qualquer alteração em qualquer evento
    muda este valor. Publicando-o em algum lugar que você não controla — um
    commit assinado, um e-mail para você mesmo, uma mensagem em grupo — passa a
    existir prova de que o livro tinha aquele conteúdo naquela data.

    É o passo que fecha o buraco que `verificar()` não fecha sozinho.
    """
    resultado = verificar(livro)
    carimbo = {
        "carimbado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eventos": resultado.eventos_conferidos,
        "hash_final": resultado.hash_final,
        "intacta": resultado.intacta,
    }
    corpo = f"{carimbo['carimbado_em']}|{carimbo['eventos']}|{carimbo['hash_final']}"
    carimbo["assinatura"] = hashlib.sha256(corpo.encode("utf-8")).hexdigest()
    return carimbo


def conferir_ancora(livro: Livro, carimbo: dict) -> tuple[bool, str]:
    """Confere se o livro atual ainda contém o que o carimbo registrou.

    Um livro pode ter crescido desde o carimbo — isso é normal. O que não pode
    é ter mudado o passado: o hash do evento N precisa continuar o mesmo.
    """
    eventos = int(carimbo["eventos"])
    total = livro.total_de_eventos()
    if total < eventos:
        return False, f"o livro tem {total} eventos, menos que os {eventos} carimbados"

    # Recalcula em vez de ler a coluna: a coluna é justamente o que alguém
    # reescreveria para esconder a alteração.
    parcial = GENESE
    for i, evento in enumerate(livro.percorrer(), start=1):
        parcial = calcular_hash(
            evento.quando, evento.tipo.value, evento.origem,
            evento.resumo, evento.dados, parcial,
        )
        if i == eventos:
            break

    if parcial != carimbo["hash_final"]:
        return False, (
            f"o hash do evento {eventos} mudou: era {carimbo['hash_final'][:16]}…, "
            f"agora é {parcial[:16]}…. Algum evento anterior foi alterado."
        )
    return True, f"íntegro até o evento {eventos}; o livro cresceu para {total}"
