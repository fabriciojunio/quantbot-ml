"""
Aprender com o passado sem aprender o passado de cor.

A ideia de "ajustar os parâmetros pelo que funcionou" é a forma mais comum de
destruir dinheiro em sistemática, e destrói justamente porque parece óbvia.
Com trinta combinações e vinte anos de histórico, sempre existe uma combinação
que teria rendido o dobro. Ela rendeu o dobro **naquele** histórico, e é isso
que ela vai continuar rendendo: naquele histórico.

O que este módulo faz em vez disso:

1. Corta a história em blocos. Ajusta num bloco de N anos, **aplica no bloco
   seguinte**, e só o resultado do bloco seguinte conta. O parâmetro nunca é
   avaliado no período em que foi escolhido.
2. Compara a escolha aprendida contra o **padrão fixo**, no mesmo período fora
   da amostra. Se o aprendizado não ganha do padrão, a resposta é "fique com o
   padrão" — e essa resposta é registrada como qualquer outra.
3. Mede a **degradação**: quanto o resultado piora ao sair da amostra. Uma
   degradação grande é a assinatura de que a busca achou ruído, não regra.
4. Grava tudo no livro de auditoria, com a janela de ajuste e a data a partir
   da qual o parâmetro passa a valer. O livro recusa parâmetro cuja vigência
   comece antes do fim da janela de ajuste — que seria usar informação do
   futuro.

O resultado esperado, e que não é fracasso nenhum, é a calibração concluir que
não há o que aprender além do padrão. Um sistema que sempre encontra algo para
aprender está encontrando ruído.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import aporte_historico as ah

logger = logging.getLogger("quantbot.carteira.calibracao")


@dataclass(frozen=True)
class Grade:
    """Um parâmetro e os valores que se admite testar.

    A grade é pequena de propósito. Cada valor a mais multiplica as chances de
    achar um vencedor por acaso, e a correção para isso (o Sharpe deflacionado)
    cresce com o número de tentativas. Testar cem combinações para escolher uma
    é pior que testar seis.
    """

    nome: str
    valores: tuple[Any, ...]

    def __post_init__(self) -> None:
        if len(self.valores) < 1:
            raise ValueError(f"grade '{self.nome}' vazia")
        if len(set(map(str, self.valores))) != len(self.valores):
            raise ValueError(f"grade '{self.nome}' tem valores repetidos")


@dataclass(frozen=True)
class Bloco:
    """Uma dobra do walk-forward: onde ajusta e onde aplica."""

    ajuste_de: pd.Timestamp
    ajuste_ate: pd.Timestamp
    uso_de: pd.Timestamp
    uso_ate: pd.Timestamp

    def __post_init__(self) -> None:
        if self.uso_de < self.ajuste_ate:
            raise ValueError(
                f"o bloco de uso começa em {self.uso_de.date()}, antes do fim do ajuste "
                f"em {self.ajuste_ate.date()}: os períodos se sobrepõem"
            )

    def resumo(self) -> dict:
        return {
            "ajuste_de": str(self.ajuste_de.date()),
            "ajuste_ate": str(self.ajuste_ate.date()),
            "uso_de": str(self.uso_de.date()),
            "uso_ate": str(self.uso_ate.date()),
        }


@dataclass
class ResultadoDoBloco:
    bloco: Bloco
    escolhido: dict[str, Any]
    metrica_dentro: float
    metrica_fora: float
    metrica_do_padrao_fora: float
    combinacoes_testadas: int

    @property
    def degradacao(self) -> float:
        """Quanto o resultado piorou ao sair da amostra, em pontos."""
        return self.metrica_dentro - self.metrica_fora

    @property
    def ganhou_do_padrao(self) -> bool:
        return self.metrica_fora > self.metrica_do_padrao_fora

    def resumo(self) -> dict:
        return {
            **self.bloco.resumo(),
            "escolhido": self.escolhido,
            "metrica_dentro": round(self.metrica_dentro, 4),
            "metrica_fora": round(self.metrica_fora, 4),
            "metrica_do_padrao_fora": round(self.metrica_do_padrao_fora, 4),
            "degradacao": round(self.degradacao, 4),
            "ganhou_do_padrao": self.ganhou_do_padrao,
            "combinacoes_testadas": self.combinacoes_testadas,
        }


@dataclass
class Veredito:
    """A conclusão sobre se vale a pena aprender."""

    aprender: bool
    motivos: tuple[str, ...]
    blocos: tuple[ResultadoDoBloco, ...] = field(default_factory=tuple)

    @property
    def vitorias(self) -> int:
        return sum(1 for b in self.blocos if b.ganhou_do_padrao)

    @property
    def degradacao_mediana(self) -> float:
        if not self.blocos:
            return 0.0
        return float(np.median([b.degradacao for b in self.blocos]))

    @property
    def vantagem_mediana(self) -> float:
        if not self.blocos:
            return 0.0
        return float(np.median([b.metrica_fora - b.metrica_do_padrao_fora for b in self.blocos]))

    def resumo(self) -> dict:
        return {
            "aprender": self.aprender,
            "motivos": list(self.motivos),
            "blocos": len(self.blocos),
            "vitorias_sobre_o_padrao": self.vitorias,
            "degradacao_mediana": round(self.degradacao_mediana, 4),
            "vantagem_mediana": round(self.vantagem_mediana, 4),
            "detalhe": [b.resumo() for b in self.blocos],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Blocos
# ─────────────────────────────────────────────────────────────────────────────

def montar_blocos(
    indice: pd.DatetimeIndex,
    anos_de_ajuste: int = 8,
    anos_de_uso: int = 2,
    embargo_dias: int = 21,
) -> list[Bloco]:
    """Corta o histórico em dobras que avançam no tempo.

    O `embargo` é o intervalo descartado entre ajuste e uso. Existe porque
    indicadores usam janela para trás: um sinal calculado no primeiro dia do
    bloco de uso enxerga preços do bloco de ajuste. Sem o embargo, um pedaço
    do "fora da amostra" está dentro dela.
    """
    if anos_de_ajuste <= 0 or anos_de_uso <= 0:
        raise ValueError("os dois períodos precisam ser positivos")
    if len(indice) < 200:
        raise ValueError("histórico curto demais para dividir em blocos")

    inicio, fim = indice[0], indice[-1]
    blocos: list[Bloco] = []
    ajuste_de = inicio

    while True:
        ajuste_ate = ajuste_de + pd.DateOffset(years=anos_de_ajuste)
        uso_de = ajuste_ate + pd.Timedelta(days=embargo_dias)
        uso_ate = uso_de + pd.DateOffset(years=anos_de_uso)
        if uso_ate > fim:
            break
        blocos.append(Bloco(ajuste_de, ajuste_ate, uso_de, uso_ate))
        ajuste_de = ajuste_de + pd.DateOffset(years=anos_de_uso)

    if not blocos:
        raise ValueError(
            f"não cabe nenhum bloco de {anos_de_ajuste}+{anos_de_uso} anos em "
            f"{(fim - inicio).days / 365.25:.1f} anos de histórico"
        )
    return blocos


def combinacoes(grades: Iterable[Grade]) -> list[dict[str, Any]]:
    """Produto cartesiano das grades, como lista de dicionários."""
    grades = list(grades)
    if not grades:
        return [{}]
    nomes = [g.nome for g in grades]
    return [dict(zip(nomes, valores)) for valores in itertools.product(*(g.valores for g in grades))]


# ─────────────────────────────────────────────────────────────────────────────
# A métrica
# ─────────────────────────────────────────────────────────────────────────────

def vantagem_sobre_cdi_por_risco(resultado: ah.ResultadoAporte) -> float:
    """Quanto a carteira bateu o CDI, por unidade de queda que fez sofrer.

    Retorno puro é a métrica errada para escolher parâmetro: ela sempre premia
    a versão mais arriscada, e "mais arriscada" é justamente a decisão que o
    usuário já tomou e que não cabe ao ajuste mexer. Dividir pela queda faz o
    ajuste escolher a forma mais eficiente de correr o risco escolhido, que é
    o que ele tem competência para decidir.
    """
    vantagem = resultado.tir_anual_pct - resultado.tir_cdi_pct
    queda = abs(resultado.queda_da_cota_pct)
    return vantagem / max(queda, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward
# ─────────────────────────────────────────────────────────────────────────────

def calibrar(
    precos: pd.DataFrame,
    cdi: pd.Series,
    grades: Iterable[Grade],
    aplicar: Callable[[dict[str, Any]], dict],
    padrao: dict[str, Any],
    *,
    aporte: Callable[[int], float] | float = 500.0,
    anos_de_ajuste: int = 8,
    anos_de_uso: int = 2,
    embargo_dias: int = 21,
    metrica: Callable[[ah.ResultadoAporte], float] = vantagem_sobre_cdi_por_risco,
    minimo_de_blocos: int = 3,
    vitorias_minimas: float = 0.60,
    degradacao_maxima: float = 0.50,
    p_maximo: float = 0.10,
) -> Veredito:
    """Ajusta em cada bloco e mede no seguinte.

    `aplicar` converte um dicionário de parâmetros nos argumentos que a
    simulação entende: devolve ``{"pesos": ..., "rebalancear_a_cada_meses": ...}``
    e o que mais fizer sentido.

    Os três limiares no fim são o filtro. Um aprendizado só é aceito se
    apareceu em blocos suficientes, ganhou do padrão na maioria deles e não
    desabou ao sair da amostra. Falhar em qualquer um deles devolve
    `aprender=False`, que é uma conclusão legítima e a mais frequente.
    """
    grades = list(grades)
    blocos = montar_blocos(precos.index, anos_de_ajuste, anos_de_uso, embargo_dias)
    todas = combinacoes(grades)
    resultados: list[ResultadoDoBloco] = []

    for bloco in blocos:
        dentro = precos.loc[(precos.index >= bloco.ajuste_de) & (precos.index <= bloco.ajuste_ate)]
        fora = precos.loc[(precos.index >= bloco.uso_de) & (precos.index <= bloco.uso_ate)]
        if len(dentro) < 200 or len(fora) < 100:
            logger.warning("Bloco %s ignorado: dados insuficientes", bloco.resumo())
            continue

        melhor, melhor_metrica = None, -np.inf
        for combo in todas:
            try:
                r = ah.simular(dentro, cdi, aporte=aporte, **aplicar(combo))
            except Exception as exc:
                logger.debug("Combinação %s falhou no ajuste: %s", combo, exc)
                continue
            m = metrica(r)
            if np.isfinite(m) and m > melhor_metrica:
                melhor, melhor_metrica = combo, m

        if melhor is None:
            logger.warning("Bloco %s: nenhuma combinação rodou", bloco.resumo())
            continue

        try:
            m_fora = metrica(ah.simular(fora, cdi, aporte=aporte, **aplicar(melhor)))
            m_padrao = metrica(ah.simular(fora, cdi, aporte=aporte, **aplicar(padrao)))
        except Exception as exc:
            logger.warning("Bloco %s falhou fora da amostra: %s", bloco.resumo(), exc)
            continue

        resultados.append(ResultadoDoBloco(
            bloco=bloco, escolhido=melhor,
            metrica_dentro=melhor_metrica, metrica_fora=m_fora,
            metrica_do_padrao_fora=m_padrao, combinacoes_testadas=len(todas),
        ))

    return _julgar(resultados, minimo_de_blocos, vitorias_minimas, degradacao_maxima, p_maximo)


def p_valor_binomial(vitorias: int, blocos: int) -> float:
    """Probabilidade de ganhar do padrão ao menos `vitorias` vezes por acaso.

    Sob a hipótese de que o aprendizado não serve para nada, cada bloco é uma
    moeda honesta: ou a escolha aprendida ganha do padrão, ou perde. Com seis
    blocos, ganhar quatro tem 34% de chance de acontecer sem nenhuma
    habilidade — e um limiar de "60% de vitórias" aprova exatamente esse caso.

    Foi o que aconteceu: com ruído puro e limiar de 60%, a calibração aprovava
    em três de cinco sementes. Trocar o limiar por este teste é a diferença
    entre um número escolhido a dedo e um que sai da própria estatística.
    """
    if blocos <= 0:
        return 1.0
    vitorias = max(0, min(vitorias, blocos))
    total = 2**blocos
    caudas = sum(math.comb(blocos, k) for k in range(vitorias, blocos + 1))
    return caudas / total


def _julgar(
    resultados: list[ResultadoDoBloco],
    minimo_de_blocos: int,
    vitorias_minimas: float,
    degradacao_maxima: float,
    p_maximo: float = 0.10,
) -> Veredito:
    motivos: list[str] = []

    if len(resultados) < minimo_de_blocos:
        return Veredito(
            aprender=False,
            motivos=(
                f"só {len(resultados)} bloco(s) fora da amostra, mínimo {minimo_de_blocos}. "
                "Com poucos blocos não dá para separar habilidade de sorte.",
            ),
            blocos=tuple(resultados),
        )

    veredito = Veredito(aprender=True, motivos=(), blocos=tuple(resultados))
    n = len(resultados)
    taxa = veredito.vitorias / n
    p = p_valor_binomial(veredito.vitorias, n)

    if taxa < vitorias_minimas:
        motivos.append(
            f"o parâmetro aprendido ganhou do padrão em {veredito.vitorias} de {n} blocos "
            f"({taxa * 100:.0f}%), abaixo do mínimo de {vitorias_minimas * 100:.0f}%."
        )

    if p > p_maximo:
        motivos.append(
            f"ganhar {veredito.vitorias} de {n} blocos tem {p * 100:.0f}% de chance de "
            f"acontecer por acaso, acima do teto de {p_maximo * 100:.0f}%. "
            f"Não dá para distinguir de moeda honesta."
        )

    if veredito.degradacao_mediana > degradacao_maxima:
        motivos.append(
            f"a degradação mediana ao sair da amostra é {veredito.degradacao_mediana:.3f}, "
            f"acima do teto de {degradacao_maxima:.2f}. Isso é assinatura de ajuste ao ruído."
        )

    if veredito.vantagem_mediana <= 0:
        motivos.append(
            "a vantagem mediana sobre o padrão é nula ou negativa fora da amostra."
        )

    # Estabilidade: uma regra de verdade escolhe a mesma coisa na maioria dos
    # blocos. Ruído escolhe um vencedor diferente a cada vez, e a média desses
    # vencedores não é um parâmetro — é uma lista de acasos.
    estavel = parametro_estavel(resultados)
    if estavel is None:
        distintos = len({str(sorted(b.escolhido.items())) for b in resultados})
        motivos.append(
            f"a busca escolheu {distintos} conjuntos diferentes em {n} blocos, sem maioria "
            f"para nenhum. Parâmetro que não se repete é ruído, não regra."
        )

    if motivos:
        return Veredito(aprender=False, motivos=tuple(motivos), blocos=tuple(resultados))

    return Veredito(
        aprender=True,
        motivos=(
            f"ganhou do padrão em {veredito.vitorias}/{n} blocos fora da amostra "
            f"(p={p:.3f}), sempre escolhendo {estavel}, com degradação mediana de "
            f"{veredito.degradacao_mediana:.3f}.",
        ),
        blocos=tuple(resultados),
    )


def parametro_estavel(resultados: Iterable[ResultadoDoBloco]) -> dict[str, Any] | None:
    """O conjunto de parâmetros que a busca escolheu mais vezes.

    Devolve `None` quando não há maioria: se cada bloco escolheu algo
    diferente, não existe "o parâmetro aprendido", existe uma lista de acasos.
    """
    resultados = list(resultados)
    if not resultados:
        return None

    contagem: dict[str, int] = {}
    representante: dict[str, dict] = {}
    for r in resultados:
        chave = str(sorted(r.escolhido.items()))
        contagem[chave] = contagem.get(chave, 0) + 1
        representante[chave] = r.escolhido

    chave, vezes = max(contagem.items(), key=lambda x: x[1])
    if vezes * 2 <= len(resultados):
        return None
    return representante[chave]


def registrar_no_livro(livro, veredito: Veredito, nome: str = "calibracao") -> list:
    """Grava cada bloco no livro de auditoria, vencendo ou perdendo.

    Registrar o fracasso importa tanto quanto registrar o acerto: é o histórico
    de fracassos que mostra, daqui a dez anos, que a busca foi honesta e não
    ficou repetindo até dar certo.
    """
    from auditoria import TipoEvento

    eventos = []
    for b in veredito.blocos:
        eventos.append(livro.registrar_parametro(
            nome=nome,
            valor=b.escolhido,
            ajustado_de=str(b.bloco.ajuste_de.date()),
            ajustado_ate=str(b.bloco.ajuste_ate.date()),
            vale_a_partir=str(b.bloco.uso_de.date()),
            metrica_dentro=b.metrica_dentro,
            metrica_fora=b.metrica_fora,
        ))

    eventos.append(livro.registrar(
        TipoEvento.VALIDACAO, "calibracao",
        ("Calibração aprovada: " if veredito.aprender else "Calibração reprovada: ")
        + " ".join(veredito.motivos),
        veredito.resumo(),
    ))
    return eventos
