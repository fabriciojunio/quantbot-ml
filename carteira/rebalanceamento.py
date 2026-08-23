"""
Quando rebalancear.

O peso alvo diz *para onde* a carteira deveria ir. Este módulo decide *quando*
vale a pena ir — e essa segunda decisão mexe mais no resultado final do que
quase qualquer sinal.

Três políticas:

* **Calendário** — toda semana, todo mês, todo trimestre. Simples e caro:
  opera mesmo quando a carteira quase não saiu do lugar.
* **Banda de tolerância** — só mexe no ativo que se afastou mais que X pontos
  percentuais do alvo. Corta a maioria das operações inúteis e é o padrão em
  gestão profissional.
* **Banda com custo** — igual à anterior, mas só executa se o desvio custar
  mais do que a operação para corrigir. É o critério economicamente correto:
  rebalancear é um investimento, e investimento com retorno negativo não se faz.

Para uma carteira pequena a diferença é grande. Rebalancear mensalmente sete
posições custa cerca de 2,5% ao ano em custo e imposto antecipado; por banda,
com dois eventos ao ano, custa perto de 0,4%.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Politica(str, Enum):
    DIARIO = "diario"
    SEMANAL = "semanal"
    MENSAL = "mensal"
    TRIMESTRAL = "trimestral"
    SEMESTRAL = "semestral"
    ANUAL = "anual"
    BANDA = "banda"
    BANDA_COM_CUSTO = "banda_com_custo"


# Quantos pregões cada política de calendário espera.
_PASSO = {
    Politica.DIARIO: 1,
    Politica.SEMANAL: 5,
    Politica.MENSAL: 21,
    Politica.TRIMESTRAL: 63,
    Politica.SEMESTRAL: 126,
    Politica.ANUAL: 252,
}


@dataclass
class ConfigRebalanceamento:
    politica: Politica = Politica.BANDA
    banda_pp: float = 5.0          # desvio absoluto em pontos percentuais
    banda_relativa: float = 0.25   # ou 25% do próprio peso alvo, o que for maior
    custo_ida_volta_pct: float = 0.004
    ganho_minimo_vs_custo: float = 1.5  # só opera se o desvio valer 1,5x o custo


def datas_de_calendario(indice: pd.DatetimeIndex, politica: Politica) -> pd.DatetimeIndex:
    """Datas em que uma política de calendário dispara."""
    if politica in (Politica.BANDA, Politica.BANDA_COM_CUSTO):
        raise ValueError("política de banda não tem calendário fixo")

    if politica == Politica.DIARIO:
        return indice

    passo = _PASSO[politica]
    return indice[::passo]


def precisa_rebalancear(
    pesos_atuais: pd.Series,
    pesos_alvo: pd.Series,
    cfg: ConfigRebalanceamento,
) -> tuple[bool, dict]:
    """Decide se vale mexer na carteira agora, e explica por quê.

    Devolve `(decisao, diagnostico)`. O diagnóstico existe para que nenhuma
    ordem seja disparada sem que dê para reconstruir o motivo depois.
    """
    alvo = pesos_alvo.reindex(pesos_atuais.index).fillna(0.0)
    desvio = (pesos_atuais - alvo).abs()

    # Banda efetiva por ativo: a maior entre a absoluta e a relativa ao alvo.
    banda = pd.Series(
        [
            max(cfg.banda_pp / 100, abs(a) * cfg.banda_relativa)
            for a in alvo.values
        ],
        index=alvo.index,
    )

    estourou = desvio > banda
    giro = float(desvio.sum() / 2)  # metade da soma dos desvios = volume a girar

    diag = {
        "maior_desvio_pp": round(float(desvio.max()) * 100, 2) if len(desvio) else 0.0,
        "ativo_mais_desviado": str(desvio.idxmax()) if len(desvio) else None,
        "ativos_fora_da_banda": [str(t) for t in estourou[estourou].index],
        "giro_necessario_pct": round(giro * 100, 2),
        "custo_estimado_pct": round(giro * cfg.custo_ida_volta_pct * 100, 4),
    }

    if cfg.politica not in (Politica.BANDA, Politica.BANDA_COM_CUSTO):
        diag["motivo"] = "política de calendário: quem manda é a data"
        return True, diag

    if not estourou.any():
        diag["motivo"] = "todo mundo dentro da banda"
        return False, diag

    if cfg.politica == Politica.BANDA:
        diag["motivo"] = f"{int(estourou.sum())} ativo(s) fora da banda"
        return True, diag

    # BANDA_COM_CUSTO: o desvio precisa ser grande o bastante para pagar a
    # conta de corrigir.
    #
    # Comparar "soma dos desvios" com "custo" seria somar grandezas diferentes:
    # desvio é distância do alvo, custo é dinheiro gasto. O que dá para comparar
    # honestamente é desvio contra custo **na mesma unidade** — os dois como
    # fração do patrimônio. O critério fica: só corrige a distância que for
    # várias vezes maior que o preço de corrigi-la.
    maior_desvio = float(desvio.max())
    limiar = cfg.custo_ida_volta_pct * cfg.ganho_minimo_vs_custo
    vale = maior_desvio > limiar

    diag["limiar_pct"] = round(limiar * 100, 4)
    diag["motivo"] = (
        f"desvio de {maior_desvio * 100:.2f}% supera o limiar de {limiar * 100:.2f}%"
        if vale
        else f"desvio de {maior_desvio * 100:.2f}% não paga o custo de corrigir "
             f"(limiar {limiar * 100:.2f}%)"
    )
    return vale, diag


def marcar_datas(
    pesos_alvo: pd.DataFrame,
    retornos: pd.DataFrame,
    cfg: ConfigRebalanceamento,
) -> pd.Series:
    """Percorre o histórico e marca em quais datas a carteira seria rebalanceada.

    A carteira entre rebalanceamentos **deriva com o mercado** — é isso que a
    simulação precisa capturar. Backtest que reaplica o peso alvo todo dia sem
    cobrar nada está medindo uma carteira que nunca existiu.
    """
    if cfg.politica in _PASSO:
        datas = set(datas_de_calendario(pesos_alvo.index, cfg.politica))
        return pd.Series(
            [d in datas for d in pesos_alvo.index], index=pesos_alvo.index, name="rebalancear"
        )

    marcas = []
    atuais: pd.Series | None = None

    for data in pesos_alvo.index:
        alvo = pesos_alvo.loc[data]

        if atuais is None or alvo.sum() == 0:
            marcas.append(True)
            atuais = alvo.copy()
            continue

        decidiu, _ = precisa_rebalancear(atuais, alvo, cfg)
        marcas.append(decidiu)

        if decidiu:
            atuais = alvo.copy()
        elif data in retornos.index:
            # Deriva: o peso muda sozinho conforme cada ativo rende.
            crescido = atuais * (1 + retornos.loc[data].fillna(0.0))
            total = crescido.sum()
            if total > 0:
                atuais = crescido / total * atuais.sum()

    return pd.Series(marcas, index=pesos_alvo.index, name="rebalancear")


def contar_eventos(marcas: pd.Series) -> dict:
    """Quantos rebalanceamentos por ano a política gerou."""
    total = int(marcas.sum())
    anos = (marcas.index[-1] - marcas.index[0]).days / 365.25 if len(marcas) > 1 else 0.0
    return {
        "eventos": total,
        "anos": round(anos, 2),
        "por_ano": round(total / anos, 2) if anos > 0 else 0.0,
    }
