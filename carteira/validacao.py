"""
Validação das carteiras.

O estudo comparativo testa dezenas de combinações e informa a melhor. Esse
número é o menos confiável de todos: com 66 tentativas sobre o mesmo histórico,
a melhor delas tem Sharpe alto por construção, mesmo que nenhuma tenha
vantagem nenhuma. É o mesmo mecanismo que fazia o painel Sentinela parecer
lucrativo, só que mais elegante.

Três camadas de defesa, reaproveitando o pacote `validation/`:

* **Walk-forward ancorado** — escolhe a estratégia usando só o passado e mede
  o que ela entrega no futuro imediato. Repete andando a janela. Só o trecho
  fora da amostra entra na conta.
* **Deflated Sharpe** — desconta o Sharpe pelo número de tentativas feitas.
* **PBO** — mede se escolher a melhor pelo passado tem relação com ir bem no
  futuro. Acima de 0,5, a seleção é pior que sortear.

E uma quarta, que não é estatística e é a mais importante:

* **Comparação com o CDI.** No Brasil, a alternativa sem risco rende dois
  dígitos. Uma estratégia com Sharpe bonito que entrega menos que o CDI não é
  uma estratégia mediana — é uma forma cara e arriscada de ter o resultado da
  renda fixa.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from validation import evaluate_strategy, probability_of_backtest_overfitting

from . import backtest, estudo
from .rebalanceamento import ConfigRebalanceamento, Politica

logger = logging.getLogger("quantbot.carteira.validacao")

DIAS_UTEIS_ANO = 252


@dataclass
class Janela:
    treino: tuple[str, str]
    teste: tuple[str, str]
    escolhida: str
    retorno_treino_pct: float
    retorno_teste_pct: float
    cdi_teste_pct: float
    bateu_cdi_fora_da_amostra: bool


def _retorno_pct(serie: pd.Series) -> float:
    if len(serie) < 2 or float(serie.iloc[0]) == 0:
        return 0.0
    return (float(serie.iloc[-1]) / float(serie.iloc[0]) - 1) * 100


def walk_forward(
    precos: pd.DataFrame,
    cdi: pd.Series,
    capital_inicial: float = 10_000.0,
    dias_treino: int = 504,      # ~2 anos
    dias_teste: int = 126,       # ~6 meses
    embargo: int = 5,
    politica: Politica = Politica.TRIMESTRAL,
) -> tuple[list[Janela], pd.Series, dict]:
    """Escolhe a melhor estratégia no treino e mede o que ela rende no teste.

    O aquecimento de 252 pregões antes do teste existe porque momentum de 12
    meses precisa de 12 meses de histórico. Sem ele, o começo de cada janela
    de teste operaria com sinal pela metade.
    """
    cdi_alinhado = cdi.reindex(precos.index).ffill().fillna(0.0)
    cdi_acumulado = (1 + cdi_alinhado).cumprod() * capital_inicial

    janelas: list[Janela] = []
    retornos_oos: list[pd.Series] = []
    curvas_treino: dict[str, list[float]] = {}

    inicio = 0
    while inicio + dias_treino + embargo + dias_teste <= len(precos):
        fim_treino = inicio + dias_treino
        inicio_teste = fim_treino + embargo
        fim_teste = inicio_teste + dias_teste

        px_treino = precos.iloc[inicio:fim_treino]
        # Aquecimento: pega 252 pregões antes do início do teste.
        aquecimento = max(0, inicio_teste - DIAS_UTEIS_ANO)
        px_teste = precos.iloc[aquecimento:fim_teste]

        cfg = backtest.ConfigBacktest(
            capital_inicial=capital_inicial,
            config_rebal=ConfigRebalanceamento(politica=politica),
        )

        # ── Treino: qual estratégia foi a melhor no passado? ────────────────
        melhor_nome, melhor_ret = None, -np.inf
        estr_treino = estudo.montar_estrategias(
            px_treino, cdi_acumulado.reindex(px_treino.index).ffill()
        )
        for nome, pesos in estr_treino.items():
            try:
                r = backtest.executar(px_treino, pesos, cfg, cdi=cdi, nome=nome)
            except Exception:
                continue
            curvas_treino.setdefault(nome, []).extend(
                r.patrimonio.pct_change().fillna(0.0).tolist()
            )
            ret = _retorno_pct(r.patrimonio)
            if ret > melhor_ret:
                melhor_ret, melhor_nome = ret, nome

        if melhor_nome is None:
            break

        # ── Teste: o que ela entrega no futuro que não viu ──────────────────
        estr_teste = estudo.montar_estrategias(
            px_teste, cdi_acumulado.reindex(px_teste.index).ffill()
        )
        pesos_teste = estr_teste.get(melhor_nome)
        if pesos_teste is None:
            break

        try:
            r_teste = backtest.executar(px_teste, pesos_teste, cfg, cdi=cdi, nome=melhor_nome)
        except Exception:
            break

        corte = precos.index[inicio_teste]
        patr_oos = r_teste.patrimonio.loc[r_teste.patrimonio.index >= corte]
        if len(patr_oos) < 2:
            break

        retornos_oos.append(patr_oos.pct_change().dropna())

        cdi_oos = cdi_acumulado.loc[(cdi_acumulado.index >= corte) & (cdi_acumulado.index <= patr_oos.index[-1])]
        ret_oos = _retorno_pct(patr_oos)
        ret_cdi = _retorno_pct(cdi_oos)

        janelas.append(
            Janela(
                treino=(str(px_treino.index[0])[:10], str(px_treino.index[-1])[:10]),
                teste=(str(corte)[:10], str(patr_oos.index[-1])[:10]),
                escolhida=melhor_nome,
                retorno_treino_pct=round(melhor_ret, 2),
                retorno_teste_pct=round(ret_oos, 2),
                cdi_teste_pct=round(ret_cdi, 2),
                bateu_cdi_fora_da_amostra=ret_oos > ret_cdi,
            )
        )
        inicio += dias_teste

    if not retornos_oos:
        raise ValueError(
            "histórico curto demais para walk-forward. Reduza dias_treino/dias_teste."
        )

    curva_oos = pd.concat(retornos_oos)

    # CDI no mesmo trecho fora da amostra, para a comparação que decide.
    primeira = pd.Timestamp(janelas[0].teste[0])
    ultima = pd.Timestamp(janelas[-1].teste[1])
    cdi_trecho = cdi_acumulado.loc[(cdi_acumulado.index >= primeira) & (cdi_acumulado.index <= ultima)]

    resumo = {
        "janelas": len(janelas),
        "pregoes_fora_da_amostra": len(curva_oos),
        "retorno_oos_total_pct": round(float((1 + curva_oos).prod() - 1) * 100, 2),
        "cdi_no_mesmo_trecho_pct": round(_retorno_pct(cdi_trecho), 2),
        "sharpe_oos": round(
            float(curva_oos.mean() / curva_oos.std(ddof=1) * np.sqrt(DIAS_UTEIS_ANO))
            if curva_oos.std(ddof=1) > 0 else 0.0,
            3,
        ),
        "janelas_que_bateram_o_cdi": sum(1 for j in janelas if j.bateu_cdi_fora_da_amostra),
        "estrategias_testadas": len(curvas_treino),
        "escolhas": {j.escolhida: sum(1 for x in janelas if x.escolhida == j.escolhida) for j in janelas},
    }
    resumo["retorno_oos_menos_cdi_pp"] = round(
        resumo["retorno_oos_total_pct"] - resumo["cdi_no_mesmo_trecho_pct"], 2
    )

    # Matriz de tentativas para DSR e PBO.
    tamanho = min((len(v) for v in curvas_treino.values()), default=0)
    matriz = (
        np.column_stack([np.asarray(v[:tamanho], dtype=float) for v in curvas_treino.values()])
        if tamanho > 20
        else None
    )

    sr_variance = 0.0
    if matriz is not None and matriz.shape[1] > 1:
        sharpes = [
            float(c.mean() / c.std(ddof=1)) if c.std(ddof=1) > 0 else 0.0 for c in matriz.T
        ]
        sr_variance = float(np.var(sharpes, ddof=1))

    try:
        resumo["deflated_sharpe"] = {
            k: (round(v, 4) if isinstance(v, (int, float)) else v)
            for k, v in evaluate_strategy(
                curva_oos.to_numpy(dtype=float),
                n_trials=len(curvas_treino) * max(1, len(janelas)),
                sr_variance=max(sr_variance, 1e-6),
                periods_per_year=DIAS_UTEIS_ANO,
            ).items()
        }
    except Exception as exc:
        logger.warning("Deflated Sharpe indisponível: %s", exc)
        resumo["deflated_sharpe"] = None

    try:
        if matriz is None or matriz.shape[1] < 2:
            raise ValueError("matriz insuficiente")
        resumo["pbo"] = {
            k: (round(v, 4) if isinstance(v, (int, float)) else v)
            for k, v in probability_of_backtest_overfitting(matriz, n_splits=8).items()
        }
    except Exception as exc:
        logger.warning("PBO indisponível: %s", exc)
        resumo["pbo"] = None

    return janelas, curva_oos, resumo


def veredito(resumo: dict, minimo_janelas: int = 3) -> dict:
    """Traduz os números numa decisão. Os cortes são conservadores de propósito."""
    motivos: list[str] = []
    aprovado = True

    if resumo.get("janelas", 0) < minimo_janelas:
        aprovado = False
        motivos.append(
            f"só {resumo.get('janelas', 0)} janelas fora da amostra "
            f"(mínimo {minimo_janelas})"
        )

    if resumo.get("retorno_oos_menos_cdi_pp", 0) <= 0:
        aprovado = False
        motivos.append(
            f"ficou {abs(resumo.get('retorno_oos_menos_cdi_pp', 0)):.1f} p.p. "
            f"ABAIXO do CDI fora da amostra"
        )

    positivas = resumo.get("janelas_que_bateram_o_cdi", 0)
    total = resumo.get("janelas", 0)
    if total and positivas / total < 0.5:
        aprovado = False
        motivos.append(f"só {positivas}/{total} janelas bateram o CDI")

    pbo_bruto = resumo.get("pbo")
    pbo = pbo_bruto.get("pbo") if isinstance(pbo_bruto, dict) else pbo_bruto
    if pbo is not None and pbo > 0.5:
        aprovado = False
        motivos.append(f"PBO {pbo:.2f} > 0,50: escolher pelo passado não prevê o futuro")

    dsr_bruto = resumo.get("deflated_sharpe")
    dsr = dsr_bruto.get("dsr") if isinstance(dsr_bruto, dict) else dsr_bruto
    if dsr is not None and dsr < 0.95:
        aprovado = False
        motivos.append(
            f"Deflated Sharpe {dsr:.2f} < 0,95: o resultado não se separa do acaso "
            f"dadas as {resumo.get('estrategias_testadas', '?')} tentativas"
        )

    return {
        "aprovado_para_papel": aprovado,
        "aprovado_para_dinheiro_real": False,  # nunca automático
        "motivos": motivos or ["passou em todos os cortes"],
    }
