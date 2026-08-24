"""
Validação anti-autoengano.

Um backtest sozinho não vale quase nada: com liberdade para escolher período,
ativo e parâmetro, qualquer pessoa acha uma curva bonita. O que separa
resultado de sorte é medir quantas configurações foram tentadas e o quanto o
melhor resultado se sustenta fora da amostra em que foi escolhido.

Três camadas, reaproveitando o pacote `validation/` que já existe no QuantBot:

* **Walk-forward ancorado** — otimiza no passado, opera no futuro imediato,
  anda a janela. Só o resultado fora da amostra conta.
* **Deflated Sharpe Ratio** — desconta o Sharpe pelo número de tentativas.
  Vinte configurações testadas produzem um "melhor Sharpe" alto só por acaso;
  o DSR diz quanto sobra depois de tirar esse efeito.
* **Probability of Backtest Overfitting** — mede se escolher a melhor
  configuração no passado tem alguma relação com ir bem no futuro. PBO acima
  de 0.5 significa que a seleção é pior que jogar dados.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from validation import evaluate_strategy, probability_of_backtest_overfitting

from .backtest import BARRAS_ANO, ConfigBacktest, Resultado, executar

logger = logging.getLogger("quantbot.cripto.walkforward")


@dataclass
class JanelaWF:
    treino_inicio: pd.Timestamp
    treino_fim: pd.Timestamp
    teste_inicio: pd.Timestamp
    teste_fim: pd.Timestamp
    melhor_parametro: dict
    sharpe_treino: float
    sharpe_teste: float
    retorno_teste_pct: float
    operacoes_teste: int


def grade_padrao() -> list[dict]:
    """Grade de parâmetros deliberadamente pequena.

    Grade grande não é sinal de rigor: é a forma mais eficiente de garantir
    overfitting. Cada dimensão aqui tem justificativa — limiar de entrada,
    filtro de tendência, e a relação entre stop e alvo.
    """
    combos = []
    for limite in (58.0, 62.0, 66.0):
        for adx_min in (0.0, 20.0, 25.0):
            for atr_stop, atr_alvo in ((1.5, 3.0), (2.0, 3.0), (2.0, 4.0)):
                combos.append(
                    {
                        "limite_compra": limite,
                        "limite_venda": 100 - limite,
                        "adx_minimo": adx_min,
                        "atr_stop": atr_stop,
                        "atr_alvo": atr_alvo,
                    }
                )
    return combos


def _config_com(base: ConfigBacktest, params: dict) -> ConfigBacktest:
    sinal = replace(base.config_sinal, **params)
    return replace(base, config_sinal=sinal)


def _retornos_por_barra(res: Resultado) -> pd.Series:
    return res.capital.pct_change().fillna(0.0)


def walk_forward(
    df: pd.DataFrame,
    simbolo: str = "BTCUSDT",
    base: ConfigBacktest | None = None,
    grade: list[dict] | None = None,
    barras_treino: int = 2_000,
    barras_teste: int = 500,
    embargo: int = 24,
) -> tuple[list[JanelaWF], pd.Series, dict]:
    """Roda walk-forward ancorado e devolve janelas, curva fora da amostra e resumo.

    `embargo` descarta as barras logo após o treino, para que uma posição
    aberta no fim do treino não contamine o teste. É o mesmo princípio do
    purge/embargo que o `validation.PurgedKFold` aplica em classificação.
    """
    base = base or ConfigBacktest()
    grade = grade or grade_padrao()

    janelas: list[JanelaWF] = []
    retornos_oos: list[pd.Series] = []
    matriz_treino: dict[str, list[float]] = {}

    inicio = 0
    while inicio + barras_treino + embargo + barras_teste <= len(df):
        fim_treino = inicio + barras_treino
        inicio_teste = fim_treino + embargo
        fim_teste = inicio_teste + barras_teste

        treino = df.iloc[inicio:fim_treino]
        teste = df.iloc[inicio_teste - 250 : fim_teste]  # 250 barras de aquecimento

        melhor_params, melhor_sharpe = None, -np.inf
        for idx, params in enumerate(grade):
            try:
                r = executar(treino, simbolo, _config_com(base, params))
            except ValueError:
                continue
            s = r.sharpe
            matriz_treino.setdefault(f"cfg_{idx}", []).extend(
                _retornos_por_barra(r).tolist()
            )
            if s > melhor_sharpe:
                melhor_sharpe, melhor_params = s, params

        if melhor_params is None:
            break

        try:
            r_teste = executar(teste, simbolo, _config_com(base, melhor_params))
        except ValueError:
            break

        # Descarta o aquecimento: só o trecho fora da amostra conta.
        corte = df.index[inicio_teste]
        capital_oos = r_teste.capital.loc[r_teste.capital.index >= corte]
        ret_oos = capital_oos.pct_change().dropna()
        retornos_oos.append(ret_oos)

        ops_oos = [o for o in r_teste.operacoes if o.entrada_em >= corte]
        janelas.append(
            JanelaWF(
                treino_inicio=treino.index[0],
                treino_fim=treino.index[-1],
                teste_inicio=corte,
                teste_fim=teste.index[-1],
                melhor_parametro=melhor_params,
                sharpe_treino=round(melhor_sharpe, 3),
                sharpe_teste=round(r_teste.sharpe, 3),
                retorno_teste_pct=round(
                    float(capital_oos.iloc[-1] / capital_oos.iloc[0] - 1) * 100, 2
                )
                if len(capital_oos) > 1
                else 0.0,
                operacoes_teste=len(ops_oos),
            )
        )

        inicio += barras_teste

    if not retornos_oos:
        raise ValueError(
            "série curta demais para walk-forward. Reduza barras_treino/barras_teste "
            "ou baixe mais histórico."
        )

    curva_oos = pd.concat(retornos_oos)
    barras_ano = BARRAS_ANO.get(base.intervalo, 8_760)

    resumo = {
        "janelas": len(janelas),
        "barras_fora_da_amostra": len(curva_oos),
        "retorno_oos_total_pct": round(float((1 + curva_oos).prod() - 1) * 100, 2),
        "sharpe_oos": round(
            float(curva_oos.mean() / curva_oos.std(ddof=1) * np.sqrt(barras_ano))
            if curva_oos.std(ddof=1) > 0
            else 0.0,
            3,
        ),
        "janelas_positivas": sum(1 for j in janelas if j.retorno_teste_pct > 0),
        "configuracoes_testadas": len(grade),
        "sharpe_treino_medio": round(float(np.mean([j.sharpe_treino for j in janelas])), 3),
        "sharpe_teste_medio": round(float(np.mean([j.sharpe_teste for j in janelas])), 3),
    }

    # Variância dos Sharpe testados: é o que mede quanta dispersão a busca
    # gerou, e portanto quanto do "melhor" pode ser sorte.
    tamanho = min((len(v) for v in matriz_treino.values()), default=0)
    matriz = (
        np.column_stack([np.asarray(v[:tamanho], dtype=float) for v in matriz_treino.values()])
        if tamanho > 10
        else None
    )

    sr_variance = 0.0
    if matriz is not None and matriz.shape[1] > 1:
        sharpes = [
            float(col.mean() / col.std(ddof=1)) if col.std(ddof=1) > 0 else 0.0
            for col in matriz.T
        ]
        sr_variance = float(np.var(sharpes, ddof=1))

    # Deflated Sharpe: corrige o Sharpe fora da amostra pelo nº de tentativas.
    try:
        resumo["deflated_sharpe"] = {
            k: (round(v, 4) if isinstance(v, (int, float)) else v)
            for k, v in evaluate_strategy(
                curva_oos.to_numpy(dtype=float),
                n_trials=len(grade) * max(1, len(janelas)),
                sr_variance=max(sr_variance, 1e-6),
                periods_per_year=barras_ano,
            ).items()
        }
    except Exception as exc:
        logger.warning("Não consegui calcular o Deflated Sharpe: %s", exc)
        resumo["deflated_sharpe"] = None

    # PBO: a seleção pelo melhor treino prevê alguma coisa?
    try:
        if matriz is None or matriz.shape[1] < 2:
            raise ValueError("matriz de tentativas insuficiente")
        resumo["pbo"] = {
            k: (round(v, 4) if isinstance(v, (int, float)) else v)
            for k, v in probability_of_backtest_overfitting(matriz, n_splits=8).items()
        }
    except Exception as exc:
        logger.warning("Não consegui calcular o PBO: %s", exc)
        resumo["pbo"] = None

    return janelas, curva_oos, resumo


def varredura(
    df: pd.DataFrame,
    simbolo: str,
    base: ConfigBacktest | None = None,
    grade: list[dict] | None = None,
) -> pd.DataFrame:
    """Roda a grade inteira na série toda e devolve a tabela de resultados.

    Serve para enxergar a distribuição, não para escolher o melhor. O melhor
    de uma varredura in-sample é justamente o número em que menos se deve
    confiar — por isso a coluna `n_configuracoes` vai junto: ela é o
    denominador que o Deflated Sharpe vai usar depois.
    """
    base = base or ConfigBacktest()
    grade = grade or grade_padrao()

    linhas = []
    for params in grade:
        try:
            r = executar(df, simbolo, _config_com(base, params))
        except ValueError as exc:
            logger.warning("Config %s falhou: %s", params, exc)
            continue
        linha = dict(params)
        linha.update(r.resumo(df["fechamento"]))
        linha["n_configuracoes"] = len(grade)
        linhas.append(linha)

    return pd.DataFrame(linhas).sort_values("sharpe_anualizado", ascending=False)


def veredito(resumo: dict, minimo_janelas: int = 3) -> dict:
    """Traduz os números em uma decisão de operar ou não.

    Os cortes são conservadores de propósito. Um sistema que não passa aqui
    não é um sistema ruim de mercado — é um sistema sobre o qual ainda não há
    evidência suficiente para arriscar dinheiro.
    """
    motivos = []
    aprovado = True

    if resumo.get("janelas", 0) < minimo_janelas:
        aprovado = False
        motivos.append(
            f"só {resumo.get('janelas', 0)} janelas fora da amostra "
            f"(mínimo {minimo_janelas}): amostra insuficiente"
        )

    sharpe = resumo.get("sharpe_oos", 0.0)
    if sharpe < 1.0:
        aprovado = False
        motivos.append(f"Sharpe fora da amostra {sharpe:.2f} < 1.0")

    if resumo.get("retorno_oos_total_pct", 0.0) <= 0:
        aprovado = False
        motivos.append(
            f"retorno fora da amostra {resumo.get('retorno_oos_total_pct', 0):.2f}% não é positivo"
        )

    pbo_bruto = resumo.get("pbo")
    pbo = pbo_bruto.get("pbo") if isinstance(pbo_bruto, dict) else pbo_bruto
    if pbo is not None and pbo > 0.5:
        aprovado = False
        motivos.append(f"PBO {pbo:.2f} > 0.50: escolher pelo passado não prevê o futuro")

    dsr = resumo.get("deflated_sharpe")
    valor_dsr = dsr.get("dsr", dsr.get("deflated_sharpe_ratio")) if isinstance(dsr, dict) else dsr
    if valor_dsr is not None and valor_dsr < 0.95:
        aprovado = False
        motivos.append(
            f"Deflated Sharpe {valor_dsr:.2f} < 0.95: o resultado não se separa do acaso "
            "dado o número de tentativas"
        )

    positivas = resumo.get("janelas_positivas", 0)
    total = resumo.get("janelas", 0)
    if total and positivas / total < 0.5:
        aprovado = False
        motivos.append(f"só {positivas}/{total} janelas fora da amostra deram lucro")

    return {
        "aprovado_para_papel": aprovado,
        "aprovado_para_dinheiro_real": False,  # nunca automático; ver docs/CRIPTO.md
        "motivos": motivos or ["passou em todos os cortes de validação"],
    }
