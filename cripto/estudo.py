"""
Estudo de viabilidade.

Roda a mesma bateria de validação em vários ativos e vários tempos gráficos e
guarda tudo em JSON. A pergunta que ele responde não é "qual é o melhor
resultado?" — é "existe alguma combinação com evidência de vantagem depois do
custo?". A diferença importa: a primeira pergunta sempre tem resposta, a
segunda quase sempre não tem, e é a segunda que decide se vale colocar
dinheiro.

Uso:
    python -m cripto.cli estudo --dias 400
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import mercado, walkforward
from .backtest import ConfigBacktest, executar
from .custos import ModeloCusto, resumo_viabilidade
from .sinais import ConfigSinal

logger = logging.getLogger("quantbot.cripto.estudo")

SAIDA = Path("data/cripto/estudos")

# Pares com liquidez suficiente para o custo modelado fazer sentido.
SIMBOLOS_PADRAO = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")

# Tempos gráficos com quantas barras de treino/teste usar em cada um.
# Quanto mais rápido o gráfico, mais barras cabem na mesma janela de calendário.
TEMPOS_PADRAO = {
    "15m": (4_000, 1_000),
    "1h": (2_000, 500),
    "4h": (900, 220),
    "1d": (400, 90),
}


def _config(intervalo: str, permitir_venda: bool, modelo: ModeloCusto) -> ConfigBacktest:
    return ConfigBacktest(
        intervalo=intervalo,
        modelo_custo=modelo,
        config_sinal=ConfigSinal(permitir_venda=permitir_venda),
    )


def rodar(
    simbolos: tuple[str, ...] = SIMBOLOS_PADRAO,
    tempos: dict[str, tuple[int, int]] | None = None,
    dias: int = 400,
    modelo_custo: ModeloCusto | None = None,
    incluir_venda: bool = True,
) -> dict:
    """Executa a bateria completa e devolve o relatório em dicionário."""
    tempos = tempos or TEMPOS_PADRAO
    modelo_custo = modelo_custo or ModeloCusto()

    relatorio: dict = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "dias_de_historico": dias,
        "custo_modelado": {
            "taxa_taker": modelo_custo.taxa_taker,
            "taxa_maker": modelo_custo.taxa_maker,
            "spread_pct": modelo_custo.spread_pct,
            "custo_ida_volta_pct_em_1000usd": round(
                modelo_custo.custo_ida_volta_pct(1_000) * 100, 4
            ),
        },
        "casos": [],
        "aprovados": [],
    }

    for simbolo in simbolos:
        for intervalo, (treino, teste) in tempos.items():
            df = None
            try:
                df = mercado.historico(simbolo, intervalo, dias=dias)
                mercado.valida_serie(df, f"{simbolo} {intervalo}")
            except Exception as exc:
                logger.warning("Pulei %s %s: %s", simbolo, intervalo, exc)
                relatorio["casos"].append(
                    {"simbolo": simbolo, "intervalo": intervalo, "erro": str(exc)}
                )
                continue

            lados = [False, True] if incluir_venda else [False]
            for permitir_venda in lados:
                nome = f"{simbolo} {intervalo} {'long+short' if permitir_venda else 'long'}"
                base = _config(intervalo, permitir_venda, modelo_custo)

                caso: dict = {
                    "simbolo": simbolo,
                    "intervalo": intervalo,
                    "permite_venda": permitir_venda,
                    "candles": len(df),
                    "periodo": [str(df.index.min()), str(df.index.max())],
                }

                try:
                    completo = executar(df, simbolo, base)
                    caso["backtest_completo"] = completo.resumo(df["fechamento"])
                except Exception as exc:
                    caso["erro_backtest"] = str(exc)

                try:
                    janelas, _, resumo = walkforward.walk_forward(
                        df,
                        simbolo,
                        base=base,
                        barras_treino=treino,
                        barras_teste=teste,
                    )
                    caso["walk_forward"] = resumo
                    caso["veredito"] = walkforward.veredito(resumo)
                    caso["janelas"] = [
                        {
                            "teste": [str(j.teste_inicio)[:10], str(j.teste_fim)[:10]],
                            "sharpe_teste": j.sharpe_teste,
                            "retorno_pct": j.retorno_teste_pct,
                            "operacoes": j.operacoes_teste,
                            "parametros": j.melhor_parametro,
                        }
                        for j in janelas
                    ]
                    if caso["veredito"]["aprovado_para_papel"]:
                        relatorio["aprovados"].append(nome)
                except Exception as exc:
                    caso["erro_walk_forward"] = str(exc)

                logger.info(
                    "%s: OOS %s%% | veredito %s",
                    nome,
                    caso.get("walk_forward", {}).get("retorno_oos_total_pct", "?"),
                    caso.get("veredito", {}).get("aprovado_para_papel", "?"),
                )
                relatorio["casos"].append(caso)

    relatorio["conclusao"] = _concluir(relatorio)
    return relatorio


def _concluir(relatorio: dict) -> dict:
    validos = [c for c in relatorio["casos"] if "walk_forward" in c]
    if not validos:
        return {"resumo": "nenhum caso completou a validação", "casos_validados": 0}

    retornos = [c["walk_forward"]["retorno_oos_total_pct"] for c in validos]
    positivos = [r for r in retornos if r > 0]

    return {
        "casos_validados": len(validos),
        "casos_com_retorno_oos_positivo": len(positivos),
        "casos_aprovados_para_papel": len(relatorio["aprovados"]),
        "retorno_oos_mediano_pct": round(float(pd.Series(retornos).median()), 2),
        "retorno_oos_melhor_pct": round(max(retornos), 2),
        "retorno_oos_pior_pct": round(min(retornos), 2),
        "resumo": (
            "há pelo menos um caso aprovado — leia os motivos antes de comemorar"
            if relatorio["aprovados"]
            else "nenhuma combinação passou nos cortes de validação"
        ),
    }


def teste_do_alvo_de_centavos(modelo: ModeloCusto | None = None) -> list[dict]:
    """Reproduz numericamente a regra do painel antigo: sair em +US$ 0,10.

    Mostra, para vários tamanhos de posição, quanto do alvo o custo já come
    antes de o preço se mexer.
    """
    modelo = modelo or ModeloCusto()
    return [
        resumo_viabilidade(alvo_usd=0.10, notional_usd=n, modelo=modelo)
        for n in (50, 100, 200, 500, 1_000, 3_000)
    ]


def salvar(relatorio: dict, nome: str | None = None) -> Path:
    SAIDA.mkdir(parents=True, exist_ok=True)
    nome = nome or f"estudo_{datetime.now(timezone.utc):%Y%m%d_%H%M}.json"
    caminho = SAIDA / nome
    caminho.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Estudo salvo em %s", caminho)
    return caminho
