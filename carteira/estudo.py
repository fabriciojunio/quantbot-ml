"""
Estudo comparativo de carteiras.

Roda todas as políticas de alocação e todas as frequências de rebalanceamento
sobre os mesmos dados, com o mesmo custo e o mesmo imposto, e coloca o
resultado ao lado das referências que importam: CDI, Ibovespa e a carteira
comprada e esquecida.

O critério de sucesso é declarado antes de rodar, e é duro de propósito:
**bater o CDI líquido depois de custo e imposto**. Num país com juro real
alto, essa é a barra. Estratégia que não passa dela não é estratégia ruim, é
uma forma cara de ter o resultado da renda fixa.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import alocacao, backtest, dados, universo
from .rebalanceamento import ConfigRebalanceamento, Politica, contar_eventos, marcar_datas

logger = logging.getLogger("quantbot.carteira.estudo")

SAIDA = Path("data/carteira/estudos")


def montar_estrategias(
    precos: pd.DataFrame, cdi_acumulado: pd.Series
) -> dict[str, pd.DataFrame]:
    """Todas as alocações a comparar, das mais simples às mais elaboradas."""
    estrategias: dict[str, pd.DataFrame] = {}

    for nome, alvo in universo.CARTEIRAS_REFERENCIA.items():
        if nome == "so_cdi":
            continue  # o CDI entra como referência, não como carteira
        disponivel = {t: p for t, p in alvo.items() if t in precos.columns}
        if disponivel:
            estrategias[f"fixa_{nome}"] = alocacao.pesos_fixos(precos, disponivel)

    estrategias["peso_igual"] = alocacao.peso_igual(precos)
    estrategias["inverso_vol"] = alocacao.inverso_da_volatilidade(precos)
    estrategias["momentum_absoluto"] = alocacao.momentum_absoluto_puro(precos, cdi_acumulado)
    estrategias["dual_momentum_3"] = alocacao.dual_momentum(precos, 3, cdi_acumulado)
    estrategias["dual_momentum_5"] = alocacao.dual_momentum(precos, 5, cdi_acumulado)

    nucleo = {t: p for t, p in universo.CARTEIRAS_REFERENCIA["diversificada"].items()
              if t in precos.columns}
    estrategias["nucleo_com_trend"] = alocacao.nucleo_com_trend(
        precos, nucleo, cdi_acumulado, fatia_protegida=0.5
    )
    estrategias["nucleo_com_trend_total"] = alocacao.nucleo_com_trend(
        precos, nucleo, cdi_acumulado, fatia_protegida=1.0
    )
    return estrategias


POLITICAS = (
    Politica.MENSAL,
    Politica.TRIMESTRAL,
    Politica.SEMESTRAL,
    Politica.ANUAL,
    Politica.BANDA,
    Politica.BANDA_COM_CUSTO,
)


def rodar(
    anos: int = 5,
    capital_inicial: float = 10_000.0,
    aporte_mensal: float = 0.0,
    politicas: tuple[Politica, ...] = POLITICAS,
    incluir_ibov: bool = True,
) -> dict:
    """Executa a bateria completa e devolve o relatório."""
    tickers = list(universo.UNIVERSO_B3.keys())
    precos = dados.alinhar(dados.precos(tickers, anos=anos))
    dados.valida_precos(precos, "ETFs B3")

    cdi = dados.cdi_diario(anos=anos)
    cdi_alinhado = cdi.reindex(precos.index).ffill().fillna(0.0)
    cdi_acumulado = (1 + cdi_alinhado).cumprod() * capital_inicial

    comparacoes = {"cdi": cdi_acumulado}
    if incluir_ibov:
        try:
            ibov = dados.precos("^BVSP", anos=anos)["^BVSP"]
            ibov = ibov.reindex(precos.index).ffill()
            comparacoes["ibov"] = ibov / float(ibov.iloc[0]) * capital_inicial
        except Exception as exc:
            logger.warning("Ibovespa indisponível: %s", exc)

    estrategias = montar_estrategias(precos, cdi_acumulado)
    retornos = precos.pct_change().fillna(0.0)

    relatorio: dict = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "periodo": [str(precos.index.min())[:10], str(precos.index.max())[:10]],
        "anos": anos,
        "capital_inicial": capital_inicial,
        "aporte_mensal": aporte_mensal,
        "ativos": tickers,
        "criterio_de_sucesso": (
            "bater o CDI líquido de custo e imposto no mesmo período"
        ),
        "resultados": [],
    }

    cdi_periodo_pct = (float(cdi_acumulado.iloc[-1]) / capital_inicial - 1) * 100
    relatorio["cdi_no_periodo_pct"] = round(cdi_periodo_pct, 2)

    for nome_est, pesos in estrategias.items():
        for politica in politicas:
            cfg = backtest.ConfigBacktest(
                capital_inicial=capital_inicial,
                aporte_mensal=aporte_mensal,
                config_rebal=ConfigRebalanceamento(politica=politica),
            )
            try:
                res = backtest.executar(
                    precos, pesos, cfg, cdi=cdi, nome=f"{nome_est}/{politica.value}"
                )
            except Exception as exc:
                logger.warning("%s/%s falhou: %s", nome_est, politica.value, exc)
                continue

            resumo = res.resumo(comparacoes)
            eventos = contar_eventos(marcar_datas(pesos, retornos, cfg.config_rebal))
            resumo["rebal_por_ano"] = eventos["por_ano"]
            resumo["estrategia"] = nome_est
            resumo["politica"] = politica.value
            resumo["bateu_cdi"] = resumo["retorno_sobre_investido_pct"] > cdi_periodo_pct

            relatorio["resultados"].append(resumo)
            logger.info(
                "%-28s %-16s ret %7.2f%%  dd %6.2f%%  atrito %5.2f%%  %s",
                nome_est, politica.value,
                resumo["retorno_sobre_investido_pct"],
                resumo["max_drawdown_pct"],
                resumo["atrito_pct_do_investido"],
                "BATEU CDI" if resumo["bateu_cdi"] else "",
            )

    relatorio["conclusao"] = _concluir(relatorio)
    return relatorio


def _concluir(relatorio: dict) -> dict:
    res = relatorio["resultados"]
    if not res:
        return {"resumo": "nenhuma combinação completou"}

    venceram = [r for r in res if r["bateu_cdi"]]
    melhor = max(res, key=lambda r: r["retorno_sobre_investido_pct"])
    melhor_sharpe = max(res, key=lambda r: r["sharpe"])
    menor_dd = max(res, key=lambda r: r["max_drawdown_pct"])

    return {
        "combinacoes_testadas": len(res),
        "bateram_o_cdi": len(venceram),
        "cdi_no_periodo_pct": relatorio["cdi_no_periodo_pct"],
        "melhor_retorno": {
            "nome": melhor["nome"],
            "retorno_pct": melhor["retorno_sobre_investido_pct"],
            "drawdown_pct": melhor["max_drawdown_pct"],
        },
        "melhor_sharpe": {
            "nome": melhor_sharpe["nome"],
            "sharpe": melhor_sharpe["sharpe"],
            "retorno_pct": melhor_sharpe["retorno_sobre_investido_pct"],
        },
        "menor_drawdown": {
            "nome": menor_dd["nome"],
            "drawdown_pct": menor_dd["max_drawdown_pct"],
            "retorno_pct": menor_dd["retorno_sobre_investido_pct"],
        },
        "atrito_mediano_pct": round(
            float(pd.Series([r["atrito_pct_do_investido"] for r in res]).median()), 2
        ),
        "resumo": (
            f"{len(venceram)} de {len(res)} combinações bateram o CDI de "
            f"{relatorio['cdi_no_periodo_pct']:.1f}% no período"
        ),
    }


def salvar(relatorio: dict, nome: str | None = None) -> Path:
    SAIDA.mkdir(parents=True, exist_ok=True)
    nome = nome or f"carteiras_{datetime.now(timezone.utc):%Y%m%d_%H%M}.json"
    caminho = SAIDA / nome
    caminho.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return caminho


def tabela(relatorio: dict, ordenar_por: str = "retorno_sobre_investido_pct") -> pd.DataFrame:
    df = pd.DataFrame(relatorio["resultados"])
    colunas = [
        "estrategia", "politica", "retorno_sobre_investido_pct", "cagr_pct",
        "vol_anual_pct", "max_drawdown_pct", "sharpe", "rebal_por_ano",
        "custo_total", "imposto_total", "atrito_pct_do_investido",
        "alpha_vs_cdi_pp", "bateu_cdi",
    ]
    existentes = [c for c in colunas if c in df.columns]
    return df[existentes].sort_values(ordenar_por, ascending=False)
