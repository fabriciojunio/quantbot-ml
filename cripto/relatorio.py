"""
Relatórios em Markdown.

Regra de estilo que vale para todo texto gerado aqui: número sem contexto é
propaganda. Todo resultado sai acompanhado do custo que ele já pagou, do
período que ele cobre e do que ele NÃO prova. Se o relatório não couber nessa
regra, é porque o resultado não deveria ser publicado.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .backtest import Resultado
from .custos import ModeloCusto, resumo_viabilidade


def _tabela(linhas: list[list[str]], cabecalho: list[str]) -> str:
    out = ["| " + " | ".join(cabecalho) + " |"]
    out.append("|" + "|".join(["---"] * len(cabecalho)) + "|")
    for linha in linhas:
        out.append("| " + " | ".join(str(c) for c in linha) + " |")
    return "\n".join(out)


def relatorio_backtest(res: Resultado, precos: pd.Series) -> str:
    r = res.resumo(precos)
    ops = [o for o in res.operacoes if o.saida_em is not None]

    motivos: dict[str, int] = {}
    for o in ops:
        motivos[o.motivo_saida] = motivos.get(o.motivo_saida, 0) + 1

    partes = [
        f"# Backtest — {r['simbolo']} {r['intervalo']}",
        "",
        f"_Gerado em {datetime.now(timezone.utc):%d/%m/%Y %H:%M} UTC._",
        "",
        "## Resultado depois do custo",
        "",
        _tabela(
            [
                ["Capital inicial", f"US$ {r['capital_inicial']:,.2f}"],
                ["Capital final", f"US$ {r['capital_final']:,.2f}"],
                ["Retorno", f"{r['retorno_total_pct']:+.2f}%"],
                ["Buy and hold no mesmo período", f"{r.get('buy_and_hold_pct', 0):+.2f}%"],
                ["Diferença para o buy and hold", f"{r.get('alpha_vs_buy_and_hold_pp', 0):+.2f} p.p."],
                ["Sharpe anualizado", f"{r['sharpe_anualizado']:.2f}"],
                ["Drawdown máximo", f"{r['max_drawdown_pct']:.2f}%"],
            ],
            ["Métrica", "Valor"],
        ),
        "",
        "## Operação a operação",
        "",
        _tabela(
            [
                ["Operações fechadas", r["operacoes"]],
                ["Taxa de acerto", f"{r['taxa_acerto_pct']:.1f}%"],
                ["Payoff (ganho médio / perda média)", f"{r['payoff']:.2f}"],
                ["Expectância por operação", f"{r['expectancy_por_operacao_pct']:+.4f}%"],
            ],
            ["Métrica", "Valor"],
        ),
        "",
        "## Onde o dinheiro foi",
        "",
        _tabela(
            [
                ["Resultado bruto (sem atrito)", f"US$ {r['pnl_bruto_usd']:+,.2f}"],
                ["Custo de execução (taxa + spread + slippage)", f"US$ -{abs(r['custo_total_usd']):,.2f}"],
                ["**Resultado líquido**", f"**US$ {r['pnl_liquido_usd']:+,.2f}**"],
            ],
            ["Componente", "Valor"],
        ),
        "",
    ]

    if r["operacoes"]:
        peso_custo = abs(r["custo_total_usd"]) / max(abs(r["pnl_bruto_usd"]), 1e-9)
        partes.append(
            f"O custo de execução equivale a {peso_custo:.1f}x o resultado bruto da estratégia. "
            "Quanto maior esse número, mais a estratégia está trabalhando para a corretora."
        )
        partes.append("")

    if motivos:
        partes += [
            "## Motivo das saídas",
            "",
            _tabela(
                [[m, n, f"{n / len(ops) * 100:.0f}%"] for m, n in sorted(motivos.items(), key=lambda x: -x[1])],
                ["Motivo", "Operações", "Fatia"],
            ),
            "",
        ]

    partes += [
        "## O que este número não prova",
        "",
        "- Um backtest é uma medida sobre um passado específico, não uma previsão.",
        "- O resultado acima é de uma configuração só; escolher a melhor entre várias",
        "  infla a métrica. Para corrigir isso existe o walk-forward com Deflated Sharpe.",
        "- Custo de execução é estimado por modelo. Em mercado estressado, spread e",
        "  slippage ficam piores do que o modelo aqui assume.",
        "",
    ]
    return "\n".join(partes)


def relatorio_estudo(estudo: dict) -> str:
    conclusao = estudo.get("conclusao", {})
    custo = estudo.get("custo_modelado", {})

    partes = [
        "# Estudo de viabilidade — motor de cripto",
        "",
        f"_Gerado em {estudo.get('gerado_em', '')[:19]} UTC, sobre {estudo.get('dias_de_historico')} dias de histórico._",
        "",
        "## Custo modelado",
        "",
        _tabela(
            [
                ["Taxa taker", f"{custo.get('taxa_taker', 0) * 100:.3f}%"],
                ["Taxa maker", f"{custo.get('taxa_maker', 0) * 100:.3f}%"],
                ["Spread médio", f"{custo.get('spread_pct', 0) * 100:.3f}%"],
                ["Custo de ida e volta em US$ 1.000", f"{custo.get('custo_ida_volta_pct_em_1000usd', 0):.3f}%"],
            ],
            ["Item", "Valor"],
        ),
        "",
        "## Conclusão",
        "",
        _tabela(
            [
                ["Casos validados", conclusao.get("casos_validados", 0)],
                ["Com retorno fora da amostra positivo", conclusao.get("casos_com_retorno_oos_positivo", 0)],
                ["Aprovados para papel", conclusao.get("casos_aprovados_para_papel", 0)],
                ["Retorno fora da amostra mediano", f"{conclusao.get('retorno_oos_mediano_pct', 0):+.2f}%"],
                ["Melhor caso", f"{conclusao.get('retorno_oos_melhor_pct', 0):+.2f}%"],
                ["Pior caso", f"{conclusao.get('retorno_oos_pior_pct', 0):+.2f}%"],
            ],
            ["Métrica", "Valor"],
        ),
        "",
        f"**{conclusao.get('resumo', '')}**",
        "",
        "## Caso a caso",
        "",
    ]

    linhas = []
    for caso in estudo.get("casos", []):
        wf = caso.get("walk_forward")
        if not wf:
            continue
        vd = caso.get("veredito", {})
        linhas.append(
            [
                caso["simbolo"],
                caso["intervalo"],
                "long+short" if caso.get("permite_venda") else "long",
                wf.get("janelas", 0),
                f"{wf.get('retorno_oos_total_pct', 0):+.2f}%",
                f"{wf.get('sharpe_oos', 0):.2f}",
                f"{wf.get('janelas_positivas', 0)}/{wf.get('janelas', 0)}",
                "sim" if vd.get("aprovado_para_papel") else "não",
            ]
        )

    partes.append(
        _tabela(
            linhas,
            ["Par", "Tempo", "Lados", "Janelas", "Retorno OOS", "Sharpe OOS", "Janelas +", "Aprovado"],
        )
    )
    partes.append("")
    return "\n".join(partes)


def relatorio_alvo_centavos(modelo: ModeloCusto | None = None) -> str:
    """Tabela que mostra por que sair em +US$ 0,10 não fecha a conta."""
    modelo = modelo or ModeloCusto()
    linhas = []
    for notional in (50, 100, 200, 500, 1_000, 3_000):
        r = resumo_viabilidade(0.10, notional, modelo)
        linhas.append(
            [
                f"US$ {notional:,}",
                f"US$ {r['custo_usd']:.4f}",
                f"US$ {r['resultado_liquido_usd']:+.4f}",
                f"{r['quantas_vezes_o_alvo_o_custo_come']:.1f}x",
                "sim" if r["viavel"] else "não",
            ]
        )

    return "\n".join(
        [
            "# A regra de sair em +US$ 0,10",
            "",
            "O painel anterior fechava a posição assim que o lucro não realizado batia",
            "dez centavos de dólar, e o simulador não cobrava taxa nenhuma. Com a taxa",
            f"real de {modelo.taxa_taker * 100:.2f}% por ponta mais spread e slippage, o custo de abrir e",
            "fechar fica assim:",
            "",
            _tabela(
                linhas,
                ["Tamanho da posição", "Custo ida e volta", "Sobra do alvo de US$ 0,10", "Custo / alvo", "Dá lucro?"],
            ),
            "",
            "Para o alvo de dez centavos pagar o próprio custo numa posição de US$ 1.000,",
            "a taxa da corretora teria que ser praticamente zero. Não existe corretora",
            "assim para pessoa física.",
            "",
        ]
    )
