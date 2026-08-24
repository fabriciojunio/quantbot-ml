"""
Linha de comando do motor de carteira.

    python -m carteira.cli aportes                 o que mais move o patrimônio
    python -m carteira.cli custo                   custo por frequência de giro
    python -m carteira.cli imposto                 quanto o giro custa em IR
    python -m carteira.cli estudo --anos 5          compara todas as carteiras
    python -m carteira.cli validar --anos 12        walk-forward + DSR + PBO
    python -m carteira.cli hoje                     situação e sinal do dia
    python -m carteira.cli meta --meta 1500000      quanto tempo até a meta
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from . import alocacao, aportes, custos_b3, dados, estudo, sinais, tributos, universo, validacao
from .rebalanceamento import Politica
from .regras_pai import guarda_de_regime
from .universo import Regime


def _log(verboso: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verboso else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def _saida(d) -> None:
    print(json.dumps(d, indent=2, ensure_ascii=False, default=str))


def cmd_aportes(args) -> int:
    print("Saldo depois de %d anos, começando com R$ %.0f" % (args.anos, args.capital))
    print()
    tabela = aportes.tabela_de_sensibilidade(args.capital, args.anos)
    print(tabela.to_string(index=False))
    print()
    r = aportes.quanto_pesa_a_estrategia(
        args.capital, args.aporte, args.anos, retorno_base_pct=13.9, retorno_melhor_pct=18.0
    )
    print(r["leitura"])
    print()
    print("Para viver de renda:")
    for pat in (100_000, 500_000, 1_000_000, 1_500_000):
        d = aportes.renda_mensal_sustentavel(pat)
        print("  R$ %9.0f de patrimônio  ->  R$ %7.0f por mês" % (pat, d["renda_mensal"]))
    return 0


def cmd_custo(args) -> int:
    ativo = universo.UNIVERSO_B3.get(args.ticker)
    if ativo is None:
        print(f"ticker desconhecido: {args.ticker}", file=sys.stderr)
        return 1
    linhas = custos_b3.comparar_frequencias(ativo, args.valor)
    df = pd.DataFrame(linhas)[
        ["frequencia", "custo_por_giro_pct", "custo_anual_brl", "custo_anual_pct"]
    ]
    print(f"Custo de girar R$ {args.valor:,.0f} em {ativo.nome} ({ativo.ticker})")
    print()
    print(df.to_string(index=False))
    print()
    print("A taxa de administração do fundo (%.2f%% a.a.) é cobrada por fora, dentro da cota."
          % (custos_b3.TAXA_ADM_ANUAL.get(ativo.ticker, 0) * 100))
    return 0


def cmd_imposto(args) -> int:
    print("Custo fiscal do giro, carteira de R$ %.0f rendendo %.1f%% ao ano" %
          (args.valor, args.ganho))
    print()
    linhas = []
    for giros in (0, 1, 2, 4, 12, 52):
        r = tributos.custo_fiscal_do_giro(args.ganho, args.valor, giros, Regime.ETF_RENDA_FIXA)
        linhas.append({
            "giros_por_ano": giros,
            "aliquota": f"{r['aliquota_girando'] * 100:.1f}%",
            "imposto_no_ano": r["imposto_girando"],
            "custo_extra_vs_parado": r["custo_extra_do_giro"],
        })
    print(pd.DataFrame(linhas).to_string(index=False))
    print()
    print("Comparação de regimes, para o mesmo rendimento líquido:")
    for liq in (8.0, 9.5, 11.0):
        bruto = tributos.equivalente_bruto(liq, dias=365)
        print("  FII isento a %.1f%%  ==  renda fixa tributada a %.2f%% bruto" % (liq, bruto))
    return 0


def cmd_estudo(args) -> int:
    rel = estudo.rodar(anos=args.anos, capital_inicial=args.capital, aporte_mensal=args.aporte)
    caminho = estudo.salvar(rel)
    print(estudo.tabela(rel).to_string(index=False))
    print()
    _saida({"arquivo": str(caminho), "conclusao": rel["conclusao"]})
    return 0


def cmd_validar(args) -> int:
    tickers = args.tickers.split(",") if args.tickers else list(universo.UNIVERSO_B3.keys())
    px = dados.alinhar(dados.precos(tickers, anos=args.anos))
    dados.valida_precos(px, "universo")
    cdi = dados.cdi_diario(anos=args.anos)

    janelas, _, resumo = validacao.walk_forward(
        px, cdi, capital_inicial=args.capital,
        dias_treino=args.treino, dias_teste=args.teste,
        politica=Politica(args.politica),
    )
    _saida({
        "resumo": resumo,
        "veredito": validacao.veredito(resumo),
        "janelas": [
            {
                "teste": list(j.teste),
                "escolhida": j.escolhida,
                "retorno_pct": j.retorno_teste_pct,
                "cdi_pct": j.cdi_teste_pct,
                "bateu_cdi": j.bateu_cdi_fora_da_amostra,
            }
            for j in janelas
        ],
    })
    return 0


def cmd_hoje(args) -> int:
    tickers = args.tickers.split(",") if args.tickers else list(universo.UNIVERSO_B3.keys())
    px = dados.alinhar(dados.precos(tickers, anos=3))
    cdi = dados.cdi_diario(anos=3)

    diag = sinais.diagnostico(px)
    cdi_acum = (1 + cdi.reindex(px.index).ffill().fillna(0.0)).cumprod()

    ret_ibov_12m = float(px["BOVA11.SA"].iloc[-1] / px["BOVA11.SA"].iloc[-252] - 1) if "BOVA11.SA" in px and len(px) > 252 else 0.0
    ret_cdi_12m = float(cdi_acum.iloc[-1] / cdi_acum.iloc[-252] - 1) if len(cdi_acum) > 252 else 0.0
    acima_200 = bool(sinais.filtro_de_tendencia(px).iloc[-1].get("BOVA11.SA", False))

    favoravel, detalhe = guarda_de_regime(ret_ibov_12m, ret_cdi_12m, acima_200)

    pesos = alocacao.dual_momentum(px, top_n=3, cdi_acumulado=cdi_acum)
    alvo = pesos.iloc[-1]
    alvo = alvo[alvo > 0]

    _saida({
        "data": diag["quando"],
        "macro": dados.macro(),
        "regime": detalhe,
        "sinais": diag["ativos"],
        "alocacao_sugerida": {t: round(float(p) * 100, 1) for t, p in alvo.items()},
        "caixa_sugerido_pct": round((1 - float(alvo.sum())) * 100, 1),
        "aviso": (
            "Sugestão de alocação a partir de regra declarada, não recomendação. "
            "A validação fora da amostra ainda NÃO aprovou esta estratégia."
        ),
    })
    return 0


def cmd_meta(args) -> int:
    print(f"Quanto tempo até R$ {args.meta:,.0f}, partindo de R$ {args.capital:,.0f}")
    print()
    linhas = []
    for aporte in (100, 300, 500, 1_000, 2_000):
        for r in (13.9, 18.0):
            d = aportes.tempo_para_meta(args.capital, aporte, r, args.meta)
            linhas.append({
                "aporte_mensal": aporte,
                "retorno_a.a.": f"{r:.1f}%",
                "anos": d["anos"] if d["alcancavel"] else "> 60",
            })
    print(pd.DataFrame(linhas).to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="carteira",
        description="Motor de carteira sistemática: custo, imposto e validação honesta.",
    )
    p.add_argument("-v", "--verboso", action="store_true")
    sub = p.add_subparsers(dest="comando", required=True)

    sp = sub.add_parser("aportes", help="o que mais move o patrimônio")
    sp.add_argument("--capital", type=float, default=1_000.0)
    sp.add_argument("--aporte", type=float, default=300.0)
    sp.add_argument("--anos", type=int, default=10)
    sp.set_defaults(func=cmd_aportes)

    sp = sub.add_parser("custo", help="custo por frequência de rebalanceamento")
    sp.add_argument("--ticker", default="BOVA11.SA")
    sp.add_argument("--valor", type=float, default=10_000.0)
    sp.set_defaults(func=cmd_custo)

    sp = sub.add_parser("imposto", help="quanto o giro custa em imposto")
    sp.add_argument("--valor", type=float, default=10_000.0)
    sp.add_argument("--ganho", type=float, default=12.0)
    sp.set_defaults(func=cmd_imposto)

    sp = sub.add_parser("estudo", help="compara todas as carteiras")
    sp.add_argument("--anos", type=int, default=5)
    sp.add_argument("--capital", type=float, default=10_000.0)
    sp.add_argument("--aporte", type=float, default=0.0)
    sp.set_defaults(func=cmd_estudo)

    sp = sub.add_parser("validar", help="walk-forward fora da amostra")
    sp.add_argument("--anos", type=int, default=12)
    sp.add_argument("--capital", type=float, default=10_000.0)
    sp.add_argument("--tickers", default="BOVA11.SA,SMAL11.SA,DIVO11.SA,IVVB11.SA")
    sp.add_argument("--treino", type=int, default=504)
    sp.add_argument("--teste", type=int, default=126)
    sp.add_argument("--politica", default="trimestral")
    sp.set_defaults(func=cmd_validar)

    sp = sub.add_parser("hoje", help="situação e sinal do dia")
    sp.add_argument("--tickers", default="")
    sp.set_defaults(func=cmd_hoje)

    sp = sub.add_parser("meta", help="tempo até uma meta de patrimônio")
    sp.add_argument("--capital", type=float, default=1_000.0)
    sp.add_argument("--meta", type=float, default=1_500_000.0)
    sp.set_defaults(func=cmd_meta)

    args = p.parse_args(argv)
    _log(args.verboso)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
