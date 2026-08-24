"""
Auditoria do book de paper trading de FIIs.

Recalcula o track record do zero, com provento acumulado do jeito certo, e
compara com o que está registrado. Escrito depois de encontrar, em 23/08/2026,
que `PaperBook.marcar_mercado` acumulava provento de forma quadrática: a função
lia `ultima_avaliacao` para saber quantos dias tinham passado, mas nunca
atualizava esse campo. Resultado: no dia 1 somava 1 dia de provento, no dia 2
somava 2 dias, no dia 3 somava 3 — em n dias somava n(n+1)/2 dias em vez de n.

Uso:
    python scripts/auditar_paper_book.py
    python scripts/auditar_paper_book.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARQUIVO = Path("data/paper_book.json")

# Custo de operar FII na B3 para pessoa física: a maioria das corretoras não
# cobra corretagem, mas emolumentos e liquidação da B3 existem, e o spread do
# book é o que realmente pesa em fundo pouco líquido.
EMOLUMENTOS_PCT = 0.0325 / 100     # B3, ida e volta somadas
SPREAD_PCT = 0.15 / 100            # metade do spread típico, por ponta


def _dias(d1: str, d2: str) -> int:
    try:
        return (datetime.fromisoformat(d2).date() - datetime.fromisoformat(d1).date()).days
    except ValueError:
        return 0


def provento_correto(valor_investido: float, dy_anual_pct: float, dias: int) -> float:
    """Provento linear: valor × DY × (dias/365). É o accrual que a função quebrada
    deveria ter feito uma vez por período, e acabou fazendo em cascata."""
    if dias <= 0 or dy_anual_pct <= 0:
        return 0.0
    return valor_investido * (dy_anual_pct / 100) * (dias / 365)


def auditar(caminho: Path = ARQUIVO, cdi_anual: float = 13.9) -> dict:
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    capital = dados["capital_inicial"]

    fechadas = dados["fechadas"]
    posicoes = dados["posicoes"]

    registrado_soma_div = 0.0
    correto_soma_div = 0.0
    linhas = []

    for f in fechadas:
        # O book não guarda o número de cotas da operação fechada; ele é
        # recuperável do retorno de preço e do valor investido implícito.
        # O que precisamos é da razão provento/investido, e essa está no
        # próprio retorno total registrado.
        ret_preco = f["retorno_preco_pct"]
        ret_div_registrado = f["retorno_total_pct"] - ret_preco
        ret_div_correto = f["dy_entrada"] * f["dias"] / 365

        registrado_soma_div += ret_div_registrado
        correto_soma_div += ret_div_correto

        custo_pct = (EMOLUMENTOS_PCT + 2 * SPREAD_PCT) * 100

        linhas.append(
            {
                "ticker": f["ticker"],
                "dias": f["dias"],
                "retorno_preco_pct": round(ret_preco, 2),
                "provento_registrado_pct": round(ret_div_registrado, 2),
                "provento_correto_pct": round(ret_div_correto, 2),
                "exagero_vezes": round(ret_div_registrado / ret_div_correto, 1)
                if ret_div_correto > 0
                else None,
                "retorno_registrado_pct": round(f["retorno_total_pct"], 2),
                "retorno_corrigido_pct": round(ret_preco + ret_div_correto - custo_pct, 2),
            }
        )

    # Posições ainda abertas: mesmo tratamento.
    hoje = date.today().isoformat()
    abertas = []
    for tk, p in posicoes.items():
        investido = p["cotas"] * p["preco_entrada"]
        dias_abertos = _dias(p["data_entrada"], hoje)
        div_correto = provento_correto(investido, p["dy_entrada"], dias_abertos)
        abertas.append(
            {
                "ticker": tk,
                "investido": round(investido, 2),
                "dias": dias_abertos,
                "provento_registrado": round(p["dividendos"], 2),
                "provento_correto": round(div_correto, 2),
                "exagero_vezes": round(p["dividendos"] / div_correto, 1)
                if div_correto > 0
                else None,
            }
        )

    div_reg_abertas = sum(a["provento_registrado"] for a in abertas)
    div_cor_abertas = sum(a["provento_correto"] for a in abertas)

    serie = dados.get("serie_valor", [])
    valor_registrado = serie[-1]["valor_total"] if serie else capital
    dias_corridos = _dias(serie[0]["data"], serie[-1]["data"]) if len(serie) > 1 else 0

    # Retorno corrigido: retira o provento inflado das fechadas e das abertas.
    # As fechadas já viraram caixa, então o excesso entrou direto no valor.
    excesso_fechadas = 0.0
    for f, linha in zip(fechadas, linhas):
        # provento em reais: reconstruído da fração sobre o investido
        # (o registro guarda o valor absoluto, que é o que entrou no caixa)
        excesso_pct = linha["provento_registrado_pct"] - linha["provento_correto_pct"]
        if f["retorno_preco_pct"] != 0 or excesso_pct:
            investido_implicito = (
                f["dividendos"] / (linha["provento_registrado_pct"] / 100)
                if linha["provento_registrado_pct"] > 0
                else 0.0
            )
            excesso_fechadas += investido_implicito * (excesso_pct / 100)

    excesso_abertas = div_reg_abertas - div_cor_abertas
    valor_corrigido = valor_registrado - excesso_fechadas - excesso_abertas

    cdi_periodo = ((1 + cdi_anual / 100) ** (dias_corridos / 365) - 1) * 100 if dias_corridos else 0.0

    return {
        "periodo": [serie[0]["data"], serie[-1]["data"]] if len(serie) > 1 else None,
        "dias_corridos": dias_corridos,
        "capital_inicial": capital,
        "registrado": {
            "valor_final": round(valor_registrado, 2),
            "retorno_pct": round((valor_registrado / capital - 1) * 100, 2),
        },
        "corrigido": {
            "valor_final": round(valor_corrigido, 2),
            "retorno_pct": round((valor_corrigido / capital - 1) * 100, 2),
        },
        "provento_inflado_em_reais": round(excesso_fechadas + excesso_abertas, 2),
        "cdi_no_periodo_pct": round(cdi_periodo, 2),
        "bateu_o_cdi_de_verdade": (valor_corrigido / capital - 1) * 100 > cdi_periodo,
        "operacoes_fechadas": len(fechadas),
        "posicoes_abertas": len(posicoes),
        "detalhe_fechadas": linhas,
        "detalhe_abertas": abertas,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Audita o track record do paper book de FIIs")
    p.add_argument("--arquivo", default=str(ARQUIVO))
    p.add_argument("--cdi", type=float, default=13.9)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    r = auditar(Path(args.arquivo), args.cdi)

    if args.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0

    print("=" * 68)
    print("AUDITORIA DO PAPER BOOK DE FIIs")
    print("=" * 68)
    print(f"Período: {r['periodo'][0]} a {r['periodo'][1]} ({r['dias_corridos']} dias)")
    print(f"Capital inicial: R$ {r['capital_inicial']:,.2f}")
    print()
    print(f"  Registrado no book:  R$ {r['registrado']['valor_final']:>10,.2f}  "
          f"({r['registrado']['retorno_pct']:+.2f}%)")
    print(f"  Corrigido:           R$ {r['corrigido']['valor_final']:>10,.2f}  "
          f"({r['corrigido']['retorno_pct']:+.2f}%)")
    print(f"  CDI no período:                          {r['cdi_no_periodo_pct']:+.2f}%")
    print()
    print(f"  Provento inflado pelo bug: R$ {r['provento_inflado_em_reais']:,.2f}")
    print(f"  Bateu o CDI de verdade? {'SIM' if r['bateu_o_cdi_de_verdade'] else 'NÃO'}")
    print()
    print("Maiores exageros nas operações fechadas:")
    piores = sorted(
        [d for d in r["detalhe_fechadas"] if d["exagero_vezes"]],
        key=lambda d: -d["exagero_vezes"],
    )[:8]
    print(f"  {'ticker':10} {'dias':>5} {'registrado':>11} {'correto':>9} {'exagero':>9}")
    for d in piores:
        print(f"  {d['ticker']:10} {d['dias']:5d} {d['provento_registrado_pct']:10.2f}% "
              f"{d['provento_correto_pct']:8.2f}% {d['exagero_vezes']:8.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
