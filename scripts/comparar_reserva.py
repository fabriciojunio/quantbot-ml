"""
Compara onde deixar a reserva de emergência.

Aporte mensal fixo, com o imposto de cada aporte calculado pelo prazo que ELE
ficou aplicado — e não pelo prazo médio da carteira. A diferença importa: numa
reserva construída em 24 meses, o primeiro aporte cai na faixa de 17,5% e o
último ainda está em 22,5%. Quem usa uma alíquota só superestima o resultado.

    python scripts/comparar_reserva.py --aporte 500 --meses 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carteira.tributos import aliquota_regressiva  # noqa: E402

# Cenário de agosto de 2026.
CDI_ANUAL = 0.139
POUPANCA_ANUAL = 0.0617   # 0,5% a.m. + TR, regra vigente com Selic acima de 8,5%
CUSTODIA_TESOURO = 0.0020  # 0,20% a.a., isenta até R$ 10 mil em Tesouro Selic
ISENCAO_CUSTODIA = 10_000.0


def _mensal(anual: float) -> float:
    return (1 + anual) ** (1 / 12) - 1


def simular(
    aporte: float,
    meses: int,
    pct_do_cdi: float = 1.0,
    isento_de_ir: bool = False,
    taxa_anual_fixa: float | None = None,
    custodia: bool = False,
) -> dict:
    """Simula aportes mensais e devolve o valor líquido no fim.

    Cada aporte é acompanhado separadamente porque o IR depende do prazo de
    cada um. `taxa_anual_fixa` substitui o CDI (usado pela poupança).
    """
    taxa_anual = taxa_anual_fixa if taxa_anual_fixa is not None else CDI_ANUAL * pct_do_cdi
    taxa_mensal = _mensal(taxa_anual)

    bruto_total = 0.0
    aportado_total = 0.0
    imposto_total = 0.0
    custodia_total = 0.0
    saldo_corrente = 0.0

    for m in range(meses):
        prazo = meses - m                      # meses que este aporte fica aplicado
        dias = int(prazo * 30.4)
        bruto = aporte * (1 + taxa_mensal) ** prazo
        rendimento = bruto - aporte

        if isento_de_ir:
            imposto = 0.0
        else:
            imposto = rendimento * aliquota_regressiva(dias)

        bruto_total += bruto
        aportado_total += aporte
        imposto_total += imposto

        # Custódia do Tesouro incide só sobre o estoque acima de R$ 10 mil.
        saldo_corrente += aporte
        if custodia and saldo_corrente > ISENCAO_CUSTODIA:
            excedente = saldo_corrente - ISENCAO_CUSTODIA
            custodia_total += excedente * CUSTODIA_TESOURO / 12 * prazo

    liquido = bruto_total - imposto_total - custodia_total
    return {
        "aportado": round(aportado_total, 2),
        "bruto": round(bruto_total, 2),
        "imposto": round(imposto_total, 2),
        "custodia": round(custodia_total, 2),
        "liquido": round(liquido, 2),
        "ganho_liquido": round(liquido - aportado_total, 2),
        "rentabilidade_liquida_pct": round((liquido / aportado_total - 1) * 100, 2),
        "taxa_anual_bruta_pct": round(taxa_anual * 100, 2),
    }


OPCOES = [
    ("Poupança", dict(taxa_anual_fixa=POUPANCA_ANUAL, isento_de_ir=True)),
    ("Tesouro Selic (100% CDI)", dict(pct_do_cdi=1.00, custodia=True)),
    ("CDB liquidez diária 100% CDI", dict(pct_do_cdi=1.00)),
    ("CDB liquidez diária 110% CDI", dict(pct_do_cdi=1.10)),
    ("LCI/LCA 90% CDI (isenta, com carência)", dict(pct_do_cdi=0.90, isento_de_ir=True)),
    ("LCI/LCA 85% CDI (isenta, com carência)", dict(pct_do_cdi=0.85, isento_de_ir=True)),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Onde deixar a reserva de emergência")
    p.add_argument("--aporte", type=float, default=500.0)
    p.add_argument("--meses", type=int, default=24)
    args = p.parse_args()

    print(f"Aporte de R$ {args.aporte:,.2f} por mês, durante {args.meses} meses")
    print(f"Total aportado: R$ {args.aporte * args.meses:,.2f}")
    print(f"CDI de referência: {CDI_ANUAL * 100:.2f}% ao ano\n")

    print(f"{'Onde':40} {'Líquido':>12} {'Ganho':>10} {'Rend. líq.':>11}")
    print("-" * 76)

    resultados = []
    for nome, kw in OPCOES:
        r = simular(args.aporte, args.meses, **kw)
        resultados.append((nome, r))
        print(
            f"{nome:40} R$ {r['liquido']:>9,.2f} "
            f"R$ {r['ganho_liquido']:>7,.2f} {r['rentabilidade_liquida_pct']:>10.2f}%"
        )

    melhor = max(resultados, key=lambda x: x[1]["liquido"])
    pior = min(resultados, key=lambda x: x[1]["liquido"])
    print("-" * 76)
    print(
        f"\nDiferença entre a melhor e a pior opção: "
        f"R$ {melhor[1]['liquido'] - pior[1]['liquido']:,.2f} "
        f"({melhor[0]} contra {pior[0]})"
    )
    print(
        "\nNenhuma destas oscila. Reserva de emergência não é lugar de buscar "
        "retorno:\né lugar de o dinheiro estar lá no dia em que precisar."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
