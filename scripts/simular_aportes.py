"""
Quanto R$ 500 por mês viram, medido em todo pedaço do histórico.

Roda a estratégia do robô sobre 22 anos de dado real reconstruído, começando
em cada trimestre possível, e mostra a **distribuição** do resultado em vez de
um número só. A diferença importa: "rende 13% ao ano" é o que aconteceu numa
janela específica; "a pior janela de 5 anos rendeu 4% e a melhor 22%" é o que
descreve o que pode acontecer com você.

    python scripts/simular_aportes.py
    python scripts/simular_aportes.py --risco 60 --anos 10
    python scripts/simular_aportes.py --json

Dois calendários de aporte são comparados:

    constante   R$ 500 todo mês
    degrau      R$ 500, subindo R$ 200 a cada 2 anos

Tudo com custo de spread e emolumento em cada ordem, imposto pago no
rebalanceamento e liquidação total no fim com o imposto devido. O número que
sai é o que sobraria na conta, não o de tela.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carteira import aporte_historico as ah  # noqa: E402
from carteira import dados, serie_longa  # noqa: E402
from carteira.perfil import Perfil  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

SAIDA = Path("relatorios/carteira/aportes.json")

CALENDARIOS = {
    "constante_500": ("R$ 500 todo mês", ah.constante(500.0)),
    "degrau_500_mais_200": ("R$ 500 subindo R$ 200 a cada 2 anos", ah.degrau(500.0, 200.0, 24)),
}


def _brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")


def carregar() -> tuple[pd.DataFrame, pd.Series]:
    precos = serie_longa.montar(anos=23)
    cdi = dados.cdi_diario(anos=23)
    dados.valida_precos(precos, "série longa reconstruída")
    return precos, cdi


def rodar(
    precos: pd.DataFrame,
    cdi: pd.Series,
    risco_pct: float,
    anos: int,
    calendario: str,
    passo_meses: int = 3,
) -> dict:
    perfil = Perfil(f"risco_{int(risco_pct)}", risco_pct, f"{risco_pct:.0f}% em risco")
    pesos = perfil.pesos_por_ativo(composicao=serie_longa.COMPOSICAO_LONGA)
    rotulo, fn = CALENDARIOS[calendario]

    # Os alvos reconstruídos usam o ticker do ETF real, então o modelo de custo
    # encontra o spread certo em UNIVERSO_B3 sem precisar de mapa extra.
    janelas = ah.janelas_moveis(
        precos, cdi, pesos, fn, anos=anos, passo_meses=passo_meses,
    )
    return {
        "risco_pct": risco_pct,
        "anos": anos,
        "calendario": calendario,
        "rotulo": rotulo,
        "pesos": {t: round(p * 100, 1) for t, p in sorted(pesos.items(), key=lambda x: -x[1])},
        "aportado_tipico": round(float(janelas["total_aportado"].median()), 2),
        "final": ah.distribuicao(janelas, "valor_liquido_final"),
        "tir": ah.distribuicao(janelas, "tir_anual_pct"),
        "cdi": ah.comparar_com_cdi(janelas),
        "pior_queda_pct": round(float(janelas["max_drawdown_pct"].min()), 2),
        "meses_no_prejuizo_pior": int(janelas["meses_no_prejuizo"].max()),
        "janelas": janelas.to_dict("records"),
    }


def imprimir(bloco: dict) -> None:
    f, t, c = bloco["final"], bloco["tir"], bloco["cdi"]
    print(f"\n  {bloco['anos']} anos · risco {bloco['risco_pct']:.0f}% · {bloco['rotulo']}")
    print(f"  {f['janelas']} janelas testadas · aportado {_brl(bloco['aportado_tipico'])}")
    print(f"    pior     {_brl(f['pior']):>16}   (TIR {t['pior']:>6.2f}% ao ano)")
    print(f"    mediana  {_brl(f['mediana']):>16}   (TIR {t['mediana']:>6.2f}% ao ano)")
    print(f"    melhor   {_brl(f['melhor']):>16}   (TIR {t['melhor']:>6.2f}% ao ano)")
    print(f"    bateu o CDI em {c['venceu_o_cdi_em']}/{c['janelas']} janelas ({c['taxa_de_vitoria_pct']:.0f}%)")
    print(f"    pior momento: patrimônio {bloco['pior_queda_pct']:.1f}% abaixo do aportado")


def main() -> int:
    p = argparse.ArgumentParser(description="Simula aporte mensal sobre 22 anos de dado real")
    p.add_argument("--risco", type=float, nargs="*", default=[20, 40, 60, 80, 100])
    p.add_argument("--anos", type=int, nargs="*", default=[5, 10])
    p.add_argument("--passo", type=int, default=3, help="passo entre janelas, em meses")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    precos, cdi = carregar()
    cob = serie_longa.cobertura(precos, serie_longa.AVISOS_DA_COMPOSICAO_LONGA)

    if not args.json:
        print(f"\n{'=' * 72}")
        print("  APORTE MENSAL SOBRE HISTÓRICO RECONSTRUÍDO")
        print(f"{'=' * 72}")
        print(f"\n  Período: {cob.inicio.date()} a {cob.fim.date()}  ({cob.anos:.1f} anos, {cob.pregoes} pregões)")
        print(f"  Ativos : {', '.join(cob.ativos)} + CDI")
        print("\n  Como cada série foi reconstruída:")
        for linha in serie_longa.procedencia():
            if linha["reconstruido_de"] != "—":
                print(f"    {linha['ativo']:<12} <- {linha['reconstruido_de']:<24} "
                      f"menos {linha['taxa_adm_descontada_pct']:.2f}% a.a.")
        print("\n  Limites conhecidos desta reconstrução:")
        for a in cob.avisos:
            print(f"    - {a}")

    resultados = []
    for calendario in CALENDARIOS:
        if not args.json:
            print(f"\n{'-' * 72}\n  {CALENDARIOS[calendario][0].upper()}\n{'-' * 72}")
        for anos in args.anos:
            for risco in args.risco:
                bloco = rodar(precos, cdi, risco, anos, calendario, args.passo)
                resultados.append(bloco)
                if not args.json:
                    imprimir(bloco)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(
        json.dumps(
            {
                "gerado_em": pd.Timestamp.utcnow().isoformat(),
                "cobertura": {
                    "inicio": str(cob.inicio.date()), "fim": str(cob.fim.date()),
                    "anos": round(cob.anos, 2), "pregoes": cob.pregoes,
                    "ativos": list(cob.ativos), "avisos": list(cob.avisos),
                },
                "procedencia": serie_longa.procedencia(),
                "resultados": resultados,
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(resultados, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\n  Detalhe de cada janela salvo em {SAIDA}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
