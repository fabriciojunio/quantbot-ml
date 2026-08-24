"""
O robô.

Um comando só: você diz quanto vai aportar e qual o seu orçamento de risco, e
ele decide o resto — lê o mercado, lê o regime, confere o estado da validação,
monta a alocação e diz exatamente quantos reais vão para cada lugar.

    python scripts/robo.py --valor 500 --risco 80

O que ele decide sozinho:
  * quais ativos entram, dentro de cada faixa de risco
  * quanto vai para cada um, respeitando o orçamento de risco declarado
  * se compra tudo de uma vez ou concentra num ativo só (rodízio)
  * se o satélite pode ou não ser usado (depende da validação, não da vontade)
  * se o regime de mercado pede exposição reduzida

O que ele **não** decide, e nunca vai decidir sozinho:
  * o seu orçamento de risco — é escolha sua, e ele mostra o custo dela
  * se você tem reserva de emergência — ele pergunta e obedece
  * usar dinheiro real — a trava é fixa no código

Toda decisão sai com o motivo escrito. Se ele não conseguir explicar, ele não
deveria estar decidindo.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carteira import dados, sinais  # noqa: E402
from carteira.custos_b3 import ModeloCustoB3  # noqa: E402
from carteira.perfil import Perfil  # noqa: E402
from carteira.plano_aporte import Situacao, montar  # noqa: E402
from carteira.regras_pai import guarda_de_regime  # noqa: E402
from carteira.universo import UNIVERSO_B3  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("quantbot.robo")

REGISTRO = Path("data/carteira/decisoes_robo.jsonl")
ESTADO = Path("data/carteira/carteira_robo.json")
UNIVERSO = ["BOVA11.SA", "SMAL11.SA", "IVVB11.SA", "IMAB11.SA", "B5P211.SA", "GOLD11.SA", "XFIX11.SA"]
REFERENCIA = "BOVA11.SA"

# Ordem mínima que vale a pena. Abaixo disso o spread come a compra e é melhor
# acumular no caixa até juntar.
ORDEM_MINIMA = 100.0


def carregar_carteira() -> dict[str, float]:
    if not ESTADO.exists():
        return {}
    try:
        return json.loads(ESTADO.read_text(encoding="utf-8")).get("posicoes", {})
    except Exception as exc:
        logger.warning("Estado ilegível (%s). Começando do zero.", exc)
        return {}


def salvar_carteira(posicoes: dict[str, float]) -> None:
    ESTADO.parent.mkdir(parents=True, exist_ok=True)
    ESTADO.write_text(
        json.dumps(
            {"atualizado_em": datetime.now(timezone.utc).isoformat(), "posicoes": posicoes},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def satelite_esta_liberado() -> tuple[bool, str]:
    """Lê o último veredito da validação. Sem aprovação, o satélite fica fechado."""
    historico = Path("data/carteira/monitor/historico.jsonl")
    if not historico.exists():
        return False, "sem histórico de validação: o satélite começa fechado"

    ultimo_veredito = None
    for linha in historico.read_text(encoding="utf-8").splitlines():
        try:
            reg = json.loads(linha)
        except Exception:
            continue
        if reg.get("validacao"):
            ultimo_veredito = reg["validacao"].get("veredito")

    if ultimo_veredito is None:
        return False, "nenhuma validação registrada ainda: satélite fechado"
    if ultimo_veredito.get("aprovado_para_papel"):
        return True, "a última validação aprovou a estratégia para papel"
    motivo = "; ".join(ultimo_veredito.get("motivos", []))
    return False, f"a validação reprovou: {motivo}"


def decidir(valor: float, risco_pct: float, situacao: Situacao) -> dict:
    """A decisão completa do mês, com o motivo de cada parte."""
    precos = dados.alinhar(dados.precos(UNIVERSO, anos=3))
    dados.valida_precos(precos, "universo do robô")
    cdi = dados.cdi_diario(anos=3)
    cdi_acum = (1 + cdi.reindex(precos.index).ffill().fillna(0.0)).cumprod()

    # ── Regime ───────────────────────────────────────────────────────────
    ret_ref = float(precos[REFERENCIA].iloc[-1] / precos[REFERENCIA].iloc[-252] - 1)
    ret_cdi = float(cdi_acum.iloc[-1] / cdi_acum.iloc[-252] - 1)
    acima_200 = bool(sinais.filtro_de_tendencia(precos).iloc[-1].get(REFERENCIA, False))
    favoravel, regime = guarda_de_regime(ret_ref, ret_cdi, acima_200)

    # ── Reserva primeiro ─────────────────────────────────────────────────
    plano = montar(situacao)
    if not situacao.reserva_completa:
        return {
            "quando": datetime.now(timezone.utc).isoformat(),
            "valor": valor,
            "decisao": "tudo para a reserva de emergência",
            "regime": regime,
            "ordens": [
                {
                    "onde": "Renda fixa de liquidez diária (CDB/RDB com FGC)",
                    "valor": round(valor, 2),
                    "motivo": plano["destinos"][0]["motivo"],
                }
            ],
            "reserva": plano["reserva"],
            "satelite": {"liberado": False, "motivo": "reserva ainda não fechou"},
        }

    # ── Orçamento de risco ───────────────────────────────────────────────
    liberado, motivo_satelite = satelite_esta_liberado()

    risco_efetivo = risco_pct
    ajuste = None
    if not favoravel and risco_pct > 40:
        # Guarda de regime herdada do painel: em regime ruim, reduz exposição
        # em vez de zerar. Zerar no fundo do poço é o erro clássico.
        risco_efetivo = risco_pct * 0.75
        ajuste = (
            f"regime desfavorável: risco reduzido de {risco_pct:.0f}% para "
            f"{risco_efetivo:.0f}% nesta compra (não zerado — sair no fundo é pior)"
        )

    p = Perfil(f"sob_medida_{int(risco_efetivo)}", risco_efetivo, "orçamento do usuário")
    alvo = p.pesos_por_ativo(satelite_liberado=liberado)

    # ── Rodízio: concentra a compra no ativo mais atrasado ───────────────
    carteira = carregar_carteira()
    total_atual = sum(carteira.values())

    ordens = []
    if total_atual <= 0 or valor < ORDEM_MINIMA * len(alvo):
        atuais = {t: (carteira.get(t, 0.0) / total_atual if total_atual else 0.0) for t in alvo}
        faltas = {t: alvo[t] - atuais.get(t, 0.0) for t in alvo}
        escolhido = max(faltas, key=faltas.get)
        ativo = UNIVERSO_B3.get(escolhido)
        custo = ModeloCustoB3().custo_ponta(ativo, valor) if ativo else 0.0
        ordens.append({
            "onde": escolhido,
            "valor": round(valor, 2),
            "custo_estimado": round(custo, 2),
            "motivo": (
                f"rodízio: está {faltas[escolhido] * 100:.1f} pontos abaixo do alvo de "
                f"{alvo[escolhido] * 100:.0f}%. Concentrar em uma ordem em vez de "
                f"{len(alvo)} economiza {len(alvo) - 1} spreads."
            ),
        })
    else:
        for ticker, peso in sorted(alvo.items(), key=lambda x: -x[1]):
            v = valor * peso
            if v < ORDEM_MINIMA:
                continue
            ativo = UNIVERSO_B3.get(ticker)
            ordens.append({
                "onde": ticker,
                "valor": round(v, 2),
                "custo_estimado": round(ModeloCustoB3().custo_ponta(ativo, v), 2) if ativo else 0.0,
                "motivo": f"peso alvo de {peso * 100:.0f}% no orçamento de {risco_efetivo:.0f}% de risco",
            })

    return {
        "quando": datetime.now(timezone.utc).isoformat(),
        "valor": valor,
        "decisao": f"aportar no núcleo com {risco_efetivo:.0f}% em risco",
        "risco_pedido_pct": risco_pct,
        "risco_aplicado_pct": risco_efetivo,
        "ajuste_de_regime": ajuste,
        "regime": regime,
        "alvo_por_ativo_pct": {t: round(p_ * 100, 1) for t, p_ in sorted(alvo.items(), key=lambda x: -x[1])},
        "exposicao_a_dolar_pct": round(alvo.get("IVVB11.SA", 0.0) * 100, 1),
        "ordens": ordens,
        "satelite": {"liberado": liberado, "motivo": motivo_satelite},
        "carteira_atual": carteira,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="O robô decide onde vai o aporte do mês")
    p.add_argument("--valor", type=float, default=500.0, help="quanto vai aportar")
    p.add_argument("--risco", type=float, default=60.0, help="%% em ativos de risco (0 a 100)")
    p.add_argument("--despesa", type=float, default=2_000.0, help="sua despesa mensal")
    p.add_argument("--reserva", type=float, default=0.0, help="quanto já tem de reserva")
    p.add_argument("--vinculo", default="pj", choices=["clt", "pj", "servidor"])
    p.add_argument("--reserva-por-fora", action="store_true",
                   help="a reserva está garantida em outro lugar; pular a fase 1")
    p.add_argument("--json", action="store_true")
    p.add_argument("--aplicar", action="store_true", help="registra a compra na carteira do robô")
    args = p.parse_args()

    situacao = Situacao(
        aporte_mensal=args.valor,
        despesa_mensal=args.despesa,
        reserva_atual=args.reserva if not args.reserva_por_fora else 10**9,
        vinculo=args.vinculo,
    )

    d = decidir(args.valor, args.risco, situacao)

    if args.json:
        print(json.dumps(d, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\n{'=' * 66}")
        print(f"  APORTE DE R$ {args.valor:,.2f}".replace(",", "."))
        print(f"{'=' * 66}\n")
        print(f"Decisão: {d['decisao']}")
        print(f"Regime : {d['regime']['leitura']}")
        if d.get("ajuste_de_regime"):
            print(f"Ajuste : {d['ajuste_de_regime']}")
        print(f"Satélite: {'liberado' if d['satelite']['liberado'] else 'fechado'} — {d['satelite']['motivo']}")

        if d.get("alvo_por_ativo_pct"):
            print("\nAlocação alvo:")
            for t, w in d["alvo_por_ativo_pct"].items():
                print(f"  {t:<12} {w:>5.1f}%")
            print(f"  {'(dólar)':<12} {d['exposicao_a_dolar_pct']:>5.1f}%")

        print("\nOrdens deste mês:")
        for o in d["ordens"]:
            custo = o.get("custo_estimado", 0)
            print(f"  R$ {o['valor']:>8,.2f} -> {o['onde']}".replace(",", "."))
            if custo:
                print(f"                 custo estimado: R$ {custo:.2f}")
            print(f"                 {o['motivo']}")
        print()

    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRO, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")

    if args.aplicar:
        carteira = carregar_carteira()
        for o in d["ordens"]:
            if o["onde"] in UNIVERSO_B3:
                carteira[o["onde"]] = carteira.get(o["onde"], 0.0) + o["valor"]
        salvar_carteira(carteira)
        print("Carteira do robô atualizada.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
