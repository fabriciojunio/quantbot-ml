"""
O robô rodando: um dia de operação, com preço real e dinheiro simulado.

    python scripts/rodar_diario.py                  # opera o pregão mais recente
    python scripts/rodar_diario.py --desde 2021-01-01  # reconstrói do zero
    python scripts/rodar_diario.py --auditar        # só confere a integridade

Roda todo dia útil pelo GitHub Actions. Cada dia entra no livro de auditoria
com o motivo da decisão, e o livro é encadeado por hash: um dia que der
resultado ruim não pode ser removido depois sem que a cadeia acuse.

O `--desde` existe para dar ao sistema um histórico simulado desde uma data
passada, usando só o preço que existia em cada dia. Não é trapaça: a decisão de
cada dia usa exclusivamente dados até aquele dia, e a mesma classe `Conta` que
a operação ao vivo usa. É a diferença entre "teria dado certo" e "está dando
certo" — o backfill dá o primeiro, e a partir de hoje o robô acumula o segundo.
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

from auditoria import abrir, verificar  # noqa: E402
from auditoria.verificacao import ancora  # noqa: E402
from carteira import aporte_historico as ah  # noqa: E402
from carteira import dados, multimoeda, operacao  # noqa: E402
from carteira.conta import Conta  # noqa: E402
from carteira.custos_b3 import ModeloCustoB3  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("quantbot.rodar_diario")

LIVRO = Path("data/auditoria/livro.sqlite3")
PAINEL = Path("relatorios/painel.json")
CARIMBO = Path("data/auditoria/carimbo.json")

# O universo que o robô opera de verdade. ETFs reais da B3 mais uma fatia de
# bitcoin, todos com cotação diária pública.
UNIVERSO = ["BOVA11.SA", "SMAL11.SA", "IVVB11.SA", "IMAB11.SA",
            "B5P211.SA", "GOLD11.SA", "XFIX11.SA"]


def _brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")


def carregar_mercado(anos: int = 6, com_cripto: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Preços reais de todo o universo, alinhados, mais o CDI."""
    precos = dados.alinhar(dados.precos(UNIVERSO, anos=anos))
    dados.valida_precos(precos, "universo da operação")

    if com_cripto:
        try:
            btc = multimoeda.serie_cripto_em_reais("BTC-BRL", anos=anos)
            precos = precos.join(btc.to_frame(), how="left")
            precos["BTC-BRL"] = precos["BTC-BRL"].ffill()
            precos = precos.dropna()
        except Exception as exc:
            logger.warning("Sem série de bitcoin (%s). Seguindo sem a fatia de cripto.", exc)

    dados.valida_continuidade(precos, "universo da operação")
    cdi = dados.cdi_diario(anos=anos + 1)
    return precos, cdi


def satelite_liberado(livro) -> tuple[bool, str]:
    """O satélite só abre com validação aprovada. Hoje não está."""
    historico = Path("data/carteira/monitor/historico.jsonl")
    if not historico.exists():
        return False, "sem histórico de validação: o satélite começa fechado"
    ultimo = None
    for linha in historico.read_text(encoding="utf-8").splitlines():
        try:
            reg = json.loads(linha)
        except Exception:
            continue
        if reg.get("validacao"):
            ultimo = reg["validacao"].get("veredito")
    if ultimo is None:
        return False, "nenhuma validação registrada ainda"
    if ultimo.get("aprovado_para_papel"):
        return True, "a última validação aprovou a estratégia"
    return False, "a validação reprovou: " + "; ".join(ultimo.get("motivos", []))


def operar(
    livro,
    config: operacao.Configuracao,
    precos: pd.DataFrame,
    cdi: pd.Series,
    dias: list[pd.Timestamp],
    liberado: bool,
) -> tuple[Conta, list[operacao.Decisao]]:
    """Roda uma sequência de dias, retomando o estado do livro."""
    estado = livro.ultimo_estado()
    conta = (
        Conta.de_dict(estado, ModeloCustoB3(), operacao.universo_da_operacao())
        if estado else
        Conta(modelo=ModeloCustoB3(), ativos=operacao.universo_da_operacao())
    )

    cdi_alinhado = cdi.reindex(precos.index).ffill().fillna(0.0)
    cdi_acumulado = (1 + cdi_alinhado).cumprod()

    decisoes = []
    cota, pat_ontem = 1.0, 0.0

    for dia in dias:
        precos_ate = precos.loc[:dia]
        hoje_precos = {t: float(v) for t, v in precos.loc[dia].items() if pd.notna(v)}

        pat_pre = conta.patrimonio(hoje_precos)
        if pat_ontem > 0:
            cota *= pat_pre / pat_ontem

        conta, decisao = operacao.rodar_dia(
            livro, config, precos_ate, cdi, hoje=dia,
            satelite_liberado=liberado, conta=conta,
        )
        operacao.registrar_dia(
            livro, conta, decisao, hoje_precos,
            float(cdi_acumulado.loc[dia]), cota,
        )
        decisoes.append(decisao)
        pat_ontem = conta.patrimonio(hoje_precos)

    return conta, decisoes


def montar_painel(livro, conta: Conta, precos: pd.DataFrame, cdi: pd.Series) -> dict:
    """O JSON que a tela lê. Só dado que já está no livro."""
    historico = livro.patrimonio()
    ultimo_precos = {t: float(v) for t, v in precos.iloc[-1].items() if pd.notna(v)}
    integridade = verificar(livro)

    # A referência de CDI precisa acumular aporte por aporte, cada um com o seu
    # prazo e a sua alíquota. Aplicar o crescimento do período inteiro sobre o
    # total aportado trata todo o dinheiro como se tivesse entrado no primeiro
    # dia — num backfill de 5 anos isso inflava a referência em mais de 20% e
    # fazia o robô parecer estar perdendo feio do CDI quando não estava.
    from auditoria import TipoEvento

    aportes = [
        (pd.Timestamp(e.dados["data"]), float(e.dados["valor"]))
        for e in livro.eventos(TipoEvento.APORTE)
        if e.dados.get("data") and e.dados.get("valor")
    ]
    fim = precos.index[-1]
    liquido_cdi, tir_cdi = ah.referencia_cdi(aportes, cdi, fim) if aportes else (None, None)

    patrimonio = conta.patrimonio(ultimo_precos)
    return {
        "gerado_em": pd.Timestamp.utcnow().isoformat(),
        "dinheiro_real": operacao.DINHEIRO_REAL,
        "aviso": "Preços reais, dinheiro simulado. Nenhuma ordem foi enviada a corretora.",
        "patrimonio": round(patrimonio, 2),
        "aportado": round(conta.total_aportado, 2),
        "resultado": round(patrimonio - conta.total_aportado, 2),
        "resultado_pct": round(
            (patrimonio / conta.total_aportado - 1) * 100, 2
        ) if conta.total_aportado > 0 else 0.0,
        "caixa": round(conta.caixa, 2),
        "custo_pago": round(conta.custo_pago, 2),
        "imposto_pago": round(conta.imposto_pago, 2),
        "referencia_cdi": round(liquido_cdi, 2) if liquido_cdi else None,
        "tir_cdi_pct": round(tir_cdi * 100, 2) if tir_cdi else None,
        "tir_pct": round(
            ah.tir_anual(
                [((d - aportes[0][0]).days / 365.25, -v) for d, v in aportes]
                + [((fim - aportes[0][0]).days / 365.25, patrimonio)]
            ) * 100, 2
        ) if aportes else None,
        "posicoes": [
            {
                "ticker": t,
                "cotas": round(p.cotas, 6),
                "preco": round(ultimo_precos.get(t, 0.0), 2),
                "valor": round(p.cotas * ultimo_precos.get(t, 0.0), 2),
                "custo_medio": round(p.custo_medio, 2),
                "resultado_pct": round(
                    (ultimo_precos.get(t, 0.0) / p.custo_medio - 1) * 100, 2
                ) if p.custo_medio > 0 else 0.0,
                "peso_pct": round(p.cotas * ultimo_precos.get(t, 0.0) / patrimonio * 100, 1)
                if patrimonio > 0 else 0.0,
            }
            for t, p in sorted(conta.posicoes.items())
        ],
        "exposicao_cambial": multimoeda.exposicao_cambial(conta.pesos(ultimo_precos)).resumo(),
        "serie": [
            {"data": h["data"], "total": round(h["total"], 2),
             "aportado": round(h["aportado"], 2), "cota": round(h["cota"], 6)}
            for h in historico
        ],
        "ultimas_ordens": [
            {"quando": e.quando, "resumo": e.resumo, **e.dados}
            for e in livro.eventos(limite=None)[-400:]
            if e.tipo.value == "ordem"
        ][-25:],
        "auditoria": {
            "eventos": integridade.eventos_conferidos,
            "cadeia_intacta": integridade.intacta,
            "quebras": len(integridade.quebras),
            "hash_final": integridade.hash_final,
            "primeiro_evento": integridade.primeiro_evento,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Roda um dia (ou vários) de operação simulada")
    p.add_argument("--desde", help="reconstrói a partir desta data (AAAA-MM-DD)")
    p.add_argument("--aporte", type=float, default=500.0)
    p.add_argument("--risco", type=float, default=80.0)
    p.add_argument("--cripto", type=float, default=1.0, help="%% em bitcoin (0 a 5)")
    p.add_argument("--anos", type=int, default=6, help="quanto histórico baixar")
    p.add_argument("--livro", default=str(LIVRO))
    p.add_argument("--auditar", action="store_true", help="só verifica a integridade")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    livro = abrir(Path(args.livro), origem="rodar_diario")

    if args.auditar:
        r = verificar(livro)
        print(json.dumps(r.resumo(), indent=2, ensure_ascii=False))
        livro.fechar()
        return 0 if r.intacta else 1

    config = operacao.Configuracao(
        aporte_mensal=args.aporte,
        risco_pct=args.risco,
        fatia_cripto=args.cripto / 100,
    )

    precos, cdi = carregar_mercado(anos=args.anos)
    liberado, motivo_satelite = satelite_liberado(livro)

    if args.desde:
        inicio = pd.Timestamp(args.desde)
        dias = [d for d in precos.index if d >= inicio]
    else:
        dias = [precos.index[-1]]

    if not dias:
        print("Nenhum pregão a processar.")
        livro.fechar()
        return 0

    conta, decisoes = operar(livro, config, precos, cdi, dias, liberado)
    painel = montar_painel(livro, conta, precos, cdi)
    painel["satelite"] = {"liberado": liberado, "motivo": motivo_satelite}
    painel["configuracao"] = config.para_dict()

    PAINEL.parent.mkdir(parents=True, exist_ok=True)
    PAINEL.write_text(json.dumps(painel, indent=2, ensure_ascii=False, default=str),
                      encoding="utf-8")

    carimbo = ancora(livro)
    CARIMBO.parent.mkdir(parents=True, exist_ok=True)
    CARIMBO.write_text(json.dumps(carimbo, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps(painel, indent=2, ensure_ascii=False, default=str))
    else:
        ultima = decisoes[-1]
        print(f"\n{'=' * 68}")
        print(f"  ROBÔ — {ultima.data}   (preço real, dinheiro simulado)")
        print(f"{'=' * 68}\n")
        print(f"  Patrimônio  {_brl(painel['patrimonio']):>16}")
        print(f"  Aportado    {_brl(painel['aportado']):>16}")
        print(f"  Resultado   {_brl(painel['resultado']):>16}   ({painel['resultado_pct']:+.2f}%)")
        if painel["referencia_cdi"]:
            print(f"  Mesmo em CDI{_brl(painel['referencia_cdi']):>16}   "
                  f"(líquido de IR, aporte a aporte)")
            print(f"  TIR         {painel['tir_pct']:>13.2f}% ao ano   "
                  f"(CDI: {painel['tir_cdi_pct']:.2f}%)")
        print(f"\n  Regime : {ultima.regime.get('leitura', '—')}")
        print(f"  Risco  : pedido {ultima.risco_pedido_pct:.0f}%, "
              f"aplicado {ultima.risco_aplicado_pct:.0f}%")
        print(f"  Satélite: {'liberado' if liberado else 'fechado'} — {motivo_satelite}")

        if painel["posicoes"]:
            print("\n  Posições:")
            for pos in painel["posicoes"]:
                print(f"    {pos['ticker']:<12} {_brl(pos['valor']):>13}  "
                      f"{pos['peso_pct']:>5.1f}%  {pos['resultado_pct']:+7.2f}%")

        movimentos = sum(len(d.movimentos) for d in decisoes)
        print(f"\n  {len(decisoes)} pregão(ões) processado(s), {movimentos} ordem(ns).")
        aud = painel["auditoria"]
        print(f"  Livro: {aud['eventos']} eventos, cadeia "
              f"{'íntegra' if aud['cadeia_intacta'] else 'QUEBRADA'}.")
        print(f"  Painel em {PAINEL}\n")

    livro.fechar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
