"""
Monitor diário de cripto.

Roda na nuvem (GitHub Actions) e acumula evidência sem arriscar um centavo.
Todo dia ele registra:

1. Estado do mercado — preço, spread real e desequilíbrio de book.
2. Carrego de funding líquido de custo, por par.
3. O score de confluência do dia, com a contribuição de cada indicador.
4. Uma vez por semana, a validação fora da amostra completa.

Por que só medir e não operar em papel: paper trading de uma estratégia que já
foi reprovada na validação não produz informação nova, só consome tempo. O que
produz informação é a série histórica de custo, carrego e veredito — se o
regime de mercado mudar e a estratégia passar a ter vantagem, o veredito muda e
isso fica registrado no dia em que aconteceu.

    python scripts/monitor_cripto.py
    python scripts/monitor_cripto.py --validar   # força a validação completa
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cripto import carrego, mercado, sinais, walkforward  # noqa: E402
from cripto.backtest import ConfigBacktest  # noqa: E402
from cripto.custos import ModeloCusto, resumo_viabilidade  # noqa: E402
from cripto.exchanges import cliente_padrao  # noqa: E402
from cripto.sinais import ConfigSinal  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("quantbot.monitor_cripto")

PARES = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
DIRETORIO = Path("data/cripto/monitor")
RELATORIOS = Path("relatorios/cripto")


def coletar_mercado(cliente) -> list[dict]:
    itens = []
    for par in PARES:
        try:
            book = cliente.book(par, profundidade=20)
            itens.append(
                {
                    "simbolo": par,
                    "melhor_bid": book.melhor_bid,
                    "melhor_ask": book.melhor_ask,
                    "meio": round((book.melhor_bid + book.melhor_ask) / 2, 6),
                    "spread_pct": round(book.spread_pct, 5),
                    "desequilibrio": round(book.desequilibrio(), 4),
                    "fonte": cliente.ultima_fonte,
                }
            )
        except Exception as exc:
            logger.warning("Sem book de %s: %s", par, exc)
            itens.append({"simbolo": par, "erro": str(exc)})
    return itens


def coletar_carrego() -> list[dict]:
    itens = []
    for par in PARES:
        try:
            r = carrego.analisar(par, capital_usd=1_000.0)
            itens.append(
                {
                    "simbolo": par,
                    "taxa_media_pct": r["taxa_media_por_periodo_pct"],
                    "negativas_pct": r["fatia_negativa_pct"],
                    "liquido_ano_pct": r["carrego_liquido_anualizado_pct"],
                    "dias_para_pagar_montagem": r["dias_para_pagar_a_montagem"],
                }
            )
        except Exception as exc:
            logger.warning("Sem funding de %s: %s", par, exc)
    return itens


def coletar_sinais() -> list[dict]:
    itens = []
    for par in PARES:
        try:
            df = mercado.historico(par, "4h", dias=120)
            dados = sinais.gerar_sinais(df, ConfigSinal())
            ultima = dados.index[-1]
            detalhe = sinais.explica_barra(dados, ultima)
            itens.append(
                {
                    "simbolo": par,
                    "score": detalhe["score"],
                    "posicao_alvo": int(dados["posicao_alvo"].iloc[-1]),
                    "fechamento": float(dados["fechamento"].iloc[-1]),
                    "rsi_14": round(float(dados["rsi_14"].iloc[-1]), 1),
                    "adx": round(float(dados["adx"].iloc[-1]), 1),
                    "atr_pct": round(float(dados["atr_pct"].iloc[-1]) * 100, 3),
                    "contribuicoes": detalhe["indicadores"],
                }
            )
        except Exception as exc:
            logger.warning("Sem sinal de %s: %s", par, exc)
    return itens


def rodar_validacao() -> list[dict]:
    """Validação fora da amostra em 4h — o tempo gráfico mais barato de operar."""
    itens = []
    for par in PARES:
        try:
            df = mercado.historico(par, "4h", dias=400)
            mercado.valida_serie(df, f"{par} 4h")
            _, _, resumo = walkforward.walk_forward(
                df, par, base=ConfigBacktest(intervalo="4h"),
                barras_treino=900, barras_teste=220,
            )
            veredito = walkforward.veredito(resumo)
            itens.append(
                {
                    "simbolo": par,
                    "janelas": resumo["janelas"],
                    "retorno_oos_pct": resumo["retorno_oos_total_pct"],
                    "sharpe_oos": resumo["sharpe_oos"],
                    "janelas_positivas": resumo["janelas_positivas"],
                    "aprovado": veredito["aprovado_para_papel"],
                    "motivos": veredito["motivos"],
                }
            )
        except Exception as exc:
            logger.warning("Validação de %s falhou: %s", par, exc)
            itens.append({"simbolo": par, "erro": str(exc)})
    return itens


def escrever_markdown(registro: dict) -> Path:
    RELATORIOS.mkdir(parents=True, exist_ok=True)
    hoje = registro["data"]
    linhas = [
        f"# Monitor de cripto — {hoje}",
        "",
        f"_Gerado em {registro['quando'][:19]} UTC._",
        "",
        "## Mercado agora",
        "",
        "| Par | Preço | Spread | Desequilíbrio do book | Fonte |",
        "|---|---|---|---|---|",
    ]
    for m in registro["mercado"]:
        if "erro" in m:
            linhas.append(f"| {m['simbolo']} | — | — | — | erro: {m['erro'][:40]} |")
        else:
            linhas.append(
                f"| {m['simbolo']} | US$ {m['meio']:,.2f} | {m['spread_pct']:.4f}% | "
                f"{m['desequilibrio']:.2f} | {m['fonte']} |"
            )

    linhas += [
        "",
        "## Carrego de funding (líquido de custo)",
        "",
        "| Par | Taxa média 8h | Liquidações negativas | Líquido a.a. | Dias p/ pagar a montagem |",
        "|---|---|---|---|---|",
    ]
    for c in registro["carrego"]:
        dias = c["dias_para_pagar_montagem"]
        linhas.append(
            f"| {c['simbolo']} | {c['taxa_media_pct']:.5f}% | {c['negativas_pct']:.1f}% | "
            f"{c['liquido_ano_pct']:+.2f}% | {dias if dias is not None else 'nunca'} |"
        )

    linhas += [
        "",
        "## Score de confluência (4h)",
        "",
        "| Par | Score | Sinal | RSI 14 | ADX | ATR |",
        "|---|---|---|---|---|---|",
    ]
    for s in registro["sinais"]:
        rotulo = {1: "compra", -1: "venda", 0: "fora"}[s["posicao_alvo"]]
        linhas.append(
            f"| {s['simbolo']} | {s['score']:.1f} | {rotulo} | {s['rsi_14']} | "
            f"{s['adx']} | {s['atr_pct']:.2f}% |"
        )

    if registro.get("validacao"):
        linhas += [
            "",
            "## Validação fora da amostra (4h)",
            "",
            "| Par | Janelas | Retorno OOS | Sharpe OOS | Janelas + | Aprovado |",
            "|---|---|---|---|---|---|",
        ]
        for v in registro["validacao"]:
            if "erro" in v:
                linhas.append(f"| {v['simbolo']} | — | — | — | — | erro |")
                continue
            linhas.append(
                f"| {v['simbolo']} | {v['janelas']} | {v['retorno_oos_pct']:+.2f}% | "
                f"{v['sharpe_oos']:.2f} | {v['janelas_positivas']}/{v['janelas']} | "
                f"{'SIM' if v['aprovado'] else 'não'} |"
            )

    custo = registro["custo_de_referencia"]
    linhas += [
        "",
        "## Custo de referência",
        "",
        f"Abrir e fechar US$ {custo['notional_usd']:,.0f} custa US$ {custo['custo_usd']:.4f} "
        f"({custo['custo_pct']:.3f}%). Um alvo de US$ 0,10 nesse tamanho sai "
        f"{custo['quantas_vezes_o_alvo_o_custo_come']:.1f}x abaixo do custo.",
        "",
        "---",
        "",
        "_Este relatório mede. Ele não recomenda, não prevê e não operou nada._",
        "",
    ]

    caminho = RELATORIOS / f"{hoje}.md"
    caminho.write_text("\n".join(linhas), encoding="utf-8")
    (RELATORIOS / "ULTIMO.md").write_text("\n".join(linhas), encoding="utf-8")
    return caminho


def main() -> int:
    p = argparse.ArgumentParser(description="Monitor diário de cripto do QuantBot")
    p.add_argument("--validar", action="store_true", help="força a validação completa")
    args = p.parse_args()

    agora = datetime.now(timezone.utc)
    # Segunda-feira roda a validação inteira; nos outros dias só se pedirem.
    validar = args.validar or agora.weekday() == 0

    cliente = cliente_padrao()
    registro = {
        "data": agora.strftime("%Y-%m-%d"),
        "quando": agora.isoformat(),
        "mercado": coletar_mercado(cliente),
        "carrego": coletar_carrego(),
        "sinais": coletar_sinais(),
        "validacao": rodar_validacao() if validar else None,
        "custo_de_referencia": resumo_viabilidade(0.10, 1_000.0, ModeloCusto()),
    }

    DIRETORIO.mkdir(parents=True, exist_ok=True)
    linha = json.dumps(registro, ensure_ascii=False, default=str)
    with open(DIRETORIO / "historico.jsonl", "a", encoding="utf-8") as f:
        f.write(linha + "\n")

    caminho = escrever_markdown(registro)
    logger.info("Relatório escrito em %s", caminho)

    if registro["validacao"]:
        aprovados = [v["simbolo"] for v in registro["validacao"] if v.get("aprovado")]
        if aprovados:
            logger.warning(
                "ATENÇÃO: %s passaram na validação hoje. Revise antes de qualquer decisão.",
                ", ".join(aprovados),
            )
        else:
            logger.info("Nenhum par passou na validação — nada muda.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
