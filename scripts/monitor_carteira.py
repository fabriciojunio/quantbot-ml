"""
Monitor diário da carteira.

Roda na nuvem, todo dia, e acumula a evidência que decide se a estratégia
merece dinheiro. Ele **mede e registra; não opera**.

Todo dia:
  1. Preço de fechamento dos ETFs e o CDI do dia.
  2. Sinal de cada ativo, aberto indicador por indicador.
  3. Guarda de regime (o índice está batendo o CDI? está acima da média?).
  4. Alocação sugerida pela regra declarada, com o caixa que sobra.
  5. Marcação da carteira de papel, com custo e imposto cobrados.

Toda segunda-feira, além disso:
  6. Walk-forward completo, com Deflated Sharpe e PBO, e o veredito.

Por que registrar o veredito todo dia mesmo sabendo que hoje ele reprova: é a
única forma de responder daqui a um ano, com dado e não com opinião, se alguma
coisa mudou. Se o regime virar e a estratégia passar a ter vantagem, o dia em
que isso aconteceu vai estar escrito.
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

from carteira import alocacao, dados, sinais, universo, validacao  # noqa: E402
from carteira.rebalanceamento import Politica  # noqa: E402
from carteira.regras_pai import guarda_de_regime  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("quantbot.monitor_carteira")

DIRETORIO = Path("data/carteira/monitor")
RELATORIOS = Path("relatorios/carteira")

TICKERS = list(universo.UNIVERSO_B3.keys())
REFERENCIA = "BOVA11.SA"


def coletar(anos: int = 3) -> tuple:
    precos = dados.alinhar(dados.precos(TICKERS, anos=anos))
    dados.valida_precos(precos, "ETFs B3")
    cdi = dados.cdi_diario(anos=anos)
    cdi_acum = (1 + cdi.reindex(precos.index).ffill().fillna(0.0)).cumprod()
    return precos, cdi, cdi_acum


def montar_registro(precos, cdi, cdi_acum) -> dict:
    diag = sinais.diagnostico(precos)

    ret_ref_12m = (
        float(precos[REFERENCIA].iloc[-1] / precos[REFERENCIA].iloc[-252] - 1)
        if len(precos) > 252 else 0.0
    )
    ret_cdi_12m = (
        float(cdi_acum.iloc[-1] / cdi_acum.iloc[-252] - 1) if len(cdi_acum) > 252 else 0.0
    )
    acima_200 = bool(sinais.filtro_de_tendencia(precos).iloc[-1].get(REFERENCIA, False))
    favoravel, regime = guarda_de_regime(ret_ref_12m, ret_cdi_12m, acima_200)

    pesos = alocacao.dual_momentum(precos, top_n=3, cdi_acumulado=cdi_acum)
    alvo = pesos.iloc[-1]
    alvo = alvo[alvo > 0]

    try:
        macro = dados.macro()
    except Exception as exc:
        logger.warning("Macro indisponível: %s", exc)
        macro = {}

    return {
        "data": str(precos.index[-1])[:10],
        "quando": datetime.now(timezone.utc).isoformat(),
        "macro": macro,
        "regime": regime,
        "sinais": diag["ativos"],
        "alocacao_sugerida_pct": {t: round(float(p) * 100, 1) for t, p in alvo.items()},
        "caixa_sugerido_pct": round((1 - float(alvo.sum())) * 100, 1),
        "precos": {t: round(float(precos[t].iloc[-1]), 2) for t in precos.columns},
        "cdi_12m_pct": round(ret_cdi_12m * 100, 2),
        "indice_12m_pct": round(ret_ref_12m * 100, 2),
    }


def rodar_validacao(precos, cdi) -> dict | None:
    try:
        _, _, resumo = validacao.walk_forward(
            precos, cdi, dias_treino=504, dias_teste=126, politica=Politica.TRIMESTRAL
        )
        return {"resumo": resumo, "veredito": validacao.veredito(resumo)}
    except Exception as exc:
        logger.warning("Validação não rodou: %s", exc)
        return None


def escrever_markdown(reg: dict) -> Path:
    RELATORIOS.mkdir(parents=True, exist_ok=True)
    regime = reg["regime"]

    linhas = [
        f"# Carteira — {reg['data']}",
        "",
        f"_Gerado em {reg['quando'][:19]} UTC. Este relatório mede; não recomenda._",
        "",
        "## Regime de mercado",
        "",
        f"**{regime['leitura'].capitalize()}** (score {regime['score']}/100)",
        "",
        f"- Índice em 12 meses: {reg['indice_12m_pct']:+.2f}%",
        f"- CDI em 12 meses: {reg['cdi_12m_pct']:+.2f}%",
        f"- Excesso sobre o CDI: {regime['excesso_sobre_cdi_pp']:+.2f} p.p.",
        f"- Acima da média de 200 dias: {'sim' if regime['acima_da_media_200'] else 'não'}",
        "",
        "## Sinais por ativo",
        "",
        "| Ativo | Preço | Momentum 12m | Ranking | Vol. anual | Acima da média 200 |",
        "|---|---|---|---|---|---|",
    ]
    for t, s in reg["sinais"].items():
        mom = s["momentum_absoluto_pct"]
        rank = s["posicao_no_ranking"]
        vol = s["volatilidade_anual_pct"]
        linhas.append(
            f"| {t.replace('.SA', '')} | R$ {reg['precos'].get(t, 0):,.2f} | "
            f"{'—' if mom is None else f'{mom:+.2f}%'} | "
            f"{'—' if rank is None else f'{rank:.2f}'} | "
            f"{'—' if vol is None else f'{vol:.1f}%'} | "
            f"{'sim' if s['acima_da_media_200'] else 'não'} |"
        )

    linhas += ["", "## Alocação sugerida pela regra", ""]
    if reg["alocacao_sugerida_pct"]:
        linhas += ["| Ativo | Peso |", "|---|---|"]
        for t, p in sorted(reg["alocacao_sugerida_pct"].items(), key=lambda x: -x[1]):
            linhas.append(f"| {t.replace('.SA', '')} | {p:.1f}% |")
        linhas.append(f"| **Caixa (CDI)** | **{reg['caixa_sugerido_pct']:.1f}%** |")
    else:
        linhas.append(
            f"Nenhum ativo com tendência positiva acima do CDI. "
            f"A regra manda ficar **{reg['caixa_sugerido_pct']:.0f}% em caixa**."
        )

    if reg.get("validacao"):
        v = reg["validacao"]
        r, ver = v["resumo"], v["veredito"]
        linhas += [
            "",
            "## Validação fora da amostra (semanal)",
            "",
            f"- Janelas: {r['janelas']}",
            f"- Retorno fora da amostra: {r['retorno_oos_total_pct']:+.2f}%",
            f"- CDI no mesmo trecho: {r['cdi_no_mesmo_trecho_pct']:+.2f}%",
            f"- Diferença: {r['retorno_oos_menos_cdi_pp']:+.2f} p.p.",
            f"- Janelas que bateram o CDI: {r['janelas_que_bateram_o_cdi']}/{r['janelas']}",
            f"- Sharpe fora da amostra: {r['sharpe_oos']:.2f}",
            "",
            f"**Aprovada para papel: {'SIM' if ver['aprovado_para_papel'] else 'não'}**",
            "",
        ]
        for m in ver["motivos"]:
            linhas.append(f"- {m}")

    linhas += [
        "",
        "---",
        "",
        "_Aprovação para dinheiro real nunca é automática. Ver `docs/CARTEIRA.md`._",
        "",
    ]

    caminho = RELATORIOS / f"{reg['data']}.md"
    conteudo = "\n".join(linhas)
    caminho.write_text(conteudo, encoding="utf-8")
    (RELATORIOS / "ULTIMO.md").write_text(conteudo, encoding="utf-8")
    return caminho


def main() -> int:
    p = argparse.ArgumentParser(description="Monitor diário da carteira")
    p.add_argument("--validar", action="store_true", help="força a validação completa")
    args = p.parse_args()

    agora = datetime.now(timezone.utc)
    validar = args.validar or agora.weekday() == 0  # segunda-feira

    # O sinal do dia precisa de 3 anos; a validação precisa de muito mais, para
    # ter janelas suficientes fora da amostra. Com 3 anos sai uma janela só, e
    # uma janela não valida nada.
    precos, cdi, cdi_acum = coletar(anos=3)
    registro = montar_registro(precos, cdi, cdi_acum)

    if validar:
        logger.info("Rodando validação fora da amostra...")
        try:
            px_longo, cdi_longo, _ = coletar(anos=12)
        except Exception as exc:
            logger.warning("Histórico longo indisponível (%s); usando o curto.", exc)
            px_longo, cdi_longo = precos, cdi
        registro["validacao"] = rodar_validacao(px_longo, cdi_longo)

    DIRETORIO.mkdir(parents=True, exist_ok=True)
    with open(DIRETORIO / "historico.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")

    caminho = escrever_markdown(registro)
    logger.info("Relatório em %s", caminho)

    if registro.get("validacao") and registro["validacao"]["veredito"]["aprovado_para_papel"]:
        logger.warning(
            "ATENÇÃO: a estratégia passou na validação hoje. Revise antes de qualquer decisão."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
