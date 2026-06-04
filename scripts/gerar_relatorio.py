"""
Gera um relatório diário em markdown a partir do que o sistema sabe hoje:
macro (Banco Central), screening de FIIs (Fundamentus) e o sentimento
acumulado das notícias.

Salva em relatorios/AAAA-MM-DD.md e atualiza relatorios/ULTIMO.md.
Pensado para rodar sozinho (GitHub Actions) e ir formando um histórico
de relatórios ao longo dos meses.
"""

import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.WARNING)

REL = ROOT / "relatorios"
HIST = ROOT / "data" / "news_history.jsonl"
INDEX = ROOT / "data" / "sentiment_index.jsonl"


def _safe(fn, fallback):
    try:
        return fn()
    except Exception as e:
        logging.warning(f"falhou: {e}")
        return fallback


def macro_bloco() -> str:
    from data.bcb_client import BCBClient
    bcb = BCBClient()
    snap = _safe(bcb.snapshot, {})
    linhas = ["## Macro (Banco Central)", ""]
    rotulos = {"selic_meta": "Selic meta", "cdi_anual": "CDI",
               "ipca_12m": "IPCA 12m", "usd_brl": "Dólar"}
    for k, label in rotulos.items():
        v = snap.get(k)
        linhas.append(f"- **{label}:** {v if v is not None else 'n/d'}")
    return "\n".join(linhas)


def screening_bloco() -> str:
    from data.fundamentus_client import FundamentusClient
    from data.bcb_client import BCBClient
    from income.fii_screener import FIIScreener

    fc = FundamentusClient()
    metrics = _safe(lambda: fc.to_fii_metrics(), [])
    if not metrics:
        return "## FIIs\n\n_Fundamentus indisponível nesta rodada._"

    selic = _safe(lambda: BCBClient().selic_meta(), 14.5) or 14.5
    screener = FIIScreener()
    # DY mínimo acompanha a Selic (FII tem que valer a pena vs renda fixa)
    screener.criteria.dy_min = max(7.0, round(selic * 0.7, 1))
    res = screener.screen(metrics)

    linhas = [
        f"## FIIs — screening do dia",
        "",
        f"Analisados: **{len(metrics)}** | Aprovados: **{len(res.aprovados)}** "
        f"| DY mínimo: {screener.criteria.dy_min:.1f}% (Selic {selic}%)",
        "",
        "| # | Ticker | Score | DY | P/VP |",
        "|---|--------|-------|-----|------|",
    ]
    for i, f in enumerate(res.top_n(12), 1):
        linhas.append(f"| {i} | {f.ticker} | {f.score:.1f} | {f.dy_12m:.1f}% | {f.pvp:.2f} |")
    return "\n".join(linhas)


def sentimento_bloco() -> str:
    if not INDEX.exists():
        return "## Sentimento\n\n_Sem coletas ainda. Rode scripts/coletar_noticias.py._"
    series = [json.loads(l) for l in INDEX.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not series:
        return "## Sentimento\n\n_Base vazia._"
    hoje = series[-1]
    ult = series[-14:]
    media = sum(s["indice_sentimento"] for s in ult) / len(ult)
    spark = " ".join(_barra(s["indice_sentimento"]) for s in ult)

    # temas mais citados nas notícias recentes
    temas = Counter()
    if HIST.exists():
        for l in HIST.read_text(encoding="utf-8").splitlines()[-400:]:
            try:
                for s in json.loads(l).get("simbolos", []):
                    temas[s] += 1
            except json.JSONDecodeError:
                continue

    linhas = [
        "## Sentimento do mercado (notícias)",
        "",
        f"- **Hoje:** {hoje['indice_sentimento']:+.2f} "
        f"(😀 {hoje['positivo']} · 😐 {hoje['neutro']} · 😟 {hoje['negativo']}) "
        f"via {hoje.get('metodo','?')}",
        f"- **Média 14 dias:** {media:+.2f}",
        f"- **Tendência:** {spark}  (índice −1 medo … +1 otimismo)",
    ]
    if temas:
        top = ", ".join(f"{t} ({n})" for t, n in temas.most_common(6))
        linhas.append(f"- **Ativos mais citados:** {top}")
    return "\n".join(linhas)


def _barra(v: float) -> str:
    if v <= -0.4:
        return "▁"
    if v <= -0.15:
        return "▃"
    if v < 0.15:
        return "▅"
    if v < 0.4:
        return "▆"
    return "█"


def main():
    REL.mkdir(parents=True, exist_ok=True)
    hoje = date.today().isoformat()
    corpo = "\n\n".join([
        f"# QuantBot — Relatório {hoje}",
        f"_Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}._",
        macro_bloco(),
        sentimento_bloco(),
        screening_bloco(),
        "---\n*O bot analisa e informa. Você decide e executa. "
        "DY futuro não é garantido; disciplina de reinvestir é o que mais importa.*",
    ])
    (REL / f"{hoje}.md").write_text(corpo, encoding="utf-8")
    (REL / "ULTIMO.md").write_text(corpo, encoding="utf-8")
    print(f"Relatório salvo: relatorios/{hoje}.md")


if __name__ == "__main__":
    main()
