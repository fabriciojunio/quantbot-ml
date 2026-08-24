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
    from data.bcb_client import BCBClient
    from data.fundamentus_client import FundamentusClient
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
        "## FIIs — screening do dia",
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


def paper_bloco() -> str:
    """Carteira simulada + o que o bot aprendeu (decisões e notícias)."""
    book_f = ROOT / "data" / "paper_book.json"
    if not book_f.exists():
        return "## Carteira simulada\n\n_O bot ainda não operou. Rode scripts/operar_paper.py._"
    book = json.loads(book_f.read_text(encoding="utf-8"))
    serie = book.get("serie_valor", [])
    valor = serie[-1]["valor_total"] if serie else book.get("caixa", 0)
    cap = book.get("capital_inicial", 1)
    ret = (valor - cap) / cap * 100
    fechadas = book.get("fechadas", [])
    acertos = sum(1 for f in fechadas if f["retorno_total_pct"] > 0)
    posicoes = book.get("posicoes", {})

    linhas = [
        "## Carteira simulada (paper — o bot operando sozinho)",
        "",
        f"- **Valor:** R${valor:,.2f}  (**{ret:+.2f}%** vs R${cap:,.0f} inicial)",
        f"- **Posições abertas:** {len(posicoes)} · **Caixa:** R${book.get('caixa',0):,.2f}",
        f"- **Operações fechadas:** {len(fechadas)} ({acertos} no lucro)",
    ]
    if posicoes:
        nomes = ", ".join(sorted(posicoes.keys())[:12])
        linhas.append(f"- **Carteira:** {nomes}")

    # Benchmark: +X% só faz sentido comparado ao custo de oportunidade (CDI).
    bench = _benchmark_bloco(serie, ret)
    if bench:
        linhas += ["", *bench]

    # Expectancy: win rate sozinho engana — o que importa é ganho médio × perda média.
    exp = _expectancy_bloco(fechadas)
    if exp:
        linhas += ["", *exp]

    learned = _safe(lambda: __import__("income.feedback", fromlist=["Feedback"])
                    .Feedback().resumo_texto(), "")
    if learned:
        linhas += ["", learned]
    return "\n".join(linhas)


def _periodo_dias(serie: list) -> int:
    """Dias corridos entre o primeiro e o último ponto da série de valor."""
    if len(serie) < 2:
        return len(serie)
    try:
        d0 = datetime.fromisoformat(str(serie[0]["data"])[:10])
        d1 = datetime.fromisoformat(str(serie[-1]["data"])[:10])
        return max(1, (d1 - d0).days)
    except Exception:
        return len(serie)


def _benchmark_bloco(serie: list, ret: float) -> list:
    """Compara o retorno do bot com o CDI acumulado no mesmo período (alpha)."""
    dias = _periodo_dias(serie)
    if dias < 1:
        return []
    cdi_anual = _safe(lambda: __import__("data.bcb_client", fromlist=["BCBClient"])
                      .BCBClient().snapshot().get("cdi_anual"), None) or 14.15
    cdi_periodo = ((1 + cdi_anual / 100.0) ** (dias / 365.0) - 1.0) * 100.0
    alpha = ret - cdi_periodo
    marca = "✅ acima do CDI" if alpha > 0 else "❌ abaixo do CDI"
    return [
        "### Desempenho vs benchmark",
        f"- **Período:** {dias} dias · **Bot:** {ret:+.2f}% · "
        f"**CDI:** {cdi_periodo:+.2f}% (a.a. {cdi_anual:.2f}%)",
        f"- **Alpha vs CDI:** {alpha:+.2f} pontos percentuais — {marca}",
        "- _Janela curta: leia como acompanhamento, não como prova de edge._"
        if dias < 90 else "",
    ]


def _expectancy_bloco(fechadas: list) -> list:
    """Win rate, ganho/perda médios, payoff e expectancy por operação."""
    rets = [f.get("retorno_total_pct", 0.0) for f in fechadas]
    if len(rets) < 1:
        return []
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    n = len(rets)
    win_rate = len(wins) / n * 100
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0  # <= 0
    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")
    # Expectancy por trade (%) = p(win)*ganho_médio + p(loss)*perda_média.
    expectancy = (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss
    payoff_txt = "∞" if payoff == float("inf") else f"{payoff:.2f}"
    sinal = "✅ positiva" if expectancy > 0 else "⚠️ negativa"
    return [
        "### Expectativa estatística (operações fechadas)",
        f"- **Win rate:** {win_rate:.0f}% ({len(wins)}/{n})",
        f"- **Ganho médio:** {avg_win:+.2f}% · **Perda média:** {avg_loss:+.2f}%",
        f"- **Payoff (ganho/perda):** {payoff_txt} · **Expectancy/trade:** "
        f"{expectancy:+.2f}% — {sinal}",
        f"- _Amostra pequena (n={n}): ainda sem significância; acumule mais trades._"
        if n < 30 else "",
    ]


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
        paper_bloco(),
        screening_bloco(),
        "---\n*O bot analisa e informa. Você decide e executa. "
        "DY futuro não é garantido; disciplina de reinvestir é o que mais importa.*",
    ])
    (REL / f"{hoje}.md").write_text(corpo, encoding="utf-8")
    (REL / "ULTIMO.md").write_text(corpo, encoding="utf-8")
    print(f"Relatório salvo: relatorios/{hoje}.md")


if __name__ == "__main__":
    main()
