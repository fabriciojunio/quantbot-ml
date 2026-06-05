"""
O bot OPERA SOZINHO (dinheiro simulado) e APRENDE.

Em cada rodada:
    1. Lê fundamentos reais (Fundamentus) + Selic/CDI (Banco Central).
    2. Faz o screening de FIIs, já aplicando o que aprendeu (ajuste de score).
    3. Decide compras/vendas com dinheiro simulado (PaperBook).
    4. Marca a mercado, acumula proventos (bola de neve).
    5. Avalia posições antigas vs CDI → acertos/erros.
    6. Aprende: das decisões (faixas que dão certo) e das notícias (sentimento
       → retorno à frente).

Pensado para rodar todo dia na nuvem, formando um track record honesto.
"""

import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_envf = ROOT / ".env"
if _envf.exists():
    for _l in _envf.read_text(encoding="utf-8").splitlines():
        if "=" in _l and not _l.strip().startswith("#"):
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")

from data.fundamentus_client import FundamentusClient
from data.bcb_client import BCBClient
from income.fii_screener import FIIScreener
from income.paper_book import PaperBook
from income.feedback import Feedback, _faixa_sent

INDEX = ROOT / "data" / "sentiment_index.jsonl"


def _sentimento_hoje() -> float:
    if not INDEX.exists():
        return 0.0
    linhas = [l for l in INDEX.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not linhas:
        return 0.0
    try:
        return float(json.loads(linhas[-1]).get("indice_sentimento", 0.0))
    except json.JSONDecodeError:
        return 0.0


def main():
    fc = FundamentusClient()
    metrics = fc.to_fii_metrics()
    if not metrics:
        print("Fundamentus indisponível — sem operação hoje.")
        return

    bcb = BCBClient()
    selic = bcb.selic_meta() or 14.5
    cdi = bcb.cdi_anual() or (selic - 0.1)

    # preços vêm da própria leitura de fundamentos
    precos = {m.ticker: m.preco_cota for m in metrics if m.preco_cota > 0}

    screener = FIIScreener()
    screener.criteria.dy_min = max(7.0, round(selic * 0.7, 1))
    resultado = screener.screen(metrics)

    fb = Feedback()
    idx_hoje = _sentimento_hoje()
    faixa_sent = _faixa_sent(idx_hoje)

    # aplica o que aprendeu: ajusta o score de cada aprovado e re-ordena
    for m in resultado.aprovados:
        m.score = round(m.score + fb.ajuste_score(m.score, m.dy_12m, faixa_sent), 1)
    resultado.aprovados.sort(key=lambda a: a.score, reverse=True)

    book = PaperBook()
    top = resultado.top_n(book.max_posicoes)
    sent_map = {a.ticker: (faixa_sent, abs(idx_hoje)) for a in top}

    decisoes = book.decidir(top, precos, sentimento_por_ticker=sent_map)
    book.marcar_mercado(precos, cdi_anual=cdi)
    resultados = book.avaliar(precos, cdi_anual=cdi)

    # aprende
    fb.atualizar_decisoes(resultados)
    fb.aprender_sentimento(_ler_jsonl(INDEX), book.estado["serie_valor"])
    fb.salvar()
    book.salvar()

    r = book.resumo(precos)
    print("=" * 52)
    print(f"🤖 OPERAÇÃO PAPER — {date.today().isoformat()}")
    print("=" * 52)
    print(f"Decisões hoje: {len(decisoes)}  | Avaliações: {len(resultados)}")
    for d in decisoes[:8]:
        if d["acao"] == "COMPRAR":
            print(f"  🟢 COMPRAR {d['ticker']} {d['cotas']:.0f}c @ R${d['preco']:.2f} (score {d['score']})")
        else:
            print(f"  🔴 VENDER {d['ticker']} @ R${d['preco']:.2f} ({d['retorno_total_pct']:+.1f}%) — {d['motivo']}")
    print(f"\nCarteira simulada: R${r['valor_total']:,.2f} "
          f"({r['retorno_pct']:+.2f}% vs R${r['capital_inicial']:,.0f})")
    print(f"Posições: {r['n_posicoes']} | Caixa: R${r['caixa']:,.2f} | "
          f"Fechadas: {r['n_fechadas']} ({r['acertos_fechadas']} no lucro)")
    print()
    print(fb.resumo_texto())
    print("=" * 52)


def _ler_jsonl(p: Path):
    if not p.exists():
        return []
    out = []
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                continue
    return out


if __name__ == "__main__":
    main()
