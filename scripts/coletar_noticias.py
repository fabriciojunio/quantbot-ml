"""
Coletor da BASE DE APRENDIZADO de notícias do QuantBot.

O que faz, em uma rodada:
    1. Coleta notícias atuais de ~28 feeds RSS gratuitos (NewsFetcher).
    2. (opcional) Puxa notícias históricas do GDELT para temas-chave do
       mercado brasileiro — é assim que o sistema ganha uma BASE do passado.
    3. Roda o sentimento (FinBERT-PT-BR se QUANTBOT_USE_FINBERT=1, senão
       léxico) em cada notícia.
    4. Anexa tudo em data/news_history.jsonl (append-only, sem duplicar).
    5. Calcula o índice de sentimento do dia e anexa em
       data/sentiment_index.jsonl.

Rodar todo dia (ex.: Task Scheduler) faz a base crescer de jun/2026 até
jan/2027, dando histórico para análises e para treinar modelos depois.

Uso:
    python scripts/coletar_noticias.py            # RSS + GDELT
    python scripts/coletar_noticias.py --sem-gdelt
    python scripts/coletar_noticias.py --dias 3
"""

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Garante saída UTF-8 mesmo em console Windows (cp1252), evitando crash
# ao imprimir emojis/acentos no resumo final.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# carrega .env simples (para QUANTBOT_USE_FINBERT etc.)
_envf = ROOT / ".env"
if _envf.exists():
    for _l in _envf.read_text(encoding="utf-8").splitlines():
        if "=" in _l and not _l.strip().startswith("#"):
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

from data.news_fetcher import NewsFetcher
from data.sentiment import SentimentAnalyzer, analyze_news_sentiment

HIST = ROOT / "data" / "news_history.jsonl"
INDEX = ROOT / "data" / "sentiment_index.jsonl"

# Temas-chave para a base histórica via GDELT (mercado brasileiro).
GDELT_QUERIES = [
    "Ibovespa OR Petrobras OR Vale",
    "Selic OR Copom OR \"Banco Central\"",
    "dividendos OR \"fundos imobiliários\"",
    "Haddad OR \"arcabouço fiscal\" OR Fazenda",
    "dólar OR câmbio OR inflação Brasil",
]


def _carregar_urls_existentes() -> set:
    urls = set()
    if HIST.exists():
        for line in HIST.read_text(encoding="utf-8").splitlines():
            try:
                urls.add(json.loads(line).get("url", ""))
            except json.JSONDecodeError:
                continue
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-gdelt", action="store_true", help="não buscar histórico no GDELT")
    ap.add_argument("--dias", type=int, default=7, help="idade máxima das notícias RSS")
    args = ap.parse_args()

    HIST.parent.mkdir(parents=True, exist_ok=True)

    # 1. RSS atual
    fetcher = NewsFetcher(max_age_days=args.dias)
    artigos = fetcher.fetch_all()

    # 2. GDELT histórico (opcional)
    if not args.sem_gdelt:
        try:
            from data.gdelt_client import GDELTClient
            g = GDELTClient()
            for q in GDELT_QUERIES:
                artigos.extend(g.buscar(q, timespan="1m", lingua="portuguese", max_registros=50))
        except Exception as e:
            logging.warning(f"GDELT indisponível: {e}")

    # 3. Sentimento
    analyzer = SentimentAnalyzer(prefer_finbert=True)
    metodo = analyzer.get_method()
    artigos = analyze_news_sentiment(artigos, analyzer)

    # 4. Persistir (dedup por URL)
    existentes = _carregar_urls_existentes()
    novos = 0
    with HIST.open("a", encoding="utf-8") as f:
        for a in artigos:
            chave = a.url or a.title
            if chave in existentes:
                continue
            existentes.add(chave)
            f.write(json.dumps({
                "coletado_em": datetime.now().isoformat(timespec="seconds"),
                "publicado": a.published.isoformat(),
                "fonte": a.source,
                "titulo": a.title,
                "url": a.url,
                "categoria": a.category,
                "simbolos": a.symbols,
                "sentimento": a.sentiment,
                "score": round(a.sentiment_score or 0.5, 4),
                "metodo": metodo,
            }, ensure_ascii=False) + "\n")
            novos += 1

    # 5. Índice de sentimento do dia
    pos = sum(1 for a in artigos if a.sentiment == "positivo")
    neg = sum(1 for a in artigos if a.sentiment == "negativo")
    neu = sum(1 for a in artigos if a.sentiment == "neutro")
    total = max(1, pos + neg + neu)
    indice = round((pos - neg) / total, 4)  # -1 (medo) .. +1 (otimismo)
    with INDEX.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "data": date.today().isoformat(),
            "positivo": pos, "negativo": neg, "neutro": neu,
            "indice_sentimento": indice, "metodo": metodo,
        }, ensure_ascii=False) + "\n")

    total_base = sum(1 for _ in HIST.open(encoding="utf-8"))
    print("\n" + "=" * 52)
    print("📰 COLETA CONCLUÍDA")
    print("=" * 52)
    print(f"Notícias nesta rodada: {len(artigos)} (método: {metodo})")
    print(f"Novas salvas na base:  {novos}")
    print(f"Sentimento hoje:       😀 {pos}  😐 {neu}  😟 {neg}")
    print(f"Índice de sentimento:  {indice:+.2f}  (-1 medo … +1 otimismo)")
    print(f"Base acumulada total:  {total_base} notícias em {HIST.name}")
    print("=" * 52)


if __name__ == "__main__":
    main()
