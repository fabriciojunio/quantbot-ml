"""
Análise de Sentimento para Notícias Financeiras.

Dois modos de operação:
1. FinBERT (recomendado) — modelo BERT treinado em textos financeiros
2. VADER + Léxico financeiro (fallback) — sem dependências pesadas

Ambos rodam localmente, sem API, sem custo por requisição.

FinBERT: ProsusAI/finbert (Araci, 2019)
- Treinado em ~50.000 frases financeiras
- Precisão ~87% em sentimento financeiro
- Modelo: ~400MB de download (uma vez)

Referência TCC:
  Araci, D. (2019). FinBERT: Financial Sentiment Analysis with
  Pre-trained Language Models. arXiv:1908.10063.

Instalação FinBERT:
  pip install transformers torch

Se não quiser instalar (400MB+), o sistema usa VADER automaticamente.

Uso:
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("Petrobras anuncia lucro recorde")
    print(result)  # {"label": "positivo", "score": 0.87}
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("quantbot.data.sentiment")


# ═══════════════════════════════════════════════════════════════
# RESULTADO
# ═══════════════════════════════════════════════════════════════

@dataclass
class SentimentResult:
    """Resultado da análise de sentimento."""
    label: str        # "positivo", "negativo", "neutro"
    score: float      # 0.0 a 1.0 (confiança)
    method: str       # "finbert" ou "lexicon"
    raw_scores: Dict[str, float] = None  # scores por classe

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "method": self.method,
        }


# ═══════════════════════════════════════════════════════════════
# FINBERT ANALYZER
# ═══════════════════════════════════════════════════════════════

class FinBERTAnalyzer:
    """
    Análise de sentimento com FinBERT via PyTorch.

    Modelo padrão: lucas-leme/FinBERT-PT-BR — BERT treinado em ~1,4 milhão
    de notícias financeiras em português brasileiro, com classificador em
    textos rotulados (POSITIVE/NEGATIVE/NEUTRAL). É a melhor escolha para
    notícias da B3, que estão em português.

    ATENÇÃO ao mapeamento de rótulos: o FinBERT-PT-BR usa uma ordem NÃO
    padrão de índices — 0=POSITIVE, 1=NEGATIVE, 2=NEUTRAL (conforme o
    cartão do modelo no Hugging Face). Por isso usamos um mapa explícito,
    em vez de confiar na ordem default — senão o sentimento sairia trocado.

    Alternativa em inglês: ProsusAI/finbert (Araci, 2019). Troque pelo
    env QUANTBOT_FINBERT_MODEL.

    Download na primeira chamada (~400MB) e cache local do Hugging Face.
    """

    # Mapas índice → rótulo, autoritativos por modelo (a config do modelo
    # pode vir genérica como LABEL_0/LABEL_1, o que inverteria o resultado).
    _LABEL_MAPS = {
        "lucas-leme/FinBERT-PT-BR": {0: "positivo", 1: "negativo", 2: "neutro"},
    }

    def __init__(self, model_name: Optional[str] = None):
        import os
        self.model_name = model_name or os.getenv(
            "QUANTBOT_FINBERT_MODEL", "lucas-leme/FinBERT-PT-BR"
        )
        self._torch = None
        self.tokenizer = None
        self.model = None
        self._index_to_label: Dict[int, str] = {}
        self._loaded = False

    def _load_model(self) -> None:
        """Carrega tokenizer + modelo. Faz download na primeira vez."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
        except ImportError as e:
            raise ImportError(
                "FinBERT requer: pip install torch transformers\n"
                "O sistema vai usar análise léxica como fallback."
            ) from e

        logger.info(
            f"🧠 Carregando FinBERT '{self.model_name}' "
            f"(primeira vez faz download ~400MB)..."
        )
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.eval()
        self._index_to_label = self._build_label_map()
        self._loaded = True
        logger.info("🧠 FinBERT carregado com sucesso!")

    def _build_label_map(self) -> Dict[int, str]:
        """Resolve o mapa índice → rótulo, priorizando o mapa conhecido."""
        if self.model_name in self._LABEL_MAPS:
            return dict(self._LABEL_MAPS[self.model_name])
        mapping: Dict[int, str] = {}
        id2label = getattr(self.model.config, "id2label", None) or {}
        for idx, raw in id2label.items():
            mapping[int(idx)] = _normalize_label(raw)
        return mapping or {0: "positivo", 1: "negativo", 2: "neutro"}

    def _scores_from_logits(self, logits_row) -> Dict[str, float]:
        """Softmax dos logits → probabilidade por rótulo em português."""
        probs = self._torch.softmax(logits_row, dim=-1).tolist()
        scores = {"positivo": 0.0, "negativo": 0.0, "neutro": 0.0}
        for idx, p in enumerate(probs):
            label = self._index_to_label.get(idx, "neutro")
            scores[label] = scores.get(label, 0.0) + float(p)
        return scores

    def analyze(self, text: str) -> SentimentResult:
        """Analisa o sentimento de um texto financeiro."""
        return self.analyze_batch([text])[0]

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analisa vários textos de uma vez (mais eficiente)."""
        self._load_model()

        clean = [(t or "")[:512] for t in texts]
        tokens = self.tokenizer(
            clean,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with self._torch.no_grad():
            logits = self.model(**tokens).logits

        results: List[SentimentResult] = []
        for row in logits:
            scores = self._scores_from_logits(row)
            best_label = max(scores, key=scores.get)
            results.append(SentimentResult(
                label=best_label,
                score=scores[best_label],
                method="finbert",
                raw_scores=scores,
            ))
        return results


def _normalize_label(raw: str) -> str:
    """Normaliza rótulos variados (positive/POS/LABEL_0...) para PT."""
    r = str(raw).lower()
    if "pos" in r:
        return "positivo"
    if "neg" in r:
        return "negativo"
    return "neutro"


# ═══════════════════════════════════════════════════════════════
# LÉXICO FINANCEIRO (FALLBACK — SEM DEPENDÊNCIAS PESADAS)
# ═══════════════════════════════════════════════════════════════

# Léxico expandido para finanças em português e inglês
POSITIVE_WORDS = {
    # Português
    "lucro", "alta", "subiu", "sobe", "positivo", "recorde", "crescimento",
    "otimismo", "otimista", "valorização", "ganho", "recuperação", "expansão",
    "dividendo", "superou", "superávit", "aprovação", "aquisição", "inovação",
    "demanda", "avanço", "melhora", "confiança", "investimento", "oportunidade",
    "resultados fortes", "acima do esperado", "surpreendeu positivamente",
    "forte", "robusto", "sólido", "estável", "favorável",
    # Inglês
    "profit", "gain", "rise", "bull", "bullish", "growth", "record",
    "optimism", "optimistic", "recovery", "expansion", "dividend",
    "beat", "surpass", "upgrade", "strong", "robust", "rally",
    "outperform", "momentum", "breakout", "upside",
}

NEGATIVE_WORDS = {
    # Português
    "perda", "queda", "caiu", "cai", "negativo", "crise", "recessão",
    "pessimismo", "pessimista", "desvalorização", "prejuízo", "colapso",
    "inflação alta", "déficit", "rebaixamento", "venda", "fuga",
    "risco", "incerteza", "volatilidade", "pressão", "deterioração",
    "abaixo do esperado", "decepcionou", "fraco", "fracos",
    "guerra", "conflito", "sanção", "multa", "investigação",
    # Inglês
    "loss", "drop", "fall", "bear", "bearish", "decline", "crisis",
    "recession", "pessimism", "crash", "downgrade", "weak",
    "underperform", "risk", "uncertainty", "volatile", "sell-off",
    "default", "bankruptcy", "investigation", "fine", "penalty",
}

# Intensificadores
INTENSIFIERS = {
    "muito", "extremamente", "fortemente", "significativamente",
    "drasticamente", "impressionante", "histórico", "inédito",
    "very", "extremely", "strongly", "significantly", "dramatically",
    "unprecedented", "massive", "huge", "enormous",
}

# Negadores (invertem o sentimento)
NEGATORS = {
    "não", "sem", "nunca", "nem", "nenhum", "jamais",
    "not", "no", "never", "neither", "without", "hardly",
}


class LexiconAnalyzer:
    """
    Análise de sentimento baseada em léxico financeiro.

    Não requer downloads pesados nem GPU.
    Precisão estimada: ~70-75% em textos financeiros.
    Boa o suficiente como fallback do FinBERT.
    """

    def analyze(self, text: str) -> SentimentResult:
        """
        Analisa sentimento usando léxico financeiro.

        Algoritmo:
        1. Tokeniza e normaliza texto
        2. Conta palavras positivas e negativas
        3. Aplica intensificadores e negadores
        4. Calcula score final normalizado

        Args:
            text: Texto da notícia

        Returns:
            SentimentResult
        """
        text_lower = text.lower()
        words = re.findall(r"\b[a-záàâãéêíóôõúç]+\b", text_lower)

        pos_count = 0
        neg_count = 0
        intensity = 1.0

        for i, word in enumerate(words):
            # Verifica negador na palavra anterior
            is_negated = (i > 0 and words[i - 1] in NEGATORS)

            if word in POSITIVE_WORDS:
                if is_negated:
                    neg_count += 1
                else:
                    pos_count += 1
            elif word in NEGATIVE_WORDS:
                if is_negated:
                    pos_count += 1
                else:
                    neg_count += 1

            if word in INTENSIFIERS:
                intensity = 1.5

        # Também checa bigramas e trigramas
        for phrase in POSITIVE_WORDS:
            if " " in phrase and phrase in text_lower:
                pos_count += 2

        for phrase in NEGATIVE_WORDS:
            if " " in phrase and phrase in text_lower:
                neg_count += 2

        # Score
        total = pos_count + neg_count
        if total == 0:
            return SentimentResult(
                label="neutro", score=0.5, method="lexicon",
                raw_scores={"positivo": 0.33, "negativo": 0.33, "neutro": 0.34},
            )

        pos_ratio = pos_count / total * intensity
        neg_ratio = neg_count / total * intensity

        # Normaliza para 0-1
        if pos_ratio > neg_ratio:
            score = min(0.5 + (pos_ratio - neg_ratio) * 0.5, 0.95)
            label = "positivo"
        elif neg_ratio > pos_ratio:
            score = min(0.5 + (neg_ratio - pos_ratio) * 0.5, 0.95)
            label = "negativo"
        else:
            score = 0.5
            label = "neutro"

        return SentimentResult(
            label=label,
            score=score,
            method="lexicon",
            raw_scores={
                "positivo": pos_ratio / (pos_ratio + neg_ratio + 0.01),
                "negativo": neg_ratio / (pos_ratio + neg_ratio + 0.01),
                "neutro": 0.01 / (pos_ratio + neg_ratio + 0.01),
            },
        )


# ═══════════════════════════════════════════════════════════════
# ANALYZER PRINCIPAL (ESCOLHE AUTOMATICAMENTE)
# ═══════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    """
    Analisador de sentimento com fallback automático.

    Tenta usar FinBERT primeiro. Se não estiver instalado,
    usa análise léxica automaticamente.

    Uso:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("Petrobras anuncia lucro recorde no trimestre")
        print(result.label)   # "positivo"
        print(result.score)   # 0.89
        print(result.method)  # "finbert" ou "lexicon"
    """

    def __init__(self, prefer_finbert: bool = True):
        """
        Args:
            prefer_finbert: Se True, tenta usar FinBERT (se também ligado
                por env). FinBERT é opt-in via QUANTBOT_USE_FINBERT=1 para
                não disparar download de 400MB no caminho padrão/testes.
        """
        import os
        self._finbert = None
        self._lexicon = LexiconAnalyzer()
        self._use_finbert = False

        opt_in = os.getenv("QUANTBOT_USE_FINBERT", "0").strip().lower() in {
            "1", "true", "yes", "sim", "on",
        }

        if prefer_finbert and opt_in:
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
                self._finbert = FinBERTAnalyzer()
                self._use_finbert = True
                logger.info(
                    f"🧠 FinBERT ligado ({self._finbert.model_name}) — deep learning"
                )
            except ImportError:
                logger.info(
                    "📝 FinBERT pedido, mas torch/transformers não instalados. "
                    "Usando análise léxica."
                )
        elif prefer_finbert and not opt_in:
            logger.info(
                "📝 FinBERT desligado (defina QUANTBOT_USE_FINBERT=1 para ativar). "
                "Usando análise léxica."
            )
        else:
            logger.info("📝 Usando análise léxica (modo leve)")

    def analyze(self, text: str) -> SentimentResult:
        """Analisa sentimento de um texto."""
        if not text or not text.strip():
            return SentimentResult(label="neutro", score=0.5, method="none")

        if self._use_finbert and self._finbert:
            try:
                return self._finbert.analyze(text)
            except Exception as e:
                logger.warning(f"FinBERT falhou, usando léxico: {e}")

        return self._lexicon.analyze(text)

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analisa múltiplos textos."""
        if self._use_finbert and self._finbert:
            try:
                return self._finbert.analyze_batch(texts)
            except Exception:
                pass

        return [self._lexicon.analyze(t) for t in texts]

    def get_method(self) -> str:
        """Retorna o método atual de análise."""
        return "finbert" if self._use_finbert else "lexicon"


# ═══════════════════════════════════════════════════════════════
# INTEGRAÇÃO: NEWS + SENTIMENT
# ═══════════════════════════════════════════════════════════════

def analyze_news_sentiment(
    articles: list,
    analyzer: SentimentAnalyzer = None,
) -> list:
    """
    Analisa sentimento de uma lista de notícias.

    Adiciona sentiment e sentiment_score a cada NewsArticle.

    Args:
        articles: Lista de NewsArticle
        analyzer: SentimentAnalyzer (cria um se None)

    Returns:
        Mesma lista com sentimento preenchido
    """
    if analyzer is None:
        analyzer = SentimentAnalyzer()

    logger.info(f"🧠 Analisando sentimento de {len(articles)} notícias...")

    texts = [f"{a.title}. {a.description}" for a in articles]
    results = analyzer.analyze_batch(texts)

    for article, result in zip(articles, results):
        article.sentiment = result.label
        article.sentiment_score = result.score

    # Estatísticas
    pos = sum(1 for a in articles if a.sentiment == "positivo")
    neg = sum(1 for a in articles if a.sentiment == "negativo")
    neu = sum(1 for a in articles if a.sentiment == "neutro")

    logger.info(
        f"🧠 Sentimento: {pos} positivo, {neg} negativo, {neu} neutro "
        f"(método: {analyzer.get_method()})"
    )

    return articles


def get_symbol_sentiment(
    articles: list, symbol: str
) -> Tuple[str, float]:
    """
    Calcula sentimento agregado para um ativo específico.

    Média ponderada pelo impacto da notícia.

    Args:
        articles: Lista de NewsArticle com sentimento preenchido
        symbol: Símbolo do ativo

    Returns:
        Tupla (label, score) agregados
    """
    relevant = [a for a in articles if symbol in a.symbols and a.sentiment]

    if not relevant:
        return "neutro", 0.5

    impact_weight = {"alta": 3.0, "média": 1.5, "baixa": 0.5}

    weighted_score = 0
    total_weight = 0

    for article in relevant:
        weight = impact_weight.get(article.impact, 1.0)
        # Converte label em score numérico
        if article.sentiment == "positivo":
            s = article.sentiment_score
        elif article.sentiment == "negativo":
            s = 1 - article.sentiment_score
        else:
            s = 0.5

        weighted_score += s * weight
        total_weight += weight

    avg_score = weighted_score / total_weight if total_weight > 0 else 0.5

    if avg_score > 0.6:
        label = "positivo"
    elif avg_score < 0.4:
        label = "negativo"
    else:
        label = "neutro"

    return label, avg_score
