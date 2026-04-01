"""
Rastreador de Acurácia e Confiança do QuantBot ML.

Monitora a taxa de acerto do bot em tempo real, registrando
cada predição e seu resultado real, calculando métricas
de confiabilidade para exibição no dashboard.

Métricas rastreadas:
- Taxa de acerto geral e por ativo
- Taxa de acerto por nível de confiança
- Calibração do modelo (confiança vs acerto real)
- Histórico de predições para auditoria
- Matriz de confusão dos sinais

Para TCC: esta classe fornece os dados para a seção de
"Resultados e Análise" da monografia.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from config.settings import Signal
from utils.logger import get_logger

logger = get_logger("quantbot.core.accuracy")


@dataclass
class Prediction:
    """Registro de uma predição individual."""
    id: int
    timestamp: str
    symbol: str
    signal: str          # "COMPRA", "VENDA", etc.
    score: float         # 0-100
    confidence: float    # 0-100
    price_at_prediction: float
    price_after: float = 0.0     # preço N dias depois
    actual_return: float = 0.0   # retorno real em %
    was_correct: bool = False    # se a predição acertou
    resolved: bool = False       # se já temos o resultado
    horizon_days: int = 5
    sentiment: str = "neutro"
    model_agreement: float = 0.0  # % dos modelos que concordam


@dataclass
class AccuracyMetrics:
    """Métricas consolidadas de acurácia."""
    total_predictions: int
    resolved_predictions: int
    correct_predictions: int
    hit_rate: float              # taxa de acerto geral (%)
    hit_rate_buys: float         # acerto só em sinais de compra
    hit_rate_sells: float        # acerto só em sinais de venda
    hit_rate_high_confidence: float  # acerto quando confiança > 70%
    hit_rate_with_sentiment: float   # acerto quando sentimento concorda
    avg_confidence: float        # confiança média
    avg_return_correct: float    # retorno médio dos acertos
    avg_return_incorrect: float  # retorno médio dos erros
    profit_factor: float         # ganho_total / perda_total
    calibration: Dict[str, float] = field(default_factory=dict)
    by_symbol: Dict[str, float] = field(default_factory=dict)
    by_market: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}


class AccuracyTracker:
    """
    Rastreia e calcula métricas de acurácia do bot.

    Uso:
        tracker = AccuracyTracker()
        tracker.record_prediction("PETR4.SA", "COMPRA", 72.5, 85.0, 38.50)
        # ... depois de N dias ...
        tracker.resolve_prediction(pred_id, actual_price=39.80)
        metrics = tracker.get_metrics()
        print(f"Taxa de acerto: {metrics.hit_rate:.1f}%")
    """

    def __init__(self):
        self.predictions: List[Prediction] = []
        self._counter = 0

    def record_prediction(
        self,
        symbol: str,
        signal: str,
        score: float,
        confidence: float,
        price: float,
        sentiment: str = "neutro",
        model_agreement: float = 0.0,
        horizon_days: int = 5,
    ) -> Prediction:
        """
        Registra uma nova predição.

        Args:
            symbol: Símbolo do ativo
            signal: Sinal gerado (COMPRA, VENDA, etc.)
            score: Score do ensemble (0-100)
            confidence: Confiança (0-100)
            price: Preço no momento da predição
            sentiment: Sentimento de notícias
            model_agreement: % de concordância entre modelos
            horizon_days: Horizonte da predição em dias

        Returns:
            Prediction registrada
        """
        self._counter += 1
        pred = Prediction(
            id=self._counter,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            signal=signal,
            score=score,
            confidence=confidence,
            price_at_prediction=price,
            sentiment=sentiment,
            model_agreement=model_agreement,
            horizon_days=horizon_days,
        )
        self.predictions.append(pred)
        return pred

    def resolve_prediction(
        self, prediction_id: int, actual_price: float
    ) -> Optional[Prediction]:
        """
        Resolve uma predição com o preço real após o horizonte.

        Args:
            prediction_id: ID da predição
            actual_price: Preço real N dias depois

        Returns:
            Prediction atualizada ou None
        """
        pred = next((p for p in self.predictions if p.id == prediction_id), None)
        if pred is None:
            return None

        pred.price_after = actual_price
        pred.actual_return = (
            (actual_price - pred.price_at_prediction)
            / pred.price_at_prediction * 100
        )

        # Verifica se acertou
        if pred.signal in ("COMPRA_FORTE", "COMPRA"):
            pred.was_correct = pred.actual_return > 0
        elif pred.signal in ("VENDA_FORTE", "VENDA"):
            pred.was_correct = pred.actual_return < 0
        else:  # NEUTRO
            pred.was_correct = abs(pred.actual_return) < 1.0  # ±1%

        pred.resolved = True
        return pred

    def resolve_all_from_prices(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> int:
        """
        Resolve automaticamente predições usando dados de preço.

        Para cada predição não resolvida, busca o preço
        N dias depois nos dados disponíveis.

        Args:
            price_data: {symbol: DataFrame com coluna 'close'}

        Returns:
            Número de predições resolvidas
        """
        resolved_count = 0

        for pred in self.predictions:
            if pred.resolved:
                continue

            if pred.symbol not in price_data:
                continue

            df = price_data[pred.symbol]
            if "close" not in df.columns:
                continue

            # Encontra o preço N dias depois
            prices = df["close"].values
            if len(prices) < pred.horizon_days + 1:
                continue

            # Usa os últimos dados disponíveis como "futuro"
            future_price = prices[-1]
            past_price = prices[-(pred.horizon_days + 1)]

            pred.price_at_prediction = past_price
            self.resolve_prediction(pred.id, future_price)
            resolved_count += 1

        if resolved_count:
            logger.info(f"📊 {resolved_count} predições resolvidas automaticamente")

        return resolved_count

    def get_metrics(self) -> AccuracyMetrics:
        """
        Calcula todas as métricas de acurácia.

        Returns:
            AccuracyMetrics completo
        """
        total = len(self.predictions)
        resolved = [p for p in self.predictions if p.resolved]
        correct = [p for p in resolved if p.was_correct]

        if not resolved:
            return AccuracyMetrics(
                total_predictions=total,
                resolved_predictions=0,
                correct_predictions=0,
                hit_rate=0, hit_rate_buys=0, hit_rate_sells=0,
                hit_rate_high_confidence=0, hit_rate_with_sentiment=0,
                avg_confidence=0, avg_return_correct=0,
                avg_return_incorrect=0, profit_factor=0,
            )

        n_resolved = len(resolved)
        n_correct = len(correct)
        hit_rate = n_correct / n_resolved * 100

        # Hit rate por tipo de sinal
        buys = [p for p in resolved if p.signal in ("COMPRA_FORTE", "COMPRA")]
        sells = [p for p in resolved if p.signal in ("VENDA_FORTE", "VENDA")]
        hr_buys = sum(1 for p in buys if p.was_correct) / len(buys) * 100 if buys else 0
        hr_sells = sum(1 for p in sells if p.was_correct) / len(sells) * 100 if sells else 0

        # Hit rate por confiança alta (>70%)
        high_conf = [p for p in resolved if p.confidence > 70]
        hr_high = sum(1 for p in high_conf if p.was_correct) / len(high_conf) * 100 if high_conf else 0

        # Hit rate quando sentimento concorda com sinal
        sentiment_agrees = [
            p for p in resolved
            if (p.signal in ("COMPRA_FORTE", "COMPRA") and p.sentiment == "positivo")
            or (p.signal in ("VENDA_FORTE", "VENDA") and p.sentiment == "negativo")
        ]
        hr_sentiment = sum(1 for p in sentiment_agrees if p.was_correct) / len(sentiment_agrees) * 100 if sentiment_agrees else 0

        # Confiança média
        avg_conf = np.mean([p.confidence for p in resolved])

        # Retornos
        correct_returns = [abs(p.actual_return) for p in correct]
        incorrect_returns = [abs(p.actual_return) for p in resolved if not p.was_correct]
        avg_ret_correct = np.mean(correct_returns) if correct_returns else 0
        avg_ret_incorrect = np.mean(incorrect_returns) if incorrect_returns else 0

        # Profit factor
        total_gain = sum(abs(p.actual_return) for p in correct)
        total_loss = sum(abs(p.actual_return) for p in resolved if not p.was_correct)
        profit_factor = total_gain / total_loss if total_loss > 0 else 0

        # Calibração: agrupa por faixas de confiança e calcula acerto real
        calibration = {}
        for bucket_start in range(0, 100, 10):
            bucket_end = bucket_start + 10
            bucket_preds = [
                p for p in resolved
                if bucket_start <= p.confidence < bucket_end
            ]
            if bucket_preds:
                actual_rate = sum(1 for p in bucket_preds if p.was_correct) / len(bucket_preds) * 100
                calibration[f"{bucket_start}-{bucket_end}%"] = round(actual_rate, 1)

        # Por ativo
        by_symbol = {}
        symbols = set(p.symbol for p in resolved)
        for sym in symbols:
            sym_preds = [p for p in resolved if p.symbol == sym]
            if sym_preds:
                by_symbol[sym] = round(
                    sum(1 for p in sym_preds if p.was_correct) / len(sym_preds) * 100, 1
                )

        # Por mercado
        by_market = defaultdict(lambda: {"correct": 0, "total": 0})
        for p in resolved:
            mk = "Crypto" if "USD" in p.symbol else "B3" if ".SA" in p.symbol else "US"
            by_market[mk]["total"] += 1
            if p.was_correct:
                by_market[mk]["correct"] += 1
        by_market_pct = {
            mk: round(v["correct"] / v["total"] * 100, 1)
            for mk, v in by_market.items() if v["total"] > 0
        }

        # Matriz de confusão simplificada
        confusion = {
            "true_buy": sum(1 for p in resolved if p.signal in ("COMPRA_FORTE", "COMPRA") and p.was_correct),
            "false_buy": sum(1 for p in resolved if p.signal in ("COMPRA_FORTE", "COMPRA") and not p.was_correct),
            "true_sell": sum(1 for p in resolved if p.signal in ("VENDA_FORTE", "VENDA") and p.was_correct),
            "false_sell": sum(1 for p in resolved if p.signal in ("VENDA_FORTE", "VENDA") and not p.was_correct),
            "true_hold": sum(1 for p in resolved if p.signal == "NEUTRO" and p.was_correct),
            "false_hold": sum(1 for p in resolved if p.signal == "NEUTRO" and not p.was_correct),
        }

        return AccuracyMetrics(
            total_predictions=total,
            resolved_predictions=n_resolved,
            correct_predictions=n_correct,
            hit_rate=round(hit_rate, 1),
            hit_rate_buys=round(hr_buys, 1),
            hit_rate_sells=round(hr_sells, 1),
            hit_rate_high_confidence=round(hr_high, 1),
            hit_rate_with_sentiment=round(hr_sentiment, 1),
            avg_confidence=round(avg_conf, 1),
            avg_return_correct=round(avg_ret_correct, 2),
            avg_return_incorrect=round(avg_ret_incorrect, 2),
            profit_factor=round(profit_factor, 2),
            calibration=calibration,
            by_symbol=by_symbol,
            by_market=by_market_pct,
            confusion_matrix=confusion,
        )

    def get_confidence_report(self) -> str:
        """Gera relatório textual de confiança para o terminal."""
        m = self.get_metrics()

        lines = [
            "",
            "═" * 70,
            "  📊 RELATÓRIO DE ACURÁCIA E CONFIANÇA",
            "═" * 70,
            "",
            f"  Predições totais:     {m.total_predictions}",
            f"  Predições resolvidas: {m.resolved_predictions}",
            f"  Acertos:              {m.correct_predictions}",
            "",
            "  ── TAXAS DE ACERTO ────────────────────────",
            f"  Geral:                {m.hit_rate:.1f}%",
            f"  Sinais de COMPRA:     {m.hit_rate_buys:.1f}%",
            f"  Sinais de VENDA:      {m.hit_rate_sells:.1f}%",
            f"  Confiança alta (>70): {m.hit_rate_high_confidence:.1f}%",
            f"  Com sentimento:       {m.hit_rate_with_sentiment:.1f}%",
            "",
            "  ── RETORNOS ──────────────────────────────",
            f"  Retorno médio acertos: {m.avg_return_correct:+.2f}%",
            f"  Retorno médio erros:   {m.avg_return_incorrect:+.2f}%",
            f"  Profit Factor:         {m.profit_factor:.2f}",
            "",
        ]

        if m.calibration:
            lines.append("  ── CALIBRAÇÃO (confiança vs acerto real) ──")
            for bucket, actual in sorted(m.calibration.items()):
                bar_len = int(actual / 2)
                bar = "█" * bar_len + "░" * (50 - bar_len)
                lines.append(f"  {bucket:>10} → {actual:5.1f}% [{bar}]")
            lines.append("")

        if m.by_symbol:
            lines.append("  ── ACERTO POR ATIVO ──────────────────────")
            for sym, rate in sorted(m.by_symbol.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {sym:<12} {rate:5.1f}%")
            lines.append("")

        if m.by_market:
            lines.append("  ── ACERTO POR MERCADO ────────────────────")
            for mk, rate in m.by_market.items():
                lines.append(f"  {mk:<12} {rate:5.1f}%")
            lines.append("")

        if m.confusion_matrix:
            lines.append("  ── MATRIZ DE CONFUSÃO ────────────────────")
            cm = m.confusion_matrix
            lines.append(f"  COMPRA: ✓ {cm.get('true_buy', 0)} | ✗ {cm.get('false_buy', 0)}")
            lines.append(f"  VENDA:  ✓ {cm.get('true_sell', 0)} | ✗ {cm.get('false_sell', 0)}")
            lines.append(f"  NEUTRO: ✓ {cm.get('true_hold', 0)} | ✗ {cm.get('false_hold', 0)}")
            lines.append("")

        lines.append("═" * 70)

        report = "\n".join(lines)
        print(report)
        return report

    def export(self, filepath: str = "accuracy_history.json"):
        """Exporta histórico de predições para JSON."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "metrics": self.get_metrics().to_dict(),
            "predictions": [asdict(p) for p in self.predictions],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Histórico exportado: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AccuracyTracker":
        """Carrega histórico de um JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tracker = cls()
        for p_data in data.get("predictions", []):
            pred = Prediction(**p_data)
            tracker.predictions.append(pred)
            tracker._counter = max(tracker._counter, pred.id)
        return tracker
