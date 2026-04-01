"""
Triple Barrier Method + Meta-Labeling — López de Prado (2018).

O método padrão de labeling (preço subiu em N dias?) ignora o caminho
do preço. O Triple Barrier simula condições reais de trade:
    - Barreira superior: take-profit atingido → label +1
    - Barreira inferior: stop-loss atingido → label -1
    - Barreira vertical: tempo expirou → label 0

Meta-Labeling: modelo secundário que aprende QUANDO agir nos sinais
do modelo primário. Melhora precision sem sacrificar recall.

Resultado comprovado (Hudson & Thames, Singh & Joubert):
    - Melhora F1-Score em 15-25%
    - Reduz falsos positivos em 30-40%
    - Sharpe Ratio sobe em média 37%

Referências:
    - López de Prado (2018) — Advances in Financial ML, Cap. 3
    - Singh & Joubert (2020) — Does Meta-Labeling Add to Signal Efficacy?
    - Hudson & Thames — mlfinlab implementation
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple

logger = logging.getLogger("quantbot.models.triple_barrier")


def get_daily_volatility(close: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Calcula volatilidade diária rolling (desvio padrão dos retornos log).

    Usada para definir barreiras dinâmicas — barreiras mais largas
    em períodos voláteis, mais apertadas em períodos calmos.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=lookback).std()


def triple_barrier_labels(
    df: pd.DataFrame,
    volatility: pd.Series,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 10,
    close_col: str = "Close",
) -> pd.Series:
    """
    Aplica o Triple Barrier Method de López de Prado.

    Para cada ponto no tempo, define 3 barreiras:
        - Take-Profit: close * (1 + tp_mult * volatility)
        - Stop-Loss: close * (1 - sl_mult * volatility)
        - Tempo máximo: max_holding barras

    Args:
        df: DataFrame com preços
        volatility: Série de volatilidade diária
        tp_mult: Multiplicador para take-profit (default 2.0 = 2x vol)
        sl_mult: Multiplicador para stop-loss (default 1.0 = 1x vol)
        max_holding: Máximo de períodos mantendo posição

    Returns:
        Series com labels: +1 (TP atingido), -1 (SL atingido), 0 (expirou)
    """
    close = df[close_col]
    labels = pd.Series(0, index=df.index, dtype=int)

    for i in range(len(df) - 1):
        entry_price = close.iloc[i]
        vol = volatility.iloc[i]

        if pd.isna(vol) or vol <= 0:
            continue

        # Barreiras dinâmicas baseadas na volatilidade
        upper_barrier = entry_price * (1 + tp_mult * vol)
        lower_barrier = entry_price * (1 - sl_mult * vol)

        # Janela de observação
        end_idx = min(i + max_holding, len(df) - 1)
        future_prices = close.iloc[i + 1:end_idx + 1]

        if future_prices.empty:
            continue

        # Verifica qual barreira é tocada primeiro
        tp_hit = future_prices >= upper_barrier
        sl_hit = future_prices <= lower_barrier

        tp_idx = tp_hit.idxmax() if tp_hit.any() else None
        sl_idx = sl_hit.idxmax() if sl_hit.any() else None

        if tp_idx is not None and sl_idx is not None:
            # Ambas tocadas — qual veio primeiro?
            if tp_idx <= sl_idx:
                labels.iloc[i] = 1
            else:
                labels.iloc[i] = -1
        elif tp_idx is not None:
            labels.iloc[i] = 1
        elif sl_idx is not None:
            labels.iloc[i] = -1
        # else: expirou → fica 0

    return labels


def meta_labels(
    primary_signal: pd.Series,
    triple_barrier_label: pd.Series,
) -> pd.Series:
    """
    Gera meta-labels para o modelo secundário.

    O meta-label é 1 quando o modelo primário ACERTOU
    (sinal e resultado concordam), 0 quando errou.

    Isso permite treinar um modelo que aprende QUANDO
    confiar no sinal primário.

    Args:
        primary_signal: Sinais do modelo primário (+1, -1, 0)
        triple_barrier_label: Labels do triple barrier (+1, -1, 0)

    Returns:
        Series com meta-labels: 1 (acertou) ou 0 (errou/neutro)
    """
    meta = pd.Series(0, index=primary_signal.index, dtype=int)

    # Sinal de compra + preço subiu = acertou
    buy_correct = (primary_signal > 0) & (triple_barrier_label > 0)
    # Sinal de venda + preço caiu = acertou
    sell_correct = (primary_signal < 0) & (triple_barrier_label < 0)

    meta[buy_correct | sell_correct] = 1

    return meta


class TripleBarrierLabeler:
    """
    Classe completa que combina Triple Barrier + Meta-Labeling.

    Uso:
        labeler = TripleBarrierLabeler(tp_mult=2.0, sl_mult=1.0)
        df = labeler.fit_transform(df, primary_signals)
        # df agora tem colunas: 'tb_label', 'meta_label', 'tb_volatility'
    """

    def __init__(
        self,
        tp_mult: float = 2.0,
        sl_mult: float = 1.0,
        max_holding: int = 10,
        vol_lookback: int = 20,
    ):
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.max_holding = max_holding
        self.vol_lookback = vol_lookback

    def fit_transform(
        self,
        df: pd.DataFrame,
        primary_signals: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Aplica Triple Barrier e opcionalmente Meta-Labeling.

        Args:
            df: DataFrame com coluna 'Close'
            primary_signals: Sinais do modelo primário (opcional)

        Returns:
            DataFrame com colunas adicionais
        """
        df = df.copy()

        # Volatilidade dinâmica
        vol = get_daily_volatility(df["Close"], self.vol_lookback)
        df["tb_volatility"] = vol

        # Triple Barrier Labels
        df["tb_label"] = triple_barrier_labels(
            df, vol,
            tp_mult=self.tp_mult,
            sl_mult=self.sl_mult,
            max_holding=self.max_holding,
        )

        # Meta-Labels (se sinais primários fornecidos)
        if primary_signals is not None:
            df["meta_label"] = meta_labels(primary_signals, df["tb_label"])

        # Estatísticas
        total = len(df)
        tp_count = (df["tb_label"] == 1).sum()
        sl_count = (df["tb_label"] == -1).sum()
        neutral = (df["tb_label"] == 0).sum()

        logger.info(
            f"Triple Barrier: TP={tp_count} ({tp_count/total:.1%}), "
            f"SL={sl_count} ({sl_count/total:.1%}), "
            f"Neutro={neutral} ({neutral/total:.1%})"
        )

        return df
