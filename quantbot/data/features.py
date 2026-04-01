"""
Feature Engineering para modelos de Machine Learning.

Gera 35+ indicadores técnicos a partir de dados OHLCV,
organizados em categorias: retornos, tendência, momentum,
volatilidade, volume e temporais.

Referências:
- Murphy, J. (1999). Technical Analysis of the Financial Markets.
- López de Prado, M. (2018). Advances in Financial ML, Cap. 5.
"""

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.security import DataSanitizer

logger = get_logger("quantbot.features")


class FeatureEngineer:
    """
    Gera features técnicas para os modelos de ML.

    Cada método é documentado com a fórmula e interpretação
    do indicador, facilitando a escrita do TCC.
    """

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todas as features técnicas.

        Args:
            df: DataFrame com colunas [open, high, low, close, volume]

        Returns:
            DataFrame com todas as features calculadas
        """
        if df is None or len(df) < 50:
            raise ValueError(
                f"Dados insuficientes para calcular features: {len(df) if df is not None else 0} pontos"
            )

        features = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── 1. RETORNOS ──────────────────────────────────────
        # Retorno simples: (P_t - P_{t-n}) / P_{t-n}
        features["return_1d"] = close.pct_change(1)
        features["return_5d"] = close.pct_change(5)
        features["return_10d"] = close.pct_change(10)
        features["return_20d"] = close.pct_change(20)

        # Retorno logarítmico: ln(P_t / P_{t-1})
        # Vantagem: aditivo no tempo, aproximadamente normal
        features["log_return"] = np.log(close / close.shift(1))

        # ── 2. MÉDIAS MÓVEIS (Tendência) ─────────────────────
        for window in [5, 10, 20, 50]:
            sma = close.rolling(window).mean()
            ema = close.ewm(span=window, adjust=False).mean()

            # Ratio preço/SMA: >1 = acima da média, <1 = abaixo
            features[f"sma_ratio_{window}"] = close / sma

            # Ratio preço/EMA
            features[f"ema_ratio_{window}"] = close / ema

        # Cross de médias: SMA curta vs SMA longa
        features["sma_cross_5_20"] = (
            close.rolling(5).mean() / close.rolling(20).mean()
        )
        features["sma_cross_10_50"] = (
            close.rolling(10).mean() / close.rolling(50).mean()
        )

        # ── 3. RSI (Relative Strength Index) ─────────────────
        # RSI = 100 - 100/(1 + RS), onde RS = ganho_médio / perda_média
        # RSI < 30: sobrevendido | RSI > 70: sobrecomprado
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features["rsi"] = 100 - (100 / (1 + rs))

        # ── 4. MACD (Moving Average Convergence Divergence) ──
        # MACD = EMA(12) - EMA(26)
        # Signal = EMA(9) do MACD
        # Histograma = MACD - Signal
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        # Normalizado pelo preço para comparabilidade entre ativos
        features["macd_normalized"] = macd / close * 100
        features["macd_signal_normalized"] = macd_signal / close * 100
        features["macd_hist_normalized"] = macd_hist / close * 100

        # ── 5. BOLLINGER BANDS ───────────────────────────────
        # Banda superior = SMA(20) + 2*σ
        # Banda inferior = SMA(20) - 2*σ
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20

        # Largura das bandas (normalizada): mede volatilidade
        features["bb_width"] = (bb_upper - bb_lower) / sma20

        # Posição nas bandas: 0=inferior, 1=superior
        bb_range = bb_upper - bb_lower
        features["bb_position"] = (close - bb_lower) / bb_range.replace(0, np.nan)

        # ── 6. VOLATILIDADE ──────────────────────────────────
        # Volatilidade realizada (anualizada)
        features["volatility_10"] = (
            features["log_return"].rolling(10).std() * np.sqrt(252)
        )
        features["volatility_20"] = (
            features["log_return"].rolling(20).std() * np.sqrt(252)
        )

        # Ratio de volatilidade: curto/longo prazo
        # > 1: volatilidade crescente | < 1: volatilidade diminuindo
        features["volatility_ratio"] = (
            features["volatility_10"] / features["volatility_20"].replace(0, np.nan)
        )

        # ── 7. ATR (Average True Range) ──────────────────────
        # True Range = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)
        # Mede a volatilidade intraday
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()

        # ATR ratio: normalizado pelo preço
        features["atr_ratio"] = atr / close

        # ── 8. VOLUME ────────────────────────────────────────
        # Volume relativo à média de 10 dias
        vol_sma = volume.rolling(10).mean()
        features["volume_ratio"] = volume / vol_sma.replace(0, np.nan)

        # Variação percentual do volume
        features["volume_change"] = volume.pct_change()

        # On-Balance Volume (OBV) - variação de 5 dias
        obv = (np.sign(close.diff()) * volume).cumsum()
        features["obv_change"] = obv.pct_change(5)

        # ── 9. MOMENTUM ─────────────────────────────────────
        # Momentum = P_t / P_{t-n} - 1
        features["momentum_10"] = close / close.shift(10) - 1
        features["momentum_20"] = close / close.shift(20) - 1

        # Rate of Change
        features["roc_5"] = (close - close.shift(5)) / close.shift(5) * 100
        features["roc_10"] = (close - close.shift(10)) / close.shift(10) * 100

        # ── 10. OSCILADORES ──────────────────────────────────
        # Estocástico: %K e %D
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        features["stoch_k"] = (
            100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
        )
        features["stoch_d"] = features["stoch_k"].rolling(3).mean()

        # Williams %R
        features["williams_r"] = (
            -100 * (high_14 - close) / (high_14 - low_14).replace(0, np.nan)
        )

        # ── 11. FEATURES TEMPORAIS ───────────────────────────
        # Codificação cíclica para capturar padrões sazonais
        features["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
        features["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)
        features["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

        # Clip outliers extremos em features
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            features[col] = DataSanitizer.clip_outliers(
                features[col].dropna(), std_threshold=5.0
            ).reindex(features.index)

        # ── 12. COLUNAS RAW PARA ESTRATÉGIAS TÉCNICAS ────────
        # Nomes em maiúsculo para compatibilidade com as estratégias
        # Adicionadas após o clip para não serem alteradas
        features["SMA_20"] = close.rolling(20).mean()
        features["SMA_50"] = close.rolling(50).mean()
        features["RSI"] = 100 - (100 / (1 + rs))
        features["MACD"] = macd
        features["MACD_signal"] = macd_signal
        features["MACD_hist"] = macd_hist

        logger.debug(f"Features calculadas: {len(features.columns)} colunas")
        return features

    @staticmethod
    def create_target(
        df: pd.DataFrame,
        horizon: int = 5,
        threshold: float = 0.0,
    ) -> pd.Series:
        """
        Cria variável alvo para classificação binária.

        Classificação:
        - 1 (BUY): retorno futuro > threshold
        - 0 (SELL/HOLD): retorno futuro <= threshold

        Args:
            df: DataFrame com coluna 'close'
            horizon: Horizonte de predição em dias
            threshold: Limiar de retorno para classificar como BUY

        Returns:
            Series com labels 0/1
        """
        if horizon < 1 or horizon > 60:
            raise ValueError(f"Horizonte deve estar entre 1-60, recebido: {horizon}")

        future_return = df["close"].pct_change(horizon).shift(-horizon)
        target = (future_return > threshold).astype(int)

        logger.debug(
            f"Target criado: horizon={horizon}d, threshold={threshold:.3f}, "
            f"buy_ratio={target.mean():.2%}"
        )

        return target

    @staticmethod
    def get_feature_descriptions() -> dict:
        """
        Retorna descrições de todas as features para documentação.

        Útil para a seção de metodologia do TCC.
        """
        return {
            "return_1d": "Retorno percentual de 1 dia: (P_t - P_{t-1}) / P_{t-1}",
            "return_5d": "Retorno percentual de 5 dias",
            "return_10d": "Retorno percentual de 10 dias",
            "return_20d": "Retorno percentual de 20 dias (aprox. 1 mês)",
            "log_return": "Retorno logarítmico diário: ln(P_t / P_{t-1})",
            "sma_ratio_N": "Razão preço/SMA(N): >1 acima da média, <1 abaixo",
            "ema_ratio_N": "Razão preço/EMA(N): similar ao SMA mas com peso exponencial",
            "rsi": "Relative Strength Index (14 períodos): <30 sobrevendido, >70 sobrecomprado",
            "macd_normalized": "MACD normalizado pelo preço: EMA(12) - EMA(26)",
            "bb_width": "Largura das Bollinger Bands normalizada: mede volatilidade",
            "bb_position": "Posição nas Bollinger Bands: 0=inferior, 1=superior",
            "volatility_N": "Volatilidade realizada anualizada em N dias",
            "volatility_ratio": "Razão vol curto/longo prazo: >1 vol crescente",
            "atr_ratio": "Average True Range / Preço: volatilidade intraday normalizada",
            "volume_ratio": "Volume / SMA(10) do volume: >1 volume acima da média",
            "momentum_N": "Momentum de N dias: P_t / P_{t-N} - 1",
            "stoch_k": "Estocástico %K: posição do preço no range de 14 dias",
            "williams_r": "Williams %R: similar ao estocástico, escala -100 a 0",
            "day_sin/cos": "Codificação cíclica do dia da semana (padrão semanal)",
            "month_sin/cos": "Codificação cíclica do mês (sazonalidade anual)",
        }
