"""
Módulo de segurança e sanitização do QuantBot ML.

Responsável por validar inputs, sanitizar dados e garantir
que o sistema opera dentro de limites seguros.
"""

import re
import numpy as np
import pandas as pd
from typing import Any, List, Optional
from utils.logger import get_logger

logger = get_logger("quantbot.security")


class InputValidator:
    """Valida e sanitiza inputs do sistema."""

    # Símbolos válidos: letras, números, ponto, hífen
    VALID_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9.\-]{1,20}$")

    # Limites de valores numéricos
    MAX_CAPITAL = 1_000_000_000  # 1 bilhão
    MIN_CAPITAL = 100
    MAX_LOOKBACK_DAYS = 3650  # 10 anos
    MIN_LOOKBACK_DAYS = 30

    @classmethod
    def validate_symbol(cls, symbol: str) -> str:
        """
        Valida e sanitiza um símbolo de ativo.

        Args:
            symbol: Símbolo do ativo (ex: "PETR4.SA")

        Returns:
            Símbolo validado

        Raises:
            ValueError: Se o símbolo é inválido
        """
        if not isinstance(symbol, str):
            raise ValueError(f"Símbolo deve ser string, recebido: {type(symbol)}")

        symbol = symbol.strip().upper()

        if not cls.VALID_SYMBOL_PATTERN.match(symbol):
            raise ValueError(
                f"Símbolo inválido: '{symbol}'. "
                "Use apenas letras, números, ponto e hífen (máx 20 chars)."
            )

        return symbol

    @classmethod
    def validate_symbols(cls, symbols: List[str]) -> List[str]:
        """Valida uma lista de símbolos."""
        if not isinstance(symbols, (list, tuple)):
            raise ValueError("Símbolos devem ser uma lista")
        if len(symbols) == 0:
            raise ValueError("Lista de símbolos vazia")
        if len(symbols) > 100:
            raise ValueError(f"Máximo de 100 símbolos, recebido: {len(symbols)}")

        return [cls.validate_symbol(s) for s in symbols]

    @classmethod
    def validate_capital(cls, capital: float) -> float:
        """Valida o capital inicial."""
        if not isinstance(capital, (int, float)):
            raise ValueError(f"Capital deve ser numérico, recebido: {type(capital)}")
        if capital < cls.MIN_CAPITAL:
            raise ValueError(f"Capital mínimo: ${cls.MIN_CAPITAL}")
        if capital > cls.MAX_CAPITAL:
            raise ValueError(f"Capital máximo: ${cls.MAX_CAPITAL:,.0f}")
        if np.isnan(capital) or np.isinf(capital):
            raise ValueError("Capital não pode ser NaN ou infinito")

        return float(capital)

    @classmethod
    def validate_lookback(cls, days: int) -> int:
        """Valida o período de lookback."""
        if not isinstance(days, int):
            raise ValueError(f"Lookback deve ser inteiro, recebido: {type(days)}")
        if days < cls.MIN_LOOKBACK_DAYS:
            raise ValueError(f"Lookback mínimo: {cls.MIN_LOOKBACK_DAYS} dias")
        if days > cls.MAX_LOOKBACK_DAYS:
            raise ValueError(f"Lookback máximo: {cls.MAX_LOOKBACK_DAYS} dias")

        return days


class DataSanitizer:
    """Sanitiza e valida dados de mercado."""

    @staticmethod
    def sanitize_dataframe(
        df: pd.DataFrame,
        required_columns: List[str] = None,
        max_missing_pct: float = 0.05,
    ) -> pd.DataFrame:
        """
        Sanitiza um DataFrame de dados de mercado.

        Verifica:
        - Colunas obrigatórias presentes
        - Valores negativos em preços
        - Valores infinitos
        - Percentual de dados faltantes
        - Outliers extremos

        Args:
            df: DataFrame com dados de mercado
            required_columns: Colunas que devem estar presentes
            max_missing_pct: Percentual máximo de dados faltantes

        Returns:
            DataFrame sanitizado

        Raises:
            ValueError: Se os dados são inválidos ou muito incompletos
        """
        if df is None or df.empty:
            raise ValueError("DataFrame vazio ou None")

        if required_columns is None:
            required_columns = ["open", "high", "low", "close", "volume"]

        # Verifica colunas obrigatórias
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colunas ausentes: {missing_cols}")

        df = df.copy()

        # Remove linhas completamente vazias
        df = df.dropna(how="all")

        # Verifica percentual de faltantes
        missing_pct = df[required_columns].isna().mean().max()
        if missing_pct > max_missing_pct:
            logger.warning(
                f"Dados faltantes ({missing_pct:.1%}) excedem limite "
                f"({max_missing_pct:.1%}). Preenchendo com forward fill."
            )

        # Forward fill para dados faltantes (padrão em séries financeiras)
        df[required_columns] = df[required_columns].ffill()

        # Remove infinitos
        for col in required_columns:
            if df[col].dtype in [np.float64, np.float32, np.int64]:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    logger.warning(f"Removendo {inf_count} valores infinitos em '{col}'")
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df[col] = df[col].ffill()

        # Verifica preços negativos
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        for col in price_cols:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                logger.warning(f"Removendo {neg_count} preços negativos em '{col}'")
                df.loc[df[col] < 0, col] = np.nan
                df[col] = df[col].ffill()

        # Verifica volume negativo
        if "volume" in df.columns:
            df.loc[df["volume"] < 0, "volume"] = 0

        # Remove linhas que ainda têm NaN após tratamento
        remaining_nan = df[required_columns].isna().sum().sum()
        if remaining_nan > 0:
            df = df.dropna(subset=required_columns)
            logger.info(f"Removidas linhas com NaN restantes: {remaining_nan}")

        return df

    @staticmethod
    def detect_outliers(
        series: pd.Series, std_threshold: float = 5.0
    ) -> pd.Series:
        """
        Detecta outliers usando z-score.

        Args:
            series: Série de dados
            std_threshold: Número de desvios padrão para considerar outlier

        Returns:
            Série booleana indicando outliers
        """
        if series.std() == 0:
            return pd.Series(False, index=series.index)

        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > std_threshold

    @staticmethod
    def clip_outliers(
        series: pd.Series, std_threshold: float = 5.0
    ) -> pd.Series:
        """Substitui outliers pelo valor no limiar."""
        mean = series.mean()
        std = series.std()
        if std == 0:
            return series
        lower = mean - std_threshold * std
        upper = mean + std_threshold * std
        return series.clip(lower, upper)


class OperationLimits:
    """Limites operacionais de segurança."""

    MAX_SYMBOLS_PER_RUN = 50
    MAX_FEATURES = 100
    MAX_TRAINING_TIME_SECONDS = 300
    MIN_SAMPLES_FOR_PREDICTION = 30

    @classmethod
    def check_feature_count(cls, n_features: int) -> None:
        """Verifica se o número de features está dentro do limite."""
        if n_features > cls.MAX_FEATURES:
            raise ValueError(
                f"Número de features ({n_features}) excede o limite "
                f"({cls.MAX_FEATURES}). Reduza a dimensionalidade."
            )

    @classmethod
    def check_sample_count(cls, n_samples: int, context: str = "") -> None:
        """Verifica se há amostras suficientes para predição."""
        if n_samples < cls.MIN_SAMPLES_FOR_PREDICTION:
            raise ValueError(
                f"Amostras insuficientes ({n_samples}) para {context}. "
                f"Mínimo: {cls.MIN_SAMPLES_FOR_PREDICTION}."
            )
