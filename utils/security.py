"""Validação de input e sanitização de dados.

Princípio: nunca confiar em dado externo. Todo símbolo, capital, janela e
DataFrame que entra no sistema passa por aqui antes de ser usado.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional

import numpy as np
import pandas as pd

# Símbolos aceitos: letras, dígitos, ponto e hífen (ex: PETR4.SA, BTC-USD).
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,19}$")

_MAX_CAPITAL = 1_000_000_000.0   # R$1 bi — teto sanitário
_MIN_LOOKBACK = 30
_MAX_LOOKBACK = 10_000

_DEFAULT_OHLCV = ["open", "high", "low", "close", "volume"]
_PRICE_COLUMNS = ["open", "high", "low", "close"]


class InputValidator:
    """Valida e normaliza inputs vindos de usuário/configuração."""

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        if not isinstance(symbol, str):
            raise ValueError(f"Símbolo deve ser string, recebido: {type(symbol)}")
        normalized = symbol.strip().upper()
        if not _SYMBOL_RE.match(normalized):
            raise ValueError(f"Símbolo inválido: {symbol!r}")
        return normalized

    @staticmethod
    def validate_symbols(symbols: List[str]) -> List[str]:
        if not symbols:
            raise ValueError("Lista de símbolos vazia")
        return [InputValidator.validate_symbol(s) for s in symbols]

    @staticmethod
    def validate_capital(capital: float) -> float:
        value = float(capital)
        if math.isnan(value):
            raise ValueError("Capital não pode ser NaN")
        if math.isinf(value):
            raise ValueError("Capital não pode ser infinito")
        if value <= 0:
            raise ValueError(f"Capital deve ser positivo, recebido: {value}")
        if value > _MAX_CAPITAL:
            raise ValueError(f"Capital acima do limite permitido: {value}")
        return value

    @staticmethod
    def validate_lookback(days: int) -> int:
        value = int(days)
        if value < _MIN_LOOKBACK:
            raise ValueError(
                f"Lookback muito curto: {value} (mínimo {_MIN_LOOKBACK})"
            )
        if value > _MAX_LOOKBACK:
            raise ValueError(
                f"Lookback muito longo: {value} (máximo {_MAX_LOOKBACK})"
            )
        return value


class DataSanitizer:
    """Limpa e valida DataFrames de mercado."""

    @staticmethod
    def sanitize_dataframe(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        max_missing_pct: float = 0.10,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("DataFrame vazio")

        required = required_columns or _DEFAULT_OHLCV
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes: {missing}")

        clean = df.copy()

        # Substitui infinitos por NaN para tratamento uniforme.
        clean = clean.replace([np.inf, -np.inf], np.nan)

        # Preços negativos não fazem sentido — viram NaN para serem preenchidos.
        for col in _PRICE_COLUMNS:
            if col in clean.columns:
                clean.loc[clean[col] < 0, col] = np.nan

        # Recupera buracos por interpolação temporal simples e remove o resto.
        clean = clean.ffill().bfill().dropna()

        if clean.empty:
            raise ValueError("DataFrame ficou vazio após sanitização")

        return clean

    @staticmethod
    def detect_outliers(data: pd.Series, std_threshold: float = 3.0) -> pd.Series:
        """Marca como True os pontos a mais de N desvios-padrão da média."""
        mean = data.mean()
        std = data.std()
        if std == 0 or pd.isna(std):
            return pd.Series(False, index=data.index)
        z = (data - mean).abs() / std
        return z > std_threshold

    @staticmethod
    def clip_outliers(data: pd.Series, std_threshold: float = 3.0) -> pd.Series:
        """Limita os valores ao intervalo [μ - kσ, μ + kσ]."""
        mean = data.mean()
        std = data.std()
        if std == 0 or pd.isna(std):
            return data.copy()
        lower = mean - std_threshold * std
        upper = mean + std_threshold * std
        return data.clip(lower=lower, upper=upper)


class OperationLimits:
    """Limites operacionais simples (anti-abuso) usados pela camada de risco."""

    def __init__(
        self,
        max_orders_per_day: int = 5,
        max_position_pct: float = 0.10,
        max_capital: float = _MAX_CAPITAL,
    ):
        self.max_orders_per_day = max_orders_per_day
        self.max_position_pct = max_position_pct
        self.max_capital = max_capital

    def within_order_limit(self, orders_today: int) -> bool:
        return orders_today < self.max_orders_per_day

    def within_position_limit(self, position_value: float, capital: float) -> bool:
        if capital <= 0:
            return False
        return (position_value / capital) <= self.max_position_pct
