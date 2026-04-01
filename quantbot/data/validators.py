"""Validação de dados de mercado."""

import numpy as np
import pandas as pd
from typing import Tuple
from utils.logger import get_logger

logger = get_logger("quantbot.data.validators")


class DataValidator:
    """Validações específicas para dados financeiros."""

    @staticmethod
    def check_data_quality(df: pd.DataFrame, symbol: str = "") -> Tuple[bool, str]:
        """
        Verifica a qualidade dos dados.

        Returns:
            Tupla (is_valid, message)
        """
        if df is None or df.empty:
            return False, f"{symbol}: DataFrame vazio"

        issues = []

        # Verifica dados mínimos
        if len(df) < 60:
            issues.append(f"poucos dados ({len(df)} < 60)")

        # Verifica gaps (dias sem dados)
        if hasattr(df.index, 'date'):
            date_diffs = pd.Series(df.index).diff().dt.days
            large_gaps = (date_diffs > 5).sum()
            if large_gaps > 0:
                issues.append(f"{large_gaps} gaps > 5 dias")

        # Verifica preços zerados
        zero_prices = (df["close"] == 0).sum()
        if zero_prices > 0:
            issues.append(f"{zero_prices} preços zerados")

        # Verifica retornos extremos (possível stock split não ajustado)
        returns = df["close"].pct_change().dropna()
        extreme_returns = (returns.abs() > 0.5).sum()
        if extreme_returns > 2:
            issues.append(f"{extreme_returns} retornos > 50% (possível split)")

        if issues:
            msg = f"{symbol}: " + "; ".join(issues)
            logger.warning(f"⚠ Qualidade de dados: {msg}")
            # Retorna True com warning se os problemas não são críticos
            return len(df) >= 60, msg

        return True, f"{symbol}: dados OK ({len(df)} pontos)"
