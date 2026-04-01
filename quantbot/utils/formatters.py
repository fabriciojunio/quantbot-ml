"""Funções de formatação para exibição de dados."""

from typing import Union


def fmt_currency(value: float, prefix: str = "$") -> str:
    """Formata valor monetário."""
    return f"{prefix}{value:,.2f}"


def fmt_pct(value: float, with_sign: bool = True) -> str:
    """Formata percentual."""
    if with_sign:
        return f"{value:+.2f}%"
    return f"{value:.2f}%"


def fmt_number(value: Union[int, float], decimals: int = 2) -> str:
    """Formata número genérico."""
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{decimals}f}"
