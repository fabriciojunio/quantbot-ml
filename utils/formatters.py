"""Formatação de valores monetários e percentuais para relatórios."""

from __future__ import annotations


def fmt_currency(value: float, symbol: str = "$") -> str:
    """Formata um valor monetário com separador de milhar e 2 casas.

    >>> fmt_currency(1234.56)
    '$1,234.56'
    >>> fmt_currency(0)
    '$0.00'
    """
    return f"{symbol}{value:,.2f}"


def fmt_pct(value: float, decimals: int = 2, with_sign: bool = True) -> str:
    """Formata um percentual (valor já em pontos percentuais).

    >>> fmt_pct(12.34)
    '+12.34%'
    >>> fmt_pct(-5.67)
    '-5.67%'
    >>> fmt_pct(12.34, with_sign=False)
    '12.34%'
    """
    if with_sign:
        return f"{value:+.{decimals}f}%"
    return f"{value:.{decimals}f}%"
