"""Logging estruturado e reutilizável.

`get_logger` devolve um logger nomeado já configurado com um handler de
console e formato padronizado. Idempotente: chamar várias vezes com o mesmo
nome não duplica handlers.
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Retorna um logger configurado.

    Args:
        name: nome hierárquico do logger (ex: "quantbot.data").
        level: nível opcional; default vem de QUANTBOT_LOG_LEVEL ou INFO.
    """
    logger = logging.getLogger(name)

    if level is None:
        env_level = os.getenv("QUANTBOT_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)
    logger.setLevel(level)

    # Evita duplicar handlers quando chamado repetidamente.
    if not any(getattr(h, "_quantbot", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DATE_FORMAT))
        handler._quantbot = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    logger.propagate = False
    return logger
