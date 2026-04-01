"""
Sistema de logging do QuantBot ML.

Fornece logging estruturado e auditável para todas as operações
do sistema, incluindo trades, sinais ML e alertas de risco.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "quantbot",
    level: int = logging.INFO,
    log_file: str = None,
) -> logging.Logger:
    """
    Configura e retorna um logger padronizado.

    Args:
        name: Nome do logger
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Caminho opcional para arquivo de log

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)

    # Evita handlers duplicados
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Formato padronizado
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (opcional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "quantbot") -> logging.Logger:
    """Retorna logger existente ou cria um novo."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
