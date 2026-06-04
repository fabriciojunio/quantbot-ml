"""Configuração de testes: garante a raiz do projeto no sys.path.

Permite rodar `pytest` a partir de qualquer diretório sem exportar
PYTHONPATH manualmente.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
