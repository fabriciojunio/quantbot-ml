"""
Script para iniciar o servidor da API.

Uso:
    python -m api.run
    # ou
    python api/run.py
"""

import sys
import os

# Adiciona o diretório raiz do quantbot ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn


def main():
    print("=" * 50)
    print("  QuantBot ML — API Server")
    print("  http://localhost:8000")
    print("  Docs: http://localhost:8000/api/docs")
    print("=" * 50)

    uvicorn.run(
        "api.server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
