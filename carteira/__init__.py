"""
Motor de carteira sistemática.

Este pacote nasceu de duas descobertas feitas em 23/08/2026, medindo dois
sistemas que pareciam funcionar:

1. O painel Sentinela operava com book, tape e indicadores sintéticos, e o
   simulador não cobrava taxa. Medido com dado real e custo real, a estratégia
   não tinha direção nenhuma: 22.568 operações, resultado bruto de −1,5% e
   custo de 16,6 vezes o bruto (ver `docs/AUDITORIA_SENTINELA.md`).
2. O próprio paper trading de FIIs do QuantBot marcava +17,14% contra CDI de
   +2,86%. O accrual de provento era quadrático — em 61 dias lançava 14 vezes
   o rendimento devido. Corrigido, o mesmo período dá **−6,32%**
   (`scripts/auditar_paper_book.py`).

A conclusão prática das duas: **prever preço no curto prazo não paga**, e o
que faz diferença é o que dá para controlar. Este pacote foca exatamente no
que é controlável e mensurável:

* **Custo** — emolumentos, spread e corretagem descontados em toda operação.
* **Imposto** — o motor de IR brasileiro em `tributos.py`. Isenção de provento
  de FII, isenção de R$ 20 mil/mês em ações, come-cotas em fundos, 15% em ETF.
  Isso não é previsão: é lei, e é a única vantagem mecânica de verdade
  disponível para pessoa física.
* **Giro** — rebalanceamento por banda em vez de calendário, porque cada
  operação evitada é custo e imposto que não acontecem.
* **Aporte** — `aportes.py` mede quanto do resultado vem de estratégia e
  quanto vem de quanto se poupa. A resposta costuma constranger a estratégia.
* **Comparação honesta** — todo resultado sai ao lado de CDI, Ibovespa e
  carteira parada. Estratégia que não bate as três não é estratégia.

Nada aqui promete retorno. O pacote mede, desconta o que precisa ser
descontado e mostra o número que sobra.
"""

__all__ = [
    "dados",
    "universo",
    "tributos",
    "custos_b3",
    "sinais",
    "alocacao",
    "rebalanceamento",
    "backtest",
    "validacao",
    "aportes",
    "relatorio",
]

__version__ = "0.1.0"
