"""
Custo de operar na B3 para pessoa física.

Corretagem virou zero na maioria das corretoras de varejo, e isso fez muita
gente concluir que operar ficou de graça. Não ficou. O que sobrou:

* **Emolumentos e liquidação da B3** — 0,0325% por ponta em operação normal,
  cobrado sempre, em qualquer corretora.
* **Spread do book** — quem compra a mercado paga o ask e quem vende recebe o
  bid. Em BOVA11 isso é 0,04%; em um ETF pouco negociado passa de 0,20%.
  É o custo maior, e é o único que não aparece em nota de corretagem.
* **Taxa de administração do ETF** — 0,30% ao ano no BOVA11, cobrada dentro
  da cota, portanto invisível no extrato e presente todo dia.
* **Custódia** — zerada nas corretoras de varejo, mantida no modelo por
  completude.

O parâmetro que mais muda o resultado final não é nenhum destes: é **quantas
vezes por ano a carteira gira**. Um custo de ida e volta de 0,3% aplicado
mensalmente vira 3,6% ao ano — mais que o dobro do que um ETF cobra de
administração.
"""

from __future__ import annotations

from dataclasses import dataclass

from .universo import Ativo

# Emolumentos + liquidação da B3, mercado à vista, pessoa física (2026).
EMOLUMENTOS_PCT = 0.0325 / 100

# Corretagem típica em corretora de varejo. Deixe diferente de zero para
# simular corretora que ainda cobra.
CORRETAGEM_PADRAO_BRL = 0.0

# Taxa de administração anual por tipo de ETF, cobrada dentro da cota.
TAXA_ADM_ANUAL = {
    "BOVA11.SA": 0.0030,
    "IVVB11.SA": 0.0023,
    "SMAL11.SA": 0.0050,
    "DIVO11.SA": 0.0050,
    "IMAB11.SA": 0.0025,
    "B5P211.SA": 0.0020,
    "FIXA11.SA": 0.0030,
    "XFIX11.SA": 0.0050,
    "GOLD11.SA": 0.0050,
}


@dataclass(frozen=True)
class ModeloCustoB3:
    emolumentos_pct: float = EMOLUMENTOS_PCT
    corretagem_brl: float = CORRETAGEM_PADRAO_BRL
    custodia_mensal_brl: float = 0.0

    def custo_ponta(self, ativo: Ativo, valor_brl: float) -> float:
        """Custo em reais de uma ponta (só a compra, ou só a venda)."""
        if valor_brl <= 0:
            return 0.0
        return (
            valor_brl * self.emolumentos_pct
            + valor_brl * ativo.custo_meio_spread
            + self.corretagem_brl
        )

    def custo_ponta_pct(self, ativo: Ativo, valor_brl: float) -> float:
        if valor_brl <= 0:
            return 0.0
        return self.custo_ponta(ativo, valor_brl) / valor_brl

    def custo_ida_volta(self, ativo: Ativo, valor_brl: float) -> float:
        return 2 * self.custo_ponta(ativo, valor_brl)

    def preco_efetivo(self, ativo: Ativo, preco: float, comprando: bool) -> float:
        """Preço realmente executado, com o meio spread do lado certo."""
        d = ativo.custo_meio_spread
        return preco * (1 + d) if comprando else preco * (1 - d)

    def taxa_administracao_diaria(self, ticker: str) -> float:
        """Fatia diária da taxa de administração, para descontar do retorno."""
        return TAXA_ADM_ANUAL.get(ticker, 0.0) / 252

    def custo_anual_do_giro(self, ativo: Ativo, valor_brl: float, giros_por_ano: float) -> dict:
        """Quanto o giro custa por ano, em reais e em percentual da carteira.

        É a conta que decide entre rebalancear todo mês e rebalancear por banda.
        """
        if giros_por_ano < 0:
            raise ValueError("giros por ano não pode ser negativo")
        por_giro = self.custo_ida_volta(ativo, valor_brl)
        anual = por_giro * giros_por_ano
        return {
            "ticker": ativo.ticker,
            "valor_operado": round(valor_brl, 2),
            "custo_por_giro_brl": round(por_giro, 2),
            "custo_por_giro_pct": round(por_giro / valor_brl * 100, 4) if valor_brl else 0.0,
            "giros_por_ano": giros_por_ano,
            "custo_anual_brl": round(anual, 2),
            "custo_anual_pct": round(anual / valor_brl * 100, 3) if valor_brl else 0.0,
            "taxa_adm_anual_pct": round(TAXA_ADM_ANUAL.get(ativo.ticker, 0.0) * 100, 3),
        }


def comparar_frequencias(
    ativo: Ativo,
    valor_brl: float,
    modelo: ModeloCustoB3 | None = None,
    # Ordenado do mais caro para o mais barato, para a tabela poder ser lida
    # de cima para baixo sem sustos.
    frequencias: tuple[tuple[str, float], ...] = (
        ("diário", 252),
        ("semanal", 52),
        ("mensal", 12),
        ("trimestral", 4),
        ("semestral", 2),
        ("por banda (~2x/ano)", 2),
        ("anual", 1),
        ("nunca", 0),
    ),
) -> list[dict]:
    """Tabela do custo anual por frequência de rebalanceamento.

    Serve para mostrar, com número, por que o giro é o inimigo silencioso de
    quem tem pouco capital.
    """
    modelo = modelo or ModeloCustoB3()
    return [
        {"frequencia": nome, **modelo.custo_anual_do_giro(ativo, valor_brl, giros)}
        for nome, giros in frequencias
    ]
