"""
Modelo de custo de execução.

Este é o módulo mais importante do pacote, e é exatamente o que faltava no
painel anterior: lá a posição abria no preço de tela e fechava no preço de
tela, sem taxa, sem spread e sem slippage. Com custo zero, qualquer alvo
pequeno vira "lucro" e a curva de capital sobe sozinha.

Cobramos três coisas em toda entrada e toda saída:

1. **Taxa** — percentual da corretora sobre o notional (maker ou taker).
2. **Spread** — metade da distância bid/ask, porque quem agride paga o lado
   ruim do book.
3. **Slippage** — deslocamento adicional quando a ordem é grande perto da
   liquidez disponível. Modelado como raiz da participação no book, que é a
   forma funcional usual de impacto de mercado.

O número que interessa é `custo_ida_volta_pct`: o piso de retorno que a
operação precisa entregar só para empatar.
"""

from __future__ import annotations

from dataclasses import dataclass

# Taxas spot públicas, sem desconto de token nem nível VIP (agosto/2026).
TAXAS_SPOT = {
    "binance": {"maker": 0.0010, "taker": 0.0010},
    "bybit": {"maker": 0.0010, "taker": 0.0010},
    "gateio": {"maker": 0.0020, "taker": 0.0020},
    "mercadobitcoin": {"maker": 0.0030, "taker": 0.0070},
    "foxbit": {"maker": 0.0025, "taker": 0.0050},
}


@dataclass(frozen=True)
class ModeloCusto:
    """Parâmetros de custo de um par em uma corretora.

    `spread_pct` é o spread relativo médio observado no book (ida e volta
    completa custa `spread_pct`, metade em cada ponta). `profundidade_usd` é
    quanto de notional existe perto do topo do book — usado para estimar o
    impacto de ordens grandes.
    """

    taxa_maker: float = 0.0010
    taxa_taker: float = 0.0010
    spread_pct: float = 0.0002
    profundidade_usd: float = 250_000.0
    impacto_base: float = 0.0005

    @classmethod
    def da_exchange(cls, nome: str, **kwargs) -> "ModeloCusto":
        taxas = TAXAS_SPOT.get(nome.lower(), TAXAS_SPOT["binance"])
        return cls(taxa_maker=taxas["maker"], taxa_taker=taxas["taker"], **kwargs)

    def slippage_pct(self, notional_usd: float) -> float:
        """Impacto extra estimado para uma ordem de `notional_usd`.

        Cresce com a raiz da participação na liquidez do topo do book: dobrar
        o tamanho não dobra o impacto, mas ordens grandes pagam caro.
        """
        if notional_usd <= 0 or self.profundidade_usd <= 0:
            return 0.0
        participacao = notional_usd / self.profundidade_usd
        return self.impacto_base * (participacao**0.5)

    def custo_ponta_pct(self, notional_usd: float, taker: bool = True) -> float:
        """Custo percentual de UMA ponta (só entrada, ou só saída)."""
        taxa = self.taxa_taker if taker else self.taxa_maker
        return taxa + (self.spread_pct / 2) + self.slippage_pct(notional_usd)

    def custo_ponta_usd(self, notional_usd: float, taker: bool = True) -> float:
        return notional_usd * self.custo_ponta_pct(notional_usd, taker)

    def custo_ida_volta_pct(self, notional_usd: float, taker_entrada: bool = True, taker_saida: bool = True) -> float:
        """Percentual que a operação tem que render só para empatar."""
        return self.custo_ponta_pct(notional_usd, taker_entrada) + self.custo_ponta_pct(
            notional_usd, taker_saida
        )

    def custo_ida_volta_usd(self, notional_usd: float, **kwargs) -> float:
        return notional_usd * self.custo_ida_volta_pct(notional_usd, **kwargs)

    def preco_efetivo(self, preco: float, notional_usd: float, comprando: bool, taker: bool = True) -> float:
        """Preço realmente pago/recebido, já com spread e slippage embutidos.

        Comprando, você paga acima do meio; vendendo, recebe abaixo. A taxa é
        cobrada à parte porque sai em moeda, não em preço.
        """
        desloc = (self.spread_pct / 2) + self.slippage_pct(notional_usd)
        return preco * (1 + desloc) if comprando else preco * (1 - desloc)

    def acerto_minimo(self, ganho_pct: float, perda_pct: float, notional_usd: float) -> float:
        """Taxa de acerto mínima para empatar, dado alvo e stop em percentual.

        Resolve `p*G - (1-p)*L - custo = 0`, com custo já somado nas duas
        pontas. Devolve valor acima de 1.0 quando nem 100% de acerto empata —
        é o caso do alvo de dez centavos do painel antigo.
        """
        custo = self.custo_ida_volta_pct(notional_usd)
        g = ganho_pct - custo
        l = perda_pct + custo
        # Se nem o acerto rende mais que o custo, não existe taxa de acerto
        # que salve: mesmo ganhando sempre a operação sai no prejuízo.
        if g <= 0 or (g + l) <= 0:
            return float("inf")
        return l / (g + l)


def resumo_viabilidade(
    alvo_usd: float,
    notional_usd: float,
    modelo: ModeloCusto | None = None,
) -> dict:
    """Compara um alvo em dólares com o custo de fazer a operação.

    É o teste direto da regra "fecha a posição em +US$ 0,10" do painel antigo.
    """
    modelo = modelo or ModeloCusto()
    custo_usd = modelo.custo_ida_volta_usd(notional_usd)
    alvo_pct = (alvo_usd / notional_usd) if notional_usd > 0 else 0.0
    custo_pct = modelo.custo_ida_volta_pct(notional_usd)

    return {
        "notional_usd": round(notional_usd, 2),
        "alvo_usd": round(alvo_usd, 4),
        "alvo_pct": round(alvo_pct * 100, 5),
        "custo_usd": round(custo_usd, 4),
        "custo_pct": round(custo_pct * 100, 5),
        "resultado_liquido_usd": round(alvo_usd - custo_usd, 4),
        "viavel": alvo_usd > custo_usd,
        "quantas_vezes_o_alvo_o_custo_come": round(custo_usd / alvo_usd, 2) if alvo_usd > 0 else float("inf"),
    }
