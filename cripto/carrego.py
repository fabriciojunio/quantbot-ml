"""
Carrego de funding (cash and carry).

Por que este módulo existe separado do resto: as estratégias de `sinais.py`
tentam adivinhar direção, e adivinhar direção é justamente o que a validação
não conseguiu sustentar. O carrego de funding não adivinha nada. Ele compra o
ativo à vista, vende o mesmo tamanho no perpétuo e fica **neutro em preço** —
o resultado vem só da taxa de financiamento que o perpétuo paga a cada 8 horas.

É a única estratégia acessível a pessoa física em cripto cujo ganho tem uma
origem mecânica identificável: quem está alavancado comprado paga para manter
a posição, e quem está do outro lado recebe.

O que este módulo faz: **mede**. Ele baixa o histórico real de funding,
desconta o custo real das quatro pernas (compra à vista, venda no perpétuo,
e as duas saídas) e diz qual foi o retorno líquido — e em quantos dias a
posição precisa ficar de pé só para pagar a montagem.

O que ele **não** faz: executar. Perpétuo envolve margem e liquidação, e
liquidação com o dinheiro da família não é um risco que se assume por causa de
um número bonito numa tabela.

Riscos que a tabela não mostra e que precisam ser lidos junto:

* **Funding vira.** A taxa pode ficar negativa e você passa a pagar.
* **Base descola.** À vista e perpétuo podem divergir por horas em estresse,
  e a posição "neutra" mostra prejuízo aberto no meio do caminho.
* **Liquidação.** A perna vendida é alavancada. Margem insuficiente numa alta
  forte liquida a perna e desmancha a neutralidade no pior momento.
* **Contraparte.** O dinheiro fica dentro da exchange.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .exchanges.base import http_json, valida_simbolo

logger = logging.getLogger("quantbot.cripto.carrego")

_BASE_FUTUROS = "https://fapi.binance.com/fapi/v1"

# Taxas padrão sem desconto, agosto/2026. Perpétuo é mais barato que spot.
TAXA_SPOT_TAKER = 0.0010
TAXA_PERP_TAKER = 0.0005
TAXA_PERP_MAKER = 0.0002


@dataclass(frozen=True)
class CustoMontagem:
    """Custo das quatro pernas de montar e desmontar o carrego."""

    taxa_spot_taker: float = TAXA_SPOT_TAKER
    taxa_perp_taker: float = TAXA_PERP_TAKER
    spread_pct: float = 0.0002

    def total_pct(self) -> float:
        """Percentual sobre o notional para montar E desmontar a estrutura."""
        por_perna_spot = self.taxa_spot_taker + self.spread_pct / 2
        por_perna_perp = self.taxa_perp_taker + self.spread_pct / 2
        return 2 * (por_perna_spot + por_perna_perp)


def historico_funding(simbolo: str = "BTCUSDT", limite: int = 1000) -> pd.DataFrame:
    """Baixa o histórico real de funding pago pelo perpétuo.

    A Binance liquida funding a cada 8 horas, então 1000 registros são cerca
    de 333 dias.
    """
    bruto = http_json(
        f"{_BASE_FUTUROS}/fundingRate",
        {"symbol": valida_simbolo(simbolo), "limit": min(limite, 1000)},
    )
    if not bruto:
        raise ValueError(f"sem histórico de funding para {simbolo}")

    df = pd.DataFrame(
        {
            "quando": [
                datetime.fromtimestamp(r["fundingTime"] / 1000, tz=timezone.utc) for r in bruto
            ],
            "taxa": [float(r["fundingRate"]) for r in bruto],
            "preco_marca": [float(r.get("markPrice") or 0.0) for r in bruto],
        }
    )
    return df.set_index("quando").sort_index()


def funding_atual(simbolo: str = "BTCUSDT") -> dict:
    """Taxa vigente e próxima liquidação."""
    d = http_json(f"{_BASE_FUTUROS}/premiumIndex", {"symbol": valida_simbolo(simbolo)})
    taxa = float(d["lastFundingRate"])
    return {
        "simbolo": simbolo.upper(),
        "taxa_por_periodo": taxa,
        "taxa_pct_por_periodo": round(taxa * 100, 5),
        "taxa_anualizada_pct": round(taxa * 3 * 365 * 100, 2),  # 3 liquidações por dia
        "preco_marca": float(d["markPrice"]),
        "preco_indice": float(d["indexPrice"]),
        "base_pct": round((float(d["markPrice"]) / float(d["indexPrice"]) - 1) * 100, 4),
        "proxima_liquidacao": datetime.fromtimestamp(
            d["nextFundingTime"] / 1000, tz=timezone.utc
        ).isoformat(),
    }


def analisar(
    simbolo: str = "BTCUSDT",
    limite: int = 1000,
    custo: CustoMontagem | None = None,
    capital_usd: float = 1_000.0,
) -> dict:
    """Mede o carrego realizado no período e o que sobra depois do custo.

    `capital_usd` é o capital total: metade vai para a perna à vista e metade
    para a margem da perna vendida, então o notional efetivo do carrego é
    metade do capital. Ignorar isso é o erro mais comum ao estimar o retorno
    dessa estrutura — a conta parece o dobro do que é.
    """
    custo = custo or CustoMontagem()
    df = historico_funding(simbolo, limite)

    dias = (df.index.max() - df.index.min()).total_seconds() / 86_400
    acumulado = float((1 + df["taxa"]).prod() - 1)
    media = float(df["taxa"].mean())
    negativos = int((df["taxa"] < 0).sum())

    custo_pct = custo.total_pct()
    liquido_pct = acumulado - custo_pct

    # Quantos períodos de 8h a taxa média leva para pagar a montagem.
    periodos_para_pagar = (custo_pct / media) if media > 0 else float("inf")

    notional = capital_usd / 2

    return {
        "simbolo": simbolo.upper(),
        "periodo": [str(df.index.min())[:10], str(df.index.max())[:10]],
        "dias_analisados": round(dias, 1),
        "liquidacoes": len(df),
        "taxa_media_por_periodo_pct": round(media * 100, 5),
        "liquidacoes_negativas": negativos,
        "fatia_negativa_pct": round(negativos / len(df) * 100, 1),
        "carrego_bruto_no_periodo_pct": round(acumulado * 100, 3),
        "carrego_bruto_anualizado_pct": round(acumulado * (365 / dias) * 100, 2) if dias > 0 else 0.0,
        "custo_de_montagem_pct": round(custo_pct * 100, 4),
        "carrego_liquido_no_periodo_pct": round(liquido_pct * 100, 3),
        "carrego_liquido_anualizado_pct": round(liquido_pct * (365 / dias) * 100, 2) if dias > 0 else 0.0,
        "periodos_de_8h_para_pagar_a_montagem": round(periodos_para_pagar, 1)
        if periodos_para_pagar != float("inf")
        else None,
        "dias_para_pagar_a_montagem": round(periodos_para_pagar / 3, 1)
        if periodos_para_pagar != float("inf")
        else None,
        "com_capital_de": {
            "capital_total_usd": capital_usd,
            "notional_do_carrego_usd": round(notional, 2),
            "ganho_liquido_estimado_no_periodo_usd": round(notional * liquido_pct, 2),
            "ganho_liquido_estimado_por_ano_usd": round(
                notional * liquido_pct * (365 / dias), 2
            )
            if dias > 0
            else 0.0,
        },
        "avisos": [
            "Carrego passado não garante carrego futuro: a taxa acompanha a euforia do mercado.",
            f"{negativos} das {len(df)} liquidações foram negativas — nesses períodos quem carrega paga.",
            "A perna vendida é alavancada e pode ser liquidada. Margem folgada não é opcional.",
            "O capital fica dentro da exchange, com o risco de contraparte que isso implica.",
        ],
    }


def comparar(
    simbolos: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"),
    capital_usd: float = 1_000.0,
) -> pd.DataFrame:
    """Tabela comparativa do carrego líquido entre pares."""
    linhas = []
    for simbolo in simbolos:
        try:
            r = analisar(simbolo, capital_usd=capital_usd)
            linhas.append(
                {
                    "simbolo": r["simbolo"],
                    "dias": r["dias_analisados"],
                    "taxa_media_pct": r["taxa_media_por_periodo_pct"],
                    "negativas_pct": r["fatia_negativa_pct"],
                    "bruto_ano_pct": r["carrego_bruto_anualizado_pct"],
                    "liquido_ano_pct": r["carrego_liquido_anualizado_pct"],
                    "ganho_ano_usd": r["com_capital_de"]["ganho_liquido_estimado_por_ano_usd"],
                }
            )
        except Exception as exc:
            logger.warning("Falha em %s: %s", simbolo, exc)
    return pd.DataFrame(linhas)
