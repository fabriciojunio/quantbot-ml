"""
Quanto o dinheiro parado está pagando, e o que isso faz com a decisão.

Este módulo nasceu de um achado que só apareceu quando o histórico foi
estendido de 23 para 26 anos e o imposto do caixa foi corrigido. Com janelas
móveis de 5 anos e 60% de risco:

    janelas começando entre 2000 e 2012:  perdeu do CDI em 52 de 52
    janelas começando entre 2013 e 2021:  ganhou do CDI em 33 de 33

Cinquenta e duas derrotas seguidas e trinta e três vitórias seguidas não é
ruído, e não é o nível de risco: o resultado foi igual em 20% e em 100% de
risco. O que separa as duas épocas é uma coisa só, e está na série do Banco
Central: o CDI rendeu entre 15% e 23% ao ano de 2000 a 2006, e entre 3% e 14%
de 2013 em diante.

A conclusão é aritmética e desconfortável: **a carteira de risco só bate a
renda fixa quando a renda fixa está barata.** Não é opinião sobre bolsa, é o
que sobra quando o ativo sem risco paga quase o que o ativo com risco paga.

O que interessa não é o CDI nominal e sim o **juro real líquido de imposto**:
o que sobra depois da inflação e depois da Receita. É esse número que compete
com o retorno esperado de uma carteira de ações, que também é real.

Duas coisas que este módulo deliberadamente **não** faz:

* **Não prevê juro.** Ele lê o juro que existe hoje. Prever Selic é outro
  problema, e um que ninguém resolve.
* **Não é parâmetro ajustado.** O ponto de corte vem da estimativa de retorno
  real de bolsa no longo prazo, que é literatura, não busca em grade. Ajustar
  esse número contra o próprio histórico seria exatamente o erro que a
  calibração deste projeto foi construída para reprovar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .tributos import aliquota_regressiva

logger = logging.getLogger("quantbot.carteira.juro_real")

# Retorno real de longo prazo de bolsa diversificada, já descontada inflação.
# Faixa consolidada na literatura de prêmio de risco para mercados
# desenvolvidos e emergentes ao longo de mais de um século. Fica aqui como
# constante e NÃO é ajustado contra o histórico deste repositório — o dia em
# que for, vira número escolhido para justificar a conclusão desejada.
RETORNO_REAL_DE_BOLSA = 0.065

# Prêmio mínimo que a bolsa precisa oferecer sobre o juro real líquido para
# valer o risco. Abaixo disso não compensa aceitar queda de 20% para ganhar
# o que o CDB paga sem queda nenhuma.
PREMIO_MINIMO = 0.015

# Prazo usado para a alíquota do juro real líquido. Dinheiro de reserva e de
# carteira de longo prazo passa de dois anos, então 15% é a alíquota certa.
PRAZO_LONGO_DIAS = 800


@dataclass(frozen=True)
class Leitura:
    """O que o juro está pagando agora, e o que isso implica."""

    cdi_anual: float
    ipca_12m: float
    juro_real_bruto: float
    juro_real_liquido: float
    premio_da_bolsa: float
    teto_de_risco_sugerido: float
    veredito: str

    def resumo(self) -> dict:
        return {
            "cdi_anual_pct": round(self.cdi_anual * 100, 2),
            "ipca_12m_pct": round(self.ipca_12m * 100, 2),
            "juro_real_bruto_pct": round(self.juro_real_bruto * 100, 2),
            "juro_real_liquido_pct": round(self.juro_real_liquido * 100, 2),
            "premio_da_bolsa_pp": round(self.premio_da_bolsa * 100, 2),
            "teto_de_risco_sugerido_pct": round(self.teto_de_risco_sugerido, 1),
            "veredito": self.veredito,
        }


def fisher(nominal: float, inflacao: float) -> float:
    """Juro real pela relação de Fisher, não pela subtração.

    Subtrair dá erro grande justamente onde importa: com CDI de 23% e IPCA de
    9%, a subtração diz 14% e a conta certa diz 12,8%. Um ponto e meio de
    diferença muda a decisão perto do limite.
    """
    if inflacao <= -1:
        raise ValueError("inflação implausível")
    return (1 + nominal) / (1 + inflacao) - 1


def liquido_de_ir(juro_nominal: float, dias: int = PRAZO_LONGO_DIAS) -> float:
    """O juro nominal depois da mordida da Receita."""
    if juro_nominal <= 0:
        return juro_nominal
    return juro_nominal * (1 - aliquota_regressiva(dias))


def ler(
    cdi_anual: float,
    ipca_12m: float,
    retorno_real_de_bolsa: float = RETORNO_REAL_DE_BOLSA,
    premio_minimo: float = PREMIO_MINIMO,
    prazo_dias: int = PRAZO_LONGO_DIAS,
) -> Leitura:
    """Lê o juro de hoje e diz quanto de risco ele justifica.

    O teto sugerido cai conforme o prêmio da bolsa some. Ele é **teto**, não
    alvo: quem escolheu 40% de risco continua com 40% mesmo em juro baixo. A
    regra só impede correr risco alto quando o risco não está sendo pago.
    """
    real_bruto = fisher(cdi_anual, ipca_12m)
    real_liquido = fisher(liquido_de_ir(cdi_anual, prazo_dias), ipca_12m)
    premio = retorno_real_de_bolsa - real_liquido

    if premio >= premio_minimo * 3:
        teto, veredito = 100.0, (
            f"a renda fixa paga {real_liquido * 100:.1f}% real líquido e a bolsa espera "
            f"{retorno_real_de_bolsa * 100:.1f}%: prêmio de {premio * 100:.1f} pontos, "
            f"que é o que justifica correr risco"
        )
    elif premio >= premio_minimo:
        teto, veredito = 70.0, (
            f"a renda fixa paga {real_liquido * 100:.1f}% real líquido contra os "
            f"{retorno_real_de_bolsa * 100:.1f}% esperados da bolsa: prêmio de "
            f"{premio * 100:.1f} pontos, que existe mas é magro. Risco alto passa a "
            f"ser pouco pago"
        )
    elif premio >= 0:
        teto, veredito = 40.0, (
            f"prêmio de apenas {premio * 100:.1f} pontos. A renda fixa paga "
            f"{real_liquido * 100:.1f}% real líquido sem queda nenhuma; a bolsa precisa "
            f"aceitar quedas de dois dígitos para tentar {retorno_real_de_bolsa * 100:.1f}%"
        )
    else:
        teto, veredito = 20.0, (
            f"prêmio NEGATIVO de {premio * 100:.1f} pontos: a renda fixa paga "
            f"{real_liquido * 100:.1f}% real líquido, acima do que a bolsa espera no "
            f"longo prazo. Correr risco aqui é pagar para ter risco"
        )

    return Leitura(
        cdi_anual=cdi_anual, ipca_12m=ipca_12m,
        juro_real_bruto=real_bruto, juro_real_liquido=real_liquido,
        premio_da_bolsa=premio, teto_de_risco_sugerido=teto, veredito=veredito,
    )


def aplicar_teto(risco_pedido: float, leitura: Leitura) -> tuple[float, str | None]:
    """Aplica o teto sem nunca aumentar o risco de ninguém."""
    if risco_pedido <= leitura.teto_de_risco_sugerido:
        return risco_pedido, None
    return leitura.teto_de_risco_sugerido, (
        f"risco reduzido de {risco_pedido:.0f}% para "
        f"{leitura.teto_de_risco_sugerido:.0f}% pelo juro real: {leitura.veredito}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Série histórica
# ─────────────────────────────────────────────────────────────────────────────

def serie_historica(
    cdi_diario: pd.Series,
    ipca_mensal: pd.Series,
    janela_dias: int = 252,
) -> pd.DataFrame:
    """Monta a série de juro real, dia a dia, olhando só para trás.

    O CDI anualizado usa os últimos 252 dias úteis e o IPCA usa os últimos 12
    meses. Os dois são retrovisores, de propósito: um indicador que usasse o
    IPCA do mês seguinte enxergaria o futuro, e a série inteira ficaria
    inutilizável para backtest.
    """
    if len(cdi_diario) < janela_dias + 1:
        raise ValueError(f"CDI curto demais: {len(cdi_diario)} dias, mínimo {janela_dias + 1}")

    acumulado = (1 + cdi_diario).cumprod()
    cdi_anual = (acumulado / acumulado.shift(janela_dias)) ** (252 / janela_dias) - 1

    ipca_12m = (1 + ipca_mensal / 100).rolling(12).apply(lambda x: x.prod(), raw=True) - 1
    ipca_diario = ipca_12m.reindex(cdi_anual.index, method="ffill")

    df = pd.DataFrame({"cdi_anual": cdi_anual, "ipca_12m": ipca_diario}).dropna()
    df["juro_real"] = (1 + df["cdi_anual"]) / (1 + df["ipca_12m"]) - 1
    df["juro_real_liquido"] = (
        1 + df["cdi_anual"] * (1 - aliquota_regressiva(PRAZO_LONGO_DIAS))
    ) / (1 + df["ipca_12m"]) - 1
    df["premio_da_bolsa"] = RETORNO_REAL_DE_BOLSA - df["juro_real_liquido"]
    df["teto_de_risco"] = df["premio_da_bolsa"].map(_teto_de)
    return df


def _teto_de(premio: float) -> float:
    if premio >= PREMIO_MINIMO * 3:
        return 100.0
    if premio >= PREMIO_MINIMO:
        return 70.0
    if premio >= 0:
        return 40.0
    return 20.0
