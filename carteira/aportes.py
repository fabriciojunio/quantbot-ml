"""
O simulador de aporte.

Este módulo existe para responder, com número, a pergunta que decide tudo:
**quanto do patrimônio final vem de estratégia e quanto vem de quanto se
poupa?**

A resposta desconforta qualquer sistema de trading. Para quem tem pouco
capital e está começando, o aporte domina o retorno por uma margem que não é
disputa. A conta é simples e implacável: R$ 500 rendendo 20% ao ano geram
R$ 100; aportar R$ 200 por mês gera R$ 2.400. Não existe estratégia que
compense uma taxa de poupança baixa, e nenhum retorno realista compete com
dinheiro novo entrando.

Por isso o pacote todo trata o aporte como variável de primeira classe, e não
como detalhe do backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Cenario:
    capital_inicial: float
    aporte_mensal: float
    retorno_anual_pct: float
    anos: int
    reajuste_aporte_anual_pct: float = 0.0  # aporte cresce com o salário


def projetar(cenario: Cenario) -> pd.DataFrame:
    """Projeção mês a mês, separando o que é aporte e o que é rendimento."""
    if cenario.anos <= 0:
        raise ValueError("horizonte precisa ser positivo")

    taxa_mensal = (1 + cenario.retorno_anual_pct / 100) ** (1 / 12) - 1
    meses = cenario.anos * 12

    saldo = cenario.capital_inicial
    aportado = cenario.capital_inicial
    aporte = cenario.aporte_mensal
    linhas = []

    for m in range(1, meses + 1):
        rendimento = saldo * taxa_mensal
        saldo += rendimento + aporte
        aportado += aporte

        if m % 12 == 0 and cenario.reajuste_aporte_anual_pct:
            aporte *= 1 + cenario.reajuste_aporte_anual_pct / 100

        linhas.append({
            "mes": m,
            "ano": round(m / 12, 2),
            "aporte_do_mes": round(aporte, 2),
            "rendimento_do_mes": round(rendimento, 2),
            "total_aportado": round(aportado, 2),
            "saldo": round(saldo, 2),
            "ganho_acumulado": round(saldo - aportado, 2),
            "fatia_que_e_aporte_pct": round(aportado / saldo * 100, 1) if saldo else 100.0,
        })

    return pd.DataFrame(linhas)


def quanto_pesa_a_estrategia(
    capital_inicial: float,
    aporte_mensal: float,
    anos: int,
    retorno_base_pct: float,
    retorno_melhor_pct: float,
) -> dict:
    """Compara ganhar mais retorno com aportar mais.

    Devolve quanto de aporte mensal a mais equivale ao ganho de retorno. É a
    forma mais direta de mostrar onde o esforço rende: estudar estratégia ou
    aumentar a taxa de poupança.
    """
    base = projetar(Cenario(capital_inicial, aporte_mensal, retorno_base_pct, anos))
    melhor = projetar(Cenario(capital_inicial, aporte_mensal, retorno_melhor_pct, anos))

    saldo_base = float(base["saldo"].iloc[-1])
    saldo_melhor = float(melhor["saldo"].iloc[-1])
    ganho_da_estrategia = saldo_melhor - saldo_base

    # Quanto de aporte extra, no retorno base, chega ao mesmo lugar?
    taxa_mensal = (1 + retorno_base_pct / 100) ** (1 / 12) - 1
    meses = anos * 12
    if abs(taxa_mensal) < 1e-9:
        fator = meses
    else:
        fator = ((1 + taxa_mensal) ** meses - 1) / taxa_mensal
    aporte_equivalente = ganho_da_estrategia / fator if fator else 0.0

    return {
        "anos": anos,
        "capital_inicial": round(capital_inicial, 2),
        "aporte_mensal": round(aporte_mensal, 2),
        "retorno_base_pct": retorno_base_pct,
        "retorno_melhor_pct": retorno_melhor_pct,
        "saldo_com_retorno_base": round(saldo_base, 2),
        "saldo_com_retorno_melhor": round(saldo_melhor, 2),
        "ganho_da_estrategia": round(ganho_da_estrategia, 2),
        "aporte_extra_equivalente_por_mes": round(aporte_equivalente, 2),
        "leitura": (
            f"ganhar {retorno_melhor_pct - retorno_base_pct:.1f} pontos de retorno ao ano "
            f"equivale a aportar R$ {aporte_equivalente:.0f} a mais por mês"
        ),
    }


def tabela_de_sensibilidade(
    capital_inicial: float = 1_000.0,
    anos: int = 10,
    aportes: tuple[float, ...] = (0, 100, 300, 500, 1_000),
    retornos: tuple[float, ...] = (0.0, 8.0, 13.9, 18.0, 25.0),
) -> pd.DataFrame:
    """Saldo final para cada combinação de aporte e retorno.

    Ler esta tabela na horizontal e na vertical costuma encerrar a discussão
    sobre onde vale gastar energia.
    """
    linhas = []
    for aporte in aportes:
        linha = {"aporte_mensal": aporte}
        for r in retornos:
            df = projetar(Cenario(capital_inicial, aporte, r, anos))
            linha[f"{r:.1f}%_a.a."] = round(float(df["saldo"].iloc[-1]), 0)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tempo_para_meta(
    capital_inicial: float,
    aporte_mensal: float,
    retorno_anual_pct: float,
    meta: float,
    max_anos: int = 60,
) -> dict:
    """Em quanto tempo o patrimônio chega na meta."""
    if meta <= capital_inicial:
        return {"meses": 0, "anos": 0.0, "alcancavel": True,
                "leitura": "a meta já está alcançada"}

    taxa = (1 + retorno_anual_pct / 100) ** (1 / 12) - 1
    saldo = capital_inicial
    for m in range(1, max_anos * 12 + 1):
        saldo = saldo * (1 + taxa) + aporte_mensal
        if saldo >= meta:
            return {
                "meses": m,
                "anos": round(m / 12, 1),
                "alcancavel": True,
                "saldo_final": round(saldo, 2),
                "leitura": f"chega em {m // 12} anos e {m % 12} meses",
            }

    return {
        "meses": None,
        "anos": None,
        "alcancavel": False,
        "saldo_em_60_anos": round(saldo, 2),
        "leitura": (
            f"com aporte de R$ {aporte_mensal:.0f} e {retorno_anual_pct:.1f}% ao ano, "
            f"a meta de R$ {meta:,.0f} não é alcançada em {max_anos} anos"
        ),
    }


def renda_mensal_sustentavel(
    patrimonio: float,
    taxa_retirada_anual_pct: float = 4.0,
) -> dict:
    """Quanto dá para retirar por mês sem consumir o patrimônio.

    A regra dos 4% vem de estudo sobre carteiras americanas em dólar. No
    Brasil, com juro real alto e inflação mais volátil, a faixa razoável é
    entre 4% e 6% — mas o número que importa é o mesmo: **para viver de renda,
    o patrimônio precisa ser cerca de 300 vezes a despesa mensal.**
    """
    anual = patrimonio * (taxa_retirada_anual_pct / 100)
    return {
        "patrimonio": round(patrimonio, 2),
        "taxa_de_retirada_pct": taxa_retirada_anual_pct,
        "renda_anual": round(anual, 2),
        "renda_mensal": round(anual / 12, 2),
        "multiplo_necessario": round(100 / taxa_retirada_anual_pct * 12, 0),
        "leitura": (
            f"R$ {patrimonio:,.0f} sustentam cerca de R$ {anual / 12:,.0f} por mês. "
            f"Para viver com R$ 5.000/mês seriam necessários "
            f"R$ {5_000 * 12 / (taxa_retirada_anual_pct / 100):,.0f}."
        ),
    }
