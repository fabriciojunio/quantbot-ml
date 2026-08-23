"""
Perfis de risco e orçamento de risco.

Aqui o usuário decide **quanto** vai para a ponta arriscada, e o motor decide
**onde**, dentro de cada faixa. É a separação certa: a tolerância a risco é
uma escolha pessoal e ninguém pode fazer por você; a distribuição dentro de
cada faixa é um problema técnico e o motor resolve.

As faixas, da mais previsível para a menos:

    caixa        CDI, liquidez diária, nunca cai
    renda fixa   IPCA+ e prefixado, oscila pouco mas oscila
    dolar        bolsa americana em reais (IVVB11) — proteção de moeda
    bolsa        bolsa local (BOVA11, SMAL11)
    alternativo  ouro e imobiliário
    satelite     só com estratégia aprovada na validação; hoje fica zerado

O que este módulo **não** faz: prometer retorno. Aumentar a fatia arriscada
aumenta o retorno esperado e aumenta a perda possível, e as duas coisas andam
juntas por construção. `medir_perfis()` mostra o que cada nível entregou de
verdade no histórico — inclusive nos piores momentos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class Faixa(str, Enum):
    CAIXA = "caixa"
    RENDA_FIXA = "renda_fixa"
    DOLAR = "dolar"
    BOLSA = "bolsa"
    ALTERNATIVO = "alternativo"
    SATELITE = "satelite"


# Quem representa cada faixa, e com que peso dentro dela.
COMPOSICAO: dict[Faixa, dict[str, float]] = {
    Faixa.CAIXA: {"CDI": 1.0},
    Faixa.RENDA_FIXA: {"IMAB11.SA": 0.60, "B5P211.SA": 0.40},
    Faixa.DOLAR: {"IVVB11.SA": 1.0},
    Faixa.BOLSA: {"BOVA11.SA": 0.70, "SMAL11.SA": 0.30},
    Faixa.ALTERNATIVO: {"GOLD11.SA": 0.60, "XFIX11.SA": 0.40},
    Faixa.SATELITE: {},
}

# As faixas que contam como risco no orçamento.
FAIXAS_DE_RISCO = (Faixa.DOLAR, Faixa.BOLSA, Faixa.ALTERNATIVO, Faixa.SATELITE)


@dataclass(frozen=True)
class Perfil:
    """Um orçamento de risco. `risco_pct` é quanto vai para a ponta arriscada."""

    nome: str
    risco_pct: float
    descricao: str
    # Como a fatia arriscada se divide entre dólar, bolsa e alternativo.
    dentro_do_risco: dict[Faixa, float] = field(
        default_factory=lambda: {Faixa.DOLAR: 0.45, Faixa.BOLSA: 0.35, Faixa.ALTERNATIVO: 0.20}
    )
    # Como a fatia segura se divide entre caixa e renda fixa.
    dentro_do_seguro: dict[Faixa, float] = field(
        default_factory=lambda: {Faixa.CAIXA: 0.40, Faixa.RENDA_FIXA: 0.60}
    )

    def __post_init__(self) -> None:
        if not 0 <= self.risco_pct <= 100:
            raise ValueError("risco_pct vai de 0 a 100")
        if abs(sum(self.dentro_do_risco.values()) - 1.0) > 1e-6:
            raise ValueError("os pesos dentro do risco precisam somar 1")
        if abs(sum(self.dentro_do_seguro.values()) - 1.0) > 1e-6:
            raise ValueError("os pesos dentro do seguro precisam somar 1")

    def pesos_por_faixa(self) -> dict[Faixa, float]:
        risco = self.risco_pct / 100
        seguro = 1 - risco
        pesos = {f: p * risco for f, p in self.dentro_do_risco.items()}
        for f, p in self.dentro_do_seguro.items():
            pesos[f] = pesos.get(f, 0.0) + p * seguro
        return pesos

    def pesos_por_ativo(
        self,
        satelite_liberado: bool = False,
        composicao: dict[Faixa, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """Explode as faixas em ativos.

        O satélite só existe quando a validação aprova E quando existe algo
        para comprar dentro dele. Enquanto uma das duas condições falta, a
        fatia dele volta para a faixa de dólar — que é a exposição arriscada
        com a evidência mais sólida do conjunto.

        As duas condições importam. Liberar o satélite sem ter composição
        fazia a fatia dele evaporar: os pesos passavam a somar 0,88 e 12% da
        carteira simplesmente não era investido, sem erro nenhum. É o tipo de
        falha silenciosa que este projeto inteiro existe para não cometer.

        `composicao` permite trocar quem representa cada faixa sem mexer no
        orçamento de risco. É o que a série longa usa: as faixas são as mesmas,
        mas os representantes mudam para os que têm história desde 2003.
        """
        comp = composicao if composicao is not None else COMPOSICAO
        faltantes = [f for f in self.pesos_por_faixa() if f not in comp]
        if faltantes:
            raise ValueError(
                f"a composição não define as faixas {[f.value for f in faltantes]}; "
                f"parte do dinheiro ficaria sem destino"
            )

        pesos_faixa = self.pesos_por_faixa()
        peso_satelite = pesos_faixa.get(Faixa.SATELITE, 0.0)
        tem_o_que_comprar = bool(comp[Faixa.SATELITE])

        if peso_satelite > 0 and not (satelite_liberado and tem_o_que_comprar):
            pesos_faixa[Faixa.DOLAR] = pesos_faixa.get(Faixa.DOLAR, 0.0) + peso_satelite
            pesos_faixa[Faixa.SATELITE] = 0.0

        ativos: dict[str, float] = {}
        for faixa, peso_faixa in pesos_faixa.items():
            for ticker, peso_dentro in comp[faixa].items():
                ativos[ticker] = ativos.get(ticker, 0.0) + peso_faixa * peso_dentro

        ativos = {t: p for t, p in ativos.items() if p > 1e-9}

        soma = sum(ativos.values())
        if abs(soma - 1.0) > 1e-6:
            raise ValueError(
                f"os pesos somaram {soma:.4f} em vez de 1. Alguma faixa com peso "
                f"tem composição vazia — parte do dinheiro ficaria sem destino."
            )
        return ativos


PERFIS: dict[str, Perfil] = {
    "conservador": Perfil(
        "conservador", 20.0,
        "Quase tudo previsível. Cai pouco, rende perto do CDI.",
    ),
    "moderado": Perfil(
        "moderado", 40.0,
        "Metade previsível. Quedas de um dígito na maioria dos anos ruins.",
    ),
    "arrojado": Perfil(
        "arrojado", 60.0,
        "Maioria em risco. Aceita passar anos atrás do CDI para tentar superá-lo.",
    ),
    "agressivo": Perfil(
        "agressivo", 80.0,
        "Pouco colchão. Quedas de dois dígitos são esperadas, não exceção.",
    ),
    "muito_agressivo": Perfil(
        "muito_agressivo", 100.0,
        "Sem colchão nenhum. É o teto do que este motor faz — e ainda não é 30% ao ano.",
    ),
}


def serie_do_perfil(
    perfil: Perfil,
    precos: pd.DataFrame,
    cdi: pd.Series,
    satelite_liberado: bool = False,
    composicao: dict[Faixa, dict[str, float]] | None = None,
) -> pd.Series:
    """Curva de um perfil, com rebalanceamento anual e o caixa rendendo CDI."""
    pesos = perfil.pesos_por_ativo(satelite_liberado, composicao)
    cdi_alinhado = cdi.reindex(precos.index).ffill().fillna(0.0)

    retornos = precos.pct_change().fillna(0.0)
    retornos["CDI"] = cdi_alinhado

    disponiveis = {t: p for t, p in pesos.items() if t in retornos.columns}
    total = sum(disponiveis.values())
    if total <= 0:
        raise ValueError("nenhum ativo do perfil está presente na série de preços")

    # Renormalizar sem checar quanto faltou é perigoso: com um universo
    # incompleto, um perfil agressivo vira 100% CDI e a curva sai bonita e
    # errada. Se mais de um quinto do peso alvo não existe na série, o
    # resultado não representa o perfil pedido e não deve ser produzido.
    if total < 0.80:
        faltando = sorted(set(pesos) - set(disponiveis))
        raise ValueError(
            f"só {total * 100:.0f}% do peso do perfil '{perfil.nome}' existe na série. "
            f"Faltam {faltando}. Renormalizar aqui produziria uma curva que não é "
            f"a do perfil pedido."
        )

    disponiveis = {t: p / total for t, p in disponiveis.items()}

    # Rebalanceamento anual: entre um rebalanceamento e outro a carteira deriva.
    valor = 1.0
    curva = []
    pesos_correntes = dict(disponiveis)
    ultimo_ano = precos.index[0].year

    for data in precos.index:
        if data.year != ultimo_ano:
            pesos_correntes = dict(disponiveis)
            ultimo_ano = data.year

        r_dia = sum(pesos_correntes[t] * float(retornos.loc[data, t]) for t in pesos_correntes)
        valor *= 1 + r_dia
        curva.append(valor)

        # Deriva dos pesos com o retorno de cada ativo.
        novos = {t: pesos_correntes[t] * (1 + float(retornos.loc[data, t])) for t in pesos_correntes}
        soma = sum(novos.values())
        if soma > 0:
            pesos_correntes = {t: v / soma for t, v in novos.items()}

    return pd.Series(curva, index=precos.index, name=perfil.nome)


def _metricas(curva: pd.Series, cdi_acumulado: pd.Series) -> dict:
    r = curva.pct_change().dropna()
    anos = (curva.index[-1] - curva.index[0]).days / 365.25

    cagr = (float(curva.iloc[-1]) ** (1 / anos) - 1) * 100 if anos > 0 else 0.0
    vol = float(r.std(ddof=1) * np.sqrt(252) * 100)
    dd = float((curva / curva.cummax() - 1).min()) * 100

    # Pior janela de 12 meses.
    janela = 252
    pior_12m = float((curva / curva.shift(janela) - 1).min()) * 100 if len(curva) > janela else 0.0

    # Fatia do tempo em que a carteira esteve à frente do CDI.
    cdi_n = cdi_acumulado / float(cdi_acumulado.iloc[0])
    curva_n = curva / float(curva.iloc[0])
    a_frente = float((curva_n > cdi_n.reindex(curva.index).ffill()).mean()) * 100

    cdi_cagr = (float(cdi_n.iloc[-1]) ** (1 / anos) - 1) * 100 if anos > 0 else 0.0

    return {
        "cagr_pct": round(cagr, 2),
        "cdi_cagr_pct": round(cdi_cagr, 2),
        "acima_do_cdi_pp": round(cagr - cdi_cagr, 2),
        "vol_anual_pct": round(vol, 2),
        "max_drawdown_pct": round(dd, 2),
        "pior_12_meses_pct": round(pior_12m, 2),
        "tempo_a_frente_do_cdi_pct": round(a_frente, 1),
        "retorno_por_unidade_de_queda": round(abs(cagr / dd), 2) if dd else 0.0,
    }


def medir_perfis(
    precos: pd.DataFrame,
    cdi: pd.Series,
    perfis: dict[str, Perfil] | None = None,
    composicao: dict[Faixa, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Mede todos os perfis no mesmo histórico.

    É a tabela que responde "quero uma fatia alta no arriscado": ela mostra o
    que a fatia alta entregou de retorno **e** o que ela custou nas quedas.
    """
    perfis = perfis or PERFIS
    cdi_acumulado = (1 + cdi.reindex(precos.index).ffill().fillna(0.0)).cumprod()

    linhas = []
    for nome, perfil in perfis.items():
        try:
            curva = serie_do_perfil(perfil, precos, cdi, composicao=composicao)
        except ValueError:
            continue
        linhas.append({
            "perfil": nome,
            "risco_pct": perfil.risco_pct,
            **_metricas(curva, cdi_acumulado),
        })

    return pd.DataFrame(linhas)


def sugerir(
    risco_desejado_pct: float,
    satelite_liberado: bool = False,
) -> dict:
    """Monta um perfil sob medida para o orçamento de risco informado."""
    perfil = Perfil(
        nome=f"sob_medida_{int(risco_desejado_pct)}",
        risco_pct=risco_desejado_pct,
        descricao=f"{risco_desejado_pct:.0f}% em ativos de risco",
    )
    faixas = perfil.pesos_por_faixa()
    ativos = perfil.pesos_por_ativo(satelite_liberado)

    return {
        "risco_pct": risco_desejado_pct,
        "por_faixa": {f.value: round(p * 100, 1) for f, p in faixas.items() if p > 0},
        "por_ativo": {t: round(p * 100, 1) for t, p in sorted(ativos.items(), key=lambda x: -x[1])},
        "exposicao_a_dolar_pct": round(ativos.get("IVVB11.SA", 0.0) * 100, 1),
        "satelite_liberado": satelite_liberado,
        "aviso": (
            "Fatia de risco maior aumenta o retorno esperado E a perda possível. "
            "As duas andam juntas: não existe a versão só com a parte boa."
        ),
    }
