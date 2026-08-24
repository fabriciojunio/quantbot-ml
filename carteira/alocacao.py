"""
Motores de alocação: dos pesos fixos ao trend overlay.

Cada motor recebe preços e devolve um DataFrame de pesos alvo por data, com
soma menor ou igual a 1 — a sobra é caixa remunerado ao CDI. Nenhum motor usa
informação futura: todo cálculo em `t` usa dados até `t`, e quem executa é o
backtest, na barra seguinte.

Ordem deliberada, do mais simples para o mais elaborado. Se o motor elaborado
não bater o simples na validação, o simples vence — e é assim que costuma
terminar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import sinais


@dataclass
class ConfigAlocacao:
    max_por_ativo: float = 0.35        # teto de concentração
    min_por_ativo: float = 0.05        # abaixo disso não vale a operação
    caixa_minimo: float = 0.0
    vol_alvo_anual: float | None = None  # quando definido, escala a carteira
    max_alavancagem: float = 1.0


def _limpar(pesos: pd.DataFrame, cfg: ConfigAlocacao) -> pd.DataFrame:
    """Aplica teto, piso e normalização, nesta ordem.

    O piso vem antes da normalização de propósito: posição de 1% da carteira
    não paga o custo de existir, e deixar ela viva só para "somar 100%" é como
    o giro nasce sem que ninguém perceba.
    """
    p = pesos.clip(lower=0.0).fillna(0.0)
    p = p.where(p >= cfg.min_por_ativo, 0.0)
    p = p.clip(upper=cfg.max_por_ativo)

    soma = p.sum(axis=1)
    limite = min(cfg.max_alavancagem, 1.0 - cfg.caixa_minimo)
    excesso = soma > limite
    p.loc[excesso] = p.loc[excesso].div(soma[excesso], axis=0) * limite
    return p


def pesos_fixos(precos: pd.DataFrame, alvo: dict[str, float]) -> pd.DataFrame:
    """Carteira de pesos constantes. É o baseline que tudo precisa bater."""
    total = sum(alvo.values())
    if total <= 0:
        raise ValueError("os pesos precisam somar mais que zero")
    if total > 1.0001:
        raise ValueError(f"os pesos somam {total:.2f}; não pode passar de 1")

    p = pd.DataFrame(0.0, index=precos.index, columns=precos.columns)
    for ticker, peso in alvo.items():
        if ticker in p.columns:
            p[ticker] = peso
    return p


def peso_igual(precos: pd.DataFrame, cfg: ConfigAlocacao | None = None) -> pd.DataFrame:
    cfg = cfg or ConfigAlocacao(min_por_ativo=0.0)
    n = precos.shape[1]
    p = pd.DataFrame(1.0 / n, index=precos.index, columns=precos.columns)
    return _limpar(p, cfg)


def inverso_da_volatilidade(
    precos: pd.DataFrame,
    dias: int = 63,
    cfg: ConfigAlocacao | None = None,
) -> pd.DataFrame:
    """Peso proporcional a 1/vol: cada ativo contribui com risco parecido.

    Não prevê nada. Só impede que o ativo mais volátil da carteira decida
    sozinho o resultado do ano.
    """
    cfg = cfg or ConfigAlocacao(min_por_ativo=0.0)
    vol = sinais.volatilidade(precos, dias)
    inv = 1 / vol.replace(0, np.nan)
    p = inv.div(inv.sum(axis=1), axis=0)
    return _limpar(p, cfg)


def momentum_absoluto_puro(
    precos: pd.DataFrame,
    cdi_acumulado: pd.Series | None = None,
    cfg: ConfigAlocacao | None = None,
) -> pd.DataFrame:
    """Fica no ativo enquanto a tendência dele for positiva; senão, caixa.

    É o time-series momentum do artigo de 140 anos, na forma mais direta.
    O ponto importante: quando tudo está em queda, a carteira fica **em CDI**,
    e é daí que vem a redução de drawdown.
    """
    cfg = cfg or ConfigAlocacao(min_por_ativo=0.0)
    mom = sinais.momentum_absoluto(precos)

    if cdi_acumulado is not None:
        cdi = cdi_acumulado.reindex(precos.index).ffill()
        ret_cdi = pd.concat(
            [cdi / cdi.shift(sinais.DIAS_UTEIS[j]) - 1 for j in ("1m", "3m", "12m")], axis=1
        ).mean(axis=1)
        mom = mom.sub(ret_cdi, axis=0)

    dentro = (mom > 0).astype(float)
    n_dentro = dentro.sum(axis=1).replace(0, np.nan)
    p = dentro.div(n_dentro, axis=0).fillna(0.0)
    return _limpar(p, cfg)


def dual_momentum(
    precos: pd.DataFrame,
    top_n: int = 3,
    cdi_acumulado: pd.Series | None = None,
    cfg: ConfigAlocacao | None = None,
) -> pd.DataFrame:
    """Escolhe os `top_n` melhores do ranking e exige tendência positiva neles.

    Relativo escolhe quem; absoluto decide se vale estar comprado em alguma
    coisa. Sem o filtro absoluto, a estratégia compra o "menos pior" no meio de
    um mercado de baixa — que é exatamente como ela perde dinheiro.
    """
    if top_n < 1:
        raise ValueError("top_n precisa ser pelo menos 1")

    cfg = cfg or ConfigAlocacao()
    rel = sinais.momentum_relativo(precos)
    mom = sinais.momentum_absoluto(precos)

    if cdi_acumulado is not None:
        cdi = cdi_acumulado.reindex(precos.index).ffill()
        ret_cdi = pd.concat(
            [cdi / cdi.shift(sinais.DIAS_UTEIS[j]) - 1 for j in ("1m", "3m", "12m")], axis=1
        ).mean(axis=1)
        mom = mom.sub(ret_cdi, axis=0)

    ranking = rel.rank(axis=1, ascending=False, method="first")
    escolhidos = (ranking <= top_n) & (mom > 0)

    n = escolhidos.sum(axis=1).replace(0, np.nan)
    p = escolhidos.astype(float).div(n, axis=0).fillna(0.0)
    return _limpar(p, cfg)


def nucleo_com_trend(
    precos: pd.DataFrame,
    nucleo: dict[str, float],
    cdi_acumulado: pd.Series | None = None,
    fatia_protegida: float = 0.5,
    cfg: ConfigAlocacao | None = None,
) -> pd.DataFrame:
    """Carteira fixa com uma parte que sai para caixa quando a tendência vira.

    Desenho que vem direto da conclusão da pesquisa: o núcleo captura o prêmio
    de risco (que é o que efetivamente paga no longo prazo) e o overlay de
    trend só mexe em `fatia_protegida` da carteira, para reduzir queda sem
    zerar a exposição no fundo do poço — que é o erro clássico de estratégia
    tática, sair na baixa e voltar depois da recuperação.
    """
    if not 0.0 <= fatia_protegida <= 1.0:
        raise ValueError("fatia_protegida vai de 0 a 1")

    cfg = cfg or ConfigAlocacao(min_por_ativo=0.0, max_por_ativo=1.0)
    base = pesos_fixos(precos, nucleo)

    mom = sinais.momentum_absoluto(precos)
    if cdi_acumulado is not None:
        cdi = cdi_acumulado.reindex(precos.index).ffill()
        ret_cdi = pd.concat(
            [cdi / cdi.shift(sinais.DIAS_UTEIS[j]) - 1 for j in ("1m", "3m", "12m")], axis=1
        ).mean(axis=1)
        mom = mom.sub(ret_cdi, axis=0)

    # Onde a tendência é negativa, a parte protegida vira caixa.
    ligado = (mom > 0).astype(float).reindex(columns=base.columns).fillna(1.0)
    fator = (1 - fatia_protegida) + fatia_protegida * ligado

    p = base * fator
    # Antes do aquecimento do sinal, mantém a carteira base em vez de zerar.
    p[mom.isna()] = base[mom.isna()]
    return _limpar(p, cfg)


def escalar_para_vol_alvo(
    pesos: pd.DataFrame,
    precos: pd.DataFrame,
    vol_alvo_anual: float = 0.10,
    dias: int = 63,
    max_escala: float = 1.0,
) -> pd.DataFrame:
    """Reduz a exposição quando a volatilidade da carteira passa do alvo.

    `max_escala=1.0` significa que o motor só desliga risco, nunca alavanca.
    Alavancar para atingir vol alvo é o que transforma uma estratégia defensiva
    numa forma cara de quebrar.
    """
    if vol_alvo_anual <= 0:
        raise ValueError("vol alvo precisa ser positiva")

    r = precos.pct_change()
    ret_carteira = (pesos.shift(1) * r).sum(axis=1)
    vol = ret_carteira.rolling(dias, min_periods=max(5, dias // 3)).std(ddof=1) * np.sqrt(252)

    escala = (vol_alvo_anual / vol.replace(0, np.nan)).clip(upper=max_escala)
    escala = escala.fillna(max_escala)
    return pesos.mul(escala, axis=0)


MOTORES = {
    "peso_igual": peso_igual,
    "inverso_vol": inverso_da_volatilidade,
    "momentum_absoluto": momentum_absoluto_puro,
    "dual_momentum": dual_momentum,
}
