"""
Sinais de alocação.

Só entram aqui construções com evidência publicada e replicada fora da amostra
em que foram descobertas. Nada de indicador escolhido por parecer bonito no
gráfico.

**Momentum absoluto (time-series momentum).** Hurst, Ooi e Pedersen mediram
1880–2016, em 67 mercados de quatro classes de ativo, e acharam retorno médio
positivo em todas as décadas, com desempenho bom em 8 das 10 maiores crises do
período. É o resultado mais robusto do conjunto — e note o que ele é: um
**diversificador que reduz drawdown**, não uma máquina de retorno.

**Momentum relativo (cross-sectional).** Comprar o que subiu mais entre pares
comparáveis. Documentado em ações e replicado em cripto. Em cripto a faixa
estável é lookback de 15 a 35 dias com rebalanceamento de 7 dias, e o efeito
some quando o custo passa de 125 pontos-base.

**Volatilidade como peso.** Peso inverso à volatilidade não prevê retorno; ele
estabiliza a contribuição de risco de cada ativo. É o único dos três que não
depende de nenhuma previsão dar certo.

A honestidade que precisa vir junto: o Global Equities Momentum, a estratégia
tática mais famosa construída sobre momentum absoluto, **ficou atrás de um
60/40 simples depois de publicada**. Isso é o padrão, não a exceção. Por isso
a validação em `validacao.py` compara sempre contra carteira parada, e por
isso o motor trata trend como redutor de queda, não como gerador de ganho.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Dias úteis por período, para converter janela de calendário em barras.
DIAS_UTEIS = {"1m": 21, "3m": 63, "6m": 126, "9m": 189, "12m": 252}


def retorno_periodo(precos: pd.DataFrame, dias: int) -> pd.DataFrame:
    """Retorno acumulado nos últimos `dias` pregões, por ativo."""
    if dias <= 0:
        raise ValueError("janela precisa ser positiva")
    return precos / precos.shift(dias) - 1


def momentum_absoluto(
    precos: pd.DataFrame,
    janelas: tuple[str, ...] = ("1m", "3m", "12m"),
) -> pd.DataFrame:
    """Média dos retornos em várias janelas — o sinal de trend seguido no artigo.

    Combinar 1, 3 e 12 meses em vez de escolher uma janela é justamente o que
    evita escolher a janela que funcionou no passado. Positivo significa
    tendência de alta; negativo, de baixa.
    """
    faltando = [j for j in janelas if j not in DIAS_UTEIS]
    if faltando:
        raise ValueError(f"janela desconhecida: {faltando}. Use {list(DIAS_UTEIS)}")

    partes = [retorno_periodo(precos, DIAS_UTEIS[j]) for j in janelas]
    return sum(partes) / len(partes)


def momentum_relativo(
    precos: pd.DataFrame,
    dias: int = 126,
    pular_ultimo_mes: bool = True,
) -> pd.DataFrame:
    """Ranking cruzado: quem subiu mais entre os pares, de 0 (pior) a 1 (melhor).

    `pular_ultimo_mes` remove os últimos 21 pregões do cálculo. Em ações, o mês
    mais recente costuma reverter, e incluir ele contamina o sinal de momentum
    com reversão de curto prazo — é a convenção 12-1 da literatura.
    """
    if pular_ultimo_mes:
        janela = precos.shift(21) / precos.shift(dias) - 1
    else:
        janela = retorno_periodo(precos, dias)

    return janela.rank(axis=1, pct=True)


def volatilidade(precos: pd.DataFrame, dias: int = 63, anualizar: bool = True) -> pd.DataFrame:
    """Volatilidade realizada dos retornos diários."""
    r = precos.pct_change()
    vol = r.rolling(dias, min_periods=max(5, dias // 3)).std(ddof=1)
    return vol * np.sqrt(252) if anualizar else vol


def correlacao_media(precos: pd.DataFrame, dias: int = 126) -> pd.Series:
    """Correlação média de cada ativo com os demais, em janela móvel.

    Sobe junto em crise: quando esta série dispara, a diversificação da carteira
    está valendo menos do que a planilha diz.
    """
    r = precos.pct_change().dropna()
    saida = []
    for fim in range(dias, len(r) + 1):
        janela = r.iloc[fim - dias : fim]
        c = janela.corr()
        n = len(c)
        media = (c.to_numpy().sum() - n) / (n * (n - 1)) if n > 1 else np.nan
        saida.append((r.index[fim - 1], media))
    if not saida:
        return pd.Series(dtype=float)
    idx, vals = zip(*saida)
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name="correlacao_media")


def filtro_de_tendencia(
    precos: pd.DataFrame,
    media_dias: int = 200,
) -> pd.DataFrame:
    """True quando o preço está acima da própria média móvel longa.

    É a versão mais simples e mais antiga do sinal de trend, e a que menos tem
    parâmetro para ajustar — o que é uma virtude, não uma limitação.
    """
    media = precos.rolling(media_dias, min_periods=media_dias).mean()
    return precos > media


def sinal_composto(
    precos: pd.DataFrame,
    cdi_acumulado: pd.Series | None = None,
) -> pd.DataFrame:
    """Combina os três sinais em um score de 0 a 1 por ativo.

    Quando o CDI é informado, o momentum absoluto é medido **em excesso ao
    CDI**, não em termos nominais. Num país com juro de dois dígitos essa
    distinção não é detalhe: um ativo que subiu 8% no ano perdeu para a renda
    fixa, e tratar isso como tendência de alta é um erro caro.
    """
    mom_abs = momentum_absoluto(precos)

    if cdi_acumulado is not None:
        cdi = cdi_acumulado.reindex(precos.index).ffill()
        retorno_cdi = pd.concat(
            [cdi / cdi.shift(DIAS_UTEIS[j]) - 1 for j in ("1m", "3m", "12m")], axis=1
        ).mean(axis=1)
        mom_abs = mom_abs.sub(retorno_cdi, axis=0)

    mom_rel = momentum_relativo(precos)
    vol = volatilidade(precos)
    # Inverso da vol, normalizado entre 0 e 1 dentro de cada data.
    inv_vol = (1 / vol.replace(0, np.nan)).rank(axis=1, pct=True)

    tem_tendencia = (mom_abs > 0).astype(float)

    score = 0.5 * mom_rel + 0.3 * tem_tendencia + 0.2 * inv_vol
    return score.where(mom_abs.notna())


def diagnostico(precos: pd.DataFrame, quando=None) -> dict:
    """Fotografia dos sinais numa data, aberta ativo por ativo.

    Todo sinal que vira ordem tem que ser explicável. Se não dá para dizer por
    que a carteira comprou, ela não deveria ter comprado.
    """
    quando = quando or precos.index[-1]
    mom_abs = momentum_absoluto(precos).loc[quando]
    mom_rel = momentum_relativo(precos).loc[quando]
    vol = volatilidade(precos).loc[quando]
    tend = filtro_de_tendencia(precos).loc[quando]

    return {
        "quando": str(quando)[:10],
        "ativos": {
            t: {
                "momentum_absoluto_pct": None if pd.isna(mom_abs[t]) else round(float(mom_abs[t]) * 100, 2),
                "posicao_no_ranking": None if pd.isna(mom_rel[t]) else round(float(mom_rel[t]), 2),
                "volatilidade_anual_pct": None if pd.isna(vol[t]) else round(float(vol[t]) * 100, 1),
                "acima_da_media_200": None if pd.isna(tend[t]) else bool(tend[t]),
            }
            for t in precos.columns
        },
    }
