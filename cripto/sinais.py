"""
Score de confluência.

Mantém a ideia do painel do Sentinela — vários indicadores votam, cada um com
um peso, e o resultado é um score de 0 a 100 — mas com três diferenças que
mudam tudo:

1. Cada voto vem do indicador calculado sobre OHLCV real (`indicadores.py`).
2. O score de uma barra usa apenas informação disponível até o fechamento
   daquela barra. A execução acontece na abertura da barra seguinte.
3. O peso é declarado num único lugar e entra na validação como parâmetro,
   então dá para medir se ele foi escolhido por evidência ou por gosto.

Score acima de 50 favorece compra, abaixo favorece venda. A zona neutra em
volta de 50 existe porque "não sei" é uma resposta legítima e barata; o
sistema antigo entrava sempre, e entrar sempre é o jeito mais rápido de doar
dinheiro para a corretora.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicadores


@dataclass(frozen=True)
class PesosConfluencia:
    """Peso de cada voto. Somam 100 depois de normalizados."""

    tendencia_ema: float = 20.0
    supertrend: float = 15.0
    macd: float = 15.0
    rsi: float = 12.0
    adx: float = 12.0
    estocastico: float = 8.0
    obv: float = 10.0
    bollinger: float = 8.0

    def normalizados(self) -> dict[str, float]:
        bruto = {
            "tendencia_ema": self.tendencia_ema,
            "supertrend": self.supertrend,
            "macd": self.macd,
            "rsi": self.rsi,
            "adx": self.adx,
            "estocastico": self.estocastico,
            "obv": self.obv,
            "bollinger": self.bollinger,
        }
        total = sum(bruto.values())
        if total <= 0:
            raise ValueError("a soma dos pesos precisa ser positiva")
        return {k: v / total for k, v in bruto.items()}


@dataclass
class ConfigSinal:
    """Regras de entrada e saída em cima do score."""

    pesos: PesosConfluencia = field(default_factory=PesosConfluencia)
    limite_compra: float = 62.0
    limite_venda: float = 38.0
    adx_minimo: float = 20.0        # abaixo disso o mercado está de lado
    permitir_venda: bool = False    # spot puro não vende descoberto
    atr_stop: float = 2.0           # stop a N ATR da entrada
    atr_alvo: float = 3.0           # alvo a N ATR da entrada
    max_barras_posicao: int = 48    # saída por tempo, em barras


def _voto_para_score(voto: pd.Series) -> pd.Series:
    """Converte um voto em [-1, 1] para contribuição em [0, 100]."""
    return (voto.clip(-1, 1) + 1) * 50


def votos(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula o voto de cada indicador, em [-1, 1].

    Espera o DataFrame já enriquecido por `indicadores.tabela_completa`.
    """
    v = pd.DataFrame(index=df.index)
    fecha = df["fechamento"]

    # Tendência por empilhamento de médias: 9 > 21 > 50 > 200 é alta plena.
    escada = (
        (df["ema_9"] > df["ema_21"]).astype(float)
        + (df["ema_21"] > df["ema_50"]).astype(float)
        + (df["ema_50"] > df["ema_200"]).astype(float)
        + (fecha > df["ema_200"]).astype(float)
    )
    v["tendencia_ema"] = (escada / 4.0) * 2 - 1
    v.loc[df["ema_200"].isna(), "tendencia_ema"] = np.nan

    v["supertrend"] = df["supertrend_tendencia"].astype(float)

    # MACD: sinal do histograma, com intensidade proporcional ao tamanho dele
    # em relação ao preço (histograma grande vale mais que histograma raso).
    hist_rel = df["macd_hist"] / fecha.replace(0.0, np.nan)
    escala = hist_rel.rolling(200, min_periods=50).std(ddof=0).replace(0.0, np.nan)
    v["macd"] = (hist_rel / (2 * escala)).clip(-1, 1)

    # RSI: neutro em 50, saturando nos extremos. Acima de 80 vira alerta de
    # esticamento e o voto recua, em vez de virar convicção máxima.
    r = df["rsi_14"]
    v["rsi"] = ((r - 50) / 25).clip(-1, 1)
    v.loc[r > 80, "rsi"] = 0.3
    v.loc[r < 20, "rsi"] = -0.3

    # ADX não tem direção própria: ele confirma o lado do DI dominante.
    forca = ((df["adx"] - 20) / 25).clip(0, 1)
    v["adx"] = np.sign(df["di_mais"] - df["di_menos"]) * forca

    v["estocastico"] = ((df["stoch_k"] - 50) / 40).clip(-1, 1)

    # OBV acima da própria média significa volume acompanhando o preço.
    v["obv"] = np.sign(df["obv"] - df["obv_ema_21"])

    # %B centrado: 0.5 no meio das bandas. Extremos entram amortecidos, porque
    # em tendência forte o preço passeia colado na banda por muito tempo.
    v["bollinger"] = ((df["bb_pct_b"] - 0.5) * 2).clip(-1, 1) * 0.7

    return v


def score_confluencia(df: pd.DataFrame, pesos: PesosConfluencia | None = None) -> pd.Series:
    """Score ponderado de 0 a 100 por barra."""
    pesos = pesos or PesosConfluencia()
    p = pesos.normalizados()
    v = votos(df)

    # Só pontua onde há voto; renormaliza pelo peso efetivamente disponível
    # para que o aquecimento dos indicadores não puxe o score para baixo.
    contribuicao = pd.Series(0.0, index=df.index)
    peso_valido = pd.Series(0.0, index=df.index)
    for nome, peso in p.items():
        col = v[nome]
        presente = col.notna()
        contribuicao += _voto_para_score(col.fillna(0.0)) * peso * presente
        peso_valido += peso * presente

    score = contribuicao / peso_valido.replace(0.0, np.nan)
    return score.where(peso_valido > 0.5)  # exige metade do peso disponível


def gerar_sinais(df: pd.DataFrame, config: ConfigSinal | None = None) -> pd.DataFrame:
    """Devolve o DataFrame com score, posição alvo e níveis de stop/alvo.

    `posicao_alvo` é 1 (comprado), -1 (vendido) ou 0 (fora), e vale para a
    barra SEGUINTE. É por isso que o backtest executa em `abertura.shift(-1)`:
    a decisão de uma barra não pode ser executada no preço que a gerou.
    """
    config = config or ConfigSinal()
    out = indicadores.tabela_completa(df) if "rsi_14" not in df.columns else df.copy()

    out["score"] = score_confluencia(out, config.pesos)

    comprar = (out["score"] >= config.limite_compra) & (out["adx"] >= config.adx_minimo)
    vender = (out["score"] <= config.limite_venda) & (out["adx"] >= config.adx_minimo)

    posicao = pd.Series(0, index=out.index, dtype=int)
    posicao[comprar] = 1
    if config.permitir_venda:
        posicao[vender] = -1

    out["posicao_alvo"] = posicao
    out["stop_long"] = out["fechamento"] - config.atr_stop * out["atr_14"]
    out["alvo_long"] = out["fechamento"] + config.atr_alvo * out["atr_14"]
    out["stop_short"] = out["fechamento"] + config.atr_stop * out["atr_14"]
    out["alvo_short"] = out["fechamento"] - config.atr_alvo * out["atr_14"]

    return out


def explica_barra(df: pd.DataFrame, quando, pesos: PesosConfluencia | None = None) -> dict:
    """Abre o score de uma barra em quanto cada indicador contribuiu.

    Serve para auditoria: nenhum sinal deve ser inexplicável, e o relatório
    diário mostra essa decomposição em vez de um texto genérico de IA.
    """
    pesos = pesos or PesosConfluencia()
    p = pesos.normalizados()
    v = votos(df).loc[quando]

    detalhe = {}
    for nome, peso in p.items():
        voto = v[nome]
        detalhe[nome] = {
            "voto": None if pd.isna(voto) else round(float(voto), 3),
            "peso_pct": round(peso * 100, 1),
            "contribuicao": None if pd.isna(voto) else round(float((voto + 1) * 50 * peso), 2),
        }

    return {
        "quando": str(quando),
        "score": round(float(score_confluencia(df, pesos).loc[quando]), 2),
        "indicadores": detalhe,
    }
