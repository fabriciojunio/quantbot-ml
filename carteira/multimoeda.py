"""
Exposição a moeda e a cripto, sem confundir as duas coisas.

Este arquivo existe para separar duas perguntas que costumam ser tratadas como
uma só, e que têm respostas opostas neste projeto:

**"Operar cripto dá dinheiro?"** — testado, respondido, não. Trinta e duas
configurações de estratégia de cripto foram validadas fora da amostra em
`cripto/`, e nenhuma passou. Custo de taker, spread e financiamento comem a
vantagem antes dela existir. Essa porta continua fechada, e o satélite do robô
segue travado.

**"Ter cripto parada na carteira faz diferença?"** — pergunta diferente, e
nunca tinha sido medida. Não envolve prever nada: é decidir uma fatia, comprar
e rebalancear, exatamente como se faz com ouro. A resposta legítima só pode vir
de medir, e é o que `estudar_fatia()` faz.

Sobre moeda: quem mora no Brasil e ganha em real já tem 100% do patrimônio
humano — salário, aposentadoria, imóvel — apostado num país só. A exposição
cambial da carteira não é aposta no dólar, é redução de concentração. E vale
notar o que o histórico mostrou: em 2008 e em 2020, quando a bolsa local caiu,
foi a fatia em dólar que segurou a carteira, porque o câmbio sobe justamente
quando o Brasil vai mal.

As séries de cripto vêm da Binance em USDT e são convertidas para real pela
PTAX. Converter importa: um brasileiro que comprou bitcoin em 2021 e vendeu em
2023 teve um resultado em dólar e outro bem diferente em real.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

logger = logging.getLogger("quantbot.carteira.multimoeda")


class Moeda(str, Enum):
    BRL = "BRL"
    USD = "USD"
    EUR = "EUR"


# Série do Banco Central para cada moeda estrangeira.
SERIE_BCB: dict[Moeda, str] = {
    Moeda.USD: "USD_BRL",
    Moeda.EUR: "EUR_BRL",
}

# A que moeda cada ativo do universo expõe de fato. Um ETF listado na B3 é
# negociado em real, mas o que está dentro dele pode ser dólar — e é o de
# dentro que manda no resultado.
MOEDA_DE_FATO: dict[str, Moeda] = {
    "BOVA11.SA": Moeda.BRL,
    "SMAL11.SA": Moeda.BRL,
    "DIVO11.SA": Moeda.BRL,
    "IMAB11.SA": Moeda.BRL,
    "B5P211.SA": Moeda.BRL,
    "FIXA11.SA": Moeda.BRL,
    "XFIX11.SA": Moeda.BRL,
    "CDI": Moeda.BRL,
    "IVVB11.SA": Moeda.USD,
    "GOLD11.SA": Moeda.USD,
    "BTC-BRL": Moeda.USD,
    "ETH-BRL": Moeda.USD,
}

# Teto duro da fatia de cripto. Não é opinião sobre o preço: é o tamanho a
# partir do qual um ativo que já caiu 77% duas vezes passa a decidir sozinho o
# resultado da carteira inteira. Com 5%, uma queda de 80% custa 4 pontos do
# patrimônio — recuperável. Com 30%, custa 24 — o que muda a vida de quem
# depende do dinheiro.
TETO_DE_CRIPTO = 0.05

SIMBOLOS_CRIPTO: dict[str, str] = {
    "BTC-BRL": "BTCUSDT",
    "ETH-BRL": "ETHUSDT",
}


class ErroMultimoeda(RuntimeError):
    """Faltou dado de câmbio ou de cripto e não havia cache."""


@dataclass(frozen=True)
class ExposicaoCambial:
    """Quanto da carteira depende de cada moeda."""

    por_moeda: dict[Moeda, float]

    @property
    def estrangeira(self) -> float:
        return sum(p for m, p in self.por_moeda.items() if m is not Moeda.BRL)

    @property
    def local(self) -> float:
        return self.por_moeda.get(Moeda.BRL, 0.0)

    def resumo(self) -> dict:
        return {
            "por_moeda": {m.value: round(p * 100, 1) for m, p in sorted(
                self.por_moeda.items(), key=lambda x: -x[1]) if p > 0},
            "estrangeira_pct": round(self.estrangeira * 100, 1),
            "local_pct": round(self.local * 100, 1),
        }


def exposicao_cambial(pesos: dict[str, float]) -> ExposicaoCambial:
    """A que moeda a carteira está exposta de fato, não a de negociação."""
    por_moeda: dict[Moeda, float] = {}
    for ticker, peso in pesos.items():
        moeda = MOEDA_DE_FATO.get(ticker)
        if moeda is None:
            logger.warning("Moeda de %s desconhecida; tratando como real.", ticker)
            moeda = Moeda.BRL
        por_moeda[moeda] = por_moeda.get(moeda, 0.0) + peso
    return ExposicaoCambial(por_moeda)


def serie_cripto_em_reais(
    ticker: str = "BTC-BRL",
    anos: int = 8,
    historico=None,
    cambista=None,
) -> pd.Series:
    """Fechamento diário de uma cripto convertido para real pela PTAX.

    A conversão não é detalhe. Entre 2021 e 2023 o bitcoin caiu 64% em dólar,
    mas o real se desvalorizou no meio do caminho, e a queda em real foi
    diferente. Medir em dólar uma carteira que vai pagar contas em real dá o
    número errado.
    """
    simbolo = SIMBOLOS_CRIPTO.get(ticker)
    if simbolo is None:
        raise ValueError(f"cripto desconhecida: {ticker}. Conhecidas: {sorted(SIMBOLOS_CRIPTO)}")

    if historico is None:
        from cripto import mercado

        historico = mercado.historico
    if cambista is None:
        from . import dados

        cambista = dados.cambio_diario

    try:
        candles = historico(simbolo, intervalo="1d", dias=anos * 365)
    except Exception as exc:
        raise ErroMultimoeda(f"sem candles de {simbolo}: {exc}") from exc

    if candles.empty or "fechamento" not in candles:
        raise ErroMultimoeda(f"série de {simbolo} vazia")

    em_dolar = candles["fechamento"].astype(float)
    em_dolar.index = pd.to_datetime(em_dolar.index)
    if em_dolar.index.tz is not None:
        em_dolar.index = em_dolar.index.tz_localize(None)
    em_dolar = em_dolar.groupby(em_dolar.index.normalize()).last()

    cambio = cambista(anos=anos + 1, moeda="USD_BRL")

    # A PTAX não sai em fim de semana e cripto negocia todo dia: a taxa que vale
    # no sábado é a do fechamento de sexta, e é isso que o preenchimento faz.
    #
    # A tolerância é obrigatória. Sem ela, um `ffill` puro sobre uma série de
    # câmbio que termina antes da de cripto arrasta a última cotação conhecida
    # por anos, sem erro nenhum: um teste com câmbio de 1995 e preço de 2020
    # produziu série completa e silenciosamente sem sentido. Dez dias cobre
    # qualquer feriado emendado e não cobre uma fonte parada.
    taxa = cambio.reindex(em_dolar.index, method="ffill", tolerance=pd.Timedelta(days=10))
    convertido = (em_dolar * taxa).dropna()
    convertido.name = ticker

    if convertido.empty:
        raise ErroMultimoeda(
            f"{ticker}: nenhum dia com preço e câmbio a menos de 10 dias de distância. "
            f"Preços de {em_dolar.index.min().date()} a {em_dolar.index.max().date()}, "
            f"câmbio de {cambio.index.min().date()} a {cambio.index.max().date()}."
        )

    cobertura = len(convertido) / len(em_dolar)
    if cobertura < 0.90:
        logger.warning(
            "%s: só %.0f%% dos dias tiveram câmbio próximo; %d dias descartados.",
            ticker, cobertura * 100, len(em_dolar) - len(convertido),
        )
    logger.info("%s: %d dias, de %s a %s", ticker, len(convertido),
                convertido.index.min().date(), convertido.index.max().date())
    return convertido


def com_cripto(
    pesos: dict[str, float],
    fatia: float,
    cripto: str = "BTC-BRL",
    teto: float = TETO_DE_CRIPTO,
) -> dict[str, float]:
    """Acrescenta cripto tirando proporcionalmente dos ativos de risco.

    Tira do risco e não do que é seguro, de propósito. Trocar renda fixa por
    bitcoin aumentaria o risco da carteira por duas vias ao mesmo tempo — mais
    ativo volátil e menos colchão — e o orçamento de risco que o usuário
    escolheu deixaria de significar o que ele acha que significa.
    """
    if fatia < 0:
        raise ValueError("a fatia de cripto não pode ser negativa")
    if fatia > teto:
        raise ValueError(
            f"fatia de {fatia * 100:.1f}% passa do teto de {teto * 100:.1f}%. "
            f"Acima disso a cripto decide sozinha o resultado da carteira."
        )
    if fatia == 0:
        return dict(pesos)

    de_risco = {t: p for t, p in pesos.items() if t not in ("CDI", "IMAB11.SA", "B5P211.SA", "FIXA11.SA")}
    total_risco = sum(de_risco.values())
    if total_risco <= fatia:
        raise ValueError(
            f"a carteira tem só {total_risco * 100:.1f}% em risco; não dá para tirar "
            f"{fatia * 100:.1f}% dela para cripto sem mexer na parte segura"
        )

    novos = dict(pesos)
    for t, p in de_risco.items():
        novos[t] = p * (1 - fatia / total_risco)
    novos[cripto] = novos.get(cripto, 0.0) + fatia

    soma = sum(novos.values())
    if abs(soma - 1.0) > 1e-9:
        raise ValueError(f"os pesos passaram a somar {soma:.6f}")
    return {t: p for t, p in novos.items() if p > 1e-12}


def estudar_fatia(
    precos: pd.DataFrame,
    cdi: pd.Series,
    pesos_base: dict[str, float],
    fatias: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05),
    aporte: float = 500.0,
    cripto: str = "BTC-BRL",
    **kwargs,
) -> pd.DataFrame:
    """Mede o que cada fatia de cripto teria feito, com aporte mensal real.

    Responde à pergunta certa — "ter uma fatia parada muda alguma coisa?" — em
    vez da que já foi respondida com não — "dá para operar cripto?". A tabela
    traz retorno e queda lado a lado, porque a fatia que melhora o retorno é a
    mesma que piora a queda, e escolher só olhando uma coluna é o erro.
    """
    from . import aporte_historico as ah

    if cripto not in precos.columns:
        raise ValueError(f"{cripto} não está na série de preços; não dá para estudar a fatia")

    linhas = []
    for fatia in fatias:
        try:
            pesos = com_cripto(pesos_base, fatia, cripto)
        except ValueError as exc:
            logger.warning("Fatia de %.1f%% inviável: %s", fatia * 100, exc)
            continue
        r = ah.simular(precos, cdi, pesos, ah.constante(aporte), **kwargs)
        linhas.append({
            "fatia_cripto_pct": round(fatia * 100, 1),
            "valor_liquido_final": round(r.valor_liquido_final, 2),
            "tir_anual_pct": round(r.tir_anual_pct, 2),
            "tir_cdi_pct": round(r.tir_cdi_pct, 2),
            "queda_da_cota_pct": round(r.queda_da_cota_pct, 2),
            "pior_mes_pct": round(r.pior_mes_pct, 2),
            "meses_no_prejuizo": r.meses_no_prejuizo,
        })

    if not linhas:
        raise ValueError("nenhuma fatia pôde ser simulada")
    return pd.DataFrame(linhas)


def estudar_fatia_em_janelas(
    precos: pd.DataFrame,
    cdi: pd.Series,
    pesos_base: dict[str, float],
    fatias: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05),
    anos: int = 5,
    passo_meses: int = 3,
    aporte: float = 500.0,
    cripto: str = "BTC-BRL",
    **kwargs,
) -> pd.DataFrame:
    """Repete o estudo começando em cada trimestre possível.

    Existe porque a versão de janela única dá a resposta errada com toda a
    aparência de estar certa. Medida de 2017 a 2026 — a melhor década que o
    bitcoin já teve e provavelmente terá —, qualquer fatia de cripto parece
    boa, e quanto maior, melhor. Isso não é uma conclusão sobre alocação, é
    uma constatação de que o ativo subiu naquele período.

    Com janelas móveis a pergunta muda para a certa: **em que fração dos
    períodos** a fatia ajudou, e não "quanto ela rendeu no período que eu
    escolhi olhar".
    """
    from . import aporte_historico as ah

    if cripto not in precos.columns:
        raise ValueError(f"{cripto} não está na série de preços")

    linhas = []
    for fatia in fatias:
        try:
            pesos = com_cripto(pesos_base, fatia, cripto)
        except ValueError as exc:
            logger.warning("Fatia de %.1f%% inviável: %s", fatia * 100, exc)
            continue
        try:
            janelas = ah.janelas_moveis(
                precos, cdi, pesos, ah.constante(aporte),
                anos=anos, passo_meses=passo_meses, **kwargs,
            )
        except ValueError as exc:
            raise ValueError(f"fatia de {fatia * 100:.0f}%: {exc}") from exc
        janelas["fatia_cripto_pct"] = round(fatia * 100, 1)
        linhas.append(janelas)

    if not linhas:
        raise ValueError("nenhuma fatia pôde ser simulada")
    return pd.concat(linhas, ignore_index=True)


def veredito_da_fatia(
    janelas: pd.DataFrame,
    tolerancia_de_queda_pp: float = 5.0,
    vitorias_minimas: float = 0.70,
) -> dict:
    """Diz se alguma fatia se paga na maioria dos períodos.

    Três exigências, e a fatia precisa das três:

    * ter rendido mais que a carteira sem cripto na **mediana** das janelas
    * ter rendido mais em pelo menos 70% delas, não só na média
    * não ter aprofundado a queda além da tolerância na janela **pior**

    A última é a que corta as fatias grandes. Uma fatia que melhora a mediana
    e destrói o pior caso não melhorou a carteira: passou o resultado bom para
    a frente e o ruim para o dia em que o dinheiro fizer falta.
    """
    if "fatia_cripto_pct" not in janelas.columns:
        raise ValueError("o estudo precisa da coluna fatia_cripto_pct")

    base = janelas[janelas["fatia_cripto_pct"] == 0]
    if base.empty:
        raise ValueError("o estudo precisa incluir a fatia de 0% como referência")

    chave = ["inicio"]
    ref = base.set_index(chave)[["tir_anual_pct", "queda_da_cota_pct"]]

    candidatas = []
    for fatia in sorted(janelas["fatia_cripto_pct"].unique()):
        if fatia == 0:
            continue
        alvo = janelas[janelas["fatia_cripto_pct"] == fatia].set_index(chave)
        comum = alvo.index.intersection(ref.index)
        if len(comum) < 3:
            continue

        ganho = alvo.loc[comum, "tir_anual_pct"] - ref.loc[comum, "tir_anual_pct"]
        piora = ref.loc[comum, "queda_da_cota_pct"] - alvo.loc[comum, "queda_da_cota_pct"]
        taxa = float((ganho > 0).mean())

        candidatas.append({
            "fatia_pct": float(fatia),
            "janelas": int(len(comum)),
            "ganho_mediano_pp": round(float(ganho.median()), 2),
            "ganho_no_pior_caso_pp": round(float(ganho.min()), 2),
            "taxa_de_vitoria_pct": round(taxa * 100, 1),
            "piora_mediana_da_queda_pp": round(float(piora.median()), 2),
            "piora_maxima_da_queda_pp": round(float(piora.max()), 2),
            "aceitavel": bool(
                float(ganho.median()) > 0
                and taxa >= vitorias_minimas
                and float(piora.max()) <= tolerancia_de_queda_pp
            ),
        })

    aceitaveis = [c for c in candidatas if c["aceitavel"]]
    # Entre as aceitáveis, a menor. Fatia maior de um ativo que já caiu 77%
    # duas vezes precisa de evidência muito melhor que "também passou".
    melhor = min(aceitaveis, key=lambda c: c["fatia_pct"]) if aceitaveis else None

    return {
        "janelas_por_fatia": int(len(base)),
        "candidatas": candidatas,
        "recomendada_pct": melhor["fatia_pct"] if melhor else 0.0,
        "conclusao": (
            f"a fatia de {melhor['fatia_pct']:.0f}% ganhou da carteira sem cripto em "
            f"{melhor['taxa_de_vitoria_pct']:.0f}% das janelas, com ganho mediano de "
            f"{melhor['ganho_mediano_pp']:.2f} ponto(s) e piora máxima de queda de "
            f"{melhor['piora_maxima_da_queda_pp']:.2f} ponto(s)"
            if melhor else
            "nenhuma fatia de cripto passou nas três exigências: a que melhora a mediana "
            "não sustenta a taxa de vitória ou estraga demais o pior caso"
        ),
        "avisos": [
            "O histórico de cripto começa em 2017 e cobre um período único — a maior "
            "alta que o ativo já teve. O teto de 5% existe porque a evidência é curta.",
            "As janelas se sobrepõem: janelas de 4 anos com passo de 3 meses dentro de "
            "9 anos de história compartilham quase todo o período. Uma taxa de vitória "
            "de 100% aqui vale muito menos do que valeria com janelas independentes.",
            "Isto mede cripto PARADA na carteira, com rebalanceamento. Não mede operar "
            "cripto, que foi testado em 32 configurações e reprovou em todas.",
        ],
    }
