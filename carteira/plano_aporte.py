"""
Plano de aporte mensal.

Responde a pergunta prática: *tenho R$ X por mês, para onde vai cada real?*

O plano é por fases, e a ordem não é negociável. Fase 2 só começa quando a
fase 1 fecha. Isso não é conservadorismo por gosto — é a única sequência em
que um imprevisto não obriga a vender posição no pior momento.

    Fase 1  Reserva de emergência    100% renda fixa líquida
    Fase 2  Núcleo diversificado     com fatia em dólar
    Fase 3  Satélite                 só depois das duas anteriores

Duas regras que existem por causa do tamanho do aporte:

**Não fatie R$ 500 em cinco ativos.** R$ 100 por ativo não compra lote decente,
paga spread proporcionalmente alto e gera cinco ordens onde caberia uma. O
plano usa **rodízio**: cada mês o aporte inteiro vai para o ativo que está
mais abaixo do peso alvo. Em cinco meses a carteira chega no mesmo lugar,
pagando um quinto do custo.

**Dólar não é ativo, é exposição.** Comprar IVVB11 dá bolsa americana *e*
dólar na mesma cota, sem conta no exterior, sem IOF de câmbio e com o ganho
tributado em 15%. Para quem tem despesa em real, é a forma mais barata de não
ficar 100% preso a uma moeda só.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .universo import UNIVERSO_B3, Ativo


class Fase(str, Enum):
    RESERVA = "reserva_de_emergencia"
    NUCLEO = "nucleo_diversificado"
    SATELITE = "satelite"


# Meses de despesa que a reserva precisa cobrir. PJ não tem FGTS, seguro-
# desemprego nem aviso prévio: quando a receita para, para inteira e no mesmo
# dia. Por isso o alvo dobra.
MESES_DE_RESERVA = {"clt": 6, "pj": 12, "servidor": 3}

# Núcleo sugerido. A fatia em dólar (IVVB11) é o que protege quem ganha em real
# de uma desvalorização da moeda; a fatia em IPCA+ protege da inflação; o resto
# é bolsa local e caixa.
NUCLEO_PADRAO: dict[str, float] = {
    "IVVB11.SA": 0.30,   # bolsa americana + dólar
    "IMAB11.SA": 0.25,   # Tesouro IPCA+ (juro real)
    "BOVA11.SA": 0.20,   # bolsa local
    "B5P211.SA": 0.15,   # IPCA+ curto, oscila pouco
    "GOLD11.SA": 0.10,   # ouro, seguro contra crise
}


@dataclass
class Situacao:
    """Onde a pessoa está hoje."""

    aporte_mensal: float
    despesa_mensal: float
    reserva_atual: float = 0.0
    carteira_atual: dict[str, float] = field(default_factory=dict)
    vinculo: str = "pj"

    @property
    def reserva_alvo(self) -> float:
        meses = MESES_DE_RESERVA.get(self.vinculo, 6)
        return self.despesa_mensal * meses

    @property
    def reserva_falta(self) -> float:
        return max(0.0, self.reserva_alvo - self.reserva_atual)

    @property
    def reserva_completa(self) -> bool:
        return self.reserva_falta <= 0

    @property
    def meses_para_completar(self) -> int | None:
        if self.reserva_completa:
            return 0
        if self.aporte_mensal <= 0:
            return None
        # Ignora o rendimento de propósito: é a estimativa conservadora, e a
        # diferença em 2 anos de CDI é pequena perto do erro da estimativa de
        # despesa.
        return int(-(-self.reserva_falta // self.aporte_mensal))


@dataclass
class Destino:
    onde: str
    valor: float
    fase: Fase
    motivo: str
    ativo: Ativo | None = None


def _pesos_atuais(carteira: dict[str, float]) -> dict[str, float]:
    total = sum(carteira.values())
    if total <= 0:
        return {}
    return {t: v / total for t, v in carteira.items()}


def escolher_por_rodizio(
    carteira: dict[str, float],
    alvo: dict[str, float],
) -> tuple[str, str]:
    """Escolhe o ativo mais abaixo do peso alvo.

    Com aporte pequeno, concentrar a compra do mês em um ativo só é o certo:
    uma ordem em vez de cinco, e a carteira converge para o alvo do mesmo
    jeito. Quem divide R$ 500 em cinco partes paga cinco spreads para chegar
    ao mesmo lugar.
    """
    atuais = _pesos_atuais(carteira)
    if not atuais:
        maior = max(alvo, key=alvo.get)
        return maior, "carteira vazia: começa pelo de maior peso alvo"

    faltas = {t: alvo.get(t, 0.0) - atuais.get(t, 0.0) for t in alvo}
    escolhido = max(faltas, key=faltas.get)
    falta_pp = faltas[escolhido] * 100
    return escolhido, (
        f"está {falta_pp:.1f} pontos abaixo do peso alvo "
        f"({alvo[escolhido] * 100:.0f}%) — é a maior lacuna da carteira"
    )


def montar(
    situacao: Situacao,
    nucleo: dict[str, float] | None = None,
    fatia_satelite: float = 0.0,
) -> dict:
    """Monta o plano do mês e explica cada real."""
    nucleo = nucleo or NUCLEO_PADRAO
    destinos: list[Destino] = []
    restante = situacao.aporte_mensal

    # ── Fase 1: reserva ──────────────────────────────────────────────────
    if not situacao.reserva_completa:
        para_reserva = min(restante, situacao.reserva_falta)
        destinos.append(
            Destino(
                onde="Renda fixa de liquidez diária (CDB/RDB com FGC ou Tesouro Selic)",
                valor=para_reserva,
                fase=Fase.RESERVA,
                motivo=(
                    f"faltam R$ {situacao.reserva_falta:,.2f} para a reserva de "
                    f"{MESES_DE_RESERVA.get(situacao.vinculo, 6)} meses de despesa"
                ),
            )
        )
        restante -= para_reserva

    # ── Fase 2: núcleo ───────────────────────────────────────────────────
    if restante > 0:
        para_satelite = restante * fatia_satelite
        para_nucleo = restante - para_satelite

        if para_nucleo >= 1:
            ticker, motivo = escolher_por_rodizio(situacao.carteira_atual, nucleo)
            destinos.append(
                Destino(
                    onde=ticker,
                    valor=para_nucleo,
                    fase=Fase.NUCLEO,
                    motivo=motivo,
                    ativo=UNIVERSO_B3.get(ticker),
                )
            )
            restante -= para_nucleo

        if para_satelite >= 1:
            destinos.append(
                Destino(
                    onde="Satélite (só com estratégia aprovada na validação)",
                    valor=para_satelite,
                    fase=Fase.SATELITE,
                    motivo="fatia de risco declarada; hoje a validação não aprovou nada",
                )
            )
            restante -= para_satelite

    exposicao_dolar = nucleo.get("IVVB11.SA", 0.0) * 100

    return {
        "aporte_mensal": round(situacao.aporte_mensal, 2),
        "fase_atual": (
            Fase.RESERVA.value if not situacao.reserva_completa else Fase.NUCLEO.value
        ),
        "reserva": {
            "alvo": round(situacao.reserva_alvo, 2),
            "atual": round(situacao.reserva_atual, 2),
            "falta": round(situacao.reserva_falta, 2),
            "completa": situacao.reserva_completa,
            "meses_para_completar": situacao.meses_para_completar,
            "meses_de_despesa_cobertos": round(
                situacao.reserva_atual / situacao.despesa_mensal, 1
            )
            if situacao.despesa_mensal > 0
            else 0.0,
        },
        "destinos": [
            {
                "onde": d.onde,
                "valor": round(d.valor, 2),
                "fatia_do_aporte_pct": round(d.valor / situacao.aporte_mensal * 100, 1)
                if situacao.aporte_mensal
                else 0.0,
                "fase": d.fase.value,
                "motivo": d.motivo,
                "spread_estimado_pct": round(d.ativo.spread_pct * 100, 3) if d.ativo else 0.0,
            }
            for d in destinos
        ],
        "sobrou": round(restante, 2),
        "nucleo_alvo": {t: round(p * 100, 1) for t, p in nucleo.items()},
        "exposicao_a_dolar_no_nucleo_pct": round(exposicao_dolar, 1),
        "avisos": _avisos(situacao, nucleo),
    }


def _avisos(situacao: Situacao, nucleo: dict[str, float]) -> list[str]:
    avisos = []

    if situacao.despesa_mensal <= 0:
        avisos.append(
            "Despesa mensal não informada: sem ela não dá para dimensionar a reserva. "
            "Some aluguel, comida, transporte, plano de saúde e contas fixas."
        )

    if not situacao.reserva_completa:
        avisos.append(
            "Enquanto a reserva não fecha, nada vai para renda variável. "
            "Emergência e crise de mercado costumam acontecer no mesmo mês."
        )

    if situacao.vinculo == "pj":
        avisos.append(
            "Como PJ você não tem FGTS, seguro-desemprego nem aviso prévio. "
            "Por isso o alvo é 12 meses de despesa, e não 6."
        )

    if situacao.aporte_mensal < 300:
        avisos.append(
            "Com aporte abaixo de R$ 300, o rodízio mensal é ainda mais importante: "
            "dividir em vários ativos faz o custo comer o aporte."
        )

    if nucleo.get("IVVB11.SA", 0) == 0:
        avisos.append(
            "Sem nenhuma exposição a dólar, todo o patrimônio depende de uma moeda só."
        )

    return avisos


def projetar_fases(situacao: Situacao, meses: int = 60) -> list[dict]:
    """Mostra em que mês cada fase começa, para o plano ter data e não só ordem."""
    linhas = []
    reserva = situacao.reserva_atual
    nucleo_acumulado = 0.0

    for m in range(1, meses + 1):
        sobra = situacao.aporte_mensal
        if reserva < situacao.reserva_alvo:
            vai = min(sobra, situacao.reserva_alvo - reserva)
            reserva += vai
            sobra -= vai
        nucleo_acumulado += sobra

        if m == 1 or m % 6 == 0 or m == meses:
            linhas.append({
                "mes": m,
                "ano": round(m / 12, 1),
                "reserva": round(reserva, 2),
                "meses_de_despesa": round(reserva / situacao.despesa_mensal, 1)
                if situacao.despesa_mensal > 0
                else 0.0,
                "nucleo": round(nucleo_acumulado, 2),
                "fase": (
                    Fase.RESERVA.value
                    if reserva < situacao.reserva_alvo
                    else Fase.NUCLEO.value
                ),
            })

    return linhas
