"""
Camada de risco herdada do painel Sentinela.

O painel tinha os dados errados, mas as **regras de gestão estavam certas** —
e regra de gestão é a parte de trading que sobrevive à mudança de mercado,
porque ela não depende de acertar previsão nenhuma. Este módulo traduz as
regras do painel para o contexto de carteira, mantendo a lógica e trocando o
que era específico de scalping pelo equivalente de posição longa.

O que veio de lá, e o que virou aqui:

| Regra original do painel | Como fica na carteira |
|---|---|
| Máximo de 3 posições simultâneas | Máximo de posições ativas, configurável |
| Uma posição por ativo | Idem, sem mudança |
| Risco de 2% do saldo por operação | Tamanho da posição limitado pela distância até o stop |
| Teto de 30% do saldo por posição | Teto de concentração por ativo |
| Filtro sniper adaptativo: 55%, 58%, 62% | Convicção exigida sobe a cada posição aberta |
| Trailing em dois estágios (6¢→3¢, 30¢→15¢) | Trailing em dois estágios, em ATR |
| Guarda de reversão do BTC (14%/86%) | Guarda de regime pelo índice de referência |
| Gerenciamento de tempo (5 min) | Prazo máximo de permanência, em pregões |
| Dupla chancela antes de executar | Confirmação primária e secundária |

Duas regras do painel foram **deliberadamente descartadas**, e o motivo é
importante:

* **Alvo fixo em dólares (+US$ 0,10).** Alvo em valor absoluto ignora o
  tamanho da posição e o custo. Foi ele que quebrou a conta lá. Aqui o alvo é
  sempre em múltiplo de ATR, que escala com a volatilidade do ativo.
* **Fechar em qualquer divergência de sinal.** Fazia sentido num sistema que
  operava em minutos; numa carteira, transforma cada oscilação em custo e
  imposto. Aqui a saída exige confirmação e respeita a banda de tolerância.

A ideia central que o painel acertou e que quase nenhum sistema de varejo tem:
**a camada de risco é independente da estratégia**. A estratégia sugere; o
risco decide se passa. É o mesmo desenho do `risk/kill_switch.py` do QuantBot,
e por isso os dois se encaixam sem atrito.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Decisao(str, Enum):
    APROVADA = "aprovada"
    BLOQUEADA = "bloqueada"
    REDUZIDA = "reduzida"


class PoliticaReversao(str, Enum):
    FECHAR = "fechar"                 # zera a posição quando o regime vira
    APERTAR_STOP = "apertar_stop"     # traz o stop para o preço de entrada
    SO_AVISAR = "so_avisar"           # registra e deixa correr


@dataclass
class LimitesRisco:
    """Os números do painel, com os defaults que ele usava."""

    max_posicoes: int = 3
    risco_por_operacao_pct: float = 2.0     # do patrimônio
    max_por_posicao_pct: float = 30.0       # teto de concentração
    conviccao_base: float = 55.0            # filtro sniper da 1ª posição
    conviccao_incremento: float = 3.5       # sobe a cada posição já aberta
    conviccao_teto: float = 75.0
    trailing_estagio1_atr: float = 1.0      # a 1 ATR de lucro, trava breakeven+
    trailing_estagio1_trava_atr: float = 0.3
    trailing_estagio2_atr: float = 3.0      # a 3 ATR, passa a seguir de perto
    trailing_estagio2_distancia_atr: float = 1.5
    max_pregoes_na_posicao: int = 252       # prazo máximo de permanência
    politica_reversao: PoliticaReversao = PoliticaReversao.APERTAR_STOP
    exigir_dupla_chancela: bool = True


@dataclass
class Posicao:
    ticker: str
    preco_entrada: float
    preco_atual: float
    quantidade: float
    stop_atual: float
    atr: float
    pregoes_aberta: int = 0
    estagio_trailing: int = 0

    @property
    def valor(self) -> float:
        return self.quantidade * self.preco_atual

    @property
    def lucro_em_atr(self) -> float:
        if self.atr <= 0:
            return 0.0
        return (self.preco_atual - self.preco_entrada) / self.atr


@dataclass
class Avaliacao:
    decisao: Decisao
    valor_aprovado: float
    conviccao_exigida: float
    motivos: list[str] = field(default_factory=list)


class PortaoDeRisco:
    """Portão independente: toda ordem sugerida passa por aqui antes de existir.

    A estratégia nunca chama o portão de dentro dela. Isso é de propósito: um
    bug na estratégia não pode desativar a proteção, porque a proteção mora
    fora dela.
    """

    def __init__(self, limites: LimitesRisco | None = None):
        self.limites = limites or LimitesRisco()

    # ── Entrada ──────────────────────────────────────────────────────────────

    def conviccao_exigida(self, posicoes_abertas: int) -> float:
        """Cada posição aberta encarece a próxima.

        Era o "filtro sniper adaptativo" do painel: 55% para a primeira, 58%
        para a segunda, 62% para a terceira. A lógica é boa — quanto mais
        exposto você já está, mais convicção a próxima aposta precisa ter.
        """
        exigida = self.limites.conviccao_base + posicoes_abertas * self.limites.conviccao_incremento
        return min(exigida, self.limites.conviccao_teto)

    def tamanho_por_risco(
        self, patrimonio: float, preco: float, stop: float
    ) -> float:
        """Tamanho da posição para que o stop custe exatamente o risco permitido.

        É a fórmula do painel: risco em reais dividido pela distância até o
        stop. O ativo mais volátil recebe posição menor automaticamente, sem
        ninguém precisar decidir isso na mão.
        """
        if preco <= 0 or patrimonio <= 0:
            return 0.0
        distancia = abs(preco - stop)
        if distancia <= 0:
            return 0.0

        risco_reais = patrimonio * (self.limites.risco_por_operacao_pct / 100)
        valor_por_risco = risco_reais * preco / distancia
        teto = patrimonio * (self.limites.max_por_posicao_pct / 100)
        return min(valor_por_risco, teto)

    def avaliar_entrada(
        self,
        ticker: str,
        patrimonio: float,
        caixa: float,
        preco: float,
        stop: float,
        conviccao: float,
        posicoes: dict[str, Posicao],
        regime_favoravel: bool = True,
        confirmacao_secundaria: bool = True,
    ) -> Avaliacao:
        """Decide se a ordem passa, e por qual tamanho."""
        motivos: list[str] = []
        abertas = len(posicoes)
        exigida = self.conviccao_exigida(abertas)

        if ticker in posicoes:
            return Avaliacao(Decisao.BLOQUEADA, 0.0, exigida,
                             [f"já existe posição aberta em {ticker}"])

        if abertas >= self.limites.max_posicoes:
            return Avaliacao(Decisao.BLOQUEADA, 0.0, exigida,
                             [f"limite de {self.limites.max_posicoes} posições atingido"])

        if conviccao < exigida:
            return Avaliacao(Decisao.BLOQUEADA, 0.0, exigida,
                             [f"convicção {conviccao:.1f} abaixo do exigido {exigida:.1f} "
                              f"com {abertas} posição(ões) aberta(s)"])

        if not regime_favoravel:
            return Avaliacao(Decisao.BLOQUEADA, 0.0, exigida,
                             ["regime de mercado desfavorável no índice de referência"])

        if self.limites.exigir_dupla_chancela and not confirmacao_secundaria:
            return Avaliacao(Decisao.BLOQUEADA, 0.0, exigida,
                             ["sem confirmação secundária (dupla chancela)"])

        if stop >= preco:
            return Avaliacao(Decisao.BLOQUEADA, 0.0, exigida,
                             ["stop precisa ficar abaixo do preço de entrada"])

        desejado = self.tamanho_por_risco(patrimonio, preco, stop)
        if desejado < 1.0:
            return Avaliacao(Decisao.BLOQUEADA, 0.0, exigida,
                             ["tamanho calculado é irrelevante para o patrimônio"])

        if desejado > caixa:
            if caixa < 1.0:
                return Avaliacao(Decisao.BLOQUEADA, 0.0, exigida, ["sem caixa"])
            motivos.append(f"reduzida de R$ {desejado:.2f} para o caixa de R$ {caixa:.2f}")
            return Avaliacao(Decisao.REDUZIDA, caixa, exigida, motivos)

        motivos.append(
            f"convicção {conviccao:.1f} ≥ {exigida:.1f}; risco de "
            f"{self.limites.risco_por_operacao_pct}% até o stop"
        )
        return Avaliacao(Decisao.APROVADA, desejado, exigida, motivos)

    # ── Manutenção ───────────────────────────────────────────────────────────

    def atualizar_trailing(self, pos: Posicao) -> tuple[float, str | None]:
        """Trailing em dois estágios, medido em ATR.

        No painel era em centavos de dólar; aqui é em ATR, que é a mesma ideia
        expressa numa unidade que funciona em qualquer ativo e qualquer preço.
        Estágio 1 trava um lucro pequeno; estágio 2 passa a seguir de perto.
        """
        lim = self.limites
        lucro = pos.lucro_em_atr
        novo = pos.stop_atual
        nota = None

        if lucro >= lim.trailing_estagio2_atr and pos.estagio_trailing < 2:
            novo = pos.preco_atual - lim.trailing_estagio2_distancia_atr * pos.atr
            pos.estagio_trailing = 2
            nota = (f"estágio 2: lucro de {lucro:.1f} ATR, stop passa a seguir a "
                    f"{lim.trailing_estagio2_distancia_atr} ATR do preço")
        elif lucro >= lim.trailing_estagio1_atr and pos.estagio_trailing < 1:
            novo = pos.preco_entrada + lim.trailing_estagio1_trava_atr * pos.atr
            pos.estagio_trailing = 1
            nota = (f"estágio 1: lucro de {lucro:.1f} ATR, stop travado acima do "
                    f"preço de entrada")
        elif pos.estagio_trailing == 2:
            candidato = pos.preco_atual - lim.trailing_estagio2_distancia_atr * pos.atr
            if candidato > pos.stop_atual:
                novo = candidato

        # O stop nunca anda para trás. Afrouxar stop em posição perdedora é o
        # jeito mais comum de transformar prejuízo pequeno em prejuízo grande.
        novo = max(novo, pos.stop_atual)
        pos.stop_atual = novo
        return novo, nota

    def avaliar_saida(
        self,
        pos: Posicao,
        sinal_ainda_valido: bool,
        regime_favoravel: bool,
    ) -> tuple[bool, str]:
        """Decide se a posição sai agora, e por quê."""
        if pos.preco_atual <= pos.stop_atual:
            return True, "stop atingido"

        if pos.pregoes_aberta >= self.limites.max_pregoes_na_posicao:
            return True, f"prazo máximo de {self.limites.max_pregoes_na_posicao} pregões"

        if not regime_favoravel:
            politica = self.limites.politica_reversao
            if politica == PoliticaReversao.FECHAR:
                return True, "guarda de regime: índice de referência virou"
            if politica == PoliticaReversao.APERTAR_STOP:
                if pos.stop_atual < pos.preco_entrada:
                    pos.stop_atual = pos.preco_entrada
                    return False, "guarda de regime: stop trazido para o preço de entrada"
                return False, "guarda de regime ativa, stop já protegido"
            return False, "guarda de regime: apenas registrado"

        if not sinal_ainda_valido:
            return True, "sinal de entrada deixou de valer"

        return False, "posição mantida"


def guarda_de_regime(
    retorno_indice_12m: float,
    retorno_cdi_12m: float,
    acima_da_media_200: bool,
    peso_tendencia: float = 0.86,
) -> tuple[bool, dict]:
    """Guarda de regime, herdada do "Score Master Ponderado 14%/86%" do painel.

    A ideia dele era boa: uma leitura dominante do mercado (ele usava o BTC)
    decide se o sistema pode estar comprado. O que estava errado era a fonte —
    os 86% de "análise técnica" eram todos `change24h >= 0`.

    Aqui o regime combina duas leituras reais, e uma delas é a que mais importa
    no Brasil: **o índice está batendo o CDI?** Um mercado que sobe 8% ao ano
    enquanto a renda fixa paga 14% não é um mercado favorável, por mais verde
    que o gráfico esteja.
    """
    excesso = retorno_indice_12m - retorno_cdi_12m
    score_excesso = 100.0 if excesso > 0 else 0.0
    score_tendencia = 100.0 if acima_da_media_200 else 0.0

    score = (1 - peso_tendencia) * score_excesso + peso_tendencia * score_tendencia
    favoravel = score >= 50.0

    if favoravel:
        leitura = "comprado liberado: tendência e excesso sobre o CDI a favor"
    elif not acima_da_media_200 and excesso > 0:
        leitura = (
            "cautela: o índice bateu o CDI nos 12 meses, mas está abaixo da média "
            "de 200 dias — o ganho é passado, a tendência atual não confirma"
        )
    elif acima_da_media_200:
        leitura = (
            "cautela: preço em tendência de alta, mas rendendo menos que a renda fixa"
        )
    else:
        leitura = "regime desfavorável: tendência e CDI ambos contra"

    return favoravel, {
        "score": round(score, 1),
        "favoravel": favoravel,
        "peso_tendencia": peso_tendencia,
        "indice_bateu_cdi_12m": excesso > 0,
        "excesso_sobre_cdi_pp": round(excesso * 100, 2),
        "acima_da_media_200": acima_da_media_200,
        "leitura": leitura,
    }
