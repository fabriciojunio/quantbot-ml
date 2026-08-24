"""
A conta: posição, caixa, base de cálculo, custo e imposto.

Existe por um motivo específico. Todo sistema destes acaba com duas
implementações da mesma contabilidade — uma no backtest e outra na operação —
e elas divergem. Divergem devagar, num arredondamento aqui, numa regra de
imposto que só foi corrigida de um lado ali, e no fim o backtest promete um
resultado que a operação não entrega. O painel do meu pai tinha exatamente
esse problema: o número da tela e o número da corretora não eram a mesma
conta.

Aqui a simulação histórica e a operação diária usam esta classe, e só ela sabe
mexer em posição e caixa. Se a regra de imposto mudar, muda num lugar.

As regras que ela implementa, todas da legislação brasileira para pessoa
física:

* **Custo médio** como base de cálculo do ganho, não PEPS. É o que a Receita
  manda usar em renda variável.
* **Custo de compra entra na base** e o de venda sai do ganho: os dois são
  dedutíveis, e ignorar isso faz o imposto sair maior que o devido.
* **ETF de renda variável paga 15%** sobre o ganho, sem a isenção de R$ 20 mil
  no mês — a isenção vale para ação individual, não para ETF, e essa confusão
  é comum o bastante para estar escrita aqui.
* **ETF de renda fixa segue a tabela regressiva**, de 22,5% a 15% conforme o
  prazo. A idade usada é a média ponderada por cota.
* **Prejuízo compensa lucro dentro da mesma modalidade**, e o saldo negativo
  atravessa meses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .custos_b3 import ModeloCustoB3
from .tributos import aliquota_regressiva
from .universo import UNIVERSO_B3, Ativo, Classe, Regime

logger = logging.getLogger("quantbot.carteira.conta")

ALIQUOTA_POR_REGIME: dict[Regime, float] = {
    Regime.ETF_RENDA_VARIAVEL: 0.15,
    Regime.ACAO_INDIVIDUAL: 0.15,
    Regime.FII: 0.20,
    Regime.ISENTO: 0.0,
    Regime.CRIPTO: 0.15,
}


def aliquota_de_venda(ativo: Ativo, dias: float) -> float:
    """Alíquota do regime do ativo, já considerando o prazo quando importa."""
    if ativo.regime in (Regime.ETF_RENDA_FIXA, Regime.TESOURO):
        return aliquota_regressiva(int(dias))
    return ALIQUOTA_POR_REGIME.get(ativo.regime, 0.15)


def ativo_generico(ticker: str) -> Ativo:
    """Para ticker fora do universo: spread conservador, regime de renda variável.

    Conservador de propósito. Errar para mais no custo faz o backtest prometer
    menos do que entrega; errar para menos faz o contrário, e o contrário é o
    que quebra gente.
    """
    return Ativo(
        ticker, ticker, Classe.ACOES_BR, Regime.ETF_RENDA_VARIAVEL,
        spread_pct=0.0015, volume_medio_brl=1e9,
    )


@dataclass
class Posicao:
    """Cotas, base de cálculo e idade média das cotas."""

    cotas: float = 0.0
    custo_total: float = 0.0
    # Soma de (cotas compradas x número do dia da compra). Dividida pelas cotas
    # dá o dia médio de entrada, e daí sai o prazo para a tabela regressiva.
    idade_ponderada: float = 0.0

    @property
    def custo_medio(self) -> float:
        return self.custo_total / self.cotas if self.cotas > 0 else 0.0

    def dias_medios(self, dia_atual: int) -> float:
        if self.cotas <= 0:
            return 0.0
        return max(0.0, dia_atual - self.idade_ponderada / self.cotas)

    def para_dict(self) -> dict:
        return {
            "cotas": self.cotas,
            "custo_total": round(self.custo_total, 6),
            "idade_ponderada": self.idade_ponderada,
        }

    @staticmethod
    def de_dict(d: dict) -> Posicao:
        return Posicao(
            cotas=float(d.get("cotas", 0.0)),
            custo_total=float(d.get("custo_total", 0.0)),
            idade_ponderada=float(d.get("idade_ponderada", 0.0)),
        )


@dataclass
class Movimento:
    """O registro de uma operação, com tudo que foi cobrado nela."""

    dia: int
    ticker: str
    tipo: str            # "compra" ou "venda"
    valor_bruto: float
    cotas: float
    preco: float
    custo: float
    imposto: float = 0.0
    ganho: float = 0.0
    motivo: str = ""

    @property
    def valor_liquido(self) -> float:
        """O que sai do caixa na compra, o que entra no caixa na venda."""
        if self.tipo == "compra":
            return self.valor_bruto
        return self.valor_bruto - self.custo - self.imposto

    def para_dict(self) -> dict:
        return {
            "dia": self.dia, "ticker": self.ticker, "tipo": self.tipo,
            "valor_bruto": round(self.valor_bruto, 2), "cotas": self.cotas,
            "preco": round(self.preco, 6), "custo": round(self.custo, 4),
            "imposto": round(self.imposto, 4), "ganho": round(self.ganho, 2),
            "motivo": self.motivo,
        }


@dataclass
class Conta:
    """Caixa, posições e o histórico do que foi cobrado.

    Não conhece preço de mercado: quem chama informa o preço de cada operação.
    Isso mantém a classe utilizável tanto com histórico quanto com cotação de
    hoje, sem nenhum caminho diferente entre os dois.
    """

    caixa: float = 0.0
    posicoes: dict[str, Posicao] = field(default_factory=dict)
    modelo: ModeloCustoB3 = field(default_factory=ModeloCustoB3)
    ativos: dict[str, Ativo] = field(default_factory=lambda: dict(UNIVERSO_B3))

    total_aportado: float = 0.0
    custo_pago: float = 0.0
    imposto_pago: float = 0.0
    # Prejuízo acumulado por modalidade, que abate lucro futuro.
    prejuizo_a_compensar: dict[str, float] = field(default_factory=dict)
    movimentos: list[Movimento] = field(default_factory=list)

    # Juros do caixa ainda não tributados, e a soma de (juros x dia) que dá a
    # idade média deles.
    #
    # Existem porque o caixa não é dinheiro embaixo do colchão: é CDB ou Tesouro
    # Selic, rende, e **paga imposto no resgate** pela tabela regressiva. Sem
    # este par, o caixa rendia livre de imposto enquanto a referência de CDI
    # pagava — e uma carteira 100% CDI "batia" a referência 100% CDI em
    # R$ 7.411 numa simulação de dez anos. O erro inflava todo perfil
    # conservador, que é justamente o que tem mais caixa.
    caixa_rendimento: float = 0.0
    caixa_idade: float = 0.0

    # ── Consulta ─────────────────────────────────────────────────────────

    def ativo(self, ticker: str) -> Ativo:
        return self.ativos.get(ticker) or ativo_generico(ticker)

    def modalidade(self, ticker: str) -> str:
        """Prejuízo só compensa lucro da mesma modalidade."""
        return self.ativo(ticker).regime.value

    def valor_investido(self, precos: dict[str, float]) -> float:
        return sum(p.cotas * precos.get(t, 0.0) for t, p in self.posicoes.items())

    def patrimonio(self, precos: dict[str, float]) -> float:
        return self.caixa + self.valor_investido(precos)

    def pesos(self, precos: dict[str, float]) -> dict[str, float]:
        total = self.patrimonio(precos)
        if total <= 0:
            return {}
        pesos = {t: (p.cotas * precos.get(t, 0.0)) / total for t, p in self.posicoes.items()}
        pesos["CDI"] = self.caixa / total
        return {t: p for t, p in pesos.items() if p > 1e-12}

    # ── Movimentação ─────────────────────────────────────────────────────

    def aportar(self, valor: float) -> None:
        if valor < 0:
            raise ValueError("aporte não pode ser negativo")
        self.caixa += valor
        self.total_aportado += valor

    def render_caixa(self, taxa_do_dia: float, dia: int = 0) -> float:
        """Aplica o rendimento do caixa. Devolve quanto rendeu.

        O rendimento fica marcado como não tributado até o resgate — que é o
        que acontece de verdade num CDB de liquidez diária ou no Tesouro Selic.
        """
        rendimento = self.caixa * taxa_do_dia
        if rendimento == 0:
            return 0.0
        self.caixa += rendimento
        self.caixa_rendimento += rendimento
        self.caixa_idade += rendimento * dia
        return rendimento

    def _idade_do_caixa(self, dia: int) -> float:
        """Idade média dos juros do caixa, ponderada pelo valor de cada crédito.

        É uma aproximação declarada, e vale saber para que lado ela erra. O
        modelo exato trataria cada aporte mensal como uma aplicação separada,
        com o próprio prazo — que é o que a referência de CDI faz. Aqui os
        juros são agrupados num pote só e envelhecem a partir do dia em que
        foram creditados, não do dia em que o principal entrou.

        Como juro creditado hoje conta como novo, a idade média sai **menor**
        que a real, a alíquota sai **maior**, e a carteira paga **mais** imposto
        do que pagaria. O erro medido é de cerca de 1% do caixa em sete anos a
        juro alto, e é contra a carteira — que é o lado certo para errar num
        sistema que decide onde vai dinheiro.
        """
        if self.caixa_rendimento <= 0:
            return 0.0
        return max(0.0, dia - self.caixa_idade / self.caixa_rendimento)

    def imposto_do_resgate(self, valor: float, dia: int) -> float:
        """Quanto de IR um resgate de `valor` do caixa geraria, sem cobrar."""
        if self.caixa <= 0 or self.caixa_rendimento <= 0 or valor <= 0:
            return 0.0
        fracao = min(1.0, valor / self.caixa)
        juros = self.caixa_rendimento * fracao
        return juros * aliquota_regressiva(int(self._idade_do_caixa(dia)))

    def _maximo_resgatavel(self, dia: int) -> float:
        """Quanto dá para gastar de fato, já descontando o IR do próprio resgate."""
        if self.caixa <= 0:
            return 0.0
        if self.caixa_rendimento <= 0:
            return self.caixa
        aliquota = aliquota_regressiva(int(self._idade_do_caixa(dia)))
        return self.caixa / (1 + self.caixa_rendimento * aliquota / self.caixa)

    def _resgatar(self, valor: float, dia: int) -> float:
        """Realiza a parte de juros de um resgate e cobra o IR. Devolve o imposto."""
        if self.caixa <= 0 or self.caixa_rendimento <= 0 or valor <= 0:
            return 0.0
        fracao = min(1.0, valor / self.caixa)
        juros = self.caixa_rendimento * fracao
        imposto = juros * aliquota_regressiva(int(self._idade_do_caixa(dia)))

        self.caixa_rendimento -= juros
        self.caixa_idade *= 1 - fracao
        self.caixa -= imposto
        self.imposto_pago += imposto
        return imposto

    def liquidar_caixa(self, dia: int) -> float:
        """Resgata tudo que sobrou no caixa, pagando o IR sobre os juros."""
        if self.caixa_rendimento <= 0:
            return 0.0
        imposto = self.caixa_rendimento * aliquota_regressiva(int(self._idade_do_caixa(dia)))
        self.caixa -= imposto
        self.imposto_pago += imposto
        self.caixa_rendimento = 0.0
        self.caixa_idade = 0.0
        return imposto

    def comprar(
        self, ticker: str, valor: float, preco: float, dia: int, motivo: str = ""
    ) -> Movimento | None:
        """Gasta `valor` do caixa comprando `ticker` ao `preco` informado."""
        if valor <= 0:
            return None
        if preco <= 0:
            raise ValueError(f"preço inválido para {ticker}: {preco}")
        # Pedir mais do que existe é erro de quem chamou, e não se conserta
        # sozinho: deixar passar vira alavancagem silenciosa.
        if valor > self.caixa + 1e-9:
            raise ValueError(
                f"compra de {valor:.2f} em {ticker} não cabe no caixa de {self.caixa:.2f}. "
                f"Deixar passar aqui vira alavancagem silenciosa."
            )

        # Pedir exatamente o caixa é pedido legítimo — é o que o rodízio faz
        # quando a carteira não tem fatia de caixa no alvo. Mas resgatar paga
        # IR sobre os juros, e o IR sai do mesmo caixa. Então a ordem encolhe
        # até caber nos dois, em vez de ser recusada.
        #
        # Resolvendo v + (v/caixa) x juros x alíquota = caixa:
        maximo = self._maximo_resgatavel(dia)
        if valor > maximo:
            valor = maximo
            if valor <= 0:
                return None

        ativo = self.ativo(ticker)
        custo = self.modelo.custo_ponta(ativo, valor)
        liquido = valor - custo
        if liquido <= 0:
            logger.warning("Ordem de %.2f em %s é menor que o próprio custo.", valor, ticker)
            return None

        cotas = liquido / preco
        p = self.posicoes.setdefault(ticker, Posicao())
        p.cotas += cotas
        p.custo_total += valor        # o custo de entrada é dedutível na venda
        p.idade_ponderada += cotas * dia

        self._resgatar(valor, dia)
        self.caixa -= valor
        self.custo_pago += custo

        mov = Movimento(dia, ticker, "compra", valor, cotas, preco, custo, motivo=motivo)
        self.movimentos.append(mov)
        return mov

    def vender(
        self, ticker: str, valor_desejado: float, preco: float, dia: int, motivo: str = ""
    ) -> Movimento | None:
        """Vende até `valor_desejado` de `ticker`, pagando custo e imposto."""
        p = self.posicoes.get(ticker)
        if p is None or p.cotas <= 0 or valor_desejado <= 0:
            return None
        if preco <= 0:
            raise ValueError(f"preço inválido para {ticker}: {preco}")

        valor = min(valor_desejado, p.cotas * preco)
        cotas = valor / preco
        ativo = self.ativo(ticker)
        custo = self.modelo.custo_ponta(ativo, valor)

        ganho_bruto = (preco - p.custo_medio) * cotas - custo
        imposto = self._imposto(ticker, ativo, ganho_bruto, p.dias_medios(dia))

        # A base sai proporcional às cotas vendidas, pelo custo médio.
        proporcao_restante = (p.cotas - cotas) / p.cotas
        p.custo_total -= p.custo_medio * cotas
        p.idade_ponderada *= proporcao_restante
        p.cotas -= cotas
        if p.cotas < 1e-12:
            self.posicoes.pop(ticker, None)

        self.caixa += valor - custo - imposto
        self.custo_pago += custo
        self.imposto_pago += imposto

        mov = Movimento(dia, ticker, "venda", valor, cotas, preco, custo,
                        imposto=imposto, ganho=ganho_bruto, motivo=motivo)
        self.movimentos.append(mov)
        return mov

    def liquidar(self, precos: dict[str, float], dia: int, motivo: str = "liquidação") -> float:
        """Vende tudo e devolve o caixa final, já com imposto pago.

        O caixa é resgatado primeiro, antes de as vendas entrarem. Depois das
        vendas o saldo mistura juros de CDB com produto de venda de ETF, e os
        dois têm regime de imposto diferente.
        """
        self.liquidar_caixa(dia)
        for ticker in list(self.posicoes):
            preco = precos.get(ticker)
            if preco is None or preco <= 0:
                raise ValueError(f"sem preço para liquidar {ticker}")
            self.vender(ticker, self.posicoes[ticker].cotas * preco, preco, dia, motivo)
        return self.caixa

    # ── Imposto ──────────────────────────────────────────────────────────

    def _imposto(self, ticker: str, ativo: Ativo, ganho: float, dias: float) -> float:
        """Imposto devido, já compensando prejuízo da mesma modalidade."""
        modalidade = self.modalidade(ticker)
        acumulado = self.prejuizo_a_compensar.get(modalidade, 0.0)

        if ganho <= 0:
            # Prejuízo entra no estoque e abate lucro futuro da mesma modalidade.
            self.prejuizo_a_compensar[modalidade] = acumulado - ganho
            return 0.0

        compensado = min(ganho, acumulado)
        self.prejuizo_a_compensar[modalidade] = acumulado - compensado
        tributavel = ganho - compensado
        return tributavel * aliquota_de_venda(ativo, dias)

    # ── Persistência ─────────────────────────────────────────────────────

    def para_dict(self) -> dict:
        return {
            "caixa": round(self.caixa, 6),
            "posicoes": {t: p.para_dict() for t, p in self.posicoes.items()},
            "total_aportado": round(self.total_aportado, 2),
            "custo_pago": round(self.custo_pago, 4),
            "imposto_pago": round(self.imposto_pago, 4),
            "prejuizo_a_compensar": {k: round(v, 4) for k, v in self.prejuizo_a_compensar.items()},
            "caixa_rendimento": round(self.caixa_rendimento, 6),
            "caixa_idade": round(self.caixa_idade, 6),
        }

    @staticmethod
    def de_dict(d: dict, modelo: ModeloCustoB3 | None = None,
                ativos: dict[str, Ativo] | None = None) -> Conta:
        return Conta(
            caixa=float(d.get("caixa", 0.0)),
            posicoes={t: Posicao.de_dict(p) for t, p in d.get("posicoes", {}).items()},
            modelo=modelo or ModeloCustoB3(),
            ativos=ativos if ativos is not None else dict(UNIVERSO_B3),
            total_aportado=float(d.get("total_aportado", 0.0)),
            custo_pago=float(d.get("custo_pago", 0.0)),
            imposto_pago=float(d.get("imposto_pago", 0.0)),
            prejuizo_a_compensar=dict(d.get("prejuizo_a_compensar", {})),
            caixa_rendimento=float(d.get("caixa_rendimento", 0.0)),
            caixa_idade=float(d.get("caixa_idade", 0.0)),
        )

    def conferir(self, precos: dict[str, float], tolerancia: float = 0.01) -> None:
        """Falha alto se a contabilidade não fechar.

        Chamada em todo fechamento. Um erro de contabilidade que não explode na
        hora vira um número errado que ninguém questiona porque "o sistema
        calculou".
        """
        if self.caixa < -tolerancia:
            raise ValueError(
                f"caixa negativo em {self.caixa:.2f}: isso é alavancagem, e o robô "
                f"não opera alavancado"
            )
        for ticker, p in self.posicoes.items():
            if p.cotas < 0:
                raise ValueError(f"posição negativa em {ticker}: {p.cotas}")
            if p.custo_total < -tolerancia:
                raise ValueError(f"base de cálculo negativa em {ticker}: {p.custo_total:.2f}")
            if ticker not in precos:
                raise ValueError(f"posição em {ticker} sem preço no fechamento")
        if self.patrimonio(precos) < -tolerancia:
            raise ValueError("patrimônio negativo")
