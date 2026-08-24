"""
A operação diária: preço de verdade, dinheiro de mentira, registro permanente.

Roda todo dia útil. Lê a cotação real do dia, aplica exatamente as mesmas
regras que o backtest aplica — pela mesma classe `Conta` —, e grava tudo no
livro de auditoria. O dinheiro é simulado; o resto não é.

Por que rodar simulado por anos antes de usar dinheiro de verdade:

* **Backtest não passa por operação.** Um backtest não sabe que a fonte de
  preço caiu na quinta, que o feriado não estava no calendário, que o ETF fez
  desdobramento. Rodar em papel encontra esses problemas de graça.
* **O histórico é o produto.** A pergunta "esse sistema funciona?" só tem
  resposta com track record de fora da amostra. Cada dia rodado é um dia de
  evidência que não pode ser fabricada depois, porque a cadeia de hash não
  deixa.
* **Dá para desistir sem custo.** Se em dois anos o resultado for indistinguível
  do CDI, o prejuízo terá sido zero.

O estado vive dentro do próprio livro, no último evento de fechamento, e não
num arquivo à parte. Arquivo de estado separado pode ser editado sem deixar
rastro — o evento não pode, porque está sob o hash.

A trava de dinheiro real é fixa no código e não tem parâmetro que a desligue.
Ligar dinheiro de verdade tem que ser uma alteração de código, revisada, e não
uma flag que alguém troca às onze da noite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from . import juro_real
from .conta import Conta
from .custos_b3 import ModeloCustoB3
from .multimoeda import TETO_DE_CRIPTO, com_cripto, exposicao_cambial
from .perfil import Perfil
from .regras_pai import guarda_de_regime
from .universo import UNIVERSO_B3, Ativo, Classe, Regime

logger = logging.getLogger("quantbot.carteira.operacao")

# Não existe parâmetro que ligue dinheiro real. É de propósito.
DINHEIRO_REAL = False


@dataclass(frozen=True)
class Configuracao:
    """O que o usuário escolhe. Tudo o mais o robô decide."""

    aporte_mensal: float = 500.0
    risco_pct: float = 80.0
    fatia_cripto: float = 0.01
    rebalancear_a_cada_meses: int = 12
    banda_de_rebalanceamento: float = 0.05
    ordem_minima: float = 100.0
    reserva_por_fora: bool = True
    reduzir_em_regime_ruim: float = 0.75
    usar_teto_de_juro: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.risco_pct <= 100:
            raise ValueError("risco_pct vai de 0 a 100")
        if self.aporte_mensal < 0:
            raise ValueError("aporte não pode ser negativo")
        if not 0 <= self.fatia_cripto <= TETO_DE_CRIPTO:
            raise ValueError(f"fatia de cripto vai de 0 a {TETO_DE_CRIPTO}")
        if self.rebalancear_a_cada_meses < 1:
            raise ValueError("o rebalanceamento precisa de ao menos um mês")

    def para_dict(self) -> dict:
        return {
            "aporte_mensal": self.aporte_mensal,
            "risco_pct": self.risco_pct,
            "fatia_cripto_pct": round(self.fatia_cripto * 100, 2),
            "rebalancear_a_cada_meses": self.rebalancear_a_cada_meses,
            "banda_de_rebalanceamento": self.banda_de_rebalanceamento,
            "ordem_minima": self.ordem_minima,
            "reserva_por_fora": self.reserva_por_fora,
        }


ATIVO_CRIPTO = Ativo(
    "BTC-BRL", "Bitcoin em reais", Classe.CRIPTO, Regime.CRIPTO,
    spread_pct=0.0006, volume_medio_brl=1e11,
    observacao="Fatia mínima e limitada. Isento de IR abaixo de R$ 35 mil vendidos no mês.",
)


def universo_da_operacao() -> dict[str, Ativo]:
    return {**UNIVERSO_B3, "BTC-BRL": ATIVO_CRIPTO}


@dataclass
class Decisao:
    """O que foi feito no dia e por quê."""

    data: str
    dia_de_aporte: bool
    regime: dict
    risco_pedido_pct: float
    risco_aplicado_pct: float
    alvo: dict[str, float]
    movimentos: list[dict] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    juro: dict | None = None

    def para_dict(self) -> dict:
        return {
            "data": self.data,
            "dia_de_aporte": self.dia_de_aporte,
            "regime": self.regime,
            "juro": self.juro,
            "risco_pedido_pct": self.risco_pedido_pct,
            "risco_aplicado_pct": self.risco_aplicado_pct,
            "alvo_pct": {t: round(p * 100, 2) for t, p in
                         sorted(self.alvo.items(), key=lambda x: -x[1])},
            "movimentos": self.movimentos,
            "avisos": self.avisos,
        }


def e_primeiro_pregao_do_mes(hoje: pd.Timestamp, calendario: pd.DatetimeIndex) -> bool:
    """O aporte cai no primeiro pregão do mês, que é o que uma ordem programada pega.

    Comparar com `hoje.day == 1` seria errado: o dia 1 quase nunca é pregão, e
    o robô simplesmente nunca aportaria.
    """
    do_mes = calendario[(calendario.year == hoje.year) & (calendario.month == hoje.month)]
    return len(do_mes) > 0 and do_mes.min() == hoje


def ler_regime(
    precos: pd.DataFrame,
    cdi_acumulado: pd.Series,
    referencia: str = "BOVA11.SA",
    janela: int = 252,
) -> tuple[bool, dict]:
    """Regime de mercado pela regra herdada do painel do meu pai.

    Devolve favorável/desfavorável e a leitura em texto. Em regime ruim a
    exposição é reduzida, nunca zerada: zerar no fundo do poço é o erro que
    transforma queda temporária em prejuízo permanente.
    """
    if referencia not in precos.columns or len(precos) < janela + 1:
        return True, {"leitura": "sem histórico suficiente para ler o regime",
                      "favoravel": True, "incerto": True}

    ret_ref = float(precos[referencia].iloc[-1] / precos[referencia].iloc[-janela] - 1)
    ret_cdi = float(cdi_acumulado.iloc[-1] / cdi_acumulado.iloc[-janela] - 1)
    media = float(precos[referencia].tail(200).mean())
    acima = bool(precos[referencia].iloc[-1] > media)

    favoravel, leitura = guarda_de_regime(ret_ref, ret_cdi, acima)
    return favoravel, leitura


def alvo_do_dia(
    config: Configuracao,
    favoravel: bool,
    satelite_liberado: bool,
    composicao: dict | None = None,
    leitura_de_juro: juro_real.Leitura | None = None,
) -> tuple[dict[str, float], float, list[str]]:
    """Os pesos alvo de hoje, com teto de juro, ajuste de regime e cripto."""
    avisos: list[str] = []
    risco = config.risco_pct

    # O teto de juro real vem antes de tudo. É a guarda mais forte do sistema,
    # e a única que se mostrou capaz de decidir o resultado: nas 34 janelas de
    # cinco anos em que ela agiu, melhorou em 29 (p = 0,00002) e reduziu a
    # queda máxima de 4,85% para 0,47%.
    #
    # Ela não cria retorno — em juro real alto nada bate a renda fixa. Ela
    # impede correr risco que não está sendo pago.
    if leitura_de_juro is not None and config.usar_teto_de_juro:
        risco, aviso = juro_real.aplicar_teto(risco, leitura_de_juro)
        if aviso:
            avisos.append(aviso)

    if not favoravel and risco > 40:
        antes = risco
        risco = risco * config.reduzir_em_regime_ruim
        avisos.append(
            f"regime desfavorável: risco reduzido de {antes:.0f}% para "
            f"{risco:.0f}% (reduzido, não zerado — sair no fundo é pior que ficar)"
        )

    perfil = Perfil(f"operacao_{int(risco)}", risco, "orçamento da operação")
    alvo = perfil.pesos_por_ativo(satelite_liberado, composicao)

    if config.fatia_cripto > 0:
        try:
            alvo = com_cripto(alvo, config.fatia_cripto)
        except ValueError as exc:
            avisos.append(f"fatia de cripto não aplicada: {exc}")

    return alvo, risco, avisos


def rodar_dia(
    livro,
    config: Configuracao,
    precos: pd.DataFrame,
    cdi: pd.Series,
    hoje: pd.Timestamp | None = None,
    satelite_liberado: bool = False,
    composicao: dict | None = None,
    conta: Conta | None = None,
    leitura_de_juro: juro_real.Leitura | None = None,
) -> tuple[Conta, Decisao]:
    """Um dia de operação, do preço ao registro.

    Não aceita rodar duas vezes o mesmo dia com aporte: o segundo aporte
    entraria como dinheiro novo que não existiu. Reexecutar um dia é normal
    (a fonte de preço pode ter corrigido), e por isso o aporte é idempotente
    enquanto o resto é recalculado.
    """
    if precos.empty:
        raise ValueError("sem preços para operar")

    hoje = hoje or precos.index[-1]
    n_dia = int(precos.index.get_loc(hoje))
    hoje_precos = {t: float(v) for t, v in precos.loc[hoje].items() if pd.notna(v)}

    cdi_alinhado = cdi.reindex(precos.index).ffill().fillna(0.0)
    cdi_acumulado = (1 + cdi_alinhado).cumprod()

    if conta is None:
        estado = livro.ultimo_estado()
        conta = (
            Conta.de_dict(estado, ModeloCustoB3(), universo_da_operacao())
            if estado else
            Conta(modelo=ModeloCustoB3(), ativos=universo_da_operacao())
        )

    favoravel, regime = ler_regime(precos.loc[:hoje], cdi_acumulado.loc[:hoje])
    alvo, risco_aplicado, avisos = alvo_do_dia(
        config, favoravel, satelite_liberado, composicao, leitura_de_juro
    )

    faltando = sorted(set(alvo) - set(hoje_precos) - {"CDI"})
    if faltando:
        avisos.append(f"sem cotação hoje para {faltando}; esses pesos ficam no caixa")
        alvo = {t: p for t, p in alvo.items() if t in hoje_precos or t == "CDI"}
        total = sum(alvo.values())
        if total <= 0:
            raise ValueError("nenhum ativo do alvo tem cotação hoje")
        alvo = {t: p / total for t, p in alvo.items()}

    decisao = Decisao(
        data=str(hoje.date()), dia_de_aporte=False, regime=regime,
        risco_pedido_pct=config.risco_pct, risco_aplicado_pct=risco_aplicado,
        alvo=alvo, avisos=avisos,
        juro=leitura_de_juro.resumo() if leitura_de_juro else None,
    )

    # 1. O caixa rende antes de qualquer movimentação.
    conta.render_caixa(float(cdi_alinhado.loc[hoje]), n_dia)

    # 2. Aporte no primeiro pregão do mês.
    if e_primeiro_pregao_do_mes(hoje, precos.index) and config.aporte_mensal > 0:
        if _ja_aportou_no_mes(livro, hoje):
            avisos.append("aporte deste mês já registrado; não repetido")
        else:
            conta.aportar(config.aporte_mensal)
            decisao.dia_de_aporte = True
            from auditoria import TipoEvento

            livro.registrar(
                TipoEvento.APORTE, "operacao",
                f"Aporte mensal de R$ {config.aporte_mensal:.2f} em {hoje.date()}",
                {"data": str(hoje.date()), "valor": config.aporte_mensal, "simulado": True},
            )

    # 3. Investe o excedente por rodízio, e rebalanceia quando for o mês.
    if decisao.dia_de_aporte:
        _investir(conta, alvo, hoje_precos, n_dia, config, decisao)
        if _mes_de_rebalanceamento(livro, hoje, config.rebalancear_a_cada_meses):
            _rebalancear(conta, alvo, hoje_precos, n_dia, config, decisao)

    conta.conferir(hoje_precos)
    return conta, decisao


def _ja_aportou_no_mes(livro, hoje: pd.Timestamp) -> bool:
    from auditoria import TipoEvento

    prefixo = f"{hoje.year:04d}-{hoje.month:02d}"
    for e in livro.eventos(TipoEvento.APORTE):
        if str(e.dados.get("data", "")).startswith(prefixo):
            return True
    return False


def _mes_de_rebalanceamento(livro, hoje: pd.Timestamp, a_cada: int) -> bool:
    """Rebalanceia quando passaram `a_cada` meses do último rebalanceamento."""
    from auditoria import TipoEvento

    ultimos = [
        e for e in livro.eventos(TipoEvento.ORDEM)
        if e.dados.get("motivo") == "rebalanceamento"
    ]
    if not ultimos:
        # O primeiro rebalanceamento só faz sentido depois de haver carteira.
        aportes = livro.eventos(TipoEvento.APORTE)
        return len(aportes) >= a_cada

    ultima = pd.Timestamp(ultimos[-1].dados["data"])
    meses = (hoje.year - ultima.year) * 12 + (hoje.month - ultima.month)
    return meses >= a_cada


def _investir(conta, alvo, precos, n_dia, config, decisao) -> None:
    """Rodízio: concentra a compra no ativo mais atrasado.

    Com R$ 500 divididos entre seis ativos, cada ordem sai perto de R$ 80 e
    paga um spread inteiro sobre isso. Concentrar num só paga um spread em vez
    de seis e chega no mesmo peso alvo depois de alguns meses.
    """
    peso_caixa = alvo.get("CDI", 0.0)
    pat = conta.patrimonio(precos)
    investivel = min(conta.caixa, conta.caixa - pat * peso_caixa)
    if investivel < config.ordem_minima:
        decisao.avisos.append(
            f"R$ {max(investivel, 0):.2f} disponível é menos que a ordem mínima de "
            f"R$ {config.ordem_minima:.2f}; fica no caixa rendendo CDI até juntar"
        )
        return

    de_risco = {t: p for t, p in alvo.items() if t != "CDI"}
    if not de_risco:
        return

    atuais = conta.pesos(precos)
    falta = {t: de_risco[t] - atuais.get(t, 0.0) for t in de_risco}
    escolhido = max(falta, key=lambda k: falta[k])

    mov = conta.comprar(
        escolhido, investivel, precos[escolhido], n_dia,
        motivo=(
            f"rodízio: está {falta[escolhido] * 100:.1f} pontos abaixo do alvo de "
            f"{de_risco[escolhido] * 100:.0f}%"
        ),
    )
    if mov:
        decisao.movimentos.append(mov.para_dict())


def _rebalancear(conta, alvo, precos, n_dia, config, decisao) -> None:
    pat = conta.patrimonio(precos)
    if pat <= 0:
        return
    de_risco = {t: p for t, p in alvo.items() if t != "CDI"}
    limite = max(config.ordem_minima, pat * config.banda_de_rebalanceamento)

    def valor_de(t):
        p = conta.posicoes.get(t)
        return p.cotas * precos[t] if p else 0.0

    for t, peso in de_risco.items():
        excesso = valor_de(t) - pat * peso
        if excesso > limite:
            mov = conta.vender(t, excesso, precos[t], n_dia, "rebalanceamento")
            if mov:
                decisao.movimentos.append(mov.para_dict())

    disponivel = min(conta.caixa, conta.caixa - pat * alvo.get("CDI", 0.0))
    for t in sorted(de_risco, key=lambda k: pat * de_risco[k] - valor_de(k), reverse=True):
        if disponivel < config.ordem_minima:
            break
        deficit = pat * de_risco[t] - valor_de(t)
        if deficit > limite:
            gasto = min(deficit, disponivel, conta.caixa)
            mov = conta.comprar(t, gasto, precos[t], n_dia, "rebalanceamento")
            if mov:
                decisao.movimentos.append(mov.para_dict())
                disponivel -= gasto


def registrar_dia(livro, conta: Conta, decisao: Decisao, precos_do_dia: dict[str, float],
                  cdi_acumulado: float, cota: float) -> None:
    """Grava a decisão, as ordens e o fechamento no livro."""
    from auditoria import TipoEvento

    livro.registrar(
        TipoEvento.DECISAO, "operacao",
        f"{decisao.data}: {decisao.regime.get('leitura', 'regime não lido')}",
        decisao.para_dict(),
    )
    for mov in decisao.movimentos:
        livro.registrar(
            TipoEvento.ORDEM, "operacao",
            f"{mov['tipo']} de R$ {mov['valor_bruto']:.2f} em {mov['ticker']}",
            {**mov, "data": decisao.data, "simulado": True},
        )

    posicoes = {
        t: {
            "quantidade": p.cotas,
            "preco": precos_do_dia.get(t, 0.0),
            "valor": p.cotas * precos_do_dia.get(t, 0.0),
            "custo_medio": p.custo_medio,
        }
        for t, p in conta.posicoes.items()
    }
    livro.registrar_fechamento(
        data=decisao.data, caixa=conta.caixa, posicoes=posicoes,
        aportado=conta.total_aportado, cota=cota, cdi_acumulado=cdi_acumulado,
        conta=conta.para_dict(),
        extra={"exposicao_cambial": exposicao_cambial(conta.pesos(precos_do_dia)).resumo()},
    )


def hoje_e_dia_util(quando: date | None = None) -> bool:
    quando = quando or date.today()
    return quando.weekday() < 5
