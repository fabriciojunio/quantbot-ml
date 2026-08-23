"""
Simulação de aporte mensal com dado real, custo real e imposto real.

A pergunta que este módulo responde é "quanto eu teria hoje se tivesse aportado
R$ 500 por mês nos últimos N anos". Ela parece simples e quase toda resposta
que circula por aí está errada, por três motivos:

1. **Ignoram o custo.** Com R$ 500 por mês, dividir o aporte entre seis ativos
   paga seis spreads sobre valores minúsculos. É por isso que o motor usa
   rodízio: concentra a compra no ativo mais atrasado e paga um spread só.
2. **Ignoram o imposto.** Rebalancear vende, vender realiza ganho, ganho paga
   15% em ETF de renda variável. Um backtest sem imposto exagera o resultado
   de forma crescente com o tempo.
3. **Usam uma janela só.** "Rendeu 14% ao ano nos últimos 5 anos" depende
   inteiramente de quais 5 anos. A resposta honesta é a distribuição de todas
   as janelas possíveis, com a pior delas em destaque.

Por isso a saída principal daqui não é um número: é `janelas_moveis()`, que
roda a mesma estratégia começando em todo mês possível do histórico e devolve
mediana, pior caso e melhor caso.

A métrica de retorno é a **TIR** (taxa interna de retorno), não o CAGR. Com
aporte mensal o CAGR não significa nada: o dinheiro do último mês ficou
investido um mês, e o do primeiro ficou o período inteiro. A TIR é a taxa que
faz o valor presente dos aportes bater com o valor final, e é a única
comparável entre estratégias com fluxos diferentes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .custos_b3 import ModeloCustoB3
from .tributos import aliquota_regressiva
from .universo import UNIVERSO_B3, Ativo, Classe, Regime

logger = logging.getLogger("quantbot.carteira.aporte")

# Abaixo disto o spread come a ordem e é melhor deixar acumulando no caixa,
# que rende CDI enquanto espera.
ORDEM_MINIMA_BRL = 100.0

# Alíquota de ganho de capital por regime, para o que não é tabela regressiva.
ALIQUOTA_POR_REGIME: dict[Regime, float] = {
    Regime.ETF_RENDA_VARIAVEL: 0.15,
    Regime.ACAO_INDIVIDUAL: 0.15,
    Regime.FII: 0.20,
    Regime.ISENTO: 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Calendários de aporte
# ─────────────────────────────────────────────────────────────────────────────

def constante(valor: float) -> Callable[[int], float]:
    """R$ X todo mês, para sempre."""
    if valor < 0:
        raise ValueError("aporte não pode ser negativo")
    return lambda _mes: valor


def degrau(inicial: float, passo: float, a_cada_meses: int) -> Callable[[int], float]:
    """Começa em `inicial` e sobe `passo` a cada `a_cada_meses`.

    `degrau(500, 200, 24)` é R$ 500 nos meses 0 a 23, R$ 700 nos meses 24 a 47,
    R$ 900 nos meses 48 a 71, e assim por diante.
    """
    if a_cada_meses <= 0:
        raise ValueError("o intervalo do degrau precisa ser positivo")
    if inicial < 0 or passo < 0:
        raise ValueError("aporte e passo não podem ser negativos")
    return lambda mes: inicial + passo * (mes // a_cada_meses)


def corrigido_pela_inflacao(inicial: float, ipca_anual: float) -> Callable[[int], float]:
    """Aporte que mantém o poder de compra. Sobe pela inflação todo mês."""
    if ipca_anual < -0.5:
        raise ValueError("inflação implausível")
    mensal = (1 + ipca_anual) ** (1 / 12) - 1
    return lambda mes: inicial * (1 + mensal) ** mes


# ─────────────────────────────────────────────────────────────────────────────
# Taxa interna de retorno
# ─────────────────────────────────────────────────────────────────────────────

def tir_anual(fluxos: list[tuple[float, float]], tolerancia: float = 1e-7) -> float:
    """TIR anualizada de uma lista de (anos_desde_o_inicio, valor).

    Aporte entra negativo, resgate entra positivo. Resolve por bisseção, que é
    lenta e não diverge — o oposto de Newton, que é rápido e diverge exatamente
    nos casos ruins que aparecem em janela de prejuízo.

    Devolve NaN quando não há troca de sinal (não existe TIR) ou quando a raiz
    cai fora de um intervalo plausível.
    """
    if len(fluxos) < 2:
        return float("nan")
    valores = [v for _, v in fluxos]
    if not (min(valores) < 0 < max(valores)):
        return float("nan")

    def vpl(taxa: float) -> float:
        return sum(v / (1 + taxa) ** t for t, v in fluxos)

    baixo, alto = -0.9999, 10.0
    f_baixo, f_alto = vpl(baixo), vpl(alto)
    if f_baixo * f_alto > 0:
        return float("nan")

    for _ in range(200):
        meio = (baixo + alto) / 2
        f_meio = vpl(meio)
        if abs(f_meio) < tolerancia or (alto - baixo) < tolerancia:
            return meio
        if f_baixo * f_meio < 0:
            alto, f_alto = meio, f_meio
        else:
            baixo, f_baixo = meio, f_meio
    return (baixo + alto) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Estado da carteira
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Posicao:
    """Uma posição, com base de cálculo do imposto e idade média das cotas.

    O custo é **médio**, que é o que a Receita manda usar para ganho de
    capital. A idade é média ponderada por cota, e serve só para escolher a
    alíquota da tabela regressiva nos ETFs de renda fixa.
    """

    cotas: float = 0.0
    custo_total: float = 0.0
    idade_ponderada: float = 0.0  # soma de cotas x dias, atualizada na compra

    @property
    def custo_medio(self) -> float:
        return self.custo_total / self.cotas if self.cotas > 0 else 0.0

    def dias_medios(self, dia_atual: int) -> float:
        if self.cotas <= 0:
            return 0.0
        return max(0.0, dia_atual - self.idade_ponderada / self.cotas)


@dataclass
class ResultadoAporte:
    """O que a simulação produziu, com tudo que foi pago pelo caminho."""

    inicio: pd.Timestamp
    fim: pd.Timestamp
    meses: int
    total_aportado: float
    valor_bruto_final: float
    custo_total_pago: float
    imposto_total_pago: float
    valor_liquido_final: float
    tir_anual_pct: float
    # Mesmo fluxo de aportes, mas em CDI, com IR regressivo por aporte.
    liquido_cdi: float
    tir_cdi_pct: float
    max_drawdown_pct: float
    pior_mes_pct: float
    queda_da_cota_pct: float
    meses_no_prejuizo: int
    curva: pd.Series = field(repr=False, default_factory=pd.Series)
    curva_cota: pd.Series = field(repr=False, default_factory=pd.Series)
    ordens: int = 0

    @property
    def lucro(self) -> float:
        return self.valor_liquido_final - self.total_aportado

    @property
    def vantagem_sobre_cdi(self) -> float:
        return self.valor_liquido_final - self.liquido_cdi

    @property
    def multiplo(self) -> float:
        return self.valor_liquido_final / self.total_aportado if self.total_aportado else 0.0

    def resumo(self) -> dict:
        return {
            "inicio": str(self.inicio.date()),
            "fim": str(self.fim.date()),
            "meses": self.meses,
            "total_aportado": round(self.total_aportado, 2),
            "valor_liquido_final": round(self.valor_liquido_final, 2),
            "lucro": round(self.lucro, 2),
            "multiplo": round(self.multiplo, 3),
            "tir_anual_pct": round(self.tir_anual_pct, 2),
            "liquido_cdi": round(self.liquido_cdi, 2),
            "tir_cdi_pct": round(self.tir_cdi_pct, 2),
            "vantagem_sobre_cdi": round(self.vantagem_sobre_cdi, 2),
            "custo_total_pago": round(self.custo_total_pago, 2),
            "imposto_total_pago": round(self.imposto_total_pago, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "pior_mes_pct": round(self.pior_mes_pct, 2),
            "queda_da_cota_pct": round(self.queda_da_cota_pct, 2),
            "meses_no_prejuizo": self.meses_no_prejuizo,
            "ordens": self.ordens,
        }


def _aliquota_de_venda(ativo: Ativo, dias: float) -> float:
    """Alíquota correta para o regime do ativo."""
    if ativo.regime in (Regime.ETF_RENDA_FIXA, Regime.TESOURO):
        return aliquota_regressiva(int(dias))
    return ALIQUOTA_POR_REGIME.get(ativo.regime, 0.15)


def _primeiros_dias_uteis(indice: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """O primeiro pregão de cada mês do índice.

    O dia 1 costuma não ser pregão, então "todo dia 5" ou "todo dia 1" não
    existe no calendário da bolsa. O primeiro pregão do mês existe sempre e é
    o que uma ordem programada de verdade acabaria pegando.
    """
    s = pd.Series(indice, index=indice)
    return list(s.groupby(indice.to_period("M"), sort=True).min())


# ─────────────────────────────────────────────────────────────────────────────
# O simulador
# ─────────────────────────────────────────────────────────────────────────────

def simular(
    precos: pd.DataFrame,
    cdi: pd.Series,
    pesos: dict[str, float],
    aporte: Callable[[int], float] | float = 500.0,
    *,
    modelo: ModeloCustoB3 | None = None,
    ativos: dict[str, Ativo] | None = None,
    rebalancear_a_cada_meses: int = 12,
    banda_de_rebalanceamento: float = 0.05,
    ordem_minima: float = ORDEM_MINIMA_BRL,
    liquidar_no_fim: bool = True,
) -> ResultadoAporte:
    """Roda o aporte mensal do começo ao fim da série de preços.

    `pesos` pode conter a chave especial ``"CDI"``, que representa o caixa
    remunerado. O caixa acumula o que ainda não foi investido e rende CDI todo
    dia útil — é o que acontece de verdade quando o aporte do mês não chega ao
    tamanho mínimo de ordem.

    Se `liquidar_no_fim`, vende tudo no último dia e paga o imposto devido.
    É o número que interessa: o que sobraria na conta, não o de tela.
    """
    if precos.empty:
        raise ValueError("sem preços para simular")
    if not precos.index.is_monotonic_increasing:
        raise ValueError("índice de preços fora de ordem")

    modelo = modelo or ModeloCustoB3()
    ativos = ativos or UNIVERSO_B3
    fn_aporte = aporte if callable(aporte) else constante(float(aporte))

    peso_caixa = float(pesos.get("CDI", 0.0))
    pesos_risco = {t: p for t, p in pesos.items() if t != "CDI"}

    ausentes = sorted(set(pesos_risco) - set(precos.columns))
    if ausentes:
        raise ValueError(
            f"os ativos {ausentes} têm peso mas não estão na série de preços. "
            f"Renormalizar aqui produziria uma carteira diferente da pedida."
        )
    soma = sum(pesos.values())
    if abs(soma - 1.0) > 1e-6:
        raise ValueError(f"os pesos somam {soma:.4f}, não 1")

    cdi_dia = cdi.reindex(precos.index).ffill().fillna(0.0)

    caixa = 0.0
    posicoes: dict[str, Posicao] = {t: Posicao() for t in pesos_risco}
    total_aportado = 0.0
    custo_pago = 0.0
    imposto_pago = 0.0
    ordens = 0
    fluxos: list[tuple[float, float]] = []
    fluxos_cdi: list[tuple[pd.Timestamp, float]] = []
    curva: list[float] = []
    # Curva de cota: patrimônio limpo do efeito do aporte. Sem ela, "o pior mês
    # da carteira" sai positivo em toda simulação — porque o dinheiro novo do
    # mês entra na conta e mascara a queda do mercado.
    cotas_twr: list[float] = []
    cota = 1.0
    pat_ontem = 0.0

    dias_de_aporte = set(_primeiros_dias_uteis(precos.index))
    inicio = precos.index[0]
    mes_do_aporte = 0
    proximo_rebalanceamento = rebalancear_a_cada_meses

    def patrimonio(dia: pd.Timestamp) -> float:
        return caixa + sum(p.cotas * float(precos.at[dia, t]) for t, p in posicoes.items())

    def vender(t: str, valor_desejado: float, dia: pd.Timestamp, n_dia: int) -> float:
        """Vende até `valor_desejado` de `t`. Devolve o caixa líquido gerado."""
        nonlocal custo_pago, imposto_pago, ordens
        p = posicoes[t]
        preco = float(precos.at[dia, t])
        valor_posicao = p.cotas * preco
        if valor_posicao <= 0 or valor_desejado <= 0:
            return 0.0

        valor = min(valor_desejado, valor_posicao)
        cotas = valor / preco
        ativo = ativos.get(t) or _ativo_generico(t)

        custo = modelo.custo_ponta(ativo, valor)
        ganho = max(0.0, (preco - p.custo_medio) * cotas - custo)
        imposto = ganho * _aliquota_de_venda(ativo, p.dias_medios(n_dia))

        # Base de cálculo sai proporcional às cotas vendidas (custo médio).
        p.idade_ponderada *= (p.cotas - cotas) / p.cotas if p.cotas > 0 else 0.0
        p.custo_total -= p.custo_medio * cotas
        p.cotas -= cotas
        if p.cotas < 1e-12:
            p.cotas, p.custo_total, p.idade_ponderada = 0.0, 0.0, 0.0

        custo_pago += custo
        imposto_pago += imposto
        ordens += 1
        return valor - custo - imposto

    def comprar(t: str, valor: float, dia: pd.Timestamp, n_dia: int) -> None:
        nonlocal caixa, custo_pago, ordens
        if valor <= 0:
            return
        ativo = ativos.get(t) or _ativo_generico(t)
        custo = modelo.custo_ponta(ativo, valor)
        liquido = valor - custo
        if liquido <= 0:
            return
        preco = float(precos.at[dia, t])
        cotas = liquido / preco

        p = posicoes[t]
        p.cotas += cotas
        # O custo de entrada entra na base: é dedutível do ganho na venda.
        p.custo_total += valor
        p.idade_ponderada += cotas * n_dia

        caixa -= valor
        custo_pago += custo
        ordens += 1

    for n_dia, dia in enumerate(precos.index):
        # 1. O caixa rende antes de qualquer movimentação do dia.
        caixa *= 1 + float(cdi_dia.iloc[n_dia])

        # 1b. Retorno do dia, medido ANTES do aporte entrar. É o que separa
        # desempenho de depósito.
        pat_pre = patrimonio(dia)
        if pat_ontem > 0:
            cota *= pat_pre / pat_ontem
        cotas_twr.append(cota)

        # 2. Aporte, no primeiro pregão do mês.
        if dia in dias_de_aporte:
            valor = float(fn_aporte(mes_do_aporte))
            if valor > 0:
                caixa += valor
                total_aportado += valor
                anos = (dia - inicio).days / 365.25
                fluxos.append((anos, -valor))
                fluxos_cdi.append((dia, valor))
            mes_do_aporte += 1

            # 3. Investe o que passa do alvo de caixa, por rodízio.
            pat = patrimonio(dia)
            investivel = caixa - pat * peso_caixa
            if investivel >= ordem_minima and pesos_risco:
                atual = {
                    t: (posicoes[t].cotas * float(precos.at[dia, t])) / pat if pat > 0 else 0.0
                    for t in pesos_risco
                }
                falta = {t: pesos_risco[t] - atual[t] for t in pesos_risco}
                escolhido = max(falta, key=lambda k: falta[k])
                comprar(escolhido, investivel, dia, n_dia)

            # 4. Rebalanceamento periódico.
            if mes_do_aporte >= proximo_rebalanceamento:
                proximo_rebalanceamento += rebalancear_a_cada_meses
                pat = patrimonio(dia)
                if pat > 0:
                    alvos = {t: pat * pesos_risco[t] for t in pesos_risco}
                    # Vende o excesso primeiro: gera o caixa que paga as compras.
                    for t in pesos_risco:
                        atual_v = posicoes[t].cotas * float(precos.at[dia, t])
                        excesso = atual_v - alvos[t]
                        if excesso > max(ordem_minima, pat * banda_de_rebalanceamento):
                            caixa += vender(t, excesso, dia, n_dia)
                    # Depois compra o que falta, com o que houver.
                    disponivel = caixa - pat * peso_caixa
                    for t in sorted(pesos_risco, key=lambda k: alvos[k] - posicoes[k].cotas * float(precos.at[dia, k]), reverse=True):
                        if disponivel < ordem_minima:
                            break
                        atual_v = posicoes[t].cotas * float(precos.at[dia, t])
                        deficit = alvos[t] - atual_v
                        if deficit > max(ordem_minima, pat * banda_de_rebalanceamento):
                            gasto = min(deficit, disponivel)
                            comprar(t, gasto, dia, n_dia)
                            disponivel -= gasto

        pat_ontem = patrimonio(dia)
        curva.append(pat_ontem)

    fim = precos.index[-1]
    n_final = len(precos) - 1
    bruto_final = patrimonio(fim)

    if liquidar_no_fim:
        for t in list(posicoes):
            caixa += vender(t, posicoes[t].cotas * float(precos.at[fim, t]), fim, n_final)
        liquido_final = caixa
    else:
        liquido_final = bruto_final

    anos_totais = (fim - inicio).days / 365.25
    fluxos.append((anos_totais, liquido_final))

    serie = pd.Series(curva, index=precos.index)
    serie_cota = pd.Series(cotas_twr, index=precos.index)
    liq_cdi, tir_cdi = _referencia_cdi(fluxos_cdi, cdi, fim)

    return ResultadoAporte(
        inicio=inicio,
        fim=fim,
        meses=mes_do_aporte,
        total_aportado=total_aportado,
        valor_bruto_final=bruto_final,
        custo_total_pago=custo_pago,
        imposto_total_pago=imposto_pago,
        valor_liquido_final=liquido_final,
        tir_anual_pct=tir_anual(fluxos) * 100,
        liquido_cdi=liq_cdi,
        tir_cdi_pct=tir_cdi * 100,
        max_drawdown_pct=_drawdown_de_aporte(serie, precos.index, fn_aporte, dias_de_aporte),
        pior_mes_pct=_pior_mes(serie_cota),
        queda_da_cota_pct=_queda_maxima(serie_cota),
        meses_no_prejuizo=_meses_no_prejuizo(serie, precos.index, fn_aporte, dias_de_aporte),
        curva=serie,
        curva_cota=serie_cota,
        ordens=ordens,
    )


def _ativo_generico(ticker: str) -> Ativo:
    """Para quando o ticker não está no universo: spread conservador."""
    return Ativo(
        ticker, ticker, Classe.ACOES_BR, Regime.ETF_RENDA_VARIAVEL,
        spread_pct=0.0015, volume_medio_brl=1e9,
    )


def _referencia_cdi(
    aportes: list[tuple[pd.Timestamp, float]],
    cdi: pd.Series,
    fim: pd.Timestamp,
) -> tuple[float, float]:
    """Os mesmos aportes em CDB de 100% do CDI, com IR regressivo por aporte.

    Cada aporte tem o próprio prazo, então cada um paga a própria alíquota. É
    a diferença entre a comparação honesta e a de propaganda, que aplica 15%
    em tudo e faz a renda fixa parecer melhor do que é no começo.
    """
    if not aportes:
        return 0.0, float("nan")

    # Cortar no fim da janela é obrigatório. Sem isto, uma janela que termina
    # em 2015 compunha CDI até 2026 e a referência ficava várias vezes maior
    # que a carteira — o backtest passava a "perder do CDI" em quase toda
    # janela antiga, por um erro de recorte e não por desempenho.
    cdi = cdi.loc[cdi.index <= fim]
    if cdi.empty:
        return 0.0, float("nan")

    fator = (1 + cdi).cumprod()
    total_liquido = 0.0
    fluxos: list[tuple[float, float]] = []
    inicio = aportes[0][0]

    for data, valor in aportes:
        idx = fator.index.searchsorted(data)
        if idx >= len(fator):
            idx = len(fator) - 1
        crescimento = float(fator.iloc[-1]) / float(fator.iloc[idx])
        bruto = valor * crescimento
        dias = (fim - data).days
        imposto = max(0.0, bruto - valor) * aliquota_regressiva(dias)
        total_liquido += bruto - imposto
        fluxos.append(((data - inicio).days / 365.25, -valor))

    fluxos.append(((fim - inicio).days / 365.25, total_liquido))
    return total_liquido, tir_anual(fluxos)


def _aportado_ate(
    indice: pd.DatetimeIndex,
    fn_aporte: Callable[[int], float],
    dias_de_aporte: set,
) -> np.ndarray:
    """Quanto já tinha sido aportado em cada dia da série."""
    acumulado = np.zeros(len(indice))
    total = 0.0
    mes = 0
    for i, dia in enumerate(indice):
        if dia in dias_de_aporte:
            total += float(fn_aporte(mes))
            mes += 1
        acumulado[i] = total
    return acumulado


def _drawdown_de_aporte(
    curva: pd.Series,
    indice: pd.DatetimeIndex,
    fn_aporte: Callable[[int], float],
    dias_de_aporte: set,
) -> float:
    """Queda máxima do patrimônio **descontando** o que foi aportado.

    Com aporte mensal o patrimônio quase nunca cai em termos absolutos: o
    dinheiro novo mascara a queda. O que dói de verdade é o patrimônio ficar
    abaixo do que foi colocado, e é isso que esta função mede.
    """
    aportado = _aportado_ate(indice, fn_aporte, dias_de_aporte)
    with np.errstate(divide="ignore", invalid="ignore"):
        razao = np.where(aportado > 0, curva.to_numpy() / aportado - 1, 0.0)
    return float(np.nanmin(razao)) * 100 if len(razao) else 0.0


def _pior_mes(curva_cota: pd.Series) -> float:
    """Pior mês **da carteira**, medido na cota.

    Precisa ser na cota e não no patrimônio. No patrimônio, o aporte do mês
    entra somado ao saldo e empurra a variação para cima: numa simulação de
    preço parado o "pior mês" saía +7,7%, que é o tamanho do depósito, não do
    resultado. Na cota o depósito não aparece, só o desempenho.
    """
    mensal = curva_cota.resample("ME").last().pct_change().dropna()
    return float(mensal.min()) * 100 if len(mensal) else 0.0


def _queda_maxima(curva_cota: pd.Series) -> float:
    """Queda máxima da cota: quanto a carteira caiu de pico a fundo."""
    if curva_cota.empty:
        return 0.0
    return float((curva_cota / curva_cota.cummax() - 1).min()) * 100


def _meses_no_prejuizo(
    curva: pd.Series,
    indice: pd.DatetimeIndex,
    fn_aporte: Callable[[int], float],
    dias_de_aporte: set,
) -> int:
    aportado = _aportado_ate(indice, fn_aporte, dias_de_aporte)
    abaixo = pd.Series(curva.to_numpy() < aportado * 0.999, index=indice)
    return int(abaixo.resample("ME").max().sum())


# ─────────────────────────────────────────────────────────────────────────────
# Janelas móveis
# ─────────────────────────────────────────────────────────────────────────────

def janelas_moveis(
    precos: pd.DataFrame,
    cdi: pd.Series,
    pesos: dict[str, float],
    aporte: Callable[[int], float] | float = 500.0,
    *,
    anos: int = 5,
    passo_meses: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """Roda a mesma estratégia começando em cada mês possível do histórico.

    É o antídoto contra a resposta de uma janela só. Se a mediana das janelas
    de 5 anos é 11% e a pior é 2%, as duas informações são verdadeiras e a
    segunda é a que decide se dá para dormir à noite.
    """
    if anos <= 0:
        raise ValueError("a janela precisa ter ao menos um ano")

    inicios = _primeiros_dias_uteis(precos.index)
    linhas = []

    for i in range(0, len(inicios), passo_meses):
        ini = inicios[i]
        fim = ini + pd.DateOffset(years=anos)
        if fim > precos.index[-1]:
            break
        recorte = precos.loc[(precos.index >= ini) & (precos.index <= fim)]
        if len(recorte) < 200 * anos:
            continue
        try:
            r = simular(recorte, cdi, pesos, aporte, **kwargs)
        except Exception as exc:
            logger.warning("Janela iniciada em %s falhou: %s", ini.date(), exc)
            continue
        linhas.append(r.resumo())

    if not linhas:
        raise ValueError(
            f"nenhuma janela de {anos} anos cabe no histórico de "
            f"{(precos.index[-1] - precos.index[0]).days / 365.25:.1f} anos"
        )
    return pd.DataFrame(linhas)


def distribuicao(janelas: pd.DataFrame, coluna: str = "valor_liquido_final") -> dict:
    """Resume as janelas em mediana, pior, melhor e quartis."""
    s = janelas[coluna].astype(float)
    return {
        "janelas": int(len(s)),
        "pior": round(float(s.min()), 2),
        "p25": round(float(s.quantile(0.25)), 2),
        "mediana": round(float(s.median()), 2),
        "p75": round(float(s.quantile(0.75)), 2),
        "melhor": round(float(s.max()), 2),
        "media": round(float(s.mean()), 2),
    }


def comparar_com_cdi(janelas: pd.DataFrame) -> dict:
    """Em que fração das janelas a carteira bateu o CDI, e por quanto."""
    venceu = janelas["valor_liquido_final"] > janelas["liquido_cdi"]
    diferenca = janelas["valor_liquido_final"] - janelas["liquido_cdi"]
    return {
        "janelas": int(len(janelas)),
        "venceu_o_cdi_em": int(venceu.sum()),
        "taxa_de_vitoria_pct": round(float(venceu.mean()) * 100, 1),
        "vantagem_mediana": round(float(diferenca.median()), 2),
        "pior_diferenca": round(float(diferenca.min()), 2),
        "melhor_diferenca": round(float(diferenca.max()), 2),
    }
