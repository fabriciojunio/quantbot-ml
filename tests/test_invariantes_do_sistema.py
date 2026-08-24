"""
Invariantes que precisam valer em todo o espaço de configuração.

Os outros arquivos testam cada peça com casos escolhidos. Este varre o espaço
inteiro: todo nível de risco de 0 a 100, todo ativo do universo, toda faixa da
tabela regressiva, todo tamanho de ordem plausível. É onde aparecem as falhas
de borda que não ocorrem a ninguém escrever à mão — foi assim que se descobriu
que liberar o satélite sem composição fazia 12% da carteira evaporar.

O critério para entrar aqui é ser uma propriedade que, se falhar, significa
dinheiro errado. Não teste de fumaça, não cobertura por cobertura.
"""

from __future__ import annotations

import pytest

from carteira.conta import Conta, aliquota_de_venda
from carteira.custos_b3 import TAXA_ADM_ANUAL, ModeloCustoB3
from carteira.multimoeda import MOEDA_DE_FATO, TETO_DE_CRIPTO, com_cripto, exposicao_cambial
from carteira.perfil import COMPOSICAO, PERFIS, Faixa, Perfil
from carteira.serie_longa import COMPOSICAO_LONGA, RECEITAS, SEM_HISTORICO_LONGO
from carteira.tributos import aliquota_regressiva
from carteira.universo import UNIVERSO_B3, Regime

RISCOS = list(range(0, 101))
VALORES = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 50_000, 100_000]
COMPOSICOES = [("padrao", COMPOSICAO), ("longa", COMPOSICAO_LONGA)]


# ─────────────────────────────────────────────────────────────────────────────
# Pesos: 101 níveis de risco x 2 composições
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("risco", RISCOS)
@pytest.mark.parametrize("nome_comp,comp", COMPOSICOES)
def test_pesos_sempre_somam_um(risco, nome_comp, comp):
    """Se não somam 1, uma parte do dinheiro fica sem destino e some calada."""
    pesos = Perfil(f"t{risco}", float(risco), "x").pesos_por_ativo(composicao=comp)
    assert sum(pesos.values()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("risco", RISCOS)
@pytest.mark.parametrize("nome_comp,comp", COMPOSICOES)
def test_nenhum_peso_negativo(risco, nome_comp, comp):
    pesos = Perfil(f"t{risco}", float(risco), "x").pesos_por_ativo(composicao=comp)
    assert all(p >= 0 for p in pesos.values())


@pytest.mark.parametrize("risco", RISCOS)
def test_fatia_segura_cai_conforme_o_risco_sobe(risco):
    """A parte previsível tem que ser exatamente o complemento do risco."""
    faixas = Perfil(f"t{risco}", float(risco), "x").pesos_por_faixa()
    segura = faixas.get(Faixa.CAIXA, 0.0) + faixas.get(Faixa.RENDA_FIXA, 0.0)
    assert segura == pytest.approx(1 - risco / 100, abs=1e-9)


@pytest.mark.parametrize("risco", range(0, 100))
def test_exposicao_a_dolar_nunca_diminui_com_mais_risco(risco):
    a = Perfil("a", float(risco), "x").pesos_por_ativo()
    b = Perfil("b", float(risco + 1), "x").pesos_por_ativo()
    assert b.get("IVVB11.SA", 0.0) >= a.get("IVVB11.SA", 0.0) - 1e-12


@pytest.mark.parametrize("risco", RISCOS)
def test_exposicao_cambial_soma_um(risco):
    pesos = Perfil(f"t{risco}", float(risco), "x").pesos_por_ativo()
    assert sum(exposicao_cambial(pesos).por_moeda.values()) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Cripto: 101 níveis de risco x fatias
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("risco", range(30, 101, 5))
@pytest.mark.parametrize("fatia", [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05])
def test_cripto_preserva_a_soma_e_a_parte_segura(risco, fatia):
    base = Perfil(f"t{risco}", float(risco), "x").pesos_por_ativo()
    novos = com_cripto(base, fatia)
    assert sum(novos.values()) == pytest.approx(1.0, abs=1e-9)
    assert novos.get("CDI", 0.0) == pytest.approx(base.get("CDI", 0.0), abs=1e-9)


@pytest.mark.parametrize("fatia", [0.051, 0.06, 0.10, 0.25, 0.50, 1.0])
def test_cripto_acima_do_teto_sempre_recusada(fatia):
    base = Perfil("t80", 80.0, "x").pesos_por_ativo()
    with pytest.raises(ValueError):
        com_cripto(base, fatia)


def test_teto_de_cripto_nao_foi_afrouxado():
    """Guarda contra alguém subir o teto sem discutir. Cinco por cento é o limite."""
    assert TETO_DE_CRIPTO <= 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Tabela regressiva: todo dia de 0 a 1200
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dias", list(range(0, 1201, 7)))
def test_aliquota_regressiva_fica_na_faixa_legal(dias):
    a = aliquota_regressiva(dias)
    assert 0.15 <= a <= 0.225


@pytest.mark.parametrize("dias", list(range(1, 1200, 11)))
def test_aliquota_nunca_aumenta_com_o_prazo(dias):
    """A tabela é regressiva: esperar mais nunca pode custar mais imposto."""
    assert aliquota_regressiva(dias + 1) <= aliquota_regressiva(dias) + 1e-12


@pytest.mark.parametrize("dias,esperado", [
    (0, 0.225), (100, 0.225), (180, 0.225),
    (181, 0.20), (300, 0.20), (360, 0.20),
    (361, 0.175), (500, 0.175), (720, 0.175),
    (721, 0.15), (2000, 0.15), (10_000, 0.15),
])
def test_degraus_da_tabela_regressiva(dias, esperado):
    assert aliquota_regressiva(dias) == pytest.approx(esperado)


# ─────────────────────────────────────────────────────────────────────────────
# Custo: todo ativo x todo valor
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ticker", sorted(UNIVERSO_B3))
@pytest.mark.parametrize("valor", VALORES)
def test_custo_e_positivo_e_proporcional(ticker, valor):
    modelo = ModeloCustoB3()
    custo = modelo.custo_ponta(UNIVERSO_B3[ticker], valor)
    assert custo > 0
    assert custo < valor * 0.02, "custo acima de 2% da ordem é implausível para ETF"


@pytest.mark.parametrize("ticker", sorted(UNIVERSO_B3))
def test_custo_cresce_com_o_valor(ticker):
    modelo = ModeloCustoB3()
    ativo = UNIVERSO_B3[ticker]
    anterior = 0.0
    for valor in VALORES:
        atual = modelo.custo_ponta(ativo, valor)
        assert atual > anterior
        anterior = atual


@pytest.mark.parametrize("ticker", sorted(UNIVERSO_B3))
def test_ida_e_volta_custa_o_dobro_de_uma_ponta(ticker):
    modelo = ModeloCustoB3()
    ativo = UNIVERSO_B3[ticker]
    assert modelo.custo_ida_volta(ativo, 1000) == pytest.approx(
        2 * modelo.custo_ponta(ativo, 1000)
    )


@pytest.mark.parametrize("ticker", sorted(UNIVERSO_B3))
def test_todo_ativo_tem_taxa_de_administracao_declarada(ticker):
    assert ticker in TAXA_ADM_ANUAL
    assert 0 < TAXA_ADM_ANUAL[ticker] < 0.02


@pytest.mark.parametrize("ticker", sorted(UNIVERSO_B3))
def test_todo_ativo_tem_spread_plausivel(ticker):
    a = UNIVERSO_B3[ticker]
    assert 0 < a.spread_pct < 0.01, "spread fora da faixa plausível para ETF líquido"


@pytest.mark.parametrize("ticker", sorted(UNIVERSO_B3))
def test_todo_ativo_tem_moeda_e_regime(ticker):
    assert ticker in MOEDA_DE_FATO
    assert isinstance(UNIVERSO_B3[ticker].regime, Regime)


# ─────────────────────────────────────────────────────────────────────────────
# Conta: compra e venda em toda combinação de ativo e valor
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ticker", sorted(UNIVERSO_B3))
@pytest.mark.parametrize("valor", [200, 1_000, 10_000])
def test_comprar_e_vender_fecha_a_conta(ticker, valor):
    """Sem variação de preço, o que volta é o que entrou menos os dois custos."""
    c = Conta(caixa=valor)
    c.comprar(ticker, valor, preco=100.0, dia=0)
    c.vender(ticker, valor, preco=100.0, dia=1)

    assert c.imposto_pago == pytest.approx(0.0, abs=1e-9)
    assert c.caixa == pytest.approx(valor - c.custo_pago, rel=1e-9)
    assert ticker not in c.posicoes


@pytest.mark.parametrize("ticker", sorted(UNIVERSO_B3))
@pytest.mark.parametrize("multiplo", [0.5, 0.8, 1.0, 1.5, 3.0])
def test_liquidacao_nunca_devolve_mais_que_o_valor_de_mercado(ticker, multiplo):
    c = Conta(caixa=10_000)
    c.comprar(ticker, 10_000, preco=100.0, dia=0)
    cotas = c.posicoes[ticker].cotas
    final = c.liquidar({ticker: 100.0 * multiplo}, dia=800)
    assert final <= cotas * 100.0 * multiplo + 1e-6


@pytest.mark.parametrize("ticker", sorted(UNIVERSO_B3))
def test_conferir_passa_depois_de_operar(ticker):
    c = Conta(caixa=5_000)
    c.comprar(ticker, 2_000, preco=50.0, dia=0)
    c.conferir({ticker: 50.0})


@pytest.mark.parametrize("dias", [0, 90, 180, 181, 360, 361, 720, 721, 1500])
@pytest.mark.parametrize("ticker", ["BOVA11.SA", "IMAB11.SA", "B5P211.SA", "XFIX11.SA"])
def test_aliquota_de_venda_respeita_o_regime(dias, ticker):
    a = aliquota_de_venda(UNIVERSO_B3[ticker], dias)
    if UNIVERSO_B3[ticker].regime is Regime.ETF_RENDA_FIXA:
        assert a == pytest.approx(aliquota_regressiva(dias))
    else:
        assert a == pytest.approx(0.15)


# ─────────────────────────────────────────────────────────────────────────────
# Catálogo: coerência entre módulos
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nome", sorted(PERFIS))
def test_perfil_de_catalogo_e_coerente(nome):
    p = PERFIS[nome]
    assert 0 <= p.risco_pct <= 100
    assert sum(p.pesos_por_ativo().values()) == pytest.approx(1.0)
    assert len(p.descricao) > 20, "todo perfil precisa dizer o que custa escolhê-lo"


@pytest.mark.parametrize("faixa", list(Faixa))
def test_toda_faixa_existe_nas_duas_composicoes(faixa):
    assert faixa in COMPOSICAO
    assert faixa in COMPOSICAO_LONGA


@pytest.mark.parametrize("faixa", list(Faixa))
def test_composicao_de_faixa_com_conteudo_soma_um(faixa):
    for comp in (COMPOSICAO, COMPOSICAO_LONGA):
        if comp[faixa]:
            assert sum(comp[faixa].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("ticker", sorted(RECEITAS))
def test_receita_longa_bate_com_o_universo_real(ticker):
    """A série reconstruída usa o ticker do ETF real: os dois têm que casar."""
    assert ticker in UNIVERSO_B3
    assert RECEITAS[ticker].taxa_adm_anual == pytest.approx(TAXA_ADM_ANUAL[ticker])


@pytest.mark.parametrize("ticker", sorted(SEM_HISTORICO_LONGO))
def test_ausencia_de_historico_tem_motivo_escrito(ticker):
    assert len(SEM_HISTORICO_LONGO[ticker]) > 20


def test_todo_ativo_do_universo_esta_coberto_ou_declarado_ausente():
    """Nenhum ativo pode ficar em silêncio entre reconstruído e ausente."""
    cobertos = set(RECEITAS) | set(SEM_HISTORICO_LONGO)
    assert set(UNIVERSO_B3) <= cobertos


@pytest.mark.parametrize("ticker", sorted(UNIVERSO_B3))
def test_todo_ativo_tem_observacao_ou_nome_util(ticker):
    a = UNIVERSO_B3[ticker]
    assert a.nome and a.nome != ticker
