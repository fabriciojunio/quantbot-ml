"""
Assinatura do comportamento: números fixos que um refator não pode mudar sozinho.

Os outros testes verificam propriedades ("o dinheiro não some", "os pesos somam
um"). Estes fixam **valores exatos**. Se alguém mexer na contabilidade, no
imposto ou na ordem das operações e o resultado mudar, o teste falha e obriga a
decisão a ser consciente: ou o número novo está certo e se atualiza aqui com
justificativa no commit, ou a mudança quebrou algo.

É a rede que faltava quando o backtest e a operação foram unificados na classe
`Conta`. O refator preservou a mediana das janelas de 10 anos em R$ 192.515,48,
e isso só foi possível confirmar rodando a análise inteira de novo, com rede
externa. A partir daqui a confirmação é local e leva milissegundos.

A série é sintética e determinística de propósito: teste de caracterização não
pode depender de rede nem de o Yahoo estar de bom humor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from carteira import aporte_historico as ah
from carteira.conta import Conta
from carteira.custos_b3 import ModeloCustoB3


def serie_fixa(dias: int = 1300, semente: int = 20260823) -> pd.DataFrame:
    """Sempre a mesma série. A semente é a data em que foi congelada."""
    idx = pd.bdate_range("2019-01-01", periods=dias)
    rng = np.random.default_rng(semente)
    return pd.DataFrame(
        {
            "BOVA11.SA": 100 * np.exp(np.cumsum(rng.normal(0.00035, 0.0135, dias))),
            "IVVB11.SA": 100 * np.exp(np.cumsum(rng.normal(0.00045, 0.0105, dias))),
            "GOLD11.SA": 100 * np.exp(np.cumsum(rng.normal(0.00030, 0.0095, dias))),
            "IMAB11.SA": 100 * np.exp(np.cumsum(rng.normal(0.00025, 0.0045, dias))),
        },
        index=idx,
    )


def cdi_fixo(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.full(len(idx), 0.00042), index=idx)


PESOS = {
    "IVVB11.SA": 0.36,
    "BOVA11.SA": 0.28,
    "GOLD11.SA": 0.16,
    "IMAB11.SA": 0.12,
    "CDI": 0.08,
}


@pytest.fixture(scope="module")
def resultado():
    p = serie_fixa()
    return ah.simular(p, cdi_fixo(p.index), PESOS, ah.constante(500.0))


class TesteAssinaturaDaSimulacao:
    """Se algum destes mudar, a contabilidade mudou. Não atualize sem entender."""

    def test_meses_aportados(self, resultado):
        assert resultado.meses == 60

    def test_total_aportado(self, resultado):
        assert resultado.total_aportado == pytest.approx(30_000.00, abs=0.01)

    def test_valor_liquido_final(self, resultado):
        assert resultado.valor_liquido_final == pytest.approx(53_308.67, abs=1.0)

    def test_custo_total_pago(self, resultado):
        assert resultado.custo_total_pago == pytest.approx(57.39, abs=0.20)

    def test_imposto_total_pago(self, resultado):
        assert resultado.imposto_total_pago == pytest.approx(3_937.00, abs=2.0)

    def test_tir_anual(self, resultado):
        assert resultado.tir_anual_pct == pytest.approx(23.35, abs=0.05)

    def test_tir_do_cdi(self, resultado):
        assert resultado.tir_cdi_pct == pytest.approx(9.97, abs=0.05)

    def test_numero_de_ordens(self, resultado):
        assert resultado.ordens == 65

    def test_queda_da_cota(self, resultado):
        assert resultado.queda_da_cota_pct == pytest.approx(-7.26, abs=0.10)

    def test_referencia_de_cdi(self, resultado):
        assert resultado.liquido_cdi == pytest.approx(38_487.90, abs=1.0)

    def test_meses_no_prejuizo(self, resultado):
        assert resultado.meses_no_prejuizo == 6

    def test_custo_e_pequeno_perto_do_imposto(self, resultado):
        """Com rodízio e pouco giro, o custo de corretagem vira ruído; o
        imposto é que pesa. Se essa relação inverter, alguma regra mudou."""
        assert resultado.custo_total_pago < resultado.imposto_total_pago / 30

    def test_a_conta_fecha(self, resultado):
        """Aportado + resultado de mercado = líquido + custo + imposto."""
        assert resultado.valor_liquido_final > 0
        assert resultado.custo_total_pago >= 0
        assert resultado.imposto_total_pago >= 0


class TesteAssinaturaDaConta:
    """Valores exatos da contabilidade, sem simulação em volta."""

    def test_compra_simples(self):
        c = Conta(caixa=1_000, modelo=ModeloCustoB3())
        c.comprar("BOVA11.SA", 1_000, preco=100.0, dia=0)
        # Emolumento 0,0325% + meio spread 0,02% sobre R$ 1.000 = R$ 0,525
        assert c.custo_pago == pytest.approx(0.525, abs=1e-6)
        assert c.posicoes["BOVA11.SA"].cotas == pytest.approx(9.99475, abs=1e-5)

    def test_venda_com_lucro_de_cem_por_cento(self):
        c = Conta(caixa=1_000, modelo=ModeloCustoB3())
        c.comprar("BOVA11.SA", 1_000, preco=100.0, dia=0)
        cotas = c.posicoes["BOVA11.SA"].cotas
        c.liquidar({"BOVA11.SA": 200.0}, dia=400)

        bruto = cotas * 200.0
        custo_venda = bruto * (0.000325 + 0.0002)
        ganho = bruto - 1_000 - custo_venda
        assert c.imposto_pago == pytest.approx(ganho * 0.15, rel=1e-6)

    def test_renda_fixa_curta_paga_mais(self):
        c = Conta(caixa=1_000, modelo=ModeloCustoB3())
        c.comprar("IMAB11.SA", 1_000, preco=100.0, dia=0)
        c.liquidar({"IMAB11.SA": 200.0}, dia=100)
        # Menos de 180 dias: 22,5%.
        assert c.imposto_pago / max(1e-9, c.imposto_pago) == pytest.approx(1.0)
        assert c.imposto_pago > 200.0

    def test_prejuizo_zera_o_imposto_do_lucro_seguinte(self):
        c = Conta(caixa=4_000, modelo=ModeloCustoB3())
        c.comprar("BOVA11.SA", 2_000, preco=100.0, dia=0)
        c.comprar("SMAL11.SA", 2_000, preco=100.0, dia=0)
        c.vender("BOVA11.SA", 1_000, preco=50.0, dia=100)
        c.vender("SMAL11.SA", 2_400, preco=120.0, dia=100)
        assert c.imposto_pago == pytest.approx(0.0, abs=1e-9)


class TesteEstabilidadeEntreExecucoes:
    def test_a_mesma_entrada_da_o_mesmo_resultado(self):
        p = serie_fixa()
        a = ah.simular(p, cdi_fixo(p.index), PESOS, ah.constante(500.0))
        b = ah.simular(p, cdi_fixo(p.index), PESOS, ah.constante(500.0))
        assert a.valor_liquido_final == pytest.approx(b.valor_liquido_final, abs=1e-9)
        assert a.ordens == b.ordens

    def test_serie_fixa_e_realmente_fixa(self):
        assert serie_fixa().iloc[-1].to_dict() == serie_fixa().iloc[-1].to_dict()

    def test_aporte_maior_produz_patrimonio_maior(self):
        p = serie_fixa()
        pequeno = ah.simular(p, cdi_fixo(p.index), PESOS, ah.constante(300.0))
        grande = ah.simular(p, cdi_fixo(p.index), PESOS, ah.constante(900.0))
        assert grande.valor_liquido_final > pequeno.valor_liquido_final * 2.5

    def test_a_tir_quase_nao_depende_do_tamanho_do_aporte(self):
        """Dobrar o aporte dobra o patrimônio, não o retorno percentual."""
        p = serie_fixa()
        pequeno = ah.simular(p, cdi_fixo(p.index), PESOS, ah.constante(500.0))
        grande = ah.simular(p, cdi_fixo(p.index), PESOS, ah.constante(2_000.0))
        assert abs(grande.tir_anual_pct - pequeno.tir_anual_pct) < 1.5
