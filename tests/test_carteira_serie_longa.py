"""
Testes da reconstrução de histórico longo.

O risco desta parte do código não é errar a conta: é montar uma série
plausível a partir de dado ruim e ninguém notar. Foi o que aconteceu com a
cotação de dólar do Yahoo, que não tem nenhum dia de 2005 — depois do
alinhamento entre ativos o buraco desaparecia e o dia seguinte virava um
retorno de 66% num pregão só.

Por isso a maioria destes testes ataca a procedência e a continuidade dos
dados, não o resultado do cálculo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from carteira import dados, serie_longa
from carteira.perfil import PERFIS, Faixa, Perfil


def indice(dias: int, inicio: str = "2004-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(inicio, periods=dias)


def fabricar_precos(idx: pd.DatetimeIndex, tickers, semente: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    return pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(idx)))) for t in tickers},
        index=idx,
    )


def baixador_falso(idx: pd.DatetimeIndex):
    def _baixar(tickers, anos=5):
        return fabricar_precos(idx, tickers)
    return _baixar


def cambista_falso(idx: pd.DatetimeIndex, valor: float = 5.0):
    def _cambio(anos=25, moeda="USD_BRL"):
        return pd.Series(np.full(len(idx), valor), index=idx, name="cambio")
    return _cambio


# ─────────────────────────────────────────────────────────────────────────────
# Receitas
# ─────────────────────────────────────────────────────────────────────────────

class TesteReceitas:
    def test_toda_receita_tem_ao_menos_um_fator(self):
        for r in serie_longa.RECEITAS.values():
            assert len(r.fatores) >= 1

    def test_receita_sem_fator_e_recusada(self):
        with pytest.raises(ValueError, match="sem nenhum fator"):
            serie_longa.Receita("X", (), 0.003, "2000", "teste")

    def test_taxa_de_administracao_implausivel_e_recusada(self):
        with pytest.raises(ValueError, match="implausível"):
            serie_longa.Receita("X", ("^BVSP",), 0.50, "2000", "teste")

    def test_receitas_em_dolar_usam_a_ptax_e_nao_o_yahoo(self):
        """A troca de fonte de câmbio é o ponto todo deste módulo."""
        for alvo in ("IVVB11.SA", "GOLD11.SA"):
            fatores = serie_longa.RECEITAS[alvo].fatores
            assert any(f.startswith(serie_longa.PREFIXO_BCB) for f in fatores)
            assert not any("USDBRL=X" in f for f in fatores)

    def test_ibovespa_nao_precisa_de_cambio(self):
        assert serie_longa.RECEITAS["BOVA11.SA"].fatores == ("^BVSP",)

    def test_taxas_batem_com_as_do_modelo_de_custo(self):
        from carteira.custos_b3 import TAXA_ADM_ANUAL

        for alvo, r in serie_longa.RECEITAS.items():
            assert r.taxa_adm_anual == pytest.approx(TAXA_ADM_ANUAL[alvo])


# ─────────────────────────────────────────────────────────────────────────────
# Montagem
# ─────────────────────────────────────────────────────────────────────────────

class TesteMontagem:
    def test_monta_uma_coluna_por_alvo(self):
        idx = indice(800)
        df = serie_longa.montar(
            anos=5, baixador=baixador_falso(idx), cambista=cambista_falso(idx)
        )
        assert set(df.columns) == set(serie_longa.RECEITAS)

    def test_comeca_em_base_cem(self):
        idx = indice(800)
        df = serie_longa.montar(
            anos=5, baixador=baixador_falso(idx), cambista=cambista_falso(idx)
        )
        assert np.allclose(df.iloc[0].to_numpy(), 100.0)

    def test_taxa_de_administracao_reduz_a_serie(self):
        idx = indice(1200)
        com = serie_longa.montar(anos=5, baixador=baixador_falso(idx),
                                 cambista=cambista_falso(idx), cobrar_taxa_adm=True)
        sem = serie_longa.montar(anos=5, baixador=baixador_falso(idx),
                                 cambista=cambista_falso(idx), cobrar_taxa_adm=False)
        assert (com.iloc[-1] < sem.iloc[-1]).all()

    def test_a_taxa_cobrada_tem_o_tamanho_certo(self):
        """Cinco anos de 0,30% ao ano tiram perto de 1,5% da série."""
        idx = indice(1260)
        com = serie_longa.montar(anos=5, baixador=baixador_falso(idx),
                                 cambista=cambista_falso(idx), cobrar_taxa_adm=True)
        sem = serie_longa.montar(anos=5, baixador=baixador_falso(idx),
                                 cambista=cambista_falso(idx), cobrar_taxa_adm=False)
        razao = float(com["BOVA11.SA"].iloc[-1] / sem["BOVA11.SA"].iloc[-1])
        esperado = (1 - 0.0030 / 252) ** (len(idx) - 1)
        assert razao == pytest.approx(esperado, rel=1e-6)

    def test_cambio_multiplica_a_serie_em_dolar(self):
        idx = indice(600)
        barato = serie_longa.montar(anos=5, baixador=baixador_falso(idx),
                                    cambista=cambista_falso(idx, 5.0))
        # Câmbio constante some na normalização base 100; o que importa é que
        # a série exista e seja finita.
        assert np.isfinite(barato["IVVB11.SA"].to_numpy()).all()

    def test_cambio_que_sobe_valoriza_o_ativo_em_dolar(self):
        idx = indice(600)
        subindo = pd.Series(np.linspace(3.0, 6.0, len(idx)), index=idx)
        plano = pd.Series(np.full(len(idx), 3.0), index=idx)

        def cambista(serie):
            return lambda anos=25, moeda="USD_BRL": serie

        com_alta = serie_longa.montar(anos=5, baixador=baixador_falso(idx),
                                      cambista=cambista(subindo))
        sem_alta = serie_longa.montar(anos=5, baixador=baixador_falso(idx),
                                      cambista=cambista(plano))
        assert float(com_alta["IVVB11.SA"].iloc[-1]) > float(sem_alta["IVVB11.SA"].iloc[-1])
        # O Ibovespa não depende de câmbio: tem que ficar igual.
        assert float(com_alta["BOVA11.SA"].iloc[-1]) == pytest.approx(
            float(sem_alta["BOVA11.SA"].iloc[-1])
        )

    def test_recusa_serie_com_buraco(self):
        """O caso real: dólar do Yahoo sem 2005."""
        cheio = indice(1200, "2004-01-01")
        furado = cheio.delete(range(300, 700))

        with pytest.raises(ValueError, match="buraco"):
            serie_longa.montar(anos=5, baixador=baixador_falso(furado),
                               cambista=cambista_falso(furado))

    def test_aceita_feriado_longo(self):
        """Natal e Ano Novo emendados não podem reprovar a série."""
        idx = indice(800)
        sem_natal = idx[~((idx.month == 12) & (idx.day >= 24))]
        serie_longa.montar(anos=5, baixador=baixador_falso(sem_natal),
                           cambista=cambista_falso(sem_natal))

    def test_recusa_quando_falta_um_fator(self):
        idx = indice(500)

        def parcial(tickers, anos=5):
            return fabricar_precos(idx, [t for t in tickers if t != "GC=F"])

        with pytest.raises(ValueError):
            serie_longa.montar(anos=5, baixador=parcial, cambista=cambista_falso(idx))

    def test_serie_montada_passa_na_validacao_de_precos(self):
        idx = indice(1500)
        df = serie_longa.montar(anos=6, baixador=baixador_falso(idx),
                                cambista=cambista_falso(idx))
        dados.valida_precos(df, "teste")
        dados.valida_continuidade(df, "teste")


# ─────────────────────────────────────────────────────────────────────────────
# Continuidade
# ─────────────────────────────────────────────────────────────────────────────

class TesteContinuidade:
    def test_aponta_a_data_do_buraco(self):
        idx = indice(600).delete(range(200, 400))
        df = pd.DataFrame({"A": np.arange(len(idx), dtype=float) + 1}, index=idx)
        with pytest.raises(ValueError) as exc:
            dados.valida_continuidade(df, "teste")
        assert "buraco" in str(exc.value)

    def test_serie_continua_passa(self):
        idx = indice(600)
        df = pd.DataFrame({"A": np.arange(len(idx), dtype=float) + 1}, index=idx)
        dados.valida_continuidade(df, "teste")

    def test_serie_de_um_ponto_nao_falha(self):
        df = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-02"]))
        dados.valida_continuidade(df, "teste")

    def test_serie_vazia_falha(self):
        with pytest.raises(ValueError, match="sem dados"):
            dados.valida_continuidade(pd.DataFrame(), "teste")

    @pytest.mark.parametrize("dias_do_buraco,esperado_falha", [(5, False), (11, False), (40, True)])
    def test_limite_de_tolerancia(self, dias_do_buraco, esperado_falha):
        base = pd.bdate_range("2020-01-01", periods=100)
        idx = base[:50].append(base[50:] + pd.Timedelta(days=dias_do_buraco))
        df = pd.DataFrame({"A": np.arange(len(idx), dtype=float) + 1}, index=idx)
        if esperado_falha:
            with pytest.raises(ValueError):
                dados.valida_continuidade(df, "teste")
        else:
            dados.valida_continuidade(df, "teste")


# ─────────────────────────────────────────────────────────────────────────────
# Composição longa
# ─────────────────────────────────────────────────────────────────────────────

class TesteComposicaoLonga:
    def test_cobre_todas_as_faixas(self):
        assert set(serie_longa.COMPOSICAO_LONGA) == set(Faixa)

    def test_cada_faixa_com_conteudo_soma_um(self):
        for faixa, comp in serie_longa.COMPOSICAO_LONGA.items():
            if comp:
                assert sum(comp.values()) == pytest.approx(1.0), faixa

    def test_so_usa_ativos_que_a_serie_longa_produz(self):
        produzidos = set(serie_longa.RECEITAS) | {"CDI"}
        for comp in serie_longa.COMPOSICAO_LONGA.values():
            assert set(comp) <= produzidos

    @pytest.mark.parametrize("risco", [0, 20, 40, 60, 80, 100])
    def test_pesos_somam_um_em_qualquer_risco(self, risco):
        p = Perfil(f"t{risco}", float(risco), "teste")
        pesos = p.pesos_por_ativo(composicao=serie_longa.COMPOSICAO_LONGA)
        assert sum(pesos.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("nome", list(PERFIS))
    def test_perfis_de_catalogo_funcionam_na_composicao_longa(self, nome):
        pesos = PERFIS[nome].pesos_por_ativo(composicao=serie_longa.COMPOSICAO_LONGA)
        assert sum(pesos.values()) == pytest.approx(1.0)

    def test_risco_maior_aumenta_a_fatia_em_dolar(self):
        baixo = Perfil("b", 20.0, "x").pesos_por_ativo(composicao=serie_longa.COMPOSICAO_LONGA)
        alto = Perfil("a", 90.0, "x").pesos_por_ativo(composicao=serie_longa.COMPOSICAO_LONGA)
        assert alto["IVVB11.SA"] > baixo["IVVB11.SA"]

    def test_composicao_incompleta_e_recusada(self):
        incompleta = {Faixa.CAIXA: {"CDI": 1.0}}
        with pytest.raises(ValueError, match="não define as faixas"):
            Perfil("t", 50.0, "x").pesos_por_ativo(composicao=incompleta)

    def test_avisos_estao_declarados(self):
        assert len(serie_longa.AVISOS_DA_COMPOSICAO_LONGA) >= 3
        texto = " ".join(serie_longa.AVISOS_DA_COMPOSICAO_LONGA).lower()
        assert "cdi" in texto and "ima-b" in texto


# ─────────────────────────────────────────────────────────────────────────────
# Procedência e cobertura
# ─────────────────────────────────────────────────────────────────────────────

class TesteProcedencia:
    def test_lista_o_que_foi_reconstruido_e_o_que_ficou_de_fora(self):
        p = serie_longa.procedencia()
        alvos = {linha["ativo"] for linha in p}
        assert set(serie_longa.RECEITAS) <= alvos
        assert set(serie_longa.SEM_HISTORICO_LONGO) <= alvos

    def test_todo_ausente_tem_motivo_escrito(self):
        for ticker, motivo in serie_longa.SEM_HISTORICO_LONGO.items():
            assert len(motivo) > 15, ticker

    def test_cobertura_conta_anos_e_janelas(self):
        idx = indice(2600)
        df = pd.DataFrame({"A": np.ones(len(idx))}, index=idx)
        c = serie_longa.cobertura(df)
        assert c.anos == pytest.approx(10.0, abs=0.3)
        assert c.janelas_de(5) > 0
        assert c.janelas_de(50) == 0

    def test_cobertura_carrega_os_avisos(self):
        idx = indice(300)
        df = pd.DataFrame({"A": np.ones(len(idx))}, index=idx)
        c = serie_longa.cobertura(df, serie_longa.AVISOS_DA_COMPOSICAO_LONGA)
        assert len(c.avisos) == len(serie_longa.AVISOS_DA_COMPOSICAO_LONGA)
