"""
Testes da camada de dados.

Nenhum toca a rede: o yfinance e o cliente do Banco Central são substituídos
por respostas gravadas. O foco é a validação de série, porque foi a ausência
dela que deixou dois sistemas rodando meses sobre número errado.
"""

import numpy as np
import pandas as pd
import pytest

from carteira import dados


def serie_boa(n=300, tickers=("A", "B")):
    idx = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for t in tickers},
        index=idx,
    )


class TestValidaPrecos:
    def test_serie_boa_passa(self):
        dados.valida_precos(serie_boa())

    def test_serie_vazia_falha(self):
        with pytest.raises(ValueError, match="sem dados"):
            dados.valida_precos(pd.DataFrame())

    def test_preco_zerado_falha(self):
        df = serie_boa()
        df.iloc[10, 0] = 0.0
        with pytest.raises(ValueError, match="zerado ou negativo"):
            dados.valida_precos(df)

    def test_preco_negativo_falha(self):
        df = serie_boa()
        df.iloc[10, 0] = -5.0
        with pytest.raises(ValueError, match="zerado ou negativo"):
            dados.valida_precos(df)

    def test_indice_fora_de_ordem_falha(self):
        with pytest.raises(ValueError, match="fora de ordem"):
            dados.valida_precos(serie_boa().iloc[::-1])

    def test_data_duplicada_falha(self):
        df = serie_boa()
        df = pd.concat([df, df.iloc[[5]]]).sort_index()
        with pytest.raises(ValueError, match="duplicadas"):
            dados.valida_precos(df)

    def test_salto_absurdo_falha(self):
        """Salto acima de 60% em um dia é desdobramento não ajustado, não mercado."""
        df = serie_boa()
        df.iloc[50, 0] = float(df.iloc[49, 0]) * 3
        with pytest.raises(ValueError, match="60%"):
            dados.valida_precos(df)

    def test_variacao_normal_passa(self):
        df = serie_boa()
        df.iloc[50, 0] = float(df.iloc[49, 0]) * 1.15  # +15%, plausível
        dados.valida_precos(df)

    def test_mensagem_traz_o_nome(self):
        with pytest.raises(ValueError, match="carteira teste"):
            dados.valida_precos(pd.DataFrame(), "carteira teste")


class TestAlinhar:
    def test_remove_dias_incompletos(self):
        df = serie_boa()
        df.iloc[10, 1] = np.nan
        assert len(dados.alinhar(df)) == len(df) - 1

    def test_serie_de_um_ativo_so_e_apenas_limpa(self):
        df = serie_boa(tickers=("A",))
        df.iloc[10, 0] = np.nan
        assert len(dados.alinhar(df)) == len(df) - 1

    def test_serie_completa_nao_muda(self):
        df = serie_boa()
        pd.testing.assert_frame_equal(dados.alinhar(df), df)

    def test_nao_preenche_para_frente(self):
        """Preencher feriado cria correlação que não existe entre praças."""
        df = serie_boa()
        df.iloc[10, 1] = np.nan
        assert pd.Timestamp(df.index[10]) not in dados.alinhar(df).index


class TestRetornos:
    def test_retorno_simples(self):
        df = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
        r = dados.retornos(df)
        assert float(r["A"].iloc[0]) == pytest.approx(0.10)

    def test_retorno_log(self):
        df = pd.DataFrame({"A": [100.0, 110.0]})
        r = dados.retornos(df, log=True)
        assert float(r["A"].iloc[0]) == pytest.approx(np.log(1.10))

    def test_primeira_linha_some(self):
        assert len(dados.retornos(serie_boa())) == len(serie_boa()) - 1

    def test_preco_constante_da_retorno_zero(self):
        df = pd.DataFrame({"A": [100.0] * 10})
        assert np.allclose(dados.retornos(df), 0.0)


class TestAchatar:
    def test_multiindex_vira_coluna_simples(self):
        colunas = pd.MultiIndex.from_product([["Close", "Open"], ["BOVA11.SA"]])
        df = pd.DataFrame(np.ones((5, 2)), columns=colunas)
        assert list(dados._achatar(df).columns) == ["Close", "Open"]

    def test_coluna_simples_nao_muda(self):
        df = pd.DataFrame({"Close": [1, 2, 3]})
        pd.testing.assert_frame_equal(dados._achatar(df), df)


class TestCache:
    def test_grava_e_le(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dados, "CACHE_DIR", tmp_path)
        s = pd.Series(
            [100.0, 101.0],
            index=pd.bdate_range("2026-01-01", periods=2),
            name="fechamento",
        )
        dados._gravar_cache("BOVA11.SA", s.to_frame())
        lido = dados._ler_cache("BOVA11.SA")
        assert lido is not None
        assert float(lido["fechamento"].iloc[0]) == pytest.approx(100.0)

    def test_cache_inexistente_devolve_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dados, "CACHE_DIR", tmp_path)
        assert dados._ler_cache("NAOEXISTE.SA") is None

    def test_cache_corrompido_nao_derruba(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dados, "CACHE_DIR", tmp_path)
        (tmp_path / "RUIM_SA.csv").write_text("isto nao e csv\x00", encoding="utf-8")
        assert dados._ler_cache("RUIM.SA") is None

    @pytest.mark.parametrize("ticker", ["BOVA11.SA", "BTC-USD", "^BVSP"])
    def test_nome_de_arquivo_e_seguro(self, ticker, tmp_path, monkeypatch):
        monkeypatch.setattr(dados, "CACHE_DIR", tmp_path)
        caminho = dados._caminho(ticker)
        assert "." not in caminho.stem
        assert "-" not in caminho.stem


class TestCDI:
    def test_usa_o_cache_quando_esta_fresco(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dados, "CACHE_MACRO", tmp_path)
        s = pd.Series(
            [0.0005] * 100,
            index=pd.bdate_range("2026-01-01", periods=100),
            name="cdi",
        )
        (tmp_path).mkdir(parents=True, exist_ok=True)
        s.to_frame().to_csv(tmp_path / "cdi_diario.csv", index_label="data")

        def nao_deveria_chamar(*a, **k):
            raise AssertionError("não deveria bater no BCB com cache fresco")

        monkeypatch.setattr(dados, "BCBClient", nao_deveria_chamar)
        r = dados.cdi_diario(anos=0)
        assert len(r) == 100

    def test_sem_bcb_e_sem_cache_levanta(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dados, "CACHE_MACRO", tmp_path)

        class Quebrado:
            def historico(self, *a, **k):
                raise RuntimeError("BCB fora do ar")

        monkeypatch.setattr(dados, "BCBClient", Quebrado)
        with pytest.raises(dados.ErroDados, match="sem CDI"):
            dados.cdi_diario(anos=5)

    def test_fatia_a_busca_em_blocos(self, tmp_path, monkeypatch):
        """A API do BCB recusa janela acima de 10 anos; a busca precisa fatiar."""
        monkeypatch.setattr(dados, "CACHE_MACRO", tmp_path)
        chamadas = []

        class Falso:
            def historico(self, serie, data_inicial=None, data_final=None, **k):
                chamadas.append((data_inicial, data_final))
                idx = pd.bdate_range(data_inicial, data_final)

                class Obs:
                    def __init__(self, d, v):
                        self.data, self.valor = d, v

                return [Obs(d.date(), 0.05) for d in idx]

        monkeypatch.setattr(dados, "BCBClient", Falso)
        s = dados.cdi_diario(anos=12)
        assert len(chamadas) >= 2
        assert len(s) > 0
