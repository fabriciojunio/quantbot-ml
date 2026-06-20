"""Testes das melhorias de metodologia.

- Walk-forward com purge + embargo (sem vazamento entre treino e teste).
- Triple barrier detectando toque de TP/SL no High/Low intradiário.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.walk_forward import WalkForwardValidator
from models.triple_barrier import TripleBarrierLabeler


class TestWalkForwardPurgeEmbargo:
    def test_gap_respeita_purge_e_embargo(self):
        purge, embargo = 10, 5
        wfv = WalkForwardValidator(
            train_size=100, test_size=30, step_size=30, purge=purge, embargo=embargo
        )
        splits = wfv.generate_splits(400)
        assert len(splits) > 0
        for train, test in splits:
            # Nenhuma barra de treino pode encostar no teste: precisa haver uma
            # folga de pelo menos (purge + embargo) entre o fim do treino e o teste.
            assert test[0] - train[-1] - 1 >= purge + embargo
            assert train[-1] < test[0]  # ordem cronológica preservada

    def test_purge_encurta_o_treino(self):
        purge = 12
        sem = WalkForwardValidator(train_size=100, test_size=30)
        com = WalkForwardValidator(train_size=100, test_size=30, purge=purge)
        # O treino purgado tem `purge` barras a menos.
        assert len(com.generate_splits(400)[0][0]) == len(sem.generate_splits(400)[0][0]) - purge

    def test_default_sem_purge_mantem_comportamento(self):
        wfv = WalkForwardValidator(train_size=100, test_size=30, step_size=30)
        for train, test in wfv.generate_splits(300):
            assert len(train) == 100
            assert train[-1] < test[0]

    def test_purge_negativo_levanta_erro(self):
        try:
            WalkForwardValidator(train_size=10, test_size=5, purge=-1)
            assert False, "deveria ter levantado ValueError"
        except ValueError:
            pass


class TestTripleBarrierIntraday:
    def _df(self, closes, highs, lows):
        return pd.DataFrame({"Close": closes, "High": highs, "Low": lows})

    def test_toque_de_tp_no_high_mesmo_com_close_parado(self):
        # Close fica de lado, mas a MÁXIMA estoura o TP numa barra → label 1.
        n = 30
        closes = [100.0] * n
        highs = [100.0] * n
        lows = [100.0] * n
        # injeta volatilidade inicial para as barreiras não serem zero
        closes[1] = 101.0
        closes[2] = 99.0
        highs[5] = 130.0  # pico intradiário forte
        df = self._df(closes, highs, lows)
        out = TripleBarrierLabeler(tp_mult=2.0, sl_mult=1.0, max_holding=10).fit_transform(df)
        assert 1 in set(out["tb_label"].tolist())

    def test_toque_de_sl_no_low(self):
        n = 30
        closes = [100.0] * n
        highs = [100.0] * n
        lows = [100.0] * n
        closes[1] = 101.0
        closes[2] = 99.0
        lows[5] = 70.0  # fundo intradiário forte
        df = self._df(closes, highs, lows)
        out = TripleBarrierLabeler(tp_mult=2.0, sl_mult=1.0, max_holding=10).fit_transform(df)
        assert -1 in set(out["tb_label"].tolist())

    def test_sem_ohlc_usa_close(self):
        # Sem High/Low, deve funcionar com o Close (compatibilidade).
        closes = pd.Series(np.linspace(100, 120, 40))
        df = pd.DataFrame({"Close": closes})
        out = TripleBarrierLabeler().fit_transform(df)
        assert "tb_label" in out.columns
        assert set(out["tb_label"].unique()).issubset({-1, 0, 1})
