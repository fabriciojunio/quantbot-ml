"""Roda a validação estatística anti-overfitting em dados REAIS.

Baixa histórico via yfinance, gera features e labels, e roda o
ValidatedTrainer (Purged CV + Deflated Sharpe + PBO) sobre o EnsembleModel.
É o PRIMEIRO PASSO antes de qualquer capital: mostra, com números, se há
evidência de edge ou se é provável overfitting/sorte.

Uso:
    python scripts/run_validation.py --symbols PETR4.SA VALE3.SA ITUB4.SA --years 3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd

from config.settings import FEATURE_COLUMNS, ML_CONFIG
from data.features import FeatureEngineer
from models.ensemble import EnsembleModel
from validation.validated_trainer import ValidatedTrainer


def _fetch(symbol: str, years: int) -> pd.DataFrame | None:
    import yfinance as yf

    df = yf.Ticker(symbol).history(period=f"{years}y")
    if df is None or len(df) < 150:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _prepare(df: pd.DataFrame, horizon: int):
    feats = FeatureEngineer.compute_all(df)
    target = FeatureEngineer.create_target(df, horizon=horizon)
    fwd = df["close"].pct_change(horizon).shift(-horizon)

    cols = [c for c in FEATURE_COLUMNS if c in feats.columns]
    data = feats[cols].copy()
    data["__y"] = target
    data["__fwd"] = fwd
    # Remove infinitos (divisões por ~0 em ratios) antes de treinar.
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    X = data[cols]
    y = data["__y"].astype(int)
    forward = data["__fwd"].astype(float)
    return X, y, forward


def validate_symbol(symbol: str, years: int, n_trials: int) -> dict | None:
    df = _fetch(symbol, years)
    if df is None:
        print(f"  ✗ {symbol}: dados insuficientes")
        return None

    horizon = ML_CONFIG.prediction_horizon
    X, y, forward = _prepare(df, horizon)
    if len(X) < 200:
        print(f"  ✗ {symbol}: amostra pequena após features ({len(X)})")
        return None

    trainer = ValidatedTrainer(
        model_factory=lambda: EnsembleModel(),
        n_splits=6,
        embargo_pct=ML_CONFIG.embargo_pct,
    )

    cv = trainer.cross_validate(X, y, forward, n_trials=n_trials)

    # Sweep de configurações para o PBO (variações honestas do modelo).
    factories = [
        lambda: EnsembleModel(random_state=1),
        lambda: EnsembleModel(random_state=2),
        lambda: EnsembleModel(random_state=3),
        lambda: EnsembleModel(random_state=4),
    ]
    try:
        sweep = trainer.evaluate_config_sweep(X, y, forward, factories, n_splits_pbo=8)
        pbo = sweep["pbo"].get("pbo")
    except Exception as e:
        pbo = None
        print(f"  (PBO indisponível: {e})")

    tr = cv["oos_track_record"]
    print(f"\n  {symbol}  (n={cv['n_samples']}, folds={cv['n_folds']})")
    print(f"    Acurácia média OOS : {cv['mean_accuracy']:.1%} (±{cv['std_accuracy']:.1%})")
    print(f"    Sharpe anualizado  : {tr['sharpe_annualized']:.2f}")
    print(f"    PSR                : {tr['psr']:.3f}")
    print(f"    DSR                : {tr['dsr']:.3f}")
    if pbo is not None:
        print(f"    PBO                : {pbo:.3f}")
    print(f"    Veredito           : {tr['verdict']}")
    return {"symbol": symbol, "dsr": tr["dsr"], "pbo": pbo, "verdict": tr["verdict"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validação estatística em dados reais")
    parser.add_argument("--symbols", nargs="+",
                        default=["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA"])
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20,
                        help="Nº de estratégias testadas no total (penaliza data snooping)")
    args = parser.parse_args()

    print("=" * 60)
    print("VALIDAÇÃO ESTATÍSTICA ANTI-OVERFITTING — DADOS REAIS")
    print("=" * 60)
    print(f"Símbolos: {', '.join(args.symbols)} | janela: {args.years} anos")
    print("Interpretação: DSR > 0.90 e PBO < 0.20 => evidência de edge.")
    print("               Caso contrário => sem evidência, NÃO arrisque capital.")

    results = []
    for sym in args.symbols:
        r = validate_symbol(sym, args.years, args.trials)
        if r:
            results.append(r)

    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    if not results:
        print("Nenhum símbolo validado.")
        return
    com_edge = [r for r in results if r["dsr"] > 0.90 and (r["pbo"] is None or r["pbo"] < 0.20)]
    for r in results:
        flag = "✅ edge" if r in com_edge else "⚠️ sem evidência"
        print(f"  {r['symbol']:<12} DSR={r['dsr']:.3f}  {flag}")
    print()
    print("LEMBRE-SE: isto mede skill estatístico, não garante lucro futuro.")
    print("Próximo passo: paper trading 3-6 meses antes de qualquer capital real.")


if __name__ == "__main__":
    main()
