"""
Script de Backtesting — Roda todas as estratégias e compara.

Uso:
    python run_backtest.py                    # Tickers padrão
    python run_backtest.py AAPL TSLA NVDA     # Específicos
    python run_backtest.py PETR4.SA VALE3.SA  # B3
    python run_backtest.py --all              # Todos os mercados
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import MarketDataFetcher as DataFetcher
from data.features import FeatureEngineer
from strategies import SMACrossoverStrategy, RSIStrategy, MACDStrategy, EnsembleVotingStrategy
from strategies.base import SIGNAL_NUMERIC
from backtest.engine import BacktestEngine
from backtest.report import ReportGenerator
from visualization import plot_backtest_signals, plot_equity_curve, plot_strategy_comparison
from config.settings import BACKTEST_CONFIG, Signal


def run_single_ticker(ticker: str, engine: BacktestEngine):
    """Executa todas as estratégias em um ticker."""
    print(f"\n{'=' * 60}")
    print(f"  Analisando: {ticker}")
    print(f"{'=' * 60}")

    fetcher = DataFetcher()
    df_raw = fetcher.fetch_single(ticker)
    if df_raw is None:
        print(f"  Sem dados para {ticker}.")
        return None

    features = FeatureEngineer.compute_all(df_raw)
    # Mescla preços originais com features calculadas
    df = df_raw.join(features, how="left")
    df["Close"] = df["close"]
    df["Volume"] = df["volume"]

    print(f"  Periodo: {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  {len(df)} registros")

    strategies = [
        SMACrossoverStrategy(),
        RSIStrategy(),
        MACDStrategy(),
        EnsembleVotingStrategy(),
    ]

    results = []
    for strategy in strategies:
        print(f"\n  Rodando: {strategy.name}...")
        df_sig = strategy.generate_signals(df.copy())

        # Converte Signal enum para numérico para o backtest engine
        df_sig["_signal_num"] = df_sig["signal"].map(lambda s: SIGNAL_NUMERIC.get(s, 0))

        signals_series = df_sig["_signal_num"]
        result = engine.run(df_sig, signals_series)
        results.append((strategy.name, result))
        ReportGenerator.print_backtest(result)

        plot_backtest_signals(df_sig, ticker, strategy.name,
                              result.total_return, result.win_rate, result.sharpe_ratio)
        if result.equity_curve is not None:
            plot_equity_curve(result.equity_curve, ticker, strategy.name)

    comparison_data = [{
        "strategy_name": name,
        "total_return_pct": r.total_return,
        "total_trades": r.total_trades,
        "win_rate": r.win_rate / 100,
        "sharpe_ratio": r.sharpe_ratio,
        "max_drawdown_pct": abs(r.max_drawdown),
    } for name, r in results]
    plot_strategy_comparison(comparison_data, ticker)

    best = max(results, key=lambda t: t[1].total_return)
    worst = min(results, key=lambda t: t[1].total_return)
    print(f"\n  Melhor: {best[0]} ({best[1].total_return:+.2f}%)")
    print(f"  Pior:  {worst[0]} ({worst[1].total_return:+.2f}%)")

    return results


def main():
    tickers = [t for t in sys.argv[1:] if not t.startswith("--")]
    if "--all" in sys.argv:
        tickers = ["AAPL", "TSLA", "NVDA", "PETR4.SA", "VALE3.SA", "BTC-USD", "ETH-USD"]
    if not tickers:
        tickers = ["AAPL", "TSLA", "NVDA", "PETR4.SA", "VALE3.SA"]

    engine = BacktestEngine()

    print("=" * 60)
    print("  SISTEMA DE BACKTESTING DE ESTRATEGIAS")
    print("=" * 60)
    print(f"  Capital: ${BACKTEST_CONFIG.initial_capital:,.0f}")
    print(f"  Tickers: {', '.join(tickers)}")

    all_results = {}
    for ticker in tickers:
        results = run_single_ticker(ticker, engine)
        if results:
            all_results[ticker] = results

    print(f"\n{'=' * 60}")
    print("  RESUMO GERAL")
    print(f"{'=' * 60}")
    for ticker, results in all_results.items():
        best = max(results, key=lambda t: t[1].total_return)
        print(f"  {ticker:12s} -> {best[0]:25s} Retorno: {best[1].total_return:+.2f}%")

    print(f"\n  Graficos salvos em: results/")


if __name__ == "__main__":
    main()
