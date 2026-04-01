#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════╗
║               QUANTBOT ML — Ponto de Entrada                  ║
║                                                               ║
║  Modos:                                                       ║
║    python main.py                → Menu interativo            ║
║    python main.py --analyze      → Análise ML completa        ║
║    python main.py --paper-trade  → Paper Trading              ║
║    python main.py --test         → Executar testes            ║
║    python -m pytest tests/ -v    → Testes detalhados          ║
╚═══════════════════════════════════════════════════════════════╝
"""
import sys
import argparse

from config.settings import Signal, Market, RiskProfile, UNIVERSE, FEATURE_COLUMNS, BACKTEST_CONFIG
from data.fetcher import MarketDataFetcher
from data.features import FeatureEngineer
from data.validators import DataValidator
from data.news_fetcher import NewsFetcher
from data.sentiment import SentimentAnalyzer, analyze_news_sentiment
from models.trainer import ModelTrainer
from models.signals import SignalGenerator
from backtest.engine import BacktestEngine
from backtest.report import ReportGenerator
from backtest.benchmarks import BenchmarkComparator
from risk.metrics import calculate_risk_metrics
from core.paper_trading import PaperTradingEngine
from core.accuracy import AccuracyTracker
from core.performance import PerformanceTracker
from core.security import SecurityManager
from utils.logger import setup_logger
from utils.formatters import fmt_currency, fmt_pct

logger = setup_logger("quantbot")


def run_full_analysis(markets=None, risk_profile=RiskProfile.MODERATE):
    """Pipeline completo: dados → features → ML → news → backtest → risco."""
    if markets is None:
        markets = [Market.B3, Market.US, Market.CRYPTO]

    print("\n╔" + "═" * 60 + "╗")
    print("║" + " QUANTBOT ML — Análise Completa".center(60) + "║")
    print("╚" + "═" * 60 + "╝\n")

    # 1. Dados
    fetcher = MarketDataFetcher()
    data = fetcher.fetch_universe(markets)
    if not data:
        print("❌ Nenhum dado carregado.")
        return None

    # 2. Notícias + Sentimento
    print("\n📰 Buscando notícias...")
    news_fetcher = NewsFetcher()
    try:
        news = news_fetcher.fetch_all()
        if news:
            analyzer = SentimentAnalyzer(prefer_finbert=False)
            analyze_news_sentiment(news, analyzer)
    except Exception as e:
        logger.warning(f"Notícias indisponíveis: {e}")
        news = []

    # 3. ML
    print("\n🤖 Treinando modelos...\n")
    trainer = ModelTrainer()
    signal_gen = SignalGenerator()
    all_signals = {}
    accuracy = AccuracyTracker()

    for symbol, df in data.items():
        is_valid, msg = DataValidator.check_data_quality(df, symbol)
        if not is_valid:
            continue
        try:
            features = FeatureEngineer.compute_all(df)
            target = FeatureEngineer.create_target(df)
            cv_results = trainer.train_and_evaluate(features, target, symbol)
            if cv_results:
                model = trainer.get_model(symbol)
                signal = signal_gen.generate(model, features, symbol)
                all_signals[symbol] = signal
                accuracy.record_prediction(
                    symbol, signal.signal.value, signal.score,
                    signal.confidence, df["close"].iloc[-1],
                    sentiment=signal.model_votes.get("sentiment", "neutro"),
                )
        except Exception as e:
            logger.debug(f"Skip {symbol}: {e}")

    # 4. Relatórios
    if all_signals:
        ReportGenerator.print_signals(all_signals)

    cv_summary = trainer.get_cv_summary()
    if not cv_summary.empty:
        print("\n" + "═" * 70)
        print("  📋 CROSS-VALIDATION")
        print("═" * 70)
        for idx, row in cv_summary.iterrows():
            print(f"  {idx:<12} accuracy={row['accuracy_mean']:.3f} (±{row['accuracy_std']:.3f})  f1={row['f1_mean']:.3f}")
        print("═" * 70)

    # 5. Backtest
    print("\n📊 Backtest...\n")
    best_symbol, best_result = None, None
    for symbol, df in data.items():
        if symbol not in all_signals:
            continue
        try:
            features = FeatureEngineer.compute_all(df)
            target = FeatureEngineer.create_target(df)
            available = [c for c in FEATURE_COLUMNS if c in features.columns]
            X = features[available].dropna()
            y = target.loc[X.index].dropna()
            mask = X.index.isin(y.index)
            X, y = X[mask], y.loc[X[mask].index]
            warmup = BACKTEST_CONFIG.warmup_period
            signals_series = y.iloc[warmup:]
            bt = BacktestEngine()
            result = bt.run(df.loc[signals_series.index], signals_series, df["close"].loc[signals_series.index])
            if best_result is None or result.sharpe_ratio > best_result.sharpe_ratio:
                best_result, best_symbol = result, symbol
        except Exception:
            pass

    if best_result:
        print(f"  Melhor: {best_symbol}")
        ReportGenerator.print_backtest(best_result)

        if best_result.equity_curve is not None:
            returns = best_result.equity_curve.pct_change().dropna()
            bench_returns = best_result.benchmark_curve.pct_change().dropna() if best_result.benchmark_curve is not None else None
            risk = calculate_risk_metrics(returns, bench_returns)
            ReportGenerator.print_risk(risk)

            # Benchmarks
            print()
            comp = BenchmarkComparator()
            report = comp.compare(best_result.equity_curve)
            report.print_summary()

            # Performance semanal/mensal/anual
            perf = PerformanceTracker(best_result.equity_curve, best_result.trades)
            perf.print_full_report()

            # Chart
            try:
                ReportGenerator.generate_charts(best_result, all_signals, risk)
            except Exception:
                pass

    print("\n✅ Análise completa!")
    return all_signals


def run_paper_trading(initial_capital=100000.0):
    """Paper Trading interativo."""
    print("\n╔" + "═" * 60 + "╗")
    print("║" + " QUANTBOT ML — Paper Trading".center(60) + "║")
    print("║" + f" Capital: {fmt_currency(initial_capital)}".center(60) + "║")
    print("╚" + "═" * 60 + "╝\n")

    engine = PaperTradingEngine(initial_capital=initial_capital)

    print("  📋 ATIVOS:")
    for market, assets in UNIVERSE.items():
        names = ", ".join(a.symbol for a in assets)
        print(f"     {market.value}: {names}")

    while True:
        print("\n  buy <SYM> <QTY> | sell <SYM> <QTY|all> | buyval <SYM> <$>")
        print("  portfolio | orders | price <SYM> | auto | save | load | quit\n")
        try:
            cmd = input("  ▶ ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not cmd:
            continue
        parts = cmd.split()
        action = parts[0].lower()
        try:
            if action == "buy" and len(parts) >= 3:
                engine.execute_buy(parts[1].upper(), quantity=float(parts[2]))
            elif action == "buyval" and len(parts) >= 3:
                engine.execute_buy(parts[1].upper(), amount=float(parts[2]))
            elif action == "sell" and len(parts) >= 3:
                sym = parts[1].upper()
                if parts[2].lower() == "all":
                    engine.execute_sell(sym, sell_all=True)
                else:
                    engine.execute_sell(sym, quantity=float(parts[2]))
            elif action == "portfolio":
                engine.print_portfolio()
            elif action == "orders":
                engine.print_orders()
            elif action == "price" and len(parts) >= 2:
                p = engine.get_current_price(parts[1].upper())
                print(f"\n  💲 {parts[1].upper()}: {fmt_currency(p)}")
            elif action == "auto":
                signals = run_full_analysis()
                if signals:
                    for sym, sig in signals.items():
                        engine.execute_signal(sym, sig.signal, sig.score)
                engine.print_portfolio()
            elif action == "save":
                engine.export_state(parts[1] if len(parts) > 1 else "paper_trading_state.json")
            elif action == "load":
                engine = PaperTradingEngine.load_state(parts[1] if len(parts) > 1 else "paper_trading_state.json")
                engine.print_portfolio()
            elif action in ("quit", "exit", "q"):
                pnl_val, pnl_pct = engine.get_total_pnl()
                print(f"\n  Resultado: {fmt_currency(pnl_val)} ({fmt_pct(pnl_pct)})")
                break
            else:
                print("  ❓ Comando não reconhecido.")
        except Exception as e:
            print(f"  ❌ {e}")


def main():
    parser = argparse.ArgumentParser(description="QuantBot ML — Finance Engine")
    parser.add_argument("--analyze", action="store_true", help="Análise completa")
    parser.add_argument("--paper-trade", action="store_true", help="Paper trading")
    parser.add_argument("--test", action="store_true", help="Rodar testes")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--risk", choices=["conservative", "moderate", "aggressive"], default="moderate")
    parser.add_argument("--markets", nargs="+", choices=["B3", "US", "CRYPTO"], default=None)
    args = parser.parse_args()

    if args.analyze:
        markets = [Market[m] for m in args.markets] if args.markets else None
        run_full_analysis(markets, RiskProfile(args.risk))
        return
    if args.paper_trade:
        run_paper_trading(args.capital)
        return
    if args.test:
        import subprocess
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
        return

    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║           🤖 QUANTBOT ML — Finance Engine            ║
    ║                                                       ║
    ║  [1] Análise Completa (ML + Backtest + Sinais)       ║
    ║  [2] Paper Trading (Dinheiro Simulado)               ║
    ║  [3] Executar Testes (147 testes)                    ║
    ║  [4] Sair                                            ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    try:
        choice = input("  Escolha [1-4]: ").strip()
    except (KeyboardInterrupt, EOFError):
        return
    if choice == "1":
        run_full_analysis()
    elif choice == "2":
        try:
            cap = input(f"  Capital [{fmt_currency(100000)}]: ").strip()
            cap = float(cap) if cap else 100000
        except ValueError:
            cap = 100000
        run_paper_trading(cap)
    elif choice == "3":
        import subprocess
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    else:
        print("  👋 Até mais!")


if __name__ == "__main__":
    main()
