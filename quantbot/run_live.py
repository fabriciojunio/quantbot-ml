"""
Trading ao Vivo — Runner Standalone via Alpaca.

Usa o AlpacaAdapter já existente em core/live_trading.py
e as novas estratégias modulares para gerar sinais.

Uso:
    python run_live.py                        # Menu interativo
    python run_live.py AAPL TSLA NVDA         # Tickers específicos
    python run_live.py --crypto               # Criptos 24/7
    python run_live.py --stocks               # Ações US
    python run_live.py --force-buy            # Força compra no 1o ciclo (teste)

IMPORTANTE: Comece SEMPRE com ALPACA_ENV=paper no .env
"""

import sys
import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Signal, UNIVERSE, Market, RISK_PROFILES, RiskProfile
from data.fetcher import MarketDataFetcher as DataFetcher
from data.features import FeatureEngineer
from strategies.ensemble_voting import EnsembleVotingStrategy
from strategies.base import SIGNAL_NUMERIC

logger = logging.getLogger("quantbot.run_live")

CHECK_INTERVAL = 300  # 5 minutos
LOOKBACK_DAYS = 100

# Tickers por categoria
CRYPTO_TICKERS = [a.symbol for a in UNIVERSE.get(Market.CRYPTO, [])]
US_TICKERS = [a.symbol for a in UNIVERSE.get(Market.US, [])]
BR_TICKERS = [a.symbol for a in UNIVERSE.get(Market.B3, [])]

# Risk params
RISK = RISK_PROFILES[RiskProfile.MODERATE]
STOP_LOSS = RISK["stop_loss_pct"]
TAKE_PROFIT = RISK["take_profit_pct"]
MAX_POS = RISK["max_position_pct"]


def get_fresh_data(ticker: str):
    """Baixa dados recentes sem cache."""
    fetcher = DataFetcher()
    df_raw = fetcher.fetch_single(ticker, lookback_days=LOOKBACK_DAYS)
    if df_raw is None:
        return pd.DataFrame()
    try:
        features = FeatureEngineer.compute_all(df_raw)
        df = df_raw.join(features, how="left")
        df["Close"] = df["close"]
        df["Volume"] = df["volume"]
    except Exception:
        return pd.DataFrame()
    return df


def calculate_position_size(capital, price, signal_val, is_crypto=False):
    """Calcula qty baseado no capital e sinal."""
    if price <= 0 or capital <= 0:
        return 0
    multiplier = 1.5 if abs(signal_val) == 2 else 1.0
    budget = min(capital * MAX_POS * multiplier, capital * 0.20)
    if is_crypto:
        return budget / price
    return max(int(budget / price), 0)


def trading_loop(tickers, force_buy=False):
    """Loop principal de trading."""
    # Tenta importar e conectar Alpaca
    try:
        from dotenv import load_dotenv
        load_dotenv()

        alpaca_env = os.getenv("ALPACA_ENV", "paper")
        from core.live_trading import AlpacaAdapter
        broker = AlpacaAdapter(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            api_secret=os.getenv("ALPACA_SECRET_KEY", ""),
            paper=(alpaca_env == "paper"),
        )

        if not broker.connect():
            print("\n  Erro ao conectar na Alpaca.")
            print("  Verifique suas chaves no .env")
            return

        env_label = "PAPER" if alpaca_env == "paper" else "LIVE"
    except Exception as e:
        print(f"\n  Erro ao inicializar Alpaca: {e}")
        print("\n  Passos para configurar:")
        print("  1. Crie conta em https://alpaca.markets")
        print("  2. Dashboard > Paper Trading > API Keys")
        print("  3. Copie .env.example para .env e preencha")
        return

    strategy = EnsembleVotingStrategy()
    daily_trades = 0
    max_daily = 10

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  TRADING AO VIVO — Alpaca ({env_label:5s})   ║")
    print(f"  ╚══════════════════════════════════════╝")
    print(f"\n  Monitorando: {', '.join(tickers)}")
    print(f"  Intervalo: {CHECK_INTERVAL}s")
    print(f"  Stop-Loss: {STOP_LOSS:.0%} | Take-Profit: {TAKE_PROFIT:.0%}")
    print(f"\n  Aguardando proximo ciclo...\n")

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Verifica mercado
        try:
            # Crypto roda 24/7
            tradable = [t for t in tickers if "-USD" in t or _is_market_open(broker)]
        except Exception:
            tradable = [t for t in tickers if "-USD" in t]

        if not tradable:
            print(f"  [{now}] Mercado fechado. Aguardando...")
            time.sleep(60)
            continue

        print(f"\n{'─' * 55}")
        print(f"  [{now}] Ciclo #{cycle}")
        print(f"{'─' * 55}")

        # Saldo
        try:
            cash = broker.get_balance()
            print(f"  Cash: ${cash:,.2f}")
        except Exception:
            cash = 0
            print("  Erro ao consultar saldo")

        # Posições
        try:
            positions = broker.get_positions()
        except Exception:
            positions = {}

        for ticker in tradable:
            try:
                if ticker.endswith(".SA"):
                    print(f"  {ticker}: Alpaca nao suporta B3.")
                    continue

                df = get_fresh_data(ticker)
                if df.empty or len(df) < 50:
                    print(f"  {ticker}: Dados insuficientes.")
                    continue

                df_signals = strategy.generate_signals(df)
                latest = df_signals.iloc[-1]
                signal = latest["signal"]
                price = latest["Close"]
                signal_val = SIGNAL_NUMERIC.get(signal, 0)

                emoji = {
                    Signal.STRONG_BUY: "++", Signal.BUY: "+",
                    Signal.HOLD: "=", Signal.SELL: "-", Signal.STRONG_SELL: "--",
                }.get(signal, "=")

                print(f"\n  [{emoji}] {ticker}: ${price:.2f} -> {signal.value}")

                if force_buy and cycle == 1 and ticker not in positions:
                    print(f"       [TESTE] Forcando COMPRA")
                    signal = Signal.BUY
                    signal_val = 1

                has_pos = ticker in positions

                if has_pos:
                    pos = positions[ticker]
                    entry = pos.get("avg_entry_price", pos.get("entry_price", price))
                    pnl_pct = (price - entry) / entry if entry > 0 else 0

                    if pnl_pct <= -STOP_LOSS:
                        print(f"       STOP-LOSS ({pnl_pct:.1%}). Vendendo...")
                        broker.sell(ticker, pos.get("qty", pos.get("quantity", 0)))
                        daily_trades += 1
                    elif pnl_pct >= TAKE_PROFIT:
                        print(f"       TAKE-PROFIT ({pnl_pct:.1%}). Vendendo...")
                        broker.sell(ticker, pos.get("qty", pos.get("quantity", 0)))
                        daily_trades += 1
                    elif signal in (Signal.SELL, Signal.STRONG_SELL):
                        print(f"       Sinal de venda. Vendendo...")
                        broker.sell(ticker, pos.get("qty", pos.get("quantity", 0)))
                        daily_trades += 1
                    else:
                        pl = pos.get("unrealized_pl", 0)
                        print(f"       Posicao aberta: P/L ${pl:+.2f}")

                elif signal in (Signal.BUY, Signal.STRONG_BUY) and daily_trades < max_daily:
                    is_crypto = "-USD" in ticker
                    qty = calculate_position_size(cash, price, signal_val, is_crypto)
                    if qty > 0:
                        qty_str = f"{qty:.4f}" if is_crypto else str(int(qty))
                        print(f"       Comprando {qty_str} unidades...")
                        broker.buy(ticker, qty, price)
                        daily_trades += 1
                        cash -= qty * price

            except Exception as e:
                print(f"  Erro em {ticker}: {e}")

        print(f"\n  Proximo ciclo em {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)


def _is_market_open(broker):
    """Tenta verificar se mercado está aberto."""
    try:
        if hasattr(broker, 'client') and broker.client:
            clock = broker.client.get_clock()
            return clock.is_open
    except Exception:
        pass
    # Fallback: horário comercial US (9:30-16:00 ET)
    from datetime import timezone
    now_utc = datetime.now(timezone.utc)
    hour_et = (now_utc.hour - 4) % 24  # Aproximação ET
    return 9 <= hour_et < 16 and now_utc.weekday() < 5


def main():
    force_buy = "--force-buy" in sys.argv
    is_crypto = "--crypto" in sys.argv
    is_stocks = "--stocks" in sys.argv

    tickers = [t for t in sys.argv[1:] if not t.startswith("--")]

    if is_crypto:
        tickers.extend(CRYPTO_TICKERS)
    if is_stocks:
        tickers.extend(US_TICKERS[:5])

    if not tickers:
        print("\n  ╔═════════════════════════════════════════╗")
        print("  ║  Opcoes de Mercado                      ║")
        print("  ╠═════════════════════════════════════════╣")
        print("  ║  [1] Criptomoedas (24/7)                ║")
        print("  ║  [2] Acoes EUA (horario comercial)      ║")
        print("  ║  [3] Ambos (Hibrido)                    ║")
        print("  ╚═════════════════════════════════════════╝")

        choice = input("\n  Qual mercado (1/2/3)? ").strip()
        if choice == "1":
            tickers = CRYPTO_TICKERS
        elif choice == "2":
            tickers = US_TICKERS[:5]
        elif choice == "3":
            tickers = CRYPTO_TICKERS + US_TICKERS[:5]
        else:
            tickers = US_TICKERS[:5]

    tickers = list(dict.fromkeys(tickers))

    try:
        trading_loop(tickers, force_buy=force_buy)
    except KeyboardInterrupt:
        print("\n\n  Trading interrompido.")
        print("  Verifique posicoes em https://app.alpaca.markets")


if __name__ == "__main__":
    main()
