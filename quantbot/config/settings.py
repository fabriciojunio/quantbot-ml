"""
Configurações centralizadas do QuantBot ML.

Todas as constantes, parâmetros de modelos e limites de risco
ficam aqui para facilitar a manutenção e auditoria.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════

class Signal(Enum):
    """Sinais de trading gerados pelo ensemble de ML."""
    STRONG_BUY = "COMPRA_FORTE"
    BUY = "COMPRA"
    HOLD = "NEUTRO"
    SELL = "VENDA"
    STRONG_SELL = "VENDA_FORTE"


class Market(Enum):
    """Mercados suportados."""
    B3 = "B3"
    US = "US"
    CRYPTO = "CRYPTO"


class RiskProfile(Enum):
    """Perfis de risco disponíveis."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# ═══════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AssetConfig:
    """Configuração de um ativo do universo de investimento."""
    symbol: str
    name: str
    market: Market
    sector: str = ""


# ═══════════════════════════════════════════════════════════════
# UNIVERSO DE ATIVOS
# ═══════════════════════════════════════════════════════════════

UNIVERSE: Dict[Market, List[AssetConfig]] = {
    Market.B3: [
        AssetConfig("PETR4.SA", "Petrobras", Market.B3, "Energy"),
        AssetConfig("VALE3.SA", "Vale", Market.B3, "Mining"),
        AssetConfig("ITUB4.SA", "Itaú Unibanco", Market.B3, "Finance"),
        AssetConfig("WEGE3.SA", "WEG", Market.B3, "Industrial"),
        AssetConfig("BBDC4.SA", "Bradesco", Market.B3, "Finance"),
        AssetConfig("ABEV3.SA", "Ambev", Market.B3, "Consumer"),
        AssetConfig("RENT3.SA", "Localiza", Market.B3, "Services"),
        AssetConfig("BBAS3.SA", "Banco do Brasil", Market.B3, "Finance"),
        AssetConfig("B3SA3.SA", "B3", Market.B3, "Finance"),
        AssetConfig("SUZB3.SA", "Suzano", Market.B3, "Paper"),
    ],
    Market.US: [
        AssetConfig("AAPL", "Apple", Market.US, "Tech"),
        AssetConfig("MSFT", "Microsoft", Market.US, "Tech"),
        AssetConfig("NVDA", "NVIDIA", Market.US, "Tech"),
        AssetConfig("GOOGL", "Alphabet", Market.US, "Tech"),
        AssetConfig("AMZN", "Amazon", Market.US, "Tech"),
        AssetConfig("META", "Meta Platforms", Market.US, "Tech"),
        AssetConfig("TSLA", "Tesla", Market.US, "Auto"),
        AssetConfig("JPM", "JPMorgan Chase", Market.US, "Finance"),
        AssetConfig("V", "Visa", Market.US, "Finance"),
        AssetConfig("JNJ", "Johnson & Johnson", Market.US, "Healthcare"),
    ],
    Market.CRYPTO: [
        AssetConfig("BTC-USD", "Bitcoin", Market.CRYPTO, "Cryptocurrency"),
        AssetConfig("ETH-USD", "Ethereum", Market.CRYPTO, "Cryptocurrency"),
        AssetConfig("SOL-USD", "Solana", Market.CRYPTO, "Cryptocurrency"),
        AssetConfig("ADA-USD", "Cardano", Market.CRYPTO, "Cryptocurrency"),
        AssetConfig("DOT-USD", "Polkadot", Market.CRYPTO, "Cryptocurrency"),
    ],
}


# ═══════════════════════════════════════════════════════════════
# PARÂMETROS DE ML
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MLConfig:
    """Configuração dos modelos de Machine Learning."""
    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: int = 10
    rf_min_samples_split: int = 20
    rf_min_samples_leaf: int = 10

    # XGBoost
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # Training
    cv_splits: int = 5
    min_training_samples: int = 100
    prediction_horizon: int = 5  # dias
    return_threshold: float = 0.0  # threshold para classificação

    # Signal thresholds (calibrado: ratio 1:3 como o bot do Luan)
    strong_buy_threshold: float = 75.0   # >75% certeza = COMPRA FORTE
    buy_threshold: float = 65.0          # >65% certeza = COMPRA
    hold_upper_threshold: float = 35.0   # zona morta: 35-65% = NÃO OPERA
    sell_threshold: float = 25.0         # <25% certeza = VENDA FORTE

    random_state: int = 42


ML_CONFIG = MLConfig()


# ═══════════════════════════════════════════════════════════════
# PARÂMETROS DE RISCO
# ═══════════════════════════════════════════════════════════════

RISK_PROFILES: Dict[RiskProfile, Dict] = {
    RiskProfile.CONSERVATIVE: {
        "max_position_pct": 0.05,
        "max_drawdown": 0.08,
        "stop_loss_pct": 0.03,          # 3%
        "take_profit_pct": 0.09,        # 9% — ratio 1:3
        "min_diversification": 15,
        "max_annual_volatility": 0.20,
        "max_correlation": 0.70,
        "rebalance_frequency": "weekly",
        "max_daily_trades": 5,
    },
    RiskProfile.MODERATE: {
        "max_position_pct": 0.10,
        "max_drawdown": 0.15,
        "stop_loss_pct": 0.05,          # 5% — corta perda rápido
        "take_profit_pct": 0.15,        # 15% — ratio 1:3 (deixa lucro correr)
        "min_diversification": 10,
        "max_annual_volatility": 0.35,
        "max_correlation": 0.80,
        "rebalance_frequency": "biweekly",
        "max_daily_trades": 10,         # limite diário
    },
    RiskProfile.AGGRESSIVE: {
        "max_position_pct": 0.20,
        "max_drawdown": 0.25,
        "stop_loss_pct": 0.07,          # 7%
        "take_profit_pct": 0.21,        # 21% — ratio 1:3
        "min_diversification": 5,
        "max_annual_volatility": 0.50,
        "max_correlation": 0.90,
        "rebalance_frequency": "monthly",
        "max_daily_trades": 20,
    },
}


# ═══════════════════════════════════════════════════════════════
# PARÂMETROS DE BACKTEST
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BacktestConfig:
    """Configuração do motor de backtesting."""
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001  # 0.1% por operação
    slippage_rate: float = 0.0005  # 0.05% de slippage estimado
    max_allocation_pct: float = 0.95  # máximo 95% em posições
    min_trade_interval_days: int = 1  # mínimo 1 dia entre trades
    warmup_period: int = 60  # dias de aquecimento para indicadores


BACKTEST_CONFIG = BacktestConfig()


# ═══════════════════════════════════════════════════════════════
# PARÂMETROS DE DADOS
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DataConfig:
    """Configuração da camada de dados."""
    lookback_days: int = 730  # 2 anos de histórico
    min_data_points: int = 60  # mínimo de pontos para análise
    max_missing_pct: float = 0.05  # máximo 5% de dados faltantes
    outlier_std_threshold: float = 5.0  # z-score para outliers
    cache_enabled: bool = True
    rate_limit_seconds: float = 0.5  # intervalo entre requisições


DATA_CONFIG = DataConfig()


# ═══════════════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════════════

# Features que serão usadas para treino (excluindo preços absolutos)
FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_20d", "log_return",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20", "sma_ratio_50",
    "rsi", "macd_normalized", "macd_hist_normalized",
    "bb_width", "bb_position",
    "volatility_10", "volatility_20", "volatility_ratio",
    "volume_ratio", "volume_change",
    "momentum_10", "momentum_20",
    "atr_ratio",
    "stoch_k", "stoch_d", "williams_r",
    "obv_change",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "roc_5", "roc_10",
]
