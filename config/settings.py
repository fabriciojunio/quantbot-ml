"""Configuração central do QuantBot.

Define os enums de domínio (sinais, mercados, perfis de risco), o universo
de ativos negociáveis e os blocos de parâmetros de dados, ML, backtest e
risco. Os valores aqui são defaults conservadores — em produção devem ser
revisados e sobrescritos por variáveis de ambiente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


# ═══════════════════════════════════════════════════════════════
# ENUMS DE DOMÍNIO
# ═══════════════════════════════════════════════════════════════

class Signal(Enum):
    """Sinal de trading discreto.

    O valor inteiro é a intensidade direcional (-2 a +2), usada para
    converter sinais em score numérico e vice-versa.
    """
    STRONG_SELL = -2
    SELL = -1
    HOLD = 0
    BUY = 1
    STRONG_BUY = 2

    def __str__(self) -> str:  # pragma: no cover - cosmético
        return self.name


class Market(Enum):
    """Mercado / praça de negociação de um ativo."""
    BR = "BR"   # B3 (Brasil)
    US = "US"   # NYSE / Nasdaq
    CRYPTO = "CRYPTO"


class RiskProfile(Enum):
    """Perfil de risco que parametriza sizing, stop e take-profit."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# ═══════════════════════════════════════════════════════════════
# UNIVERSO DE ATIVOS
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AssetConfig:
    """Configuração de um ativo negociável."""
    symbol: str
    name: str
    market: Market
    sector: str = "N/A"


# Universo enxuto e líquido. Mantido pequeno de propósito: López de Prado
# alerta que testar centenas de ativos/estratégias infla o overfitting.
UNIVERSE: Dict[Market, List[AssetConfig]] = {
    Market.BR: [
        AssetConfig("PETR4.SA", "Petrobras PN", Market.BR, "Energia"),
        AssetConfig("VALE3.SA", "Vale ON", Market.BR, "Mineração"),
        AssetConfig("ITUB4.SA", "Itaú Unibanco PN", Market.BR, "Bancos"),
        AssetConfig("BBDC4.SA", "Bradesco PN", Market.BR, "Bancos"),
        AssetConfig("BBAS3.SA", "Banco do Brasil ON", Market.BR, "Bancos"),
        AssetConfig("WEGE3.SA", "WEG ON", Market.BR, "Indústria"),
    ],
    Market.US: [
        AssetConfig("AAPL", "Apple Inc.", Market.US, "Tech"),
        AssetConfig("MSFT", "Microsoft Corp.", Market.US, "Tech"),
        AssetConfig("SPY", "SPDR S&P 500 ETF", Market.US, "ETF"),
    ],
    Market.CRYPTO: [
        AssetConfig("BTC-USD", "Bitcoin", Market.CRYPTO, "Crypto"),
        AssetConfig("ETH-USD", "Ethereum", Market.CRYPTO, "Crypto"),
    ],
}


# ═══════════════════════════════════════════════════════════════
# BLOCOS DE CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════

@dataclass
class DataConfig:
    """Parâmetros de coleta e validação de dados de mercado."""
    lookback_days: int = 730              # ~2 anos de histórico
    min_data_points: int = 100            # mínimo para gerar features
    cache_enabled: bool = True
    rate_limit_seconds: float = 1.0       # intervalo mínimo entre requisições
    max_missing_pct: float = 0.05         # tolerância de dados faltantes


@dataclass
class MLConfig:
    """Hiperparâmetros e janela de treino dos modelos."""
    train_window: int = 504               # ~2 anos úteis
    test_window: int = 63                 # ~3 meses
    embargo_pct: float = 0.01             # embargo do purged CV
    n_estimators: int = 200
    max_depth: int = 5
    random_state: int = 42
    prediction_horizon: int = 5           # dias à frente para o label
    min_confidence: float = 0.55          # confiança mínima para operar


@dataclass
class BacktestConfig:
    """Custos e premissas do backtest (mercado brasileiro)."""
    initial_capital: float = 10_000.0
    commission_pct: float = 0.0          # corretagem (muitas zeram hoje)
    b3_emolument_pct: float = 0.00005     # emolumentos B3 (~0,005%)
    slippage_pct: float = 0.0005          # escorregamento estimado
    spread_pct: float = 0.0005            # metade do spread bid-ask
    tax_exempt_monthly_limit: float = 20_000.0  # isenção de IR (swing)
    swing_tax_rate: float = 0.15          # IR swing acima da isenção
    daytrade_tax_rate: float = 0.20       # IR day trade (sempre)


@dataclass
class RiskConfig:
    """Limites por perfil de risco."""
    max_position_pct: float = 0.10        # exposição máxima por ativo
    stop_loss_pct: float = 0.035          # stop-loss
    take_profit_pct: float = 0.15         # take-profit
    max_daily_loss_pct: float = 0.03      # perda diária máxima
    max_drawdown_pct: float = 0.20        # kill-switch permanente
    max_trades_per_day: int = 5


# Instâncias singleton usadas pelo restante do sistema.
DATA_CONFIG = DataConfig()
ML_CONFIG = MLConfig()
BACKTEST_CONFIG = BacktestConfig()

RISK_PROFILES: Dict[RiskProfile, RiskConfig] = {
    RiskProfile.CONSERVATIVE: RiskConfig(
        max_position_pct=0.05, stop_loss_pct=0.025, take_profit_pct=0.10,
        max_daily_loss_pct=0.02, max_drawdown_pct=0.15, max_trades_per_day=3,
    ),
    RiskProfile.MODERATE: RiskConfig(),
    RiskProfile.AGGRESSIVE: RiskConfig(
        max_position_pct=0.20, stop_loss_pct=0.05, take_profit_pct=0.20,
        max_daily_loss_pct=0.05, max_drawdown_pct=0.30, max_trades_per_day=10,
    ),
}


# ═══════════════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════════════

# Subconjunto econômico de features usado pelos modelos. Os nomes batem
# com as colunas geradas em data/features.py::FeatureEngineer.compute_all.
FEATURE_COLUMNS: List[str] = [
    "return_1d", "return_5d", "return_10d", "return_20d", "log_return",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20", "sma_ratio_50",
    "ema_ratio_5", "ema_ratio_10", "ema_ratio_20", "ema_ratio_50",
    "sma_cross_5_20", "sma_cross_10_50",
    "rsi",
    "macd_normalized", "macd_signal_normalized", "macd_hist_normalized",
    "bb_width", "bb_position",
    "volatility_10", "volatility_20", "volatility_ratio", "atr_ratio",
    "volume_ratio", "volume_change", "obv_change",
    "momentum_10", "momentum_20", "roc_5", "roc_10",
    "stoch_k", "stoch_d", "williams_r",
    "day_sin", "day_cos", "month_sin", "month_cos",
]
