"""
Módulo de coleta de dados de mercado.

Utiliza yfinance para obter dados históricos de preços,
com cache, rate limiting e validação automática.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from config.settings import DATA_CONFIG, UNIVERSE, Market, AssetConfig
from utils.logger import get_logger
from utils.security import InputValidator, DataSanitizer

logger = get_logger("quantbot.data")


class MarketDataFetcher:
    """
    Coleta dados de mercado com cache e validação.

    Responsabilidades:
    - Baixar dados históricos via yfinance
    - Validar e sanitizar dados recebidos
    - Cachear dados para evitar requisições duplicadas
    - Respeitar rate limits da API
    """

    def __init__(self, config=None):
        self.config = config or DATA_CONFIG
        self._cache: Dict[str, pd.DataFrame] = {}
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Aplica rate limiting entre requisições."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_seconds:
            time.sleep(self.config.rate_limit_seconds - elapsed)
        self._last_request_time = time.time()

    def fetch_single(
        self,
        symbol: str,
        lookback_days: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Baixa dados históricos de um ativo.

        Args:
            symbol: Símbolo do ativo (ex: "PETR4.SA")
            lookback_days: Dias de histórico (padrão: config)

        Returns:
            DataFrame com colunas [open, high, low, close, volume]
            ou None se falhar
        """
        symbol = InputValidator.validate_symbol(symbol)
        days = lookback_days or self.config.lookback_days

        # Verifica cache
        if self.config.cache_enabled and symbol in self._cache:
            logger.debug(f"Cache hit: {symbol}")
            return self._cache[symbol]

        self._rate_limit()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty or len(df) < self.config.min_data_points:
                logger.warning(
                    f"{symbol}: dados insuficientes "
                    f"({len(df)}/{self.config.min_data_points} pontos)"
                )
                return None

            # Padroniza colunas
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]

            # Sanitiza dados
            df = DataSanitizer.sanitize_dataframe(
                df,
                max_missing_pct=self.config.max_missing_pct,
            )

            # Cache
            if self.config.cache_enabled:
                self._cache[symbol] = df

            logger.info(f"  ✓ {symbol}: {len(df)} registros carregados")
            return df

        except Exception as e:
            logger.error(f"  ✗ {symbol}: {e}")
            return None

    def fetch_universe(
        self,
        markets: Optional[List[Market]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Baixa dados de todos os ativos dos mercados selecionados.

        Args:
            markets: Lista de mercados (padrão: todos)

        Returns:
            Dicionário {símbolo: DataFrame}
        """
        if markets is None:
            markets = list(Market)

        symbols = []
        for market in markets:
            assets = UNIVERSE.get(market, [])
            symbols.extend([a.symbol for a in assets])

        logger.info(f"📊 Baixando dados para {len(symbols)} ativos...")

        results = {}
        for symbol in symbols:
            df = self.fetch_single(symbol)
            if df is not None:
                results[symbol] = df

        logger.info(
            f"📊 Concluído: {len(results)}/{len(symbols)} ativos carregados"
        )
        return results

    def get_cached(self, symbol: str) -> Optional[pd.DataFrame]:
        """Retorna dados em cache para um símbolo."""
        return self._cache.get(symbol)

    def clear_cache(self):
        """Limpa o cache de dados."""
        self._cache.clear()
        logger.info("Cache de dados limpo")

    @staticmethod
    def find_asset(symbol: str) -> Optional[AssetConfig]:
        """Encontra a configuração de um ativo pelo símbolo."""
        for assets in UNIVERSE.values():
            for asset in assets:
                if asset.symbol == symbol:
                    return asset
        return None
