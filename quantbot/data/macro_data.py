"""
Conhecimento do Professor Braguim — Integrado ao QuantBot.

Técnicas extraídas dos 25 notebooks de Python para Investimentos:

1. API Banco Central (NB 10, 24, 25) — CDI, SELIC, IPCA, câmbio
2. Correlação Dólar/Ibovespa (NB 03) — hedge e diversificação
3. Análise Fundamentalista (NB 08) — Fórmula Mágica (Greenblatt)
4. Análise Estatística (NB 07) — distribuição t-Student, VaR probabilístico
5. Backtesting com Rebalanceamento (NB 24, 25) — rebalance mensal
6. Pyfolio Tear Sheet (NB 05) — rolling beta, benchmark comparison
7. CVM Fundos (NB 16, 17) — ranking de fundos para benchmark
8. Dados FED/FRED (NB 23) — macro US: GDP, CPI, desemprego, M2
9. Simulação Monte Carlo (NB 06) — 500 carteiras aleatórias
10. Tesouro Direto (NB 20) — taxas e preços dos títulos
11. Carteira por Cotas (NB 27) — rentabilidade real com aportes
12. B3 Web Scraping (NB 18) — carteira teórica do Ibovespa

Referências:
    - Prof. Braguim — Canal Python para Investimentos
    - Greenblatt — The Little Book That Beats the Market
    - Markowitz (1952) — Portfolio Selection
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger("quantbot.data.macro_data")


# ═══════════════════════════════════════════════════════════════
# 1. API BANCO CENTRAL DO BRASIL (Notebooks 10, 24, 25)
# ═══════════════════════════════════════════════════════════════

class BancoCentralAPI:
    """
    Acessa séries temporais do Banco Central do Brasil.
    Baseado em API aberta do BCB.
    """

    SERIES = {
        "selic_meta": 432,
        "cdi": 12,
        "ipca": 433,
        "igpm": 189,
        "cambio_venda": 1,
        "cambio_compra": 10813,
        "reservas_internacionais": 13621,
        "desemprego_pnad": 24369,
        "divida_liquida_pib": 4513,
        "ibc_br": 24364,  # Índice de Atividade Econômica
    }

    @staticmethod
    def consulta_bc(codigo_bcb: int) -> pd.DataFrame:
        """Busca série temporal do BCB via API pública."""
        try:
            url = f"http://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_bcb}/dados?formato=json"
            df = pd.read_json(url)
            df["data"] = pd.to_datetime(df["data"], dayfirst=True)
            df.set_index("data", inplace=True)
            return df
        except Exception as e:
            logger.warning(f"Erro BCB série {codigo_bcb}: {e}")
            return pd.DataFrame()

    @classmethod
    def cdi_acumulado(cls, data_inicio: str, data_fim: str) -> pd.Series:
        """CDI acumulado no período."""
        cdi = cls.consulta_bc(cls.SERIES["cdi"])
        if cdi.empty:
            return pd.Series()
        cdi_periodo = cdi[data_inicio:data_fim]
        acumulado = (1 + cdi_periodo["valor"] / 100).cumprod()
        if len(acumulado) > 0:
            acumulado.iloc[0] = 1
        return acumulado

    @classmethod
    def get_macro_features(cls, start: str = "2020-01-01") -> pd.DataFrame:
        """
        Features macroeconômicas para o modelo ML.

        Retorna DataFrame com:
            - selic: taxa SELIC meta
            - selic_change: variação da SELIC
            - ipca: inflação mensal
            - ipca_12m: inflação acumulada 12 meses
            - cambio: dólar comercial
            - cambio_change: variação do dólar
            - ibc_br: índice de atividade econômica
        """
        features = pd.DataFrame()

        try:
            selic = cls.consulta_bc(cls.SERIES["selic_meta"])
            if not selic.empty:
                features["selic"] = selic["valor"]
                features["selic_change"] = selic["valor"].diff()

            ipca = cls.consulta_bc(cls.SERIES["ipca"])
            if not ipca.empty:
                features["ipca"] = ipca["valor"]
                features["ipca_12m"] = ipca["valor"].rolling(12).sum()

            cambio = cls.consulta_bc(cls.SERIES["cambio_venda"])
            if not cambio.empty:
                features["cambio"] = cambio["valor"]
                features["cambio_change"] = cambio["valor"].pct_change()

            ibc = cls.consulta_bc(cls.SERIES["ibc_br"])
            if not ibc.empty:
                features["ibc_br"] = ibc["valor"]
                features["ibc_br_change"] = ibc["valor"].pct_change()

        except Exception as e:
            logger.warning(f"Erro ao buscar macro features: {e}")

        if start and not features.empty:
            features = features[features.index >= start]

        return features


# ═══════════════════════════════════════════════════════════════
# 2. DADOS FED / FRED (Notebook 23)
# ═══════════════════════════════════════════════════════════════

class FedDataAPI:
    """
    Acessa dados econômicos dos EUA via FRED.
    Dados econômicos US via FRED API.

    Séries: GDP, CPI (inflação), desemprego, M2, Fed Funds Rate.
    """

    @staticmethod
    def fetch_fred(series_id: str, start: str = "2020-01-01") -> pd.DataFrame:
        """Busca série do FRED (Federal Reserve Economic Data)."""
        try:
            import pandas_datareader.data as web
            return web.DataReader(series_id, "fred", start)
        except ImportError:
            logger.warning("pandas_datareader não instalado. pip install pandas-datareader")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Erro FRED {series_id}: {e}")
            return pd.DataFrame()

    @classmethod
    def get_us_macro_features(cls, start: str = "2020-01-01") -> pd.DataFrame:
        """Features macro US para o modelo (ativos americanos)."""
        features = pd.DataFrame()

        series = {
            "fed_funds": "FEDFUNDS",      # Taxa de juros Fed
            "cpi": "CPIAUCSL",            # Inflação US
            "unemployment": "UNRATE",      # Desemprego
            "m2": "M2SL",                 # Oferta monetária
            "treasury_10y": "DGS10",       # Treasury 10 anos
            "treasury_3m": "DGS3MO",       # Treasury 3 meses
        }

        for name, code in series.items():
            try:
                data = cls.fetch_fred(code, start)
                if not data.empty:
                    features[name] = data.iloc[:, 0]
            except Exception:
                pass

        # Yield curve (spread 10y - 3m) — indicador de recessão
        if "treasury_10y" in features and "treasury_3m" in features:
            features["yield_curve"] = features["treasury_10y"] - features["treasury_3m"]

        return features


# ═══════════════════════════════════════════════════════════════
# 3. CORRELAÇÃO DÓLAR/IBOVESPA (Notebook 03)
# ═══════════════════════════════════════════════════════════════

class CorrelationAnalyzer:
    """
    Análise de correlação entre ativos (notebook 03).

    A correlação Dólar/Ibovespa é historicamente negativa (~-0.5).
    Quando o dólar sobe, a bolsa cai. Isso é útil para:
        - Hedging: comprar dólar quando posicionado em B3
        - Regime detection: correlação quebrando indica mudança de regime
        - Diversificação: evitar ativos muito correlacionados
    """

    @staticmethod
    def rolling_correlation(
        series_a: pd.Series,
        series_b: pd.Series,
        window: int = 252,
    ) -> pd.Series:
        """
        Correlação rolling entre duas séries.

        Janela padrão: 252 dias (1 ano de trading).
        Quando a correlação muda bruscamente, algo mudou no mercado.
        """
        returns_a = series_a.pct_change()
        returns_b = series_b.pct_change()
        return returns_a.rolling(window).corr(returns_b)

    @staticmethod
    def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Matriz de correlação dos retornos."""
        returns = df.pct_change().dropna()
        return returns.corr()

    @staticmethod
    def should_hedge(
        ibov_series: pd.Series,
        dolar_series: pd.Series,
        threshold: float = -0.3,
    ) -> bool:
        """
        Verifica se correlação dólar/ibov está forte (hedge útil).

        Se correlação < -0.3, comprar dólar protege posições em B3.
        """
        corr = CorrelationAnalyzer.rolling_correlation(
            ibov_series, dolar_series, window=60
        )
        latest = corr.dropna().iloc[-1] if len(corr.dropna()) > 0 else 0
        return latest < threshold


# ═══════════════════════════════════════════════════════════════
# 4. ANÁLISE FUNDAMENTALISTA (Notebook 08) — Fórmula Mágica
# ═══════════════════════════════════════════════════════════════

class FundamentalAnalysis:
    """
    Screening fundamentalista via Fundamentus (notebook 08).

    Implementa a Fórmula Mágica de Greenblatt:
        - Rankeia por EV/EBIT (mais barato = melhor)
        - Rankeia por ROIC (mais rentável = melhor)
        - Soma os rankings: menores somas são as melhores ações
    """

    @staticmethod
    def fetch_fundamentus() -> pd.DataFrame:
        """Busca dados fundamentalistas do Fundamentus.com.br."""
        try:
            import requests
            url = "http://www.fundamentus.com.br/resultado.php"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
            }
            r = requests.get(url, headers=headers)
            df = pd.read_html(r.text, decimal=",", thousands=".")[0]

            # Limpa colunas percentuais
            for col in ["Div.Yield", "Mrg Ebit", "Mrg. Líq.", "ROIC", "ROE", "Cresc. Rec.5a"]:
                if col in df.columns:
                    df[col] = (
                        df[col].str.replace(".", "")
                        .str.replace(",", ".")
                        .str.rstrip("%")
                        .astype(float) / 100
                    )

            return df
        except Exception as e:
            logger.warning(f"Erro ao acessar Fundamentus: {e}")
            return pd.DataFrame()

    @staticmethod
    def magic_formula(df: pd.DataFrame = None, top_n: int = 15) -> pd.DataFrame:
        """
        Fórmula Mágica de Greenblatt.

        Retorna top N ações rankeadas pela combinação de:
            - EV/EBIT (valuation) — quanto menor, mais barato
            - ROIC (qualidade) — quanto maior, mais eficiente
        """
        if df is None:
            df = FundamentalAnalysis.fetch_fundamentus()

        if df.empty:
            return pd.DataFrame()

        # Filtra liquidez mínima
        df = df[df["Liq.2meses"] > 1_000_000].copy()

        # Ranking EV/EBIT (menor = melhor)
        ev_ebit_valid = df[df["EV/EBIT"] > 0].sort_values("EV/EBIT")
        ranking_ev = pd.DataFrame({
            "pos_ev": range(1, len(ev_ebit_valid) + 1)
        }, index=ev_ebit_valid["Papel"].values)

        # Ranking ROIC (maior = melhor)
        roic_sorted = df.sort_values("ROIC", ascending=False)
        ranking_roic = pd.DataFrame({
            "pos_roic": range(1, len(roic_sorted) + 1)
        }, index=roic_sorted["Papel"].values)

        # Combina rankings
        combined = ranking_ev.join(ranking_roic, how="inner")
        combined["total"] = combined["pos_ev"] + combined["pos_roic"]
        combined = combined.sort_values("total")

        return combined.head(top_n)


# ═══════════════════════════════════════════════════════════════
# 5. ANÁLISE ESTATÍSTICA DE EVENTOS EXTREMOS (Notebook 07)
# ═══════════════════════════════════════════════════════════════

class ExtremeEventAnalysis:
    """
    Análise probabilística de eventos extremos (notebook 07).

    Calcula probabilidade de quedas extremas como circuit
    breaker de -12% do Ibovespa usando distribuição t-Student
    (mais realista que a Normal para caudas pesadas).
    """

    @staticmethod
    def analyze_tail_risk(
        returns: pd.Series,
        threshold: float = -0.05,
    ) -> Dict:
        """
        Calcula probabilidade de eventos extremos.

        Args:
            returns: Série de retornos diários
            threshold: Nível de queda a analisar (ex: -0.05 = -5%)

        Returns:
            Dict com probabilidades e frequências
        """
        from scipy.stats import norm, t

        mean = returns.mean()
        std = returns.std()

        # Probabilidade Normal
        prob_normal = norm.cdf(threshold, loc=mean, scale=std)
        freq_normal = 1 / prob_normal if prob_normal > 0 else float("inf")

        # Probabilidade t-Student (caudas pesadas — mais realista)
        dof, loc_t, scale_t = t.fit(returns)
        prob_t = t.cdf(threshold, dof, loc=loc_t, scale=scale_t)
        freq_t = 1 / prob_t if prob_t > 0 else float("inf")

        # Eventos históricos reais
        extreme_days = returns[returns < threshold]

        return {
            "threshold": threshold,
            "prob_normal": prob_normal,
            "freq_normal_days": int(freq_normal),
            "freq_normal_years": int(freq_normal / 252),
            "prob_t_student": prob_t,
            "freq_t_days": int(freq_t),
            "freq_t_years": int(freq_t / 252),
            "t_student_dof": dof,
            "historical_count": len(extreme_days),
            "historical_freq_days": int(len(returns) / max(len(extreme_days), 1)),
        }

    @staticmethod
    def impact_of_missing_days(
        returns: pd.Series,
        n_days: int = 10,
    ) -> Dict:
        """
        Impacto de perder os melhores/piores dias (notebook 21).

        Mostra por que o bot deve estar posicionado nos dias certos.
        """
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1

        # Sem os N melhores dias
        best_days = returns.sort_values(ascending=False).head(n_days)
        without_best = (1 + returns.drop(best_days.index)).cumprod()
        return_without_best = without_best.iloc[-1] - 1

        # Sem os N piores dias
        worst_days = returns.sort_values(ascending=True).head(n_days)
        without_worst = (1 + returns.drop(worst_days.index)).cumprod()
        return_without_worst = without_worst.iloc[-1] - 1

        return {
            "total_return": total_return * 100,
            f"without_{n_days}_best_days": return_without_best * 100,
            f"without_{n_days}_worst_days": return_without_worst * 100,
            "best_day_impact": (total_return - return_without_best) * 100,
            "worst_day_impact": (return_without_worst - total_return) * 100,
        }


# ═══════════════════════════════════════════════════════════════
# 6. SIMULAÇÃO MONTE CARLO DE CARTEIRAS (Notebook 06)
# ═══════════════════════════════════════════════════════════════

class MonteCarloPortfolio:
    """
    Simulação Monte Carlo de carteiras (notebook 06).

    Gera N carteiras aleatórias e identifica a fronteira eficiente.
    Útil para encontrar a alocação ótima entre os ativos do bot.
    """

    @staticmethod
    def simulate(
        returns: pd.DataFrame,
        n_portfolios: int = 5000,
        risk_free_rate: float = 0.125,  # CDI ~12.5% a.a.
    ) -> pd.DataFrame:
        """
        Simula N carteiras com pesos aleatórios.

        Args:
            returns: DataFrame com retornos diários de cada ativo
            n_portfolios: Número de carteiras a simular
            risk_free_rate: Taxa livre de risco (CDI)

        Returns:
            DataFrame com retorno, volatilidade, sharpe e pesos de cada carteira
        """
        n_assets = returns.shape[1]
        results = []

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        for _ in range(n_portfolios):
            # Pesos aleatórios normalizados
            weights = np.random.random(n_assets)
            weights /= weights.sum()

            # Retorno esperado
            port_return = np.dot(weights, mean_returns)

            # Volatilidade (risco)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            # Sharpe
            sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

            results.append({
                "return": port_return * 100,
                "volatility": port_vol * 100,
                "sharpe": sharpe,
                **{f"w_{returns.columns[i]}": weights[i] for i in range(n_assets)},
            })

        df = pd.DataFrame(results)

        # Marca carteira ótima (maior Sharpe)
        best_idx = df["sharpe"].idxmax()
        df["is_optimal"] = False
        df.loc[best_idx, "is_optimal"] = True

        # Marca mínima variância
        min_vol_idx = df["volatility"].idxmin()
        df["is_min_vol"] = False
        df.loc[min_vol_idx, "is_min_vol"] = True

        return df

    @staticmethod
    def get_optimal_weights(simulation_result: pd.DataFrame) -> Dict[str, float]:
        """Retorna os pesos da carteira ótima (maior Sharpe)."""
        optimal = simulation_result[simulation_result["is_optimal"]].iloc[0]
        weight_cols = [c for c in optimal.index if c.startswith("w_")]
        return {col.replace("w_", ""): optimal[col] for col in weight_cols}


# ═══════════════════════════════════════════════════════════════
# 7. RENTABILIDADE POR COTAS (Notebook 27)
# ═══════════════════════════════════════════════════════════════

class CotaCalculator:
    """
    Cálculo de rentabilidade por cotas com aportes (notebook 27).

    Método profissional de calcular retorno real de uma carteira
    que recebe aportes ao longo do tempo (diferente de retorno simples).
    """

    @staticmethod
    def calculate(
        portfolio_value: pd.Series,
        cash_flows: pd.Series,
    ) -> pd.DataFrame:
        """
        Calcula valor da cota e rentabilidade real.

        Args:
            portfolio_value: Série do valor total da carteira
            cash_flows: Série de aportes (positivo) e resgates (negativo)

        Returns:
            DataFrame com vl_cota, qtd_cotas, retorno
        """
        df = pd.DataFrame({
            "saldo": portfolio_value,
            "aporte": cash_flows.reindex(portfolio_value.index, fill_value=0),
        })

        df["vl_cota"] = 1.0
        df["qtd_cotas"] = 0.0
        df["retorno"] = 0.0

        for i in range(len(df)):
            if i == 0:
                df.iloc[i, df.columns.get_loc("vl_cota")] = 1.0
                df.iloc[i, df.columns.get_loc("qtd_cotas")] = df.iloc[i]["saldo"]
            else:
                aporte = df.iloc[i]["aporte"]
                if aporte != 0:
                    prev_cota = df.iloc[i - 1]["vl_cota"]
                    prev_qtd = df.iloc[i - 1]["qtd_cotas"]
                    new_qtd = prev_qtd + (aporte / prev_cota)
                    df.iloc[i, df.columns.get_loc("qtd_cotas")] = new_qtd
                else:
                    df.iloc[i, df.columns.get_loc("qtd_cotas")] = df.iloc[i - 1]["qtd_cotas"]

                qtd = df.iloc[i]["qtd_cotas"]
                df.iloc[i, df.columns.get_loc("vl_cota")] = (
                    df.iloc[i]["saldo"] / qtd if qtd > 0 else 1.0
                )

                prev_cota = df.iloc[i - 1]["vl_cota"]
                cur_cota = df.iloc[i]["vl_cota"]
                df.iloc[i, df.columns.get_loc("retorno")] = (
                    (cur_cota / prev_cota) - 1 if prev_cota > 0 else 0
                )

        return df
