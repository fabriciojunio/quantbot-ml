"""
Backtest com custos reais de transação.

Por que custos reais importam tanto
------------------------------------
A maioria dos backtests ignora custos e chega a resultados lindos que
desaparecem completamente ao vivo. Os custos reais no Brasil incluem:

1. CORRETAGEM — varia por corretora. Muitas cobram R$0 para ações, mas
   derivativos e custos indiretos existem. Modelamos 0.05% conservador.

2. SPREAD BID-ASK — você nunca compra/vende exatamente no "preço de
   fechamento" que o dado histórico mostra. Na B3, o spread médio de ações
   líquidas é ~0.05%; ações menos líquidas podem chegar a 0.3%.

3. SLIPPAGE — quando a ordem é grande em relação ao volume do dia, você
   move o mercado contra si. Modelado como função do tamanho da ordem
   relativo ao volume.

4. IMPOSTO DE RENDA — swing trade acima de R$20k/mês: 15%. Day trade: 20%.
   Abaixo de R$20k em vendas/mês: isento (Lei 11.033/2004).

5. EMOLUMENTOS B3 — taxa da própria bolsa: ~0.0325% sobre o valor negociado.

Referências:
    - B3 (2024). Tabela de tarifas de negociação.
    - Receita Federal (2026). IR sobre renda variável — Lei 11.033/2004.
    - Kissell (2013). The Science of Algorithmic Trading and Portfolio
      Management. Cap. 3 — Transaction Cost Analysis.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class BrazilianCostModel:
    """
    Modelo de custos reais para operações na B3.

    Todos os parâmetros têm defaults conservadores (realistas para varejo).
    Valores podem ser calibrados para sua corretora específica.
    """
    # Corretagem (percentual do valor negociado por lado)
    brokerage_pct: float = 0.0005          # 0.05% por lado (conservador)
    # Spread bid-ask (percentual do preço, pago ao cruzar)
    spread_pct: float = 0.0005             # 0.05% para ações líquidas B3
    # Emolumentos B3 (percentual por lado)
    b3_fee_pct: float = 0.000325           # 0.0325%
    # Slippage base (percentual sobre valor da ordem)
    slippage_base_pct: float = 0.0002      # 0.02% base
    # Fração do volume diário que a ordem representa (afeta slippage)
    order_volume_fraction: float = 0.001   # 0.1% do volume (conta pequena)
    # Alíquota IR swing trade (acima de R$20k/mês em vendas)
    ir_swing_rate: float = 0.15
    # Alíquota IR day trade (sempre)
    ir_day_trade_rate: float = 0.20
    # Limite mensal de isenção de IR (swing trade, R$)
    ir_exemption_monthly: float = 20_000.0
    # Capital inicial para cálculos de isenção
    is_day_trade: bool = False

    def total_transaction_cost_pct(self) -> float:
        """
        Custo total por operação COMPLETA (entrada + saída), em percentual.

        Retorna o percentual do capital que vai para custos em cada round-trip.
        """
        slippage = self.slippage_base_pct * (1 + self.order_volume_fraction * 10)
        per_side = self.brokerage_pct + self.spread_pct / 2 + self.b3_fee_pct + slippage
        return per_side * 2  # entrada + saída


@dataclass
class Trade:
    """Registro de uma operação executada."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    symbol: str
    direction: int            # 1 = compra, -1 = venda
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    transaction_costs: float
    ir_paid: float
    net_pnl: float


@dataclass
class BacktestResult:
    """Resultado completo do backtest com custos reais."""
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    monthly_returns: pd.Series

    # Métricas
    total_gross_return: float
    total_net_return: float
    total_transaction_costs: float
    total_ir_paid: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    avg_trade_duration_days: float
    profit_factor: float
    cost_drag_pct: float       # quanto os custos reduziram o retorno

    def summary(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "retorno_bruto_pct": round(self.total_gross_return * 100, 2),
            "retorno_liquido_pct": round(self.total_net_return * 100, 2),
            "custos_transacao_pct": round(self.total_transaction_costs * 100, 2),
            "ir_pago_pct": round(self.total_ir_paid * 100, 2),
            "impacto_custos_pct": round(self.cost_drag_pct * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 3),
            "duracao_media_dias": round(self.avg_trade_duration_days, 1),
        }


class RealisticBacktester:
    """
    Engine de backtest com custos reais e cálculo de IR.

    Recebe uma série de sinais (1 = comprar, 0 = ficar de fora) e retornos
    diários, aplica todos os custos e retorna o track record honesto.
    """

    def __init__(
        self,
        cost_model: Optional[BrazilianCostModel] = None,
        initial_capital: float = 1_000.0,
    ):
        self.cost_model = cost_model or BrazilianCostModel()
        self.initial_capital = initial_capital

    def run(
        self,
        signals: pd.Series,
        prices: pd.Series,
        symbol: str = "ATIVO",
    ) -> BacktestResult:
        """
        Executa o backtest completo.

        Args:
            signals: Série de inteiros (1 = em posição, 0 = fora).
                     Alinhada com prices pelo índice.
            prices: Preços de fechamento diários do ativo.
            symbol: Nome do ativo (para logging e relatórios).

        Returns:
            BacktestResult com todas as métricas e custos reais.
        """
        signals, prices = signals.align(prices, join="inner")
        n = len(prices)
        if n < 2:
            raise ValueError("São necessários pelo menos 2 dias de dados.")

        # cash = dinheiro disponível; separate from position value
        cash = self.initial_capital
        equity = np.zeros(n)
        equity[0] = cash
        daily_returns = np.zeros(n)
        trades: List[Trade] = []

        in_position = False
        entry_idx = 0
        entry_price = 0.0
        quantity = 0.0
        monthly_gross_pnl: dict = {}

        cost_pct = self.cost_model.total_transaction_cost_pct()
        half_cost = cost_pct / 2

        for i in range(1, n):
            sig_prev = int(signals.iloc[i - 1])
            price_prev = float(prices.iloc[i - 1])
            price_curr = float(prices.iloc[i])
            date = prices.index[i]

            prev_equity = equity[i - 1]

            # — ENTRADA: sinal muda de 0 → 1 —
            if not in_position and sig_prev == 1:
                entry_cost = cash * half_cost          # custo de corretagem entrada
                spend = cash - entry_cost              # capital disponível para compra
                quantity = spend / price_curr          # nº de ações adquiridas
                entry_price = price_curr
                entry_idx = i
                cash = 0.0                             # todo o caixa foi para a posição
                in_position = True

            # — SAÍDA: sinal muda de 1 → 0 —
            elif in_position and sig_prev == 0:
                exit_price = price_curr
                proceeds = quantity * exit_price
                gross_pnl = quantity * (exit_price - entry_price)
                exit_cost = proceeds * half_cost
                ir = self._compute_ir(gross_pnl, proceeds, date, monthly_gross_pnl)
                net_pnl = gross_pnl - exit_cost - ir
                cash = proceeds - exit_cost - ir

                entry_cost_total = (quantity * entry_price) * half_cost
                trades.append(Trade(
                    entry_date=prices.index[entry_idx],
                    exit_date=date,
                    symbol=symbol,
                    direction=1,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    gross_pnl=gross_pnl,
                    transaction_costs=exit_cost + entry_cost_total,
                    ir_paid=ir,
                    net_pnl=net_pnl,
                ))
                in_position = False
                quantity = 0.0

            # Equity = caixa + valor da posição aberta
            position_value = quantity * price_curr if in_position else 0.0
            equity[i] = cash + position_value

            # Retorno diário baseado em variação do equity
            daily_ret = (equity[i] - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            daily_returns[i] = daily_ret

        # Fecha posição aberta no último bar (mark-to-market final)
        if in_position and quantity > 0:
            exit_price = float(prices.iloc[-1])
            proceeds = quantity * exit_price
            gross_pnl = quantity * (exit_price - entry_price)
            exit_cost = proceeds * half_cost
            ir = self._compute_ir(gross_pnl, proceeds, prices.index[-1], monthly_gross_pnl)
            net_pnl = gross_pnl - exit_cost - ir
            cash = proceeds - exit_cost - ir
            equity[-1] = cash
            entry_cost_total = (quantity * entry_price) * half_cost
            trades.append(Trade(
                entry_date=prices.index[entry_idx],
                exit_date=prices.index[-1],
                symbol=symbol,
                direction=1,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                gross_pnl=gross_pnl,
                transaction_costs=exit_cost + entry_cost_total,
                ir_paid=ir,
                net_pnl=net_pnl,
            ))

        equity_series = pd.Series(equity, index=prices.index)
        returns_series = pd.Series(daily_returns, index=prices.index)
        monthly_ret = (1 + returns_series).resample("ME").prod() - 1

        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            returns=returns_series,
            monthly_returns=monthly_ret,
            total_gross_return=self._total_gross_return(trades),
            total_net_return=(equity[-1] - self.initial_capital) / self.initial_capital,
            total_transaction_costs=sum(t.transaction_costs for t in trades) / self.initial_capital,
            total_ir_paid=sum(t.ir_paid for t in trades) / self.initial_capital,
            sharpe_ratio=self._sharpe(returns_series),
            max_drawdown=self._max_drawdown(equity_series),
            win_rate=self._win_rate(trades),
            n_trades=len(trades),
            avg_trade_duration_days=self._avg_duration(trades),
            profit_factor=self._profit_factor(trades),
            cost_drag_pct=self._cost_drag(trades),
        )

    def _compute_ir(
        self,
        gross_pnl: float,
        sale_value: float,
        date: pd.Timestamp,
        monthly_gross_pnl: dict,
    ) -> float:
        """Calcula IR devido conforme legislação brasileira 2026."""
        if gross_pnl <= 0:
            return 0.0

        month_key = (date.year, date.month)
        monthly_gross_pnl[month_key] = monthly_gross_pnl.get(month_key, 0) + sale_value

        if self.cost_model.is_day_trade:
            return gross_pnl * self.cost_model.ir_day_trade_rate

        # Isenção: vendas totais no mês <= R$20.000
        if monthly_gross_pnl.get(month_key, 0) <= self.cost_model.ir_exemption_monthly:
            return 0.0

        return gross_pnl * self.cost_model.ir_swing_rate

    def _total_gross_return(self, trades: List[Trade]) -> float:
        if not trades:
            return 0.0
        gross = sum(t.gross_pnl for t in trades)
        return gross / self.initial_capital

    def _sharpe(self, returns: pd.Series, periods: int = 252) -> float:
        r = returns.dropna()
        if r.std(ddof=1) < 1e-10:
            return 0.0
        return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))

    def _max_drawdown(self, equity: pd.Series) -> float:
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max
        return float(dd.min())

    def _win_rate(self, trades: List[Trade]) -> float:
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.net_pnl > 0)
        return wins / len(trades)

    def _avg_duration(self, trades: List[Trade]) -> float:
        if not trades:
            return 0.0
        durations = [(t.exit_date - t.entry_date).days for t in trades]
        return float(np.mean(durations))

    def _profit_factor(self, trades: List[Trade]) -> float:
        gross_wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_losses = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        if gross_losses == 0:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    def _cost_drag(self, trades: List[Trade]) -> float:
        if not trades:
            return 0.0
        total_costs = sum(t.transaction_costs + t.ir_paid for t in trades)
        total_gross = sum(t.gross_pnl for t in trades)
        if total_gross == 0:
            return 0.0
        return total_costs / abs(total_gross) if total_gross > 0 else 1.0
