"""
Orchestrator — o cérebro que coordena tudo automaticamente.

Ciclos de execução:
    Diário (todo dia útil às 18h30 — após fechamento B3):
        1. Verifica dividendos pendentes para os próximos 7 dias
        2. Atualiza cotações dos FIIs e ações do portfólio
        3. Verifica alertas de saúde (vacância, PVP, etc)
        4. Envia relatório diário via Telegram/email

    Semanal (toda segunda-feira às 08h00):
        1. Executa screening completo com dados reais da brapi.dev
        2. Gera recomendações de compra/venda
        3. Envia relatório semanal com recomendações

    Mensal (dia 1 de cada mês):
        1. Consolida renda recebida
        2. Compara vs Selic e vs meta de renda
        3. Atualiza projeção da bola de neve
        4. Envia relatório mensal completo

    Anual (2 de janeiro):
        1. Relatório anual completo com todas as métricas
        2. Avaliação de performance vs benchmarks
        3. Planejamento do próximo ano
"""

import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from data.brapi_client import BrapiClient
from income.auto_screener import AutoScreener
from income.portfolio_allocator import PortfolioAllocator, PortfolioAllocation
from income.income_projector import IncomePortfolio, AssetPosition, IncomeProjector
from income.dividend_calendar import DividendCalendar
from core.decision_logger import DecisionLogger, DecisionType
from core.reports import NotificationService, ReportFormatter
from risk.kill_switch import RiskManager, RiskLimits

logger = logging.getLogger("quantbot.core.orchestrator")


class Orchestrator:
    """
    Coordena todos os módulos do QuantBot.

    Em vez de ficar rodando em loop infinito (que consumiria recursos
    e poderia travar), o Orchestrator é chamado periodicamente por um
    agendador externo (cron job, Task Scheduler, GitHub Actions, etc).

    Uso típico (crontab Linux):
        # Ciclo diário — 18:30 em dias úteis
        30 18 * * 1-5 cd /path/to/quantbot && python main_income.py --daily

        # Ciclo semanal — Segunda 8h
        0 8 * * 1 cd /path/to/quantbot && python main_income.py --weekly

        # Ciclo mensal — Dia 1 às 9h
        0 9 1 * * cd /path/to/quantbot && python main_income.py --monthly
    """

    def __init__(
        self,
        portfolio_file: str = "data/portfolio.json",
        allocation_file: str = "data/allocation.json",
        decisions_dir: str = "data/decisions",
        dividends_file: str = "data/dividends.json",
    ):
        self.portfolio_file = Path(portfolio_file)
        self.client = BrapiClient()
        self.screener = AutoScreener(brapi_client=self.client)
        self.allocator = PortfolioAllocator()
        self.projector = IncomeProjector()
        self.dividend_calendar = DividendCalendar(data_file=dividends_file)
        self.dlog = DecisionLogger(log_dir=decisions_dir)
        self.notifier = NotificationService()
        self.formatter = ReportFormatter()
        self.risk_manager = RiskManager()

    # ── Ciclos de execução ────────────────────────────────────────────────────

    def run_daily(self) -> None:
        """Ciclo diário: cotações, dividendos, alertas, relatório."""
        logger.info("📅 Iniciando ciclo DIÁRIO...")
        self.dlog.log(DecisionType.BOT_STARTED, notas="ciclo diário")

        try:
            portfolio = self._load_portfolio()
            allocation = self.allocator.load(str(self.portfolio_file.parent / "allocation.json"))

            # Atualizar dividendos da carteira
            div_alerts = self.dividend_calendar.alert_messages(days_ahead=7)
            div_summary = self.dividend_calendar.summary_text()

            # Status de risco
            risk_status = self.risk_manager.get_status()

            # Renda do mês atual
            today = date.today()
            mi = self.dividend_calendar.get_monthly_income(today.year, today.month)

            # Montar dados do relatório diário
            capital = allocation.capital_total if allocation else 0
            daily_report_data = {
                "date_str": today.strftime("%d/%m/%Y"),
                "capital": capital,
                "daily_pnl": 0.0,   # preenchido pelo tracker quando integrado
                "daily_return_pct": 0.0,
                "n_trades": 0,
                "regime": "N/A",
                "drawdown_pct": risk_status.get("drawdown_total_pct", 0),
                "signals": [],
                "risk_status": risk_status,
            }

            # Mensagem customizada com foco em renda
            msg = self._format_daily_income_report(
                today, mi, div_alerts, risk_status, allocation
            )
            logger.info(f"Relatório diário:\n{msg}")
            self.notifier.send_daily(daily_report_data)

        except Exception as e:
            logger.error(f"Erro no ciclo diário: {e}", exc_info=True)
            self.dlog.log(DecisionType.ERROR, notas=str(e))

    def run_weekly(self) -> None:
        """Ciclo semanal: screening completo + recomendações."""
        logger.info("📊 Iniciando ciclo SEMANAL...")
        self.dlog.log(DecisionType.BOT_STARTED, notas="ciclo semanal")

        try:
            portfolio = self._load_portfolio()
            bot_portfolio = portfolio.get("bot", {})

            # Screening com dados reais
            report = self.screener.run(
                current_portfolio=bot_portfolio,
                max_fiis_to_analyze=80,
            )

            # Formatar e enviar
            msg = self.screener.format_report_text(report)
            logger.info(f"Relatório semanal:\n{msg}")

            # Notificar via Telegram/email
            self.notifier.send_alert("Screening Semanal", msg)

        except Exception as e:
            logger.error(f"Erro no ciclo semanal: {e}", exc_info=True)
            self.dlog.log(DecisionType.ERROR, notas=str(e))

    def run_monthly(self) -> None:
        """Ciclo mensal: consolidação + comparativo vs Selic + projeção."""
        logger.info("📈 Iniciando ciclo MENSAL...")

        try:
            today = date.today()
            prev_month = today.month - 1 if today.month > 1 else 12
            prev_year = today.year if today.month > 1 else today.year - 1

            mi = self.dividend_calendar.get_monthly_income(prev_year, prev_month)
            selic = self.client.get_selic() or 14.5
            selic_mensal = ((1 + selic / 100) ** (1 / 12) - 1) * 100

            allocation = self.allocator.load("data/allocation.json")
            capital_end = allocation.capital_total if allocation else 0

            monthly_data = {
                "month_str": f"{prev_month:02d}/{prev_year}",
                "capital_start": capital_end * 0.99,  # aproximação
                "capital_end": capital_end,
                "monthly_return_pct": mi.total_bruto / max(1, capital_end) * 100,
                "selic_monthly_pct": round(selic_mensal, 2),
                "n_trades": 0,
                "win_rate_pct": 0,
                "sharpe_ratio": 0,
                "max_drawdown_pct": self.risk_manager.get_status().get("drawdown_total_pct", 0),
                "ir_pago": 0.0,
                "custos_pct": 0.0,
                "retorno_bruto_pct": mi.total_bruto / max(1, capital_end) * 100,
                "weekly_returns": [],
            }
            self.notifier.send_monthly(monthly_data)
            logger.info("Relatório mensal enviado.")

        except Exception as e:
            logger.error(f"Erro no ciclo mensal: {e}", exc_info=True)

    def run_setup(
        self,
        capital_total: float,
        meta_renda_mensal: float = 500.0,
        aporte_mensal: float = 200.0,
    ) -> None:
        """
        Configuração inicial do sistema — roda UMA VEZ no primeiro uso.

        Cria os arquivos de configuração, calcula a alocação inicial,
        e mostra as projeções de renda passiva.
        """
        logger.info("⚙️ Setup inicial do QuantBot...")

        # Calcula alocação
        allocation = self.allocator.compute(capital_total, meta_renda_mensal)
        self.allocator.save(allocation)
        logger.info(f"Alocação calculada:\n{allocation.resumo_texto()}")

        # Salva portfólio vazio
        self._save_portfolio({"bot": {}, "user": {}})

        # Projeção de renda
        portfolio_income = IncomePortfolio()
        portfolio_income.adicionar(AssetPosition(
            ticker="CARTEIRA",
            nome="Portfólio inicial",
            tipo="FII",
            valor_atual=allocation.capital_bot,
            dy_anual=11.5,
            pagamento="mensal",
            isento_ir=True,
        ))

        projecao = self.projector.resumo_formatado(
            portfolio_income,
            aporte_mensal=aporte_mensal,
            dy_anual_medio=11.5,
        )
        logger.info(f"Projeção de renda:\n{projecao}")

        print("\n" + "="*50)
        print("✅ QUANTBOT CONFIGURADO COM SUCESSO")
        print("="*50)
        print(allocation.resumo_texto())
        print()
        print(projecao)
        print()
        print("PRÓXIMOS PASSOS:")
        print("1. Configure .env com suas credenciais (Telegram, brapi)")
        print("2. Execute: python main_income.py --weekly")
        print("3. Siga as recomendações do primeiro screening")
        print("4. Confirme as compras com: python main_income.py --confirm TICKER N_COTAS PRECO")
        print("="*50)

    def confirm_trade(self, ticker: str, n_cotas: float, preco: float, acao: str = "COMPRAR") -> None:
        """
        Você confirma uma operação que executou manualmente na corretora.

        Args:
            ticker: ex "HGLG11"
            n_cotas: quantas cotas comprou/vendeu
            preco: preço médio da operação
            acao: "COMPRAR" ou "VENDER"
        """
        portfolio = self._load_portfolio()
        bot_portfolio = portfolio.get("bot", {})

        if acao == "COMPRAR":
            if ticker in bot_portfolio:
                # Média ponderada
                cotas_atuais = bot_portfolio[ticker].get("n_cotas", 0)
                preco_atual = bot_portfolio[ticker].get("preco_medio", preco)
                total_cotas = cotas_atuais + n_cotas
                preco_medio = (cotas_atuais * preco_atual + n_cotas * preco) / total_cotas
                bot_portfolio[ticker] = {"n_cotas": total_cotas, "preco_medio": preco_medio}
            else:
                bot_portfolio[ticker] = {"n_cotas": n_cotas, "preco_medio": preco}
            logger.info(f"✅ COMPRA confirmada: {ticker} {n_cotas} cotas @ R${preco:.2f}")
        elif acao == "VENDER":
            if ticker in bot_portfolio:
                cotas_atuais = bot_portfolio[ticker].get("n_cotas", 0)
                restantes = cotas_atuais - n_cotas
                if restantes <= 0:
                    del bot_portfolio[ticker]
                else:
                    bot_portfolio[ticker]["n_cotas"] = restantes
            logger.info(f"✅ VENDA confirmada: {ticker} {n_cotas} cotas @ R${preco:.2f}")

        portfolio["bot"] = bot_portfolio
        self._save_portfolio(portfolio)

        self.dlog.log(
            DecisionType.ORDER_FILLED,
            ticker=ticker,
            valor_envolvido=n_cotas * preco,
            contexto={"acao": acao, "n_cotas": n_cotas, "preco": preco},
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_portfolio(self) -> Dict:
        if self.portfolio_file.exists():
            import json
            return json.loads(self.portfolio_file.read_text(encoding="utf-8"))
        return {"bot": {}, "user": {}}

    def _save_portfolio(self, portfolio: Dict) -> None:
        import json
        self.portfolio_file.parent.mkdir(parents=True, exist_ok=True)
        self.portfolio_file.write_text(
            json.dumps(portfolio, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _format_daily_income_report(
        self,
        today: date,
        mi,
        div_alerts: list,
        risk_status: Dict,
        allocation,
    ) -> str:
        capital = allocation.capital_total if allocation else 0
        linhas = [
            f"📊 QUANTBOT — RENDA PASSIVA DIÁRIA",
            f"📅 {today.strftime('%d/%m/%Y')}",
            "=" * 40,
            f"Capital total: R${capital:,.2f}",
            f"Renda recebida em {today.strftime('%m/%Y')}: R${mi.total_bruto:,.2f}",
            f"Previsto para o mês: R${mi.total_previsto:,.2f}",
        ]
        if div_alerts:
            linhas.append("")
            linhas.append("⏰ DIVIDENDOS ESTA SEMANA:")
            linhas.extend(f"  {a}" for a in div_alerts[:5])

        linhas += [
            "",
            f"⚠️ Drawdown total: {risk_status.get('drawdown_total_pct', 0):.1f}%",
            f"Kill-switch: {'ATIVO ⛔' if risk_status.get('kill_switch_ativo') else 'OK ✅'}",
        ]
        return "\n".join(linhas)
