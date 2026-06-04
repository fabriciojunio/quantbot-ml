"""
Sistema de Relatórios Automatizados — Email + Telegram.

Entrega relatórios estruturados e fáceis de ler:
    - Diário: fim de cada sessão de trading
    - Semanal: toda segunda-feira, resumo da semana anterior
    - Mensal: primeiro dia útil do mês
    - Anual: 2 de janeiro de cada ano

Formato: dados estruturados em texto com emojis para facilitar leitura
no celular. Tabelas simples. Nenhum jargão desnecessário.
"""

import json
import logging
import smtplib
import os
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional
import urllib.request
import urllib.parse

logger = logging.getLogger("quantbot.reports")


class ReportFormatter:
    """Formata os dados de performance em texto legível."""

    def daily_report(
        self,
        date_str: str,
        capital: float,
        daily_pnl: float,
        daily_return_pct: float,
        n_trades: int,
        regime: str,
        drawdown_pct: float,
        signals: List[Dict],
        risk_status: Dict,
    ) -> str:
        emoji = "🟢" if daily_pnl >= 0 else "🔴"
        return f"""
📊 QUANTBOT — RELATÓRIO DIÁRIO
📅 {date_str}
{'='*40}

{emoji} RESULTADO DO DIA
   PnL: R${daily_pnl:+.2f} ({daily_return_pct:+.2f}%)
   Capital: R${capital:.2f}
   Trades: {n_trades}
   Regime de mercado: {regime}

⚠️ RISCO
   Drawdown total: {drawdown_pct:.2f}%
   Kill-switch: {'ATIVO ⛔' if risk_status.get('kill_switch_ativo') else 'OK ✅'}
   Trades hoje: {risk_status.get('trades_hoje', 0)} / {5}

📡 SINAIS GERADOS
{self._format_signals(signals)}

💡 OBSERVAÇÃO
   Este relatório é informativo. Não é recomendação de investimento.
   Dados passados não garantem resultados futuros.
""".strip()

    def weekly_report(
        self,
        week_start: str,
        week_end: str,
        capital_start: float,
        capital_end: float,
        weekly_return_pct: float,
        n_trades: int,
        win_rate_pct: float,
        best_day_pct: float,
        worst_day_pct: float,
        daily_returns: List[float],
    ) -> str:
        emoji = "🟢" if weekly_return_pct >= 0 else "🔴"
        bar = self._mini_chart(daily_returns)
        return f"""
📊 QUANTBOT — RELATÓRIO SEMANAL
📅 {week_start} → {week_end}
{'='*40}

{emoji} DESEMPENHO DA SEMANA
   Retorno: {weekly_return_pct:+.2f}%
   Capital inicial: R${capital_start:.2f}
   Capital final: R${capital_end:.2f}
   Variação: R${capital_end - capital_start:+.2f}

📈 ESTATÍSTICAS
   Trades: {n_trades}
   Win rate: {win_rate_pct:.1f}%
   Melhor dia: {best_day_pct:+.2f}%
   Pior dia: {worst_day_pct:+.2f}%

📉 RETORNOS DIÁRIOS
   {bar}

💡 LEMBRE-SE
   Uma semana não define nada. O que importa é a consistência
   ao longo de meses. Continue monitorando.
""".strip()

    def monthly_report(
        self,
        month_str: str,
        capital_start: float,
        capital_end: float,
        monthly_return_pct: float,
        selic_monthly_pct: float,
        n_trades: int,
        win_rate_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
        ir_pago: float,
        custos_pct: float,
        retorno_bruto_pct: float,
        weekly_returns: List[float],
    ) -> str:
        vs_selic = monthly_return_pct - selic_monthly_pct
        emoji = "🟢" if monthly_return_pct >= 0 else "🔴"
        selic_emoji = "✅" if vs_selic >= 0 else "❌"
        return f"""
📊 QUANTBOT — RELATÓRIO MENSAL
📅 {month_str}
{'='*40}

{emoji} RESULTADO DO MÊS
   Retorno líquido: {monthly_return_pct:+.2f}%
   Retorno bruto: {retorno_bruto_pct:+.2f}%
   Custos (corretagem+spread+IR): -{custos_pct:.2f}%
   IR pago: R${ir_pago:.2f}

💰 PATRIMÔNIO
   Início do mês: R${capital_start:.2f}
   Fim do mês: R${capital_end:.2f}
   Variação: R${capital_end - capital_start:+.2f}

📊 COMPARATIVO
   {selic_emoji} vs Selic mensal: {vs_selic:+.2f}% (Selic: {selic_monthly_pct:.2f}%/mês)

🎯 QUALIDADE
   Trades: {n_trades}
   Win rate: {win_rate_pct:.1f}%
   Sharpe ratio: {sharpe_ratio:.2f}
   Drawdown máximo: {max_drawdown_pct:.2f}%

⚖️ IMPOSTOS
   Operação tipo: SWING TRADE
   IR pago este mês: R${ir_pago:.2f}
   (Vendas abaixo de R$20k/mês = ISENTO)

🔍 ANÁLISE HONESTA
   {'✅ Mês positivo.' if monthly_return_pct > 0 else '⚠️ Mês negativo — normal em qualquer sistema.'}
   {'✅ Bateu a Selic.' if vs_selic > 0 else '⚠️ Abaixo da Selic — considere o risco assumido.'}
   {'⚠️ Win rate baixo. Revise os sinais.' if win_rate_pct < 40 else ''}

📌 PRÓXIMAS AÇÕES
   Continue o paper trading / monitoramento.
   Não tome decisões com base em 1 mês de dados.
""".strip()

    def annual_report(
        self,
        year: int,
        capital_start: float,
        capital_end: float,
        annual_return_pct: float,
        selic_annual_pct: float,
        cdi_annual_pct: float,
        ibovespa_annual_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
        n_trades: int,
        win_rate_pct: float,
        total_ir_pago: float,
        validation_dsr: Optional[float],
        validation_pbo: Optional[float],
        monthly_returns: List[float],
    ) -> str:
        vs_selic = annual_return_pct - selic_annual_pct
        vs_ibov = annual_return_pct - ibovespa_annual_pct
        bar = self._mini_chart(monthly_returns)
        return f"""
📊 QUANTBOT — RELATÓRIO ANUAL {year}
{'='*40}

💰 RESULTADO DO ANO
   Retorno líquido: {annual_return_pct:+.2f}%
   Capital inicial: R${capital_start:.2f}
   Capital final: R${capital_end:.2f}
   Lucro líquido: R${capital_end - capital_start:+.2f}

📊 COMPARATIVOS
   vs Selic ({selic_annual_pct:.1f}%): {vs_selic:+.2f}% {'✅' if vs_selic > 0 else '❌'}
   vs CDI ({cdi_annual_pct:.1f}%): {annual_return_pct - cdi_annual_pct:+.2f}% {'✅' if annual_return_pct > cdi_annual_pct else '❌'}
   vs Ibovespa ({ibovespa_annual_pct:.1f}%): {vs_ibov:+.2f}% {'✅' if vs_ibov > 0 else '❌'}

🎯 QUALIDADE
   Sharpe ratio: {sharpe_ratio:.2f}  {'✅ bom' if sharpe_ratio > 1.0 else '⚠️ fraco'}
   Drawdown máximo: {max_drawdown_pct:.2f}%
   Trades: {n_trades}
   Win rate: {win_rate_pct:.1f}%

🔬 VALIDAÇÃO ESTATÍSTICA
   DSR (Deflated Sharpe): {f'{validation_dsr:.3f}' if validation_dsr else 'n/d'}
   PBO (risco overfitting): {f'{validation_pbo:.3f}' if validation_pbo else 'n/d'}
   {'✅ Evidência de edge real' if validation_dsr and validation_dsr > 0.90 else '⚠️ Ainda sem evidência estatística suficiente'}

⚖️ IMPOSTOS
   IR total pago: R${total_ir_pago:.2f}
   (Mantenha todos os DARFs para declaração anual)

📈 RETORNOS MENSAIS
   {bar}

🔍 AVALIAÇÃO HONESTA DO ANO
   {'✅ Sistema gerou alpha sobre renda fixa.' if vs_selic > 0 else '⚠️ Sistema não superou a renda fixa. Reavalie se o risco compensa.'}
   1 ano de dados NÃO é suficiente para concluir que o sistema tem edge.
   Continue acumulando dados. Decisões importantes só após 2-3 anos de track record.

💡 META REALISTA
   Alvo profissional: superar Selic + 5-10% com Sharpe > 1.0
   Meta de 25% ao ano consistente: requer vários anos de validação
""".strip()

    def _format_signals(self, signals: List[Dict]) -> str:
        if not signals:
            return "   Nenhum sinal gerado hoje."
        lines = []
        for s in signals[:5]:  # máximo 5
            emoji = "🟢" if s.get("direction") == "COMPRA" else "🔴" if s.get("direction") == "VENDA" else "⚪"
            lines.append(f"   {emoji} {s.get('symbol', '?')}: {s.get('direction', '?')} | Conf: {s.get('confidence', 0)*100:.0f}%")
        return "\n".join(lines)

    def _mini_chart(self, returns: List[float]) -> str:
        if not returns:
            return "sem dados"
        chars = []
        for r in returns:
            if r > 0.02:
                chars.append("▲")
            elif r > 0:
                chars.append("△")
            elif r > -0.02:
                chars.append("▽")
            else:
                chars.append("▼")
        return " ".join(chars)


class NotificationService:
    """
    Envia relatórios via email e/ou Telegram.

    Configuração via variáveis de ambiente (NUNCA hardcode credenciais):
        QUANTBOT_EMAIL_FROM, QUANTBOT_EMAIL_TO, QUANTBOT_EMAIL_PASSWORD
        QUANTBOT_TELEGRAM_TOKEN, QUANTBOT_TELEGRAM_CHAT_ID
    """

    def __init__(self):
        self.formatter = ReportFormatter()
        self._email_cfg = self._load_email_config()
        self._telegram_cfg = self._load_telegram_config()

    def send_daily(self, data: Dict) -> None:
        msg = self.formatter.daily_report(**data)
        subject = f"QuantBot 📊 Diário {data.get('date_str', '')}"
        self._send(subject, msg)

    def send_weekly(self, data: Dict) -> None:
        msg = self.formatter.weekly_report(**data)
        subject = f"QuantBot 📊 Semanal {data.get('week_start', '')} → {data.get('week_end', '')}"
        self._send(subject, msg)

    def send_monthly(self, data: Dict) -> None:
        msg = self.formatter.monthly_report(**data)
        subject = f"QuantBot 📊 Mensal {data.get('month_str', '')}"
        self._send(subject, msg)

    def send_annual(self, data: Dict) -> None:
        msg = self.formatter.annual_report(**data)
        subject = f"QuantBot 📊 Anual {data.get('year', '')}"
        self._send(subject, msg)

    def send_alert(self, title: str, message: str) -> None:
        """Alerta urgente — kill-switch, drawdown, falha de dados."""
        full_msg = f"🚨 ALERTA QUANTBOT\n{title}\n\n{message}"
        self._send(f"🚨 ALERTA: {title}", full_msg)

    def _send(self, subject: str, body: str) -> None:
        """Tenta enviar via Telegram (primário) e email (fallback)."""
        sent = False
        if self._telegram_cfg:
            sent = self._send_telegram(body)
        if self._email_cfg:
            self._send_email(subject, body)
            sent = True
        if not sent:
            logger.warning(
                "Notificação não enviada: configure QUANTBOT_TELEGRAM_TOKEN "
                "ou QUANTBOT_EMAIL_* nas variáveis de ambiente."
            )
            logger.info(f"[RELATÓRIO]\n{body}")

    def _send_telegram(self, message: str) -> bool:
        try:
            token = self._telegram_cfg["token"]
            chat_id = self._telegram_cfg["chat_id"]
            # Divide mensagens longas (Telegram: máx 4096 chars)
            for chunk in self._split_message(message, 4000):
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = urllib.parse.urlencode({
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                }).encode()
                req = urllib.request.Request(url, data=data)
                urllib.request.urlopen(req, timeout=10)
            logger.info("Relatório enviado via Telegram.")
            return True
        except Exception as e:
            logger.error(f"Falha ao enviar Telegram: {e}")
            return False

    def _send_email(self, subject: str, body: str) -> bool:
        try:
            cfg = self._email_cfg
            msg = MIMEMultipart()
            msg["From"] = cfg["from"]
            msg["To"] = cfg["to"]
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))
            with smtplib.SMTP_SSL(cfg.get("smtp_host", "smtp.gmail.com"), 465) as s:
                s.login(cfg["from"], cfg["password"])
                s.sendmail(cfg["from"], cfg["to"], msg.as_string())
            logger.info("Relatório enviado por email.")
            return True
        except Exception as e:
            logger.error(f"Falha ao enviar email: {e}")
            return False

    def _load_email_config(self) -> Optional[Dict]:
        frm = os.getenv("QUANTBOT_EMAIL_FROM")
        to = os.getenv("QUANTBOT_EMAIL_TO")
        pwd = os.getenv("QUANTBOT_EMAIL_PASSWORD")
        if not all([frm, to, pwd]):
            return None
        return {"from": frm, "to": to, "password": pwd,
                "smtp_host": os.getenv("QUANTBOT_EMAIL_SMTP", "smtp.gmail.com")}

    def _load_telegram_config(self) -> Optional[Dict]:
        token = os.getenv("QUANTBOT_TELEGRAM_TOKEN")
        chat_id = os.getenv("QUANTBOT_TELEGRAM_CHAT_ID")
        if not all([token, chat_id]):
            return None
        return {"token": token, "chat_id": chat_id}

    @staticmethod
    def _split_message(message: str, max_len: int) -> List[str]:
        if len(message) <= max_len:
            return [message]
        chunks = []
        while message:
            chunks.append(message[:max_len])
            message = message[max_len:]
        return chunks
