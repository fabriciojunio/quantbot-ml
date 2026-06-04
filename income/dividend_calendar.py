"""
Calendário de Dividendos — rastreia pagamentos e antecipa renda.

O que este módulo faz
----------------------
1. Registra dividendos anunciados e pagos na carteira do bot
2. Calcula a renda mensal real recebida vs projetada
3. Identifica meses sem pagamento de algum ativo
4. Gera alertas: "HGLG11 paga amanhã — R$1.15/cota"
5. Projeta o próximo mês com base no histórico
6. Alimenta os relatórios com dados reais de proventos

Por que isso importa: FIIs pagam mensalmente, mas a DATA COM
(data limite para ter direito ao dividendo) e a DATA DE PAGAMENTO
são diferentes. O investidor precisa estar cotista ANTES da data com.
O sistema monitora automaticamente e alerta com antecedência.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("quantbot.income.dividend_calendar")


@dataclass
class DividendEvent:
    """Um evento de dividendo agendado ou pago."""
    id: str                     # ticker + data_pagamento
    ticker: str
    tipo_ativo: str             # "FII" ou "ACAO"
    valor_por_cota: float       # R$ por cota
    data_com: Optional[str]     # "YYYY-MM-DD" — data limite para ter direito
    data_pagamento: str         # "YYYY-MM-DD" — quando cai na conta
    status: str                 # "agendado", "pago", "cancelado"

    # Calculados a partir da posição
    n_cotas: float = 0.0        # quantas cotas você tem
    total_recebido: float = 0.0 # calculado: valor_por_cota * n_cotas
    isento_ir: bool = True      # FIIs para PF: sempre True

    def __post_init__(self):
        if self.n_cotas > 0:
            self.total_recebido = round(self.valor_por_cota * self.n_cotas, 2)


@dataclass
class MonthlyIncome:
    """Renda passiva de um mês específico."""
    ano: int
    mes: int
    dividendos: List[DividendEvent] = field(default_factory=list)

    @property
    def total_bruto(self) -> float:
        return sum(d.total_recebido for d in self.dividendos if d.status == "pago")

    @property
    def total_previsto(self) -> float:
        return sum(d.total_recebido for d in self.dividendos)

    @property
    def por_ativo(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for d in self.dividendos:
            result[d.ticker] = result.get(d.ticker, 0) + d.total_recebido
        return result

    @property
    def label(self) -> str:
        return f"{self.mes:02d}/{self.ano}"


class DividendCalendar:
    """
    Gerencia o calendário de dividendos da carteira do bot.

    Persiste os dados em disco (data/dividends.json) e carrega
    automaticamente ao iniciar. Permite registrar novos eventos
    (da API brapi.dev), marcar como pagos, e gerar relatórios.
    """

    def __init__(self, data_file: str = "data/dividends.json"):
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self._events: Dict[str, DividendEvent] = {}
        self._load()

    # ── Registro de eventos ────────────────────────────────────────────────

    def add_event(self, event: DividendEvent) -> None:
        """Adiciona ou atualiza um evento de dividendo."""
        self._events[event.id] = event
        self._save()
        logger.debug(f"Dividendo adicionado: {event.ticker} R${event.valor_por_cota} em {event.data_pagamento}")

    def from_brapi(
        self,
        ticker: str,
        tipo_ativo: str,
        brapi_dividends: List[Dict],
        n_cotas: float,
    ) -> List[DividendEvent]:
        """
        Converte dados da brapi.dev em DividendEvents e os registra.

        A brapi retorna: paymentDate, rate (valor por cota), type.
        """
        novos = []
        for d in brapi_dividends:
            payment_date = d.get("paymentDate") or d.get("data_pagamento")
            if not payment_date:
                continue
            valor = float(d.get("rate", d.get("value", d.get("valor_por_cota", 0))) or 0)
            if valor <= 0:
                continue

            ev_id = f"{ticker}_{payment_date}"
            if ev_id in self._events:
                # Atualiza n_cotas se mudou
                self._events[ev_id].n_cotas = n_cotas
                self._events[ev_id].total_recebido = round(valor * n_cotas, 2)
                continue

            event = DividendEvent(
                id=ev_id,
                ticker=ticker.upper(),
                tipo_ativo=tipo_ativo,
                valor_por_cota=round(valor, 4),
                data_com=d.get("date") or d.get("data_com"),
                data_pagamento=payment_date[:10],  # só a data, sem hora
                status="agendado" if payment_date[:10] >= str(date.today()) else "pago",
                n_cotas=n_cotas,
                isento_ir=(tipo_ativo == "FII"),
            )
            event.total_recebido = round(valor * n_cotas, 2)
            self._events[ev_id] = event
            novos.append(event)

        if novos:
            self._save()
        return novos

    def mark_paid(self, event_id: str) -> bool:
        """Marca um evento como pago."""
        if event_id in self._events:
            self._events[event_id].status = "pago"
            self._save()
            return True
        return False

    # ── Consultas ─────────────────────────────────────────────────────────

    def upcoming(self, days_ahead: int = 30) -> List[DividendEvent]:
        """Dividendos agendados para os próximos N dias."""
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        result = []
        for ev in self._events.values():
            if ev.status != "agendado":
                continue
            try:
                pay_date = date.fromisoformat(ev.data_pagamento)
                if today <= pay_date <= cutoff:
                    result.append(ev)
            except ValueError:
                continue
        return sorted(result, key=lambda e: e.data_pagamento)

    def get_monthly_income(self, ano: int, mes: int) -> MonthlyIncome:
        """Renda do mês: pagos + agendados."""
        mi = MonthlyIncome(ano=ano, mes=mes)
        for ev in self._events.values():
            try:
                pay = date.fromisoformat(ev.data_pagamento)
                if pay.year == ano and pay.month == mes:
                    mi.dividendos.append(ev)
            except ValueError:
                continue
        return mi

    def get_ytd_income(self) -> float:
        """Total recebido no ano corrente (pagos)."""
        ano = date.today().year
        total = 0.0
        for ev in self._events.values():
            if ev.status == "pago":
                try:
                    if date.fromisoformat(ev.data_pagamento).year == ano:
                        total += ev.total_recebido
                except ValueError:
                    continue
        return round(total, 2)

    def project_next_month(self) -> Tuple[float, List[Dict]]:
        """
        Projeta a renda do próximo mês com base no histórico.

        Assume que cada ativo pagará o mesmo valor do último mês.
        Retorna: (total_projetado, lista de projeções por ativo)
        """
        today = date.today()
        next_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
        year, month = next_month.year, next_month.month

        # Busca último pagamento de cada ticker
        last_by_ticker: Dict[str, DividendEvent] = {}
        for ev in self._events.values():
            if ev.status in ("pago", "agendado"):
                ticker = ev.ticker
                if ticker not in last_by_ticker:
                    last_by_ticker[ticker] = ev
                else:
                    if ev.data_pagamento > last_by_ticker[ticker].data_pagamento:
                        last_by_ticker[ticker] = ev

        projecoes = []
        total = 0.0
        for ticker, ev in last_by_ticker.items():
            proj = round(ev.valor_por_cota * ev.n_cotas, 2)
            total += proj
            projecoes.append({
                "ticker": ticker,
                "valor_projetado": proj,
                "base": f"último pagamento {ev.data_pagamento}",
            })

        return round(total, 2), sorted(projecoes, key=lambda x: -x["valor_projetado"])

    def alert_messages(self, days_ahead: int = 5) -> List[str]:
        """
        Mensagens de alerta para dividendos iminentes.
        Enviadas pelo NotificationService nos relatórios diários.
        """
        upcoming = self.upcoming(days_ahead)
        if not upcoming:
            return []
        msgs = []
        today = date.today()
        for ev in upcoming:
            pay_date = date.fromisoformat(ev.data_pagamento)
            diff = (pay_date - today).days
            if diff == 0:
                when = "HOJE"
            elif diff == 1:
                when = "AMANHÃ"
            else:
                when = f"em {diff} dias"
            msgs.append(
                f"💰 {ev.ticker}: R${ev.valor_por_cota:.4f}/cota "
                f"({when} — {ev.data_pagamento}) "
                f"= R${ev.total_recebido:.2f} total"
            )
        return msgs

    def summary_text(self) -> str:
        """Texto formatado para relatórios."""
        today = date.today()
        mi = self.get_monthly_income(today.year, today.month)
        proximo, projecoes = self.project_next_month()

        linhas = [
            f"📅 CALENDÁRIO DE DIVIDENDOS",
            f"{'='*40}",
            f"Mês atual ({mi.label}):",
            f"  Previsto: R${mi.total_previsto:,.2f}",
            f"  Recebido: R${mi.total_bruto:,.2f}",
            f"  Acumulado no ano: R${self.get_ytd_income():,.2f}",
            "",
            f"Próximo mês (projeção): R${proximo:,.2f}",
        ]
        for p in projecoes[:5]:
            linhas.append(f"  {p['ticker']}: R${p['valor_projetado']:.2f}")

        alerts = self.alert_messages(7)
        if alerts:
            linhas.append("")
            linhas.append("⏰ PRÓXIMOS PAGAMENTOS:")
            linhas.extend(f"  {a}" for a in alerts)

        return "\n".join(linhas)

    # ── Persistência ──────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            data = {eid: asdict(ev) for eid, ev in self._events.items()}
            self.data_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Falha ao salvar calendário de dividendos: {e}")

    def _load(self) -> None:
        if not self.data_file.exists():
            return
        try:
            data = json.loads(self.data_file.read_text(encoding="utf-8"))
            for eid, ev_data in data.items():
                self._events[eid] = DividendEvent(**ev_data)
        except Exception as e:
            logger.warning(f"Falha ao carregar calendário: {e}")
