"""
Decision Logger — logs estruturados de cada decisão do bot.

Por que logs detalhados são essenciais
----------------------------------------
Cada decisão do bot deve ser registrada com contexto suficiente para:
    1. Auditoria humana — você pode revisar o que o bot fez e por quê
    2. Aprendizado do bot — analisar padrões de erro e acerto ao longo do tempo
    3. Melhoria contínua — o que funcionou em qual regime de mercado?
    4. Conformidade — trilha auditável para fins fiscais e pessoais
    5. Debugging — quando algo der errado, o contexto está todo aqui

Cada log contém:
    - Timestamp preciso (UTC)
    - Tipo de evento (signal, order, screening, alert, report, error)
    - Contexto completo (preço, indicadores, motivo da decisão)
    - Resultado esperado vs resultado real (preenchido posteriormente)
    - Score de confiança do modelo

Formato: JSON Lines (.jsonl) — uma decisão por linha, fácil de ler e analisar.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("quantbot.core.decision_logger")


class DecisionType(str, Enum):
    # Screening e análise
    FII_SCREENING = "fii_screening"
    STOCK_SCREENING = "stock_screening"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"

    # Sinais e ordens
    SIGNAL_GENERATED = "signal_generated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_CLOSED = "position_closed"

    # Risco
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_VIOLATION = "risk_violation"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"

    # Dividendos e renda
    DIVIDEND_RECEIVED = "dividend_received"
    DIVIDEND_REINVESTED = "dividend_reinvested"
    INCOME_REPORT = "income_report"

    # Sistema
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"
    DATA_FETCH = "data_fetch"
    MODEL_RETRAIN = "model_retrain"
    ERROR = "error"
    ALERT = "alert"


class DecisionOutcome(str, Enum):
    PENDING = "pending"         # ainda não se concretizou
    SUCCESS = "success"         # resultado positivo
    FAILURE = "failure"         # resultado negativo
    CANCELLED = "cancelled"     # cancelado antes de se concretizar
    PARTIAL = "partial"         # resultado parcial


class DecisionLogger:
    """
    Registra todas as decisões do bot em formato JSON Lines estruturado.

    Cada instância grava em um arquivo diário: decisions/YYYY-MM-DD.jsonl
    O arquivo pode ser aberto no Excel, Python ou qualquer ferramenta JSON.

    Uso:
        dlog = DecisionLogger()
        decision_id = dlog.log(
            tipo=DecisionType.SIGNAL_GENERATED,
            ticker="HGLG11",
            contexto={"dy": 11.5, "pvp": 0.92, "motivo": "DY acima da Selic"},
            confianca=0.78,
        )
        # Depois, quando souber o resultado:
        dlog.update_outcome(decision_id, DecisionOutcome.SUCCESS, {"pnl": 12.50})
    """

    def __init__(
        self,
        log_dir: str = "data/decisions",
        also_log_to_console: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.also_log_to_console = also_log_to_console
        self._index: Dict[str, str] = {}  # decision_id -> linha no arquivo

    def log(
        self,
        tipo: DecisionType,
        ticker: Optional[str] = None,
        contexto: Optional[Dict[str, Any]] = None,
        confianca: float = 1.0,
        valor_envolvido: float = 0.0,
        notas: Optional[str] = None,
        regime_mercado: Optional[str] = None,
        modelo_versao: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Registra uma decisão e retorna seu ID único para atualização futura.

        Args:
            tipo: tipo do evento (sinal, ordem, alerta, etc)
            ticker: ativo envolvido (opcional)
            contexto: dict com TODOS os dados relevantes para a decisão
            confianca: score de confiança do modelo (0.0-1.0)
            valor_envolvido: valor em R$ da operação
            notas: texto livre com observações
            regime_mercado: regime detectado (BULL_LOW_VOL, etc)
            modelo_versao: versão/hash do modelo usado
            tags: etiquetas para filtragem

        Returns:
            decision_id (UUID) para poder atualizar o outcome depois
        """
        decision_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)

        entry = {
            "id": decision_id,
            "timestamp": ts.isoformat(),
            "data": ts.strftime("%Y-%m-%d"),
            "hora": ts.strftime("%H:%M:%S"),
            "tipo": tipo.value,
            "ticker": ticker,
            "confianca": round(confianca, 4),
            "valor_envolvido": round(valor_envolvido, 2),
            "regime_mercado": regime_mercado,
            "modelo_versao": modelo_versao,
            "notas": notas,
            "tags": tags or [],
            "contexto": contexto or {},
            "outcome": DecisionOutcome.PENDING.value,
            "outcome_data": None,
            "outcome_timestamp": None,
        }

        self._write(entry)
        self._index[decision_id] = ts.strftime("%Y-%m-%d")

        if self.also_log_to_console:
            ticker_str = f"[{ticker}]" if ticker else ""
            conf_str = f"conf={confianca:.0%}" if confianca < 1.0 else ""
            logger.info(
                f"DECISÃO {tipo.value.upper()} {ticker_str} "
                f"R${valor_envolvido:.2f} {conf_str} id={decision_id[:8]}"
            )

        return decision_id

    def update_outcome(
        self,
        decision_id: str,
        outcome: DecisionOutcome,
        outcome_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Atualiza o resultado de uma decisão após se concretizar.

        Cria uma entrada de "update" linkada ao ID original. Não modifica
        a entrada original (imutabilidade — importante para auditoria).
        """
        if decision_id not in self._index:
            logger.warning(f"Decision ID {decision_id} não encontrado no índice.")
            return False

        ts = datetime.now(timezone.utc)
        update_entry = {
            "id": str(uuid.uuid4()),
            "ref_id": decision_id,
            "timestamp": ts.isoformat(),
            "tipo": "outcome_update",
            "outcome": outcome.value,
            "outcome_data": outcome_data or {},
            "outcome_timestamp": ts.isoformat(),
        }
        self._write(update_entry)

        if self.also_log_to_console:
            logger.info(
                f"OUTCOME {outcome.value.upper()} para {decision_id[:8]}: "
                f"{outcome_data}"
            )
        return True

    def log_dividend(
        self,
        ticker: str,
        valor_por_cota: float,
        n_cotas: float,
        data_pagamento: str,
        reinvestido: bool = False,
        ticker_reinvestimento: Optional[str] = None,
    ) -> str:
        """Registra recebimento de dividendo."""
        total = valor_por_cota * n_cotas
        return self.log(
            tipo=DecisionType.DIVIDEND_RECEIVED,
            ticker=ticker,
            valor_envolvido=total,
            contexto={
                "valor_por_cota": valor_por_cota,
                "n_cotas": n_cotas,
                "total_recebido": round(total, 2),
                "data_pagamento": data_pagamento,
                "reinvestido": reinvestido,
                "ticker_reinvestimento": ticker_reinvestimento,
                "isento_ir": True,  # FIIs para PF
            },
            tags=["dividendo", "renda_passiva"],
        )

    def log_screening(
        self,
        tipo: str,  # "FII" ou "ACAO"
        n_analisados: int,
        n_aprovados: int,
        top_tickers: List[str],
        criterios: Dict[str, Any],
    ) -> str:
        """Registra resultado de um ciclo de screening."""
        return self.log(
            tipo=DecisionType.FII_SCREENING if tipo == "FII" else DecisionType.STOCK_SCREENING,
            contexto={
                "n_analisados": n_analisados,
                "n_aprovados": n_aprovados,
                "aprovados_pct": round(n_aprovados / max(1, n_analisados) * 100, 1),
                "top_tickers": top_tickers[:10],
                "criterios_usados": criterios,
            },
            confianca=1.0,
            tags=["screening", tipo.lower()],
        )

    def get_today_summary(self) -> Dict[str, Any]:
        """Resumo das decisões de hoje para o relatório diário."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entries = self._read_day(today)

        tipos = {}
        total_valor = 0.0
        dividendos = 0.0

        for e in entries:
            t = e.get("tipo", "outro")
            tipos[t] = tipos.get(t, 0) + 1
            total_valor += e.get("valor_envolvido", 0)
            if t == DecisionType.DIVIDEND_RECEIVED.value:
                dividendos += e.get("valor_envolvido", 0)

        return {
            "data": today,
            "total_decisoes": len(entries),
            "por_tipo": tipos,
            "valor_total_movimentado": round(total_valor, 2),
            "dividendos_recebidos": round(dividendos, 2),
        }

    def get_learning_features(self, days: int = 90) -> List[Dict]:
        """
        Extrai features de aprendizado dos últimos N dias.

        Retorna decisões com seu outcome, para análise de o que funcionou.
        Usado para treinar versões futuras do modelo com dados reais.
        """
        features = []
        for day_file in sorted(self.log_dir.glob("*.jsonl"))[-days:]:
            entries = self._read_file(day_file)
            # Agrupa decisões com seus updates
            decisions = {e["id"]: e for e in entries if "ref_id" not in e}
            updates = {e["ref_id"]: e for e in entries if "ref_id" in e}

            for did, dec in decisions.items():
                if did in updates:
                    dec["outcome"] = updates[did].get("outcome", "pending")
                    dec["outcome_data"] = updates[did].get("outcome_data", {})
                features.append(dec)

        return features

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _write(self, entry: Dict) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.log_dir / f"{today}.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.error(f"Falha ao gravar log de decisão: {e}")

    def _read_day(self, day: str) -> List[Dict]:
        return self._read_file(self.log_dir / f"{day}.jsonl")

    def _read_file(self, path: Path) -> List[Dict]:
        if not path.exists():
            return []
        entries = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Falha ao ler log {path}: {e}")
        return entries
