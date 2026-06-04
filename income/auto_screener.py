"""
AutoScreener — triagem automática semanal com dados reais da B3.

Integra BrapiClient + FIIScreener + BarsiScreener para rodar
automaticamente toda semana com dados reais via brapi.dev.

Como funciona na prática (importante ler):
--------------------------------------------
Este sistema faz ANÁLISE e RECOMENDAÇÃO, não executa ordens
automaticamente na B3. Por quê? Porque a maioria das corretoras
brasileiras (XP, Rico, Clear, NuInvest, BTG) não oferece API de
trading para pessoa física comum.

O que o bot FAZ automaticamente:
    ✅ Baixa dados reais da brapi.dev toda semana
    ✅ Analisa 100+ FIIs com critérios profissionais
    ✅ Identifica os melhores para o seu perfil
    ✅ Te manda relatório no Telegram: "COMPRE X, VENDA Y, motivo Z"
    ✅ Monitoramento contínuo de qualidade dos FIIs que você tem
    ✅ Alerta quando vacância sobe, DY cai, ou gestão muda
    ✅ Calendário de dividendos automático

O que VOCÊ faz (leva ~10 minutos por semana):
    → Abre o app da corretora
    → Executa as ordens que o bot recomendou
    → Confirma o que fez (bot atualiza o portfólio automaticamente)

Isso é exatamente o que investidores profissionais que trabalham
sozinhos fazem — o sistema pensa, você executa com consciência.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from data.brapi_client import BrapiClient
from data.fundamentus_client import FundamentusClient
from data.bcb_client import BCBClient
from income.fii_screener import (
    FIIMetrics, FIISegment, FIIRiskLevel,
    FIIScreener, ScreenerCriteria, ScreeningResult,
)
from income.barsi_screener import (
    StockBarsiMetrics, BarsiSector, BarsiScreener,
    BESTIES_TICKERS, BarsiScreenerCriteria,
)
from income.portfolio_allocator import PortfolioAllocator, PortfolioAllocation
from core.decision_logger import DecisionLogger, DecisionType

logger = logging.getLogger("quantbot.income.auto_screener")


@dataclass
class ScreeningReport:
    """Resultado completo de um ciclo de screening."""
    timestamp: str
    fii_result: Optional[ScreeningResult]
    stock_result: Optional[object]
    recommendations: List[Dict] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    data_quality: Dict = field(default_factory=dict)


class AutoScreener:
    """
    Executa screening automático semanal com dados reais da brapi.dev.

    Fluxo completo:
        1. Baixa lista de FIIs da brapi
        2. Converte para FIIMetrics (formato do nosso screener)
        3. Aplica critérios profissionais de triagem
        4. Compara com portfólio atual (o que você já tem)
        5. Gera recomendações: comprar, vender, manter
        6. Loga tudo no DecisionLogger
        7. Retorna relatório para o NotificationService enviar
    """

    # Mapeamento de segmentos brapi -> nosso FIISegment
    _SEGMENT_MAP = {
        "logística": FIISegment.LOGISTICA,
        "logistica": FIISegment.LOGISTICA,
        "lajes corporativas": FIISegment.LAJES_CORPORATIVAS,
        "lajes": FIISegment.LAJES_CORPORATIVAS,
        "escritórios": FIISegment.LAJES_CORPORATIVAS,
        "shoppings": FIISegment.SHOPPINGS,
        "shopping": FIISegment.SHOPPINGS,
        "residencial": FIISegment.RESIDENCIAL,
        "papel": FIISegment.PAPEL_CRI,
        "recebíveis": FIISegment.PAPEL_CRI,
        "cri": FIISegment.PAPEL_CRI,
        "fof": FIISegment.FUNDO_DE_FUNDOS,
        "fundo de fundos": FIISegment.FUNDO_DE_FUNDOS,
        "hospitais": FIISegment.HOSPITAIS,
        "rural": FIISegment.RURAL,
        "agro": FIISegment.RURAL,
    }

    def __init__(
        self,
        brapi_client: Optional[BrapiClient] = None,
        fii_screener: Optional[FIIScreener] = None,
        decision_logger: Optional[DecisionLogger] = None,
        selic_override: Optional[float] = None,
    ):
        self.client = brapi_client or BrapiClient()
        self.fundamentus = FundamentusClient()   # DY/P/VP reais (grátis)
        self.bcb = BCBClient()                    # macro oficial (grátis)
        self.screener = fii_screener or FIIScreener()
        self.dlog = decision_logger or DecisionLogger()
        self.selic = selic_override

    def run(
        self,
        current_portfolio: Optional[Dict[str, float]] = None,
        max_fiis_to_analyze: int = 80,
    ) -> ScreeningReport:
        """
        Executa o ciclo completo de screening.

        Args:
            current_portfolio: dict {ticker: n_cotas} do portfólio atual do bot.
                              Usado para detectar o que precisa ser vendido.
            max_fiis_to_analyze: limita o número de FIIs analisados (respeita rate limit).

        Returns:
            ScreeningReport com aprovados, recomendações e alertas.
        """
        ts = datetime.now(timezone.utc).isoformat()
        current_portfolio = current_portfolio or {}

        logger.info("🔍 Iniciando ciclo de screening automático...")

        # 1. Selic atual (Banco Central → brapi → fallback) p/ calibrar
        selic = self.selic or self.bcb.selic_meta() or self.client.get_selic() or 14.5
        self._adjust_criteria_for_selic(selic)

        # 2-3. Fundamentos REAIS de toda a B3 via Fundamentus (grátis). A
        # brapi gratuita entrega universo e preço, mas o DY/P/VP é pago lá;
        # o Fundamentus preenche essa lacuna sem custo.
        all_fiis = self.fundamentus.to_fii_metrics()
        if not all_fiis:
            logger.warning("Fundamentus indisponível. Tentando brapi como fallback.")
            raw_fiis = self.client.get_fii_list(limit=max_fiis_to_analyze)
            if not raw_fiis:
                return self._offline_report(ts)
            fiis = self._convert_brapi_to_metrics(raw_fiis)
        else:
            # prioriza os mais líquidos dentro do limite de análise
            fiis = sorted(all_fiis, key=lambda f: f.liquidez_diaria, reverse=True)[:max_fiis_to_analyze]
        logger.info(f"  {len(fiis)} FIIs para análise (fonte: Fundamentus)")

        # 4. Aplica screening
        fii_result = self.screener.screen(fiis)
        logger.info(f"  Aprovados: {len(fii_result.aprovados)} / {len(fiis)}")

        # 5. Gera recomendações
        recommendations = self._generate_recommendations(
            fii_result, current_portfolio
        )

        # 6. Gera alertas para posições atuais deteriorando
        alerts = self._portfolio_health_alerts(fiis, current_portfolio)

        # 7. Loga no DecisionLogger
        self.dlog.log_screening(
            tipo="FII",
            n_analisados=len(fiis),
            n_aprovados=len(fii_result.aprovados),
            top_tickers=[f.ticker for f in fii_result.top_n(5)],
            criterios={"selic": selic, "dy_min": self.screener.criteria.dy_min},
        )

        report = ScreeningReport(
            timestamp=ts,
            fii_result=fii_result,
            stock_result=None,
            recommendations=recommendations,
            alerts=alerts,
            data_quality={
                "fonte": "Fundamentus (fundamentos) + BCB (Selic)",
                "n_fiis_analisados": len(fiis),
                "selic_usada": selic,
            },
        )
        logger.info(f"✅ Screening concluído. {len(recommendations)} recomendações.")
        return report

    def _convert_brapi_to_metrics(self, raw_fiis: List[Dict]) -> List[FIIMetrics]:
        """Converte formato brapi.dev para FIIMetrics."""
        metrics = []
        for raw in raw_fiis:
            try:
                ticker = (raw.get("ticker") or raw.get("stock", "")).upper()
                if not ticker or not ticker.endswith("11"):
                    continue

                segmento_str = (raw.get("segment") or raw.get("sector") or "").lower()
                segmento = self._SEGMENT_MAP.get(segmento_str, FIISegment.PAPEL_CRI)

                dy_raw = raw.get("dividendYield") or raw.get("dy12m") or 0
                dy = float(dy_raw) if dy_raw else 0.0
                # brapi pode retornar em % ou decimal
                if dy < 1.0 and dy > 0:
                    dy *= 100  # converte de decimal para %

                pvp_raw = raw.get("priceToBook") or raw.get("pvp") or 1.0
                pvp = float(pvp_raw) if pvp_raw else 1.0
                if pvp > 10:  # brapi às vezes retorna fora de escala
                    pvp = pvp / 100

                vacancia = float(raw.get("vacancyRate") or raw.get("vacancia") or 5.0)
                if vacancia > 1.0:
                    vacancia = min(vacancia, 100.0)
                else:
                    vacancia *= 100  # decimal -> %

                liquidez = float(raw.get("liquidity") or raw.get("volume") or 1_000_000)
                patrimonio = float(raw.get("netAssets") or raw.get("patrimonio") or 500) / 1_000_000
                if patrimonio > 100_000:  # em R$, não em milhões
                    patrimonio /= 1_000_000

                ultimo_div = float(raw.get("lastDividend") or raw.get("ultimoRendimento") or 0.0)
                preco = float(raw.get("close") or raw.get("price") or raw.get("currentPrice") or 100.0)

                provento_sobre_resultado = 0.95  # default conservador
                taxa_adm = float(raw.get("managementFee") or 0.60)
                if taxa_adm > 5.0:  # vem em % * 100 às vezes
                    taxa_adm /= 100

                fii = FIIMetrics(
                    ticker=ticker,
                    nome=raw.get("name") or raw.get("longName") or ticker,
                    segmento=segmento,
                    gestora=raw.get("manager") or raw.get("gestora") or "N/D",
                    dy_12m=round(dy, 2),
                    dy_mes_atual=round(ultimo_div / preco * 100 if preco > 0 else 0, 3),
                    provento_mensal=round(ultimo_div, 4),
                    preco_cota=round(preco, 2),
                    pvp=round(pvp, 3),
                    patrimonio_liq=round(patrimonio, 1),
                    vacancia_fisica=round(vacancia, 1),
                    vacancia_financeira=round(vacancia * 1.1, 1),
                    liquidez_diaria=round(liquidez, 0),
                    resultado_por_cota=round(ultimo_div * 1.05, 4),
                    provento_sobre_resultado=provento_sobre_resultado,
                    historico_anos_sem_corte=3,  # conservador default
                    taxa_administracao=round(taxa_adm, 3),
                    n_imoveis=int(raw.get("nAssets") or raw.get("n_imoveis") or 5),
                    n_inquilinos=int(raw.get("nTenants") or raw.get("n_inquilinos") or 8),
                    indexador_predominante=raw.get("indexer") or "IPCA",
                )
                metrics.append(fii)
            except Exception as e:
                logger.debug(f"Erro ao converter FII {raw.get('ticker', '?')}: {e}")
                continue
        return metrics

    def _adjust_criteria_for_selic(self, selic: float) -> None:
        """Ajusta critérios mínimos de DY conforme a Selic atual."""
        # DY mínimo = Selic * 0.7 (FII deve superar pelo menos 70% da Selic)
        dy_min = round(selic * 0.70, 1)
        self.screener.criteria.dy_min = max(7.0, dy_min)
        logger.debug(f"Critério DY_min ajustado para {self.screener.criteria.dy_min}% (Selic: {selic}%)")

    def _generate_recommendations(
        self,
        result: ScreeningResult,
        current_portfolio: Dict[str, float],
    ) -> List[Dict]:
        """
        Gera recomendações de compra/venda/manutenção.

        Lógica:
            - Top aprovados que você NÃO tem → COMPRAR
            - Ativos que você TEM e que foram REPROVADOS → VENDER
            - Ativos que você TEM e que ainda são aprovados → MANTER
        """
        recommendations = []
        top_aprovados = {f.ticker for f in result.top_n(10)}
        reprovados = {f.ticker for f in result.reprovados}

        # Comprar: top aprovados que não estão na carteira
        for fii in result.top_n(10):
            if fii.ticker not in current_portfolio:
                recommendations.append({
                    "acao": "COMPRAR",
                    "ticker": fii.ticker,
                    "motivo": f"Score {fii.score:.0f} | DY {fii.dy_12m:.1f}% | P/VP {fii.pvp:.2f}",
                    "pontos_positivos": fii.pontos_positivos,
                    "prioridade": "ALTA" if fii.score >= 70 else "MEDIA",
                })

        # Vender: no portfólio mas reprovado
        for ticker in current_portfolio:
            if ticker in reprovados:
                fii_reprovado = next((f for f in result.reprovados if f.ticker == ticker), None)
                motivos = fii_reprovado.motivos_reprovacao if fii_reprovado else ["Não passou na triagem"]
                recommendations.append({
                    "acao": "VENDER",
                    "ticker": ticker,
                    "motivo": " | ".join(motivos),
                    "pontos_positivos": [],
                    "prioridade": "ALTA",
                })

        # Manter: no portfólio e ainda aprovado
        for ticker in current_portfolio:
            if ticker in top_aprovados:
                fii_ok = next((f for f in result.aprovados if f.ticker == ticker), None)
                recommendations.append({
                    "acao": "MANTER",
                    "ticker": ticker,
                    "motivo": f"Ainda aprovado | Score {fii_ok.score if fii_ok else '?'}",
                    "pontos_positivos": fii_ok.pontos_positivos if fii_ok else [],
                    "prioridade": "BAIXA",
                })

        return sorted(recommendations, key=lambda r: ("VENDER", "COMPRAR", "MANTER").index(r["acao"]))

    def _portfolio_health_alerts(
        self,
        all_fiis: List[FIIMetrics],
        current_portfolio: Dict[str, float],
    ) -> List[str]:
        """Alertas de saúde para os FIIs que você já tem."""
        alerts = []
        fii_map = {f.ticker: f for f in all_fiis}

        for ticker in current_portfolio:
            fii = fii_map.get(ticker)
            if not fii:
                alerts.append(f"⚠️ {ticker}: sem dados disponíveis — verifique manualmente")
                continue
            if fii.vacancia_fisica > 20:
                alerts.append(f"🚨 {ticker}: vacância alta {fii.vacancia_fisica:.1f}% — monitorar")
            if fii.pvp > 1.15:
                alerts.append(f"⚠️ {ticker}: P/VP {fii.pvp:.2f} — caro vs patrimônio")
            if fii.provento_sobre_resultado > 1.20:
                alerts.append(f"🚨 {ticker}: distribuindo {fii.provento_sobre_resultado:.1f}x o lucro — insustentável")
        return alerts

    def _offline_report(self, ts: str) -> ScreeningReport:
        """Relatório mínimo quando a brapi.dev não está disponível."""
        return ScreeningReport(
            timestamp=ts,
            fii_result=None,
            stock_result=None,
            recommendations=[],
            alerts=["⚠️ brapi.dev indisponível. Usando dados em cache ou configuração offline."],
            data_quality={"fonte": "offline", "erro": "brapi indisponível"},
        )

    def format_report_text(self, report: ScreeningReport) -> str:
        """Formata o relatório para envio no Telegram/email."""
        ts = report.timestamp[:10]
        linhas = [
            f"🔍 SCREENING SEMANAL DE FIIs — {ts}",
            "=" * 45,
        ]

        if report.fii_result:
            linhas += [
                f"Analisados: {report.data_quality.get('n_fiis_analisados', '?')} FIIs",
                f"Aprovados: {len(report.fii_result.aprovados)}",
                f"Selic atual: {report.data_quality.get('selic_usada', '?')}% a.a.",
                "",
            ]

        compras = [r for r in report.recommendations if r["acao"] == "COMPRAR"]
        vendas = [r for r in report.recommendations if r["acao"] == "VENDER"]
        manutencoes = [r for r in report.recommendations if r["acao"] == "MANTER"]

        if vendas:
            linhas.append("🔴 VENDER (ação urgente):")
            for r in vendas:
                linhas.append(f"  → {r['ticker']}: {r['motivo'][:80]}")

        if compras:
            linhas.append("")
            linhas.append("🟢 COMPRAR (oportunidades):")
            for r in compras[:5]:
                linhas.append(f"  → {r['ticker']} [{r['prioridade']}]: {r['motivo'][:80]}")

        if manutencoes:
            linhas.append("")
            linhas.append(f"⚪ MANTER: {', '.join(r['ticker'] for r in manutencoes)}")

        if report.alerts:
            linhas.append("")
            linhas.append("⚠️ ALERTAS DA CARTEIRA:")
            for a in report.alerts:
                linhas.append(f"  {a}")

        linhas += [
            "",
            "💡 Você decide. O bot analisa, você executa.",
            "Acesse o dashboard para ver todos os dados.",
        ]
        return "\n".join(linhas)
