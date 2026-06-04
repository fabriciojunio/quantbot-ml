"""
Barsi Stock Screener — triagem de ações pelo método Barsi.

O método Barsi em 7 princípios
--------------------------------
1. Invista em empresas PERENES — setores que existirão daqui a 30 anos.
   Energia elétrica, bancos, seguros, saneamento, telecom.

2. Foco em DIVIDENDOS, não em valorização — o preço oscila, a renda persiste.
   "As altas e baixas do mercado pouco importam. A atenção está na renda."

3. Compre em CRISES — os melhores preços aparecem no pânico. Barsi comprou
   muito em 2020 (COVID), 2016 (impeachment), 2008 (crise financeira).

4. REINVISTA TUDO — os dividendos compram mais ações, que pagam mais dividendos.
   A partir do 8º mês, os próprios dividendos financiam novos aportes.

5. HORIZONTE de 20-30 anos — não é trading, é construção de patrimônio.

6. Evite empresas CÍCLICAS — commodities, empresas de crescimento, startups.
   "Não investimos em moda, investimos em necessidade."

7. SETORES BESTIES (Louise Barsi, 2024-2026):
   B - Bancos
   E - Energia
   S - Seguros
   T - Telecomunicações
   I - Infraestrutura (saneamento, rodovias)

Critérios quantitativos de seleção Barsi:
    - Payout histórico > 50% do lucro (empresa distribui consistentemente)
    - Histórico de pagamentos sem interrupção por 5+ anos
    - Dividend yield > 5% a.a. (competitivo com renda fixa longo prazo)
    - Geração de caixa operacional positiva e crescente
    - Dividend Yield sobre o Preço Bazin (preço justo onde DY = 6%)
    - Empresa não endividada excessivamente (dívida/ebitda < 3x)
    - Setor perene (BESTIES obrigatório para o método puro)

Referências:
    - Expert XP (2026). Luiz Barsi: como o Warren Buffett brasileiro fez fortuna.
    - CNN Brasil (Dez 2024). Louise Barsi: setores BESTIES para 2025.
    - Blog Daycoval (2026). Empresas que pagam dividendos mensais.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("quantbot.income.barsi_screener")


class BarsiSector(Enum):
    """Setores BESTIES validados pelo método Barsi."""
    BANCO = "B — Bancos"
    ENERGIA = "E — Energia"
    SEGURO = "S — Seguros"
    TELECOM = "T — Telecomunicações"
    INFRAESTRUTURA = "I — Infraestrutura"
    OUTRO = "Outro (não-BESTIES)"


# Mapeamento de tickers conhecidos para setor Barsi (atualizado jun/2026)
BESTIES_TICKERS: Dict[str, BarsiSector] = {
    # Bancos
    "ITUB4": BarsiSector.BANCO, "ITUB3": BarsiSector.BANCO,
    "BBDC4": BarsiSector.BANCO, "BBDC3": BarsiSector.BANCO,
    "BBAS3": BarsiSector.BANCO, "SANB11": BarsiSector.BANCO,
    "BEES4": BarsiSector.BANCO, "BEES3": BarsiSector.BANCO,
    "BRSR6": BarsiSector.BANCO,
    # Energia
    "EGIE3": BarsiSector.ENERGIA, "CPFE3": BarsiSector.ENERGIA,
    "ENGI11": BarsiSector.ENERGIA, "TAEE11": BarsiSector.ENERGIA,
    "TAEE3": BarsiSector.ENERGIA, "TAEE4": BarsiSector.ENERGIA,
    "ENBR3": BarsiSector.ENERGIA, "EQTL3": BarsiSector.ENERGIA,
    "ALUP11": BarsiSector.ENERGIA, "TRPL4": BarsiSector.ENERGIA,
    "ISAE4": BarsiSector.ENERGIA,
    # Seguros
    "BBSE3": BarsiSector.SEGURO, "IRBR3": BarsiSector.SEGURO,
    "PSSA3": BarsiSector.SEGURO,
    # Telecom
    "VIVT3": BarsiSector.TELECOM, "TIMS3": BarsiSector.TELECOM,
    "OIBR3": BarsiSector.TELECOM,
    # Infraestrutura (saneamento, rodovias, portos)
    "SAPR11": BarsiSector.INFRAESTRUTURA, "SAPR4": BarsiSector.INFRAESTRUTURA,
    "CSMG3": BarsiSector.INFRAESTRUTURA, "SBSP3": BarsiSector.INFRAESTRUTURA,
    "ECOR3": BarsiSector.INFRAESTRUTURA, "CCRO3": BarsiSector.INFRAESTRUTURA,
}


@dataclass
class StockBarsiMetrics:
    """Métricas de uma ação para análise pelo método Barsi."""
    ticker: str
    nome: str
    setor: BarsiSector

    # Dividendos
    dy_12m: float                    # Dividend yield últimos 12 meses (%)
    dy_medio_5a: float               # DY médio dos últimos 5 anos (%)
    payout_ratio: float              # % do lucro distribuído como dividendo
    historico_pagamentos_anos: int   # Anos consecutivos sem interrupção
    frequencia_pagamento: str        # "mensal", "trimestral", "semestral", "anual"
    crescimento_dividendo_5a: float  # CAGR do dividendo nos últimos 5 anos (%)

    # Fundamentos
    preco_atual: float               # R$
    lpa: float                       # Lucro por ação (R$)
    p_l: float                       # Preço / Lucro
    divida_ebitda: float            # Alavancagem
    roe: float                       # Return on Equity (%)
    crescimento_lucro_5a: float      # CAGR do lucro 5 anos (%)

    # Preço justo Bazin
    # Bazin: preço justo = dividendo anual / 0.06 (onde DY = 6%)
    # Barsi adapta para 8% em contexto de Selic alta
    dy_referencia_bazin: float = 8.0  # % — ajustado para Selic 14.5%

    # Preenchidos pela análise
    aprovado: bool = False
    score: float = 0.0
    preco_teto_bazin: float = 0.0
    desconto_sobre_teto: float = 0.0  # % — negativo = caro
    motivos_reprovacao: List[str] = field(default_factory=list)
    pontos_positivos: List[str] = field(default_factory=list)

    # Sentimento das notícias (opcional, entra só como desempate).
    # Preenchido por BarsiScreener.aplicar_sentimento_noticias().
    sentimento_noticias: Optional[str] = None  # "positivo"|"negativo"|"neutro"
    sentimento_score: float = 0.5              # 0..1 (confiança)
    score_sem_sentimento: Optional[float] = None  # score antes do desempate

    @property
    def dividendo_anual(self) -> float:
        """Dividendo anual estimado por ação (R$)."""
        return self.preco_atual * (self.dy_12m / 100)


@dataclass
class BarsiScreenerCriteria:
    """Critérios Barsi ajustáveis."""
    apenas_besties: bool = True           # Exige setor BESTIES
    dy_min: float = 5.0                   # % — mínimo para ser relevante
    payout_min: float = 40.0             # % — empresa que distribui pouco
    historico_min_anos: int = 3          # anos consecutivos de pagamento
    divida_ebitda_max: float = 4.0       # alavancagem máxima
    roe_min: float = 10.0                # % — eficiência mínima
    preco_abaixo_teto: bool = True       # exige preço abaixo do teto Bazin
    dy_referencia_bazin: float = 8.0    # % — ajustado pelo ciclo de juros


class BarsiScreener:
    """
    Aplica os critérios do método Barsi a uma lista de ações.

    Diferente do FII Screener, este foca no longo prazo: não importa
    se o preço sobe ou desce amanhã. Importa se a empresa vai continuar
    pagando dividendos crescentes por décadas.
    """

    def __init__(self, criteria: Optional[BarsiScreenerCriteria] = None):
        self.criteria = criteria or BarsiScreenerCriteria()

    def screen(self, acoes: List[StockBarsiMetrics]) -> "BarsiScreeningResult":
        for acao in acoes:
            self._analyze(acao)
        aprovadas = [a for a in acoes if a.aprovado]
        reprovadas = [a for a in acoes if not a.aprovado]
        aprovadas.sort(key=lambda a: a.score, reverse=True)
        return BarsiScreeningResult(
            total=len(acoes),
            aprovadas=aprovadas,
            reprovadas=reprovadas,
        )

    def aplicar_sentimento_noticias(
        self,
        result: "BarsiScreeningResult",
        sentimento_por_ticker: Dict[str, Tuple[str, float]],
    ) -> "BarsiScreeningResult":
        """
        Usa o sentimento das notícias como DESEMPATE entre as ações já
        aprovadas — nunca como critério de aprovação.

        Filosofia Barsi: "as altas e baixas do mercado pouco importam".
        A manchete não decide a compra; o fundamento decide. Por isso o
        sentimento só ajusta o score em ±2 pontos (proporcional à
        confiança) para reordenar empates, e a aprovação não muda.

        Args:
            result: resultado de screen() — só os aprovados são afetados.
            sentimento_por_ticker: {ticker: (label, score)} onde label é
                "positivo"|"negativo"|"neutro" e score é 0..1. Costuma vir
                de data.sentiment.get_symbol_sentiment() sobre as notícias.

        Returns:
            O mesmo result, com aprovados reordenados pelo score ajustado.
        """
        for acao in result.aprovadas:
            info = sentimento_por_ticker.get(acao.ticker)
            if not info:
                continue
            label, conf = info
            conf = max(0.0, min(1.0, float(conf)))
            acao.sentimento_noticias = label
            acao.sentimento_score = conf
            acao.score_sem_sentimento = acao.score

            # Desempate limitado: no máximo ±2 pontos, e só na parte da
            # confiança acima de 0.5 (0.5 = neutro/indeciso => sem efeito).
            forca = max(0.0, (conf - 0.5) * 2)
            if label == "positivo":
                acao.score = round(acao.score + 2.0 * forca, 1)
                acao.pontos_positivos.append(
                    f"Notícias recentes com sentimento positivo ({conf:.0%})"
                )
            elif label == "negativo":
                acao.score = round(acao.score - 2.0 * forca, 1)
                acao.pontos_positivos.append(
                    f"Atenção: notícias recentes com sentimento negativo ({conf:.0%})"
                )

        result.aprovadas.sort(key=lambda a: a.score, reverse=True)
        return result

    def _analyze(self, acao: StockBarsiMetrics) -> None:
        acao.motivos_reprovacao = []
        acao.pontos_positivos = []

        # Preço teto Bazin
        acao.preco_teto_bazin = round(
            acao.dividendo_anual / (acao.dy_referencia_bazin / 100), 2
        )
        acao.desconto_sobre_teto = round(
            (acao.preco_teto_bazin - acao.preco_atual) / acao.preco_teto_bazin * 100, 1
        )

        # Verificações de reprovação
        c = self.criteria

        if c.apenas_besties and acao.setor == BarsiSector.OUTRO:
            acao.motivos_reprovacao.append("Fora dos setores BESTIES")

        if acao.dy_12m < c.dy_min:
            acao.motivos_reprovacao.append(
                f"DY {acao.dy_12m:.1f}% abaixo do mínimo ({c.dy_min:.0f}%)"
            )

        if acao.payout_ratio < c.payout_min:
            acao.motivos_reprovacao.append(
                f"Payout {acao.payout_ratio:.0f}% — empresa retém muito lucro"
            )

        if acao.historico_pagamentos_anos < c.historico_min_anos:
            acao.motivos_reprovacao.append(
                f"Histórico de {acao.historico_pagamentos_anos} anos — "
                f"mínimo {c.historico_min_anos} anos sem interrupção"
            )

        if acao.divida_ebitda > c.divida_ebitda_max:
            acao.motivos_reprovacao.append(
                f"Dívida/EBITDA {acao.divida_ebitda:.1f}x — endividamento alto"
            )

        if acao.roe < c.roe_min:
            acao.motivos_reprovacao.append(
                f"ROE {acao.roe:.1f}% — retorno sobre patrimônio fraco"
            )

        if c.preco_abaixo_teto and acao.desconto_sobre_teto < 0:
            acao.motivos_reprovacao.append(
                f"Preço R${acao.preco_atual:.2f} acima do teto Bazin "
                f"R${acao.preco_teto_bazin:.2f}"
            )

        acao.aprovado = len(acao.motivos_reprovacao) == 0

        # Score
        score = 0.0
        score += min(30, acao.dy_12m * 2.5)
        score += min(15, acao.historico_pagamentos_anos * 2)
        score += max(0, min(15, acao.desconto_sobre_teto * 0.5))
        score += min(10, acao.crescimento_dividendo_5a)
        score += min(10, (acao.roe - 10) * 0.5)
        score += min(10, max(0, (c.divida_ebitda_max - acao.divida_ebitda) * 2))
        score += 10 if acao.setor != BarsiSector.OUTRO else 0
        acao.score = round(score, 1)

        # Pontos positivos
        if acao.historico_pagamentos_anos >= 10:
            acao.pontos_positivos.append(
                f"{acao.historico_pagamentos_anos} anos pagando sem interrupção"
            )
        if acao.desconto_sobre_teto > 15:
            acao.pontos_positivos.append(
                f"{acao.desconto_sobre_teto:.0f}% abaixo do preço teto Bazin"
            )
        if acao.crescimento_dividendo_5a > 5:
            acao.pontos_positivos.append(
                f"Dividendo crescendo {acao.crescimento_dividendo_5a:.1f}% a.a."
            )
        if acao.frequencia_pagamento == "mensal":
            acao.pontos_positivos.append("Pagamento MENSAL — fluxo de caixa constante")
        if acao.setor != BarsiSector.OUTRO:
            acao.pontos_positivos.append(f"Setor BESTIES: {acao.setor.value}")


@dataclass
class BarsiScreeningResult:
    """Resultado da triagem Barsi."""
    total: int
    aprovadas: List[StockBarsiMetrics]
    reprovadas: List[StockBarsiMetrics]

    def summary(self) -> str:
        linhas = [
            "🏦 TRIAGEM MÉTODO BARSI",
            "=" * 45,
            f"Total analisadas: {self.total}",
            f"Aprovadas: {len(self.aprovadas)} | Reprovadas: {len(self.reprovadas)}",
            "",
            "✅ APROVADAS (por score Barsi):",
        ]
        for i, a in enumerate(self.aprovadas[:10], 1):
            teto = f"R${a.preco_teto_bazin:.2f}"
            desc = f"{a.desconto_sobre_teto:+.0f}% teto"
            linhas.append(
                f"  {i:2}. {a.ticker:8} | Score {a.score:5.1f} | "
                f"DY {a.dy_12m:.1f}% | Teto {teto} ({desc}) | "
                f"{a.setor.value[:12]}"
            )
        return "\n".join(linhas)
