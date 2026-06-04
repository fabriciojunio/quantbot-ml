"""
FII Screener — Triagem de Fundos Imobiliários com critérios profissionais.

Por que um screener é necessário
----------------------------------
Existem mais de 500 FIIs listados na B3. A maioria é inadequada: gestão
fraca, taxa excessiva, dividendo insustentável baseado em ganho de capital
pontual que vai secar em 12 meses, concentração em imóveis problemáticos.

Este screener aplica os critérios que analistas profissionais usam para
filtrar os ~30 fundos que merecem atenção real dos mais de 500 listados.

Critérios de análise (baseados em Kinea, XP Research, BTG Pactual FII, 2026):

1. DIVIDEND YIELD (DY 12m) — acima da Selic não é suficiente; precisa ser
   sustentável. DY muito alto pode ser armadilha (ganho de capital pontual).

2. P/VP (Preço / Valor Patrimonial) — abaixo de 1 significa comprar o ativo
   com desconto. Acima de 1.2 indica prêmio excessivo.

3. VACÂNCIA FÍSICA — percentual da área não alugada. Acima de 15% é sinal
   de problema estrutural para FIIs de tijolo.

4. LIQUIDEZ DIÁRIA — você precisa conseguir entrar e sair. Volume < R$500k/dia
   cria risco de liquidez para contas maiores.

5. SUSTENTABILIDADE — o provento distribuído deve estar coberto pelo resultado
   operacional, não por ganhos de capital ou amortização de patrimônio.

6. PATRIMÔNIO LÍQUIDO — fundos muito pequenos têm custo fixo alto relativo
   e menor poder de barganha com inquilinos.

7. GESTORA — gestoras consolidadas têm histórico verificável e alinhamento
   de interesses.

Referências:
    - Genial Analisa (2026). Carteira Recomendada de FIIs — Jan/Mai 2026.
    - Renova Invest (2026). Melhores FIIs para 2026.
    - brapi.dev (2026). Guia Completo de Análise de FIIs.
    - Investing.com BR (2026). Critérios para selecionar FIIs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("quantbot.income.fii_screener")


class FIISegment(Enum):
    """Segmentos de FIIs com características distintas de risco/retorno."""
    LOGISTICA = "logística"           # Galpões. Vacância estrutural baixa.
    LAJES_CORPORATIVAS = "lajes"      # Escritórios. Mais volatilidade de vacância.
    SHOPPINGS = "shoppings"           # Cíclico. Depende do varejo e consumo.
    RESIDENCIAL = "residencial"       # Emergente no Brasil. Menor liquidez.
    PAPEL_CRI = "papel_cri"           # CRIs indexados a IPCA/CDI. Sensível a juros.
    FUNDO_DE_FUNDOS = "fof"           # Diversificação via gestão profissional.
    HOSPITAIS = "hospitais"           # Contratos longos. Previsível.
    RURAL = "rural"                   # CRAs. Exposição ao agronegócio.


class FIIRiskLevel(Enum):
    BAIXO = "baixo"       # Tijolo prime, contratos longos, gestora top
    MEDIO = "médio"       # Boa qualidade, alguma concentração de risco
    ALTO = "alto"         # Crédito, desenvolvimento, concentração elevada


@dataclass
class FIIMetrics:
    """Métricas completas de um FII para análise."""
    ticker: str
    nome: str
    segmento: FIISegment
    gestora: str

    # Métricas de retorno
    dy_12m: float           # Dividend yield 12 meses (%)
    dy_mes_atual: float     # Provento do mês / cotação atual (%)
    provento_mensal: float  # R$ por cota no último mês

    # Métricas de valor
    preco_cota: float       # R$
    pvp: float              # Preço / Valor Patrimonial
    patrimonio_liq: float   # R$ milhões

    # Métricas de qualidade operacional
    vacancia_fisica: float  # % da área sem inquilino
    vacancia_financeira: float  # % da receita perdida
    liquidez_diaria: float  # Volume médio diário em R$

    # Sustentabilidade
    resultado_por_cota: float   # R$ — lucro operacional / nº cotas
    provento_sobre_resultado: float  # provento / resultado (>1 = insustentável)
    historico_anos_sem_corte: int    # nº de anos sem cortar dividendo

    # Dados de contexto
    taxa_administracao: float  # % a.a.
    n_imoveis: int
    n_inquilinos: int
    indexador_predominante: str  # "IPCA", "CDI", "IGP-M", "FIXO"

    nivel_risco: FIIRiskLevel = FIIRiskLevel.MEDIO
    aprovado: bool = False
    score: float = 0.0
    motivos_reprovacao: List[str] = field(default_factory=list)
    pontos_positivos: List[str] = field(default_factory=list)


# ── Limites de qualidade (calibrados para Selic 14.5% / jun 2026) ────────────

@dataclass
class ScreenerCriteria:
    """
    Critérios ajustáveis de triagem.

    Os defaults refletem o cenário de jun/2026 com Selic a 14.5%.
    Quando a Selic cair, o DY mínimo pode ser reduzido proportionalmente.
    """
    dy_min: float = 8.0            # % ao ano — mínimo para competir com renda fixa
    dy_max: float = 20.0           # % ao ano — acima disso, suspeito de insustentável
    pvp_max: float = 1.20          # Máximo P/VP aceito (pagando prêmio alto)
    pvp_min: float = 0.50          # Abaixo disso, sinal de problema grave
    vacancia_max: float = 15.0     # % — acima, risco operacional alto
    liquidez_min: float = 500_000  # R$/dia — liquidez mínima
    patrimonio_min: float = 200    # R$ milhões — fundos muito pequenos
    taxa_adm_max: float = 1.5      # % a.a. — acima corrói muito o retorno
    prov_sobre_resultado_max: float = 1.10  # até 10% acima do lucro ok
    historico_min: int = 2         # anos sem cortar dividendo


class FIIScreener:
    """
    Aplica critérios de qualidade a uma lista de FIIs e produz ranking.

    Uso:
        screener = FIIScreener()
        resultado = screener.screen([fii1, fii2, ...])
        aprovados = resultado.aprovados
    """

    def __init__(self, criteria: Optional[ScreenerCriteria] = None):
        self.criteria = criteria or ScreenerCriteria()

    def screen(self, fiis: List[FIIMetrics]) -> "ScreeningResult":
        """
        Triagem completa da lista de FIIs.

        Para cada FII: calcula score, identifica problemas e pontos positivos,
        aprova/reprova, e retorna ranking dos aprovados.
        """
        for fii in fiis:
            self._analyze(fii)

        aprovados = [f for f in fiis if f.aprovado]
        reprovados = [f for f in fiis if not f.aprovado]

        aprovados.sort(key=lambda f: f.score, reverse=True)

        return ScreeningResult(
            total_analisados=len(fiis),
            aprovados=aprovados,
            reprovados=reprovados,
            criteria=self.criteria,
        )

    def _analyze(self, fii: FIIMetrics) -> None:
        """Analisa um FII individualmente e popula aprovado/score/motivos."""
        fii.motivos_reprovacao = []
        fii.pontos_positivos = []
        score = 0.0

        # ── Verificações de reprovação direta ────────────────────────────────

        if fii.dy_12m < self.criteria.dy_min:
            fii.motivos_reprovacao.append(
                f"DY {fii.dy_12m:.1f}% abaixo do mínimo ({self.criteria.dy_min:.0f}%)"
            )
        if fii.dy_12m > self.criteria.dy_max:
            fii.motivos_reprovacao.append(
                f"DY {fii.dy_12m:.1f}% suspeito — pode ser insustentável"
            )
        if fii.pvp > self.criteria.pvp_max:
            fii.motivos_reprovacao.append(
                f"P/VP {fii.pvp:.2f} — pagando prêmio excessivo"
            )
        if fii.pvp < self.criteria.pvp_min:
            fii.motivos_reprovacao.append(
                f"P/VP {fii.pvp:.2f} — possível problema estrutural grave"
            )
        if fii.vacancia_fisica > self.criteria.vacancia_max:
            fii.motivos_reprovacao.append(
                f"Vacância {fii.vacancia_fisica:.1f}% — risco operacional alto"
            )
        if fii.liquidez_diaria < self.criteria.liquidez_min:
            fii.motivos_reprovacao.append(
                f"Liquidez R${fii.liquidez_diaria/1e6:.2f}M/dia — abaixo do mínimo"
            )
        if fii.patrimonio_liq < self.criteria.patrimonio_min:
            fii.motivos_reprovacao.append(
                f"Patrimônio R${fii.patrimonio_liq:.0f}M — fundo muito pequeno"
            )
        if fii.taxa_administracao > self.criteria.taxa_adm_max:
            fii.motivos_reprovacao.append(
                f"Taxa de adm {fii.taxa_administracao:.2f}% a.a. — corrói retorno"
            )
        if fii.provento_sobre_resultado > self.criteria.prov_sobre_resultado_max:
            fii.motivos_reprovacao.append(
                f"Distribuição {fii.provento_sobre_resultado:.2f}x o resultado — insustentável"
            )

        fii.aprovado = len(fii.motivos_reprovacao) == 0

        # ── Score de qualidade (0 a 100) ─────────────────────────────────────

        # DY: ideal entre 10-14% com Selic 14.5%
        dy_ideal = 12.0
        score += max(0, 30 - abs(fii.dy_12m - dy_ideal) * 3)

        # P/VP: abaixo de 1 é bom, 0.85-0.95 é ótimo
        pvp_score = 20 - abs(fii.pvp - 0.90) * 25
        score += max(0, pvp_score)

        # Vacância: quanto menor, melhor
        score += max(0, 20 - fii.vacancia_fisica * 1.5)

        # Sustentabilidade: provento coberto pelo resultado
        sust = 1 - max(0, fii.provento_sobre_resultado - 1.0) * 5
        score += max(0, 15 * sust)

        # Histórico sem corte
        score += min(10, fii.historico_anos_sem_corte * 2)

        # Diversificação (mais imóveis/inquilinos = melhor)
        score += min(5, fii.n_inquilinos * 0.5)

        fii.score = round(score, 1)

        # ── Pontos positivos ─────────────────────────────────────────────────

        if fii.pvp < 0.90:
            fii.pontos_positivos.append(f"Desconto patrimonial de {(1-fii.pvp)*100:.0f}%")
        if fii.vacancia_fisica < 5:
            fii.pontos_positivos.append(f"Vacância excelente: {fii.vacancia_fisica:.1f}%")
        if fii.historico_anos_sem_corte >= 5:
            fii.pontos_positivos.append(f"{fii.historico_anos_sem_corte} anos sem cortar dividendo")
        if fii.provento_sobre_resultado < 0.95:
            fii.pontos_positivos.append("Distribuição conservadora — espaço para crescer")
        if fii.n_imoveis >= 10:
            fii.pontos_positivos.append(f"Portfólio diversificado: {fii.n_imoveis} imóveis")

        gestoras_top = {"kinea", "xp", "btg", "cshg", "vinci", "rb capital",
                        "bb vbi", "barigui", "more", "bresco"}
        if any(g in fii.gestora.lower() for g in gestoras_top):
            fii.pontos_positivos.append(f"Gestora consolidada: {fii.gestora}")


@dataclass
class ScreeningResult:
    """Resultado da triagem com ranking e estatísticas."""
    total_analisados: int
    aprovados: List[FIIMetrics]
    reprovados: List[FIIMetrics]
    criteria: ScreenerCriteria

    def top_n(self, n: int = 10) -> List[FIIMetrics]:
        return self.aprovados[:n]

    def summary(self) -> str:
        linhas = [
            f"📊 RESULTADO DA TRIAGEM DE FIIs",
            f"{'='*45}",
            f"Total analisados: {self.total_analisados}",
            f"Aprovados: {len(self.aprovados)} ({len(self.aprovados)/max(1,self.total_analisados)*100:.0f}%)",
            f"Reprovados: {len(self.reprovados)}",
            "",
            "🏆 TOP 10 APROVADOS (por score de qualidade):",
        ]
        for i, fii in enumerate(self.top_n(10), 1):
            linhas.append(
                f"  {i:2}. {fii.ticker:8} | Score {fii.score:5.1f} | "
                f"DY {fii.dy_12m:5.1f}% | P/VP {fii.pvp:.2f} | "
                f"Vac {fii.vacancia_fisica:4.1f}% | {fii.segmento.value}"
            )
        return "\n".join(linhas)

    def to_dict_list(self) -> List[Dict]:
        return [
            {
                "ticker": f.ticker,
                "nome": f.nome,
                "segmento": f.segmento.value,
                "gestora": f.gestora,
                "dy_12m": f.dy_12m,
                "pvp": f.pvp,
                "vacancia": f.vacancia_fisica,
                "provento_mensal": f.provento_mensal,
                "score": f.score,
                "aprovado": f.aprovado,
                "pontos_positivos": f.pontos_positivos,
                "motivos_reprovacao": f.motivos_reprovacao,
            }
            for f in self.aprovados + self.reprovados
        ]
