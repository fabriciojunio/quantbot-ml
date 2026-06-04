"""
Projetor de Renda Passiva — quanto tempo para atingir sua meta mensal.

A pergunta que todo investidor Barsi quer responder: "quando vou receber
R$X por mês em dividendos?"

Este módulo calcula:
    1. Renda mensal atual com base na carteira atual
    2. Projeção com aportes mensais + reinvestimento dos dividendos
    3. Quanto tempo para atingir metas específicas (R$500, R$1k, R$3k/mês)
    4. Composição ideal de carteira para maximizar renda mensal

O efeito "bola de neve":
    - Você recebe dividendos todo mês
    - Reinveste comprando mais cotas/ações
    - Mais cotas = mais dividendos no próximo mês
    - Esse ciclo acelera exponencialmente com o tempo

Isso é exatamente o que Barsi faz há 50 anos. A mágica não está no retorno
mensal, está na disciplina de reinvestir por ANOS.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


@dataclass
class AssetPosition:
    """Uma posição na carteira de renda passiva."""
    ticker: str
    nome: str
    tipo: str              # "FII" ou "ACAO"
    valor_atual: float     # R$ total investido
    dy_anual: float        # Dividend yield anual (%)
    pagamento: str         # "mensal", "trimestral", "semestral", "anual"
    isento_ir: bool        # True para FIIs e dividendos de ações PF
    dy_crescimento_anual: float = 0.0  # crescimento esperado do DY (%)

    @property
    def renda_mensal(self) -> float:
        """Renda bruta mensal desta posição."""
        anual = self.valor_atual * (self.dy_anual / 100)
        return anual / 12

    @property
    def renda_mensal_liquida(self) -> float:
        """Renda mensal líquida (após IR onde aplicável)."""
        bruto = self.renda_mensal
        if self.isento_ir:
            return bruto
        # JCP tem 15% retido na fonte; dividendos ações PF isentos
        return bruto * 0.85


@dataclass
class IncomePortfolio:
    """Carteira completa de renda passiva."""
    posicoes: List[AssetPosition] = field(default_factory=list)

    @property
    def valor_total(self) -> float:
        return sum(p.valor_atual for p in self.posicoes)

    @property
    def renda_mensal_total(self) -> float:
        return sum(p.renda_mensal for p in self.posicoes)

    @property
    def renda_mensal_liquida_total(self) -> float:
        return sum(p.renda_mensal_liquida for p in self.posicoes)

    @property
    def dy_medio_carteira(self) -> float:
        if self.valor_total == 0:
            return 0.0
        return sum(p.valor_atual * p.dy_anual for p in self.posicoes) / self.valor_total

    def adicionar(self, posicao: AssetPosition) -> None:
        self.posicoes.append(posicao)

    def resumo(self) -> dict:
        return {
            "valor_total": round(self.valor_total, 2),
            "renda_mensal_bruta": round(self.renda_mensal_total, 2),
            "renda_mensal_liquida": round(self.renda_mensal_liquida_total, 2),
            "dy_medio": round(self.dy_medio_carteira, 2),
            "n_ativos": len(self.posicoes),
            "fiis": sum(1 for p in self.posicoes if p.tipo == "FII"),
            "acoes": sum(1 for p in self.posicoes if p.tipo == "ACAO"),
        }


@dataclass
class ProjectionResult:
    """Resultado de uma projeção de renda passiva."""
    meses_para_meta: Optional[int]         # None se não atingir no período
    anos_para_meta: Optional[float]
    renda_no_final_do_periodo: float       # R$/mês ao final do horizonte
    capital_no_final: float               # R$ total acumulado
    total_aportado: float                 # R$ que você efetivamente colocou
    total_dividendos_recebidos: float     # R$ que os dividendos te deram
    snapshots: List[dict]                 # estado mês a mês (para gráfico)
    meta_atingida: bool


class IncomeProjector:
    """
    Projeta crescimento da renda passiva ao longo do tempo.

    Considera: aporte mensal + reinvestimento dos dividendos + crescimento
    do DY ao longo do tempo.

    Tudo isso compõe o efeito bola de neve que Barsi descreve.
    """

    def project(
        self,
        portfolio: IncomePortfolio,
        aporte_mensal: float,
        meta_renda_mensal: float,
        horizonte_meses: int = 360,    # 30 anos padrão
        dy_anual_medio: float = 10.0,  # % — yield médio dos novos aportes
        crescimento_dy_ao_ano: float = 3.0,  # % — crescimento anual dos proventos
        reinvestir_dividendos: bool = True,
    ) -> ProjectionResult:
        """
        Projeta a renda passiva ao longo do tempo.

        Args:
            portfolio: carteira atual com posições e rendimentos.
            aporte_mensal: quanto você vai colocar todo mês (R$).
            meta_renda_mensal: renda passiva líquida desejada (R$/mês).
            horizonte_meses: período de projeção.
            dy_anual_medio: yield médio esperado para os novos aportes.
            crescimento_dy_ao_ano: crescimento anual dos proventos (%.
                                   Barsi historicamente tem ~5-8% a.a.).
            reinvestir_dividendos: se True, todo provento é reinvestido.

        Returns:
            ProjectionResult com trajetória completa.
        """
        capital = portfolio.valor_total
        dy_atual = max(portfolio.dy_medio_carteira, dy_anual_medio)
        dy_mensal = dy_atual / 100 / 12

        total_aportado = 0.0
        total_dividendos = 0.0
        meses_para_meta = None
        snapshots = []

        crescimento_mensal = (1 + crescimento_dy_ao_ano / 100) ** (1 / 12)
        fator_crescimento_dy = 1.0

        for mes in range(1, horizonte_meses + 1):
            # Dividendo do mês
            renda_bruta = capital * dy_mensal * fator_crescimento_dy
            renda_liquida = renda_bruta  # FIIs isentos como padrão

            # Reinvestimento
            novo_capital = aporte_mensal
            if reinvestir_dividendos:
                novo_capital += renda_liquida

            capital += novo_capital
            total_aportado += aporte_mensal
            total_dividendos += renda_liquida

            # Crescimento do DY ao longo do tempo
            fator_crescimento_dy *= crescimento_mensal

            # Snapshot trimestral para gráficos
            if mes % 3 == 0 or mes == 1:
                snapshots.append({
                    "mes": mes,
                    "ano": round(mes / 12, 1),
                    "capital": round(capital, 2),
                    "renda_mensal": round(renda_liquida, 2),
                    "total_aportado": round(total_aportado, 2),
                    "total_dividendos": round(total_dividendos, 2),
                })

            # Verificar se meta foi atingida
            if meses_para_meta is None and renda_liquida >= meta_renda_mensal:
                meses_para_meta = mes

        renda_final = capital * dy_mensal * fator_crescimento_dy

        return ProjectionResult(
            meses_para_meta=meses_para_meta,
            anos_para_meta=round(meses_para_meta / 12, 1) if meses_para_meta else None,
            renda_no_final_do_periodo=round(renda_final, 2),
            capital_no_final=round(capital, 2),
            total_aportado=round(total_aportado, 2),
            total_dividendos_recebidos=round(total_dividendos, 2),
            snapshots=snapshots,
            meta_atingida=meses_para_meta is not None,
        )

    def multi_meta(
        self,
        portfolio: IncomePortfolio,
        aporte_mensal: float,
        metas: List[float],
        dy_anual_medio: float = 10.0,
        horizonte_anos: int = 30,
    ) -> List[dict]:
        """
        Projeta o tempo para atingir múltiplas metas de renda.

        Retorna uma lista de dicionários, uma por meta, mostrando
        quando e com quanto capital cada meta será atingida.
        """
        resultados = []
        for meta in metas:
            resultado = self.project(
                portfolio=portfolio,
                aporte_mensal=aporte_mensal,
                meta_renda_mensal=meta,
                horizonte_meses=horizonte_anos * 12,
                dy_anual_medio=dy_anual_medio,
            )
            resultados.append({
                "meta_mensal": meta,
                "atingida": resultado.meta_atingida,
                "anos": resultado.anos_para_meta,
                "meses": resultado.meses_para_meta,
                "capital_necessario": round(meta * 12 / (dy_anual_medio / 100), 2),
                "capital_no_momento": round(
                    next((s["capital"] for s in resultado.snapshots
                          if resultado.meses_para_meta and s["mes"] >= resultado.meses_para_meta),
                         resultado.capital_no_final), 2
                ),
            })
        return resultados

    def capital_necessario(
        self,
        meta_renda_mensal: float,
        dy_anual: float = 10.0,
    ) -> float:
        """Capital necessário HOJE para gerar a meta sem aportes adicionais."""
        return (meta_renda_mensal * 12) / (dy_anual / 100)

    def resumo_formatado(
        self,
        portfolio: IncomePortfolio,
        aporte_mensal: float,
        metas: Optional[List[float]] = None,
        dy_anual_medio: float = 10.0,
    ) -> str:
        """Texto formatado com projeções — para usar nos relatórios."""
        metas = metas or [500, 1000, 2000, 3000, 5000, 10000]
        resumo_carteira = portfolio.resumo()
        resultados = self.multi_meta(portfolio, aporte_mensal, metas, dy_anual_medio)

        linhas = [
            "💰 PROJEÇÃO DE RENDA PASSIVA",
            "=" * 45,
            f"Capital atual: R${resumo_carteira['valor_total']:,.2f}",
            f"Renda mensal atual: R${resumo_carteira['renda_mensal_liquida']:,.2f}",
            f"DY médio da carteira: {resumo_carteira['dy_medio']:.1f}% a.a.",
            f"Aporte mensal planejado: R${aporte_mensal:,.2f}",
            "",
            "📈 METAS DE RENDA (com reinvestimento total):",
        ]
        for r in resultados:
            if r["atingida"]:
                linhas.append(
                    f"  R${r['meta_mensal']:,.0f}/mês → {r['anos']:.1f} anos "
                    f"(capital: R${r['capital_necessario']:,.0f})"
                )
            else:
                linhas.append(
                    f"  R${r['meta_mensal']:,.0f}/mês → não atingido em 30 anos "
                    f"com aporte atual"
                )
        linhas += [
            "",
            "⚠️ IMPORTANTE:",
            "  DY futuro não é garantido. Projeção assume DY constante.",
            "  Reinvestimento disciplinado é o fator mais importante.",
            "  Aportes consistentes superam timing de mercado no longo prazo.",
        ]
        return "\n".join(linhas)
