"""
Portfolio Allocator — divide o capital entre os "baldes" de investimento.

Filosofia de dois baldes
--------------------------
O sistema opera com dois mundos separados:

    BALDE 1 — VOCÊ CONTROLA (Conservador / Barsi):
    ├── Renda fixa / Tesouro SELIC (emergência + liquidez)
    ├── Ações BESTIES (ITUB4, BBAS3, EGIE3, TAEE11, SAPR11, BBSE3...)
    └── Buy & hold puro — você decide, bot apenas monitora

    BALDE 2 — BOT CONTROLA (Renda Passiva Ativa):
    ├── FIIs diversificados (renda mensal isenta de IR)
    ├── Ações de maior DY fora do núcleo Barsi
    └── Reserva de oportunidade (comprar na queda)

Por que separar?
    - Você tem controle total sobre o que não quer que o bot toque
    - O bot opera APENAS no balde 2, com limites rígidos
    - Se o bot perder, o balde 1 continua intacto
    - Psicologicamente mais fácil: você sabe exatamente o que está em risco

Alocação sugerida (baseada em perfil conservador com R$1.000 iniciais):
    Com R$1.000: 70% Balde 1 (você) / 30% Balde 2 (bot)
    Com R$5.000: 60% Balde 1 / 40% Balde 2
    Com R$20.000+: 50% Balde 1 / 50% Balde 2

    O bot NUNCA aloca mais do que o limite do Balde 2. É hard-coded.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger("quantbot.income.portfolio_allocator")


@dataclass
class AllocationBucket:
    """Um balde de alocação."""
    nome: str
    descricao: str
    pct_total: float                    # % do capital total neste balde
    quem_controla: str                  # "voce" ou "bot"
    ativos_alvo: List[str] = field(default_factory=list)
    valor_alocado: float = 0.0
    valor_atual: float = 0.0
    renda_mensal: float = 0.0           # renda passiva mensal deste balde

    @property
    def retorno_pct(self) -> float:
        if self.valor_alocado == 0:
            return 0.0
        return (self.valor_atual - self.valor_alocado) / self.valor_alocado * 100


@dataclass
class PortfolioAllocation:
    """
    Alocação completa de portfólio entre os dois baldes.

    Percentuais dinâmicos baseados no capital total disponível.
    Quanto maior o capital, mais o bot pode gerenciar proporcionalmente.
    """
    capital_total: float
    perfil: str                         # "conservador", "moderado", "arrojado"
    baldes: List[AllocationBucket] = field(default_factory=list)
    meta_renda_mensal: float = 0.0

    @property
    def capital_bot(self) -> float:
        return sum(b.valor_alocado for b in self.baldes if b.quem_controla == "bot")

    @property
    def capital_usuario(self) -> float:
        return sum(b.valor_alocado for b in self.baldes if b.quem_controla == "voce")

    @property
    def renda_total_mensal(self) -> float:
        return sum(b.renda_mensal for b in self.baldes)

    @property
    def pct_bot(self) -> float:
        return self.capital_bot / self.capital_total * 100 if self.capital_total > 0 else 0

    def resumo_texto(self) -> str:
        linhas = [
            f"💰 ALOCAÇÃO DE PORTFÓLIO — {self.perfil.upper()}",
            f"{'='*50}",
            f"Capital total: R${self.capital_total:,.2f}",
            f"Renda mensal estimada: R${self.renda_total_mensal:,.2f}",
            "",
            "📊 DISTRIBUIÇÃO:",
        ]
        for b in self.baldes:
            emoji = "🤖" if b.quem_controla == "bot" else "👤"
            linhas.append(
                f"  {emoji} {b.nome}: {b.pct_total:.0f}% = "
                f"R${b.valor_alocado:,.2f} ({b.quem_controla})"
            )
            if b.renda_mensal > 0:
                linhas.append(f"      Renda mensal: R${b.renda_mensal:,.2f}")
        return "\n".join(linhas)


class PortfolioAllocator:
    """
    Calcula e mantém a alocação ideal do portfólio.

    A alocação é ajustada conforme o capital cresce:
        < R$5.000: conservador (bot tem 25-30%)
        R$5-20k: moderado (bot tem 35-40%)
        > R$20k: arrojado permitido (bot pode ter até 50%)

    O usuário sempre tem pelo menos 50% sob seu próprio controle.
    """

    # Estrutura de baldes por capital total
    _PROFILES = {
        "conservador": {
            "threshold": 5_000,
            "baldes": [
                {
                    "nome": "Reserva de Emergência",
                    "descricao": "Tesouro SELIC ou CDB liquidez diária",
                    "pct": 30.0,
                    "quem": "voce",
                    "ativos": ["TESOURO SELIC", "CDB", "LCI"],
                    "dy_estimado": 14.5,
                },
                {
                    "nome": "Ações BESTIES (Barsi)",
                    "descricao": "Buy & hold long-term — você compra e segura",
                    "pct": 40.0,
                    "quem": "voce",
                    "ativos": ["ITUB4", "BBAS3", "EGIE3", "TAEE11", "BBSE3"],
                    "dy_estimado": 7.0,
                },
                {
                    "nome": "FIIs (bot gerencia)",
                    "descricao": "Renda mensal isenta — bot monitora e rebalanceia",
                    "pct": 20.0,
                    "quem": "bot",
                    "ativos": ["HGLG11", "XPLG11", "KNCR11", "MXRF11", "PVBI11"],
                    "dy_estimado": 11.5,
                },
                {
                    "nome": "Ações DY alto (bot)",
                    "descricao": "Ações pagadoras de alto DY não-BESTIES",
                    "pct": 7.0,
                    "quem": "bot",
                    "ativos": ["BEES4", "ALOS3", "CXSE3"],
                    "dy_estimado": 9.0,
                },
                {
                    "nome": "Caixa Oportunidade (bot)",
                    "descricao": "Reserva para comprar em quedas",
                    "pct": 3.0,
                    "quem": "bot",
                    "ativos": [],
                    "dy_estimado": 14.5,
                },
            ],
        },
        "moderado": {
            "threshold": 20_000,
            "baldes": [
                {
                    "nome": "Reserva de Emergência",
                    "descricao": "Tesouro SELIC ou CDB",
                    "pct": 20.0,
                    "quem": "voce",
                    "ativos": ["TESOURO SELIC", "CDB"],
                    "dy_estimado": 14.5,
                },
                {
                    "nome": "Ações BESTIES (Barsi)",
                    "descricao": "Buy & hold — você controla",
                    "pct": 35.0,
                    "quem": "voce",
                    "ativos": ["ITUB4", "BBAS3", "EGIE3", "TAEE11", "BBSE3", "SAPR11", "VIVT3"],
                    "dy_estimado": 7.5,
                },
                {
                    "nome": "FIIs diversificados (bot)",
                    "descricao": "50% papel + 50% tijolo — gestão automática",
                    "pct": 30.0,
                    "quem": "bot",
                    "ativos": ["HGLG11", "XPLG11", "KNCR11", "PVBI11", "XPML11", "HGRU11"],
                    "dy_estimado": 11.8,
                },
                {
                    "nome": "Ações DY alto (bot)",
                    "descricao": "Pagadoras mensais e ações BESTIES de maior yield",
                    "pct": 10.0,
                    "quem": "bot",
                    "ativos": ["BEES4", "ALOS3", "CXSE3", "BRSR6"],
                    "dy_estimado": 10.0,
                },
                {
                    "nome": "Caixa Oportunidade (bot)",
                    "descricao": "Reserva para aproveitar quedas",
                    "pct": 5.0,
                    "quem": "bot",
                    "ativos": [],
                    "dy_estimado": 14.5,
                },
            ],
        },
        "arrojado": {
            "threshold": float("inf"),
            "baldes": [
                {
                    "nome": "Reserva de Emergência",
                    "descricao": "Tesouro SELIC",
                    "pct": 10.0,
                    "quem": "voce",
                    "ativos": ["TESOURO SELIC"],
                    "dy_estimado": 14.5,
                },
                {
                    "nome": "Ações BESTIES (Barsi)",
                    "descricao": "Buy & hold — você controla",
                    "pct": 35.0,
                    "quem": "voce",
                    "ativos": ["ITUB4", "BBAS3", "EGIE3", "TAEE11", "BBSE3", "SAPR11", "VIVT3", "ALUP11"],
                    "dy_estimado": 8.0,
                },
                {
                    "nome": "FIIs diversificados (bot)",
                    "descricao": "Carteira ampla com gestão ativa",
                    "pct": 35.0,
                    "quem": "bot",
                    "ativos": ["HGLG11", "XPLG11", "KNCR11", "PVBI11", "XPML11", "HGRU11", "BTLG11", "HCTR11"],
                    "dy_estimado": 12.0,
                },
                {
                    "nome": "Ações DY alto (bot)",
                    "descricao": "Diversificação em pagadoras de alto yield",
                    "pct": 15.0,
                    "quem": "bot",
                    "ativos": ["BEES4", "ALOS3", "CXSE3", "BRSR6", "BBDC4", "ISAE4"],
                    "dy_estimado": 10.5,
                },
                {
                    "nome": "Caixa Oportunidade (bot)",
                    "descricao": "Reserva tática — compra em crises",
                    "pct": 5.0,
                    "quem": "bot",
                    "ativos": [],
                    "dy_estimado": 14.5,
                },
            ],
        },
    }

    def compute(
        self,
        capital_total: float,
        meta_renda_mensal: float = 0.0,
        perfil_override: Optional[str] = None,
    ) -> PortfolioAllocation:
        """
        Calcula a alocação ideal para o capital disponível.

        Args:
            capital_total: capital em R$ para alocar.
            meta_renda_mensal: meta de renda passiva mensal (R$).
            perfil_override: forçar um perfil específico.

        Returns:
            PortfolioAllocation com todos os baldes calculados.
        """
        if perfil_override:
            perfil_nome = perfil_override
        elif capital_total < 5_000:
            perfil_nome = "conservador"
        elif capital_total < 20_000:
            perfil_nome = "moderado"
        else:
            perfil_nome = "arrojado"

        perfil = self._PROFILES[perfil_nome]
        baldes = []

        for b in perfil["baldes"]:
            valor = capital_total * (b["pct"] / 100)
            renda_mensal = valor * (b["dy_estimado"] / 100 / 12)
            baldes.append(AllocationBucket(
                nome=b["nome"],
                descricao=b["descricao"],
                pct_total=b["pct"],
                quem_controla=b["quem"],
                ativos_alvo=b["ativos"],
                valor_alocado=round(valor, 2),
                valor_atual=round(valor, 2),
                renda_mensal=round(renda_mensal, 2),
            ))

        return PortfolioAllocation(
            capital_total=capital_total,
            perfil=perfil_nome,
            baldes=baldes,
            meta_renda_mensal=meta_renda_mensal,
        )

    def save(self, allocation: PortfolioAllocation, path: str = "data/allocation.json") -> None:
        """Persiste a alocação em disco."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "capital_total": allocation.capital_total,
            "perfil": allocation.perfil,
            "meta_renda_mensal": allocation.meta_renda_mensal,
            "baldes": [asdict(b) for b in allocation.baldes],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str = "data/allocation.json") -> Optional[PortfolioAllocation]:
        """Carrega alocação do disco."""
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        baldes = [AllocationBucket(**b) for b in data.get("baldes", [])]
        return PortfolioAllocation(
            capital_total=data["capital_total"],
            perfil=data["perfil"],
            baldes=baldes,
            meta_renda_mensal=data.get("meta_renda_mensal", 0),
        )
