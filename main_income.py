"""
QuantBot ML — Ponto de entrada principal.

Como usar:
    python main_income.py --setup 1000 --meta 200 --aporte 100
    python main_income.py --daily
    python main_income.py --weekly
    python main_income.py --monthly
    python main_income.py --status
    python main_income.py --confirm HGLG11 10 115.50 COMPRAR
    python main_income.py --project 1000 200 10

Exemplos práticos:
    # Primeira vez — configurar com R$1.000
    python main_income.py --setup 1000 --meta 200 --aporte 100

    # Ver status atual da carteira
    python main_income.py --status

    # Rodar screening semanal (recomendado: toda segunda-feira)
    python main_income.py --weekly

    # Após executar uma compra na corretora, confirmar para o bot:
    python main_income.py --confirm HGLG11 5 123.50 COMPRAR

    # Ver projeção de renda (capital atual, aporte mensal, anos)
    python main_income.py --project 1000 150 15
"""

import argparse
import logging
import sys
from pathlib import Path

# Garante saída UTF-8 mesmo em consoles Windows (cp1252), evitando crash ao
# imprimir emojis/acentos nos relatórios.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # streams já fixos ou sem suporte
        pass

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent))

# Configura logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Silencia loggers muito verbosos
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib.request").setLevel(logging.WARNING)

logger = logging.getLogger("quantbot.main")


def main():
    parser = argparse.ArgumentParser(
        description="QuantBot ML — Sistema de Renda Passiva com IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--setup", type=float, metavar="CAPITAL",
                        help="Configuração inicial com capital em R$")
    parser.add_argument("--meta", type=float, default=500.0,
                        help="Meta de renda mensal em R$ (padrão: 500)")
    parser.add_argument("--aporte", type=float, default=200.0,
                        help="Aporte mensal planejado em R$ (padrão: 200)")
    parser.add_argument("--daily", action="store_true",
                        help="Executa ciclo diário")
    parser.add_argument("--weekly", action="store_true",
                        help="Executa ciclo semanal (screening)")
    parser.add_argument("--monthly", action="store_true",
                        help="Executa ciclo mensal")
    parser.add_argument("--status", action="store_true",
                        help="Mostra status atual do portfólio")
    parser.add_argument("--project", nargs=3, metavar=("CAPITAL", "APORTE", "ANOS"),
                        help="Projeção de renda: capital aporte anos")
    parser.add_argument("--confirm", nargs=4,
                        metavar=("TICKER", "COTAS", "PRECO", "ACAO"),
                        help="Confirma operação: TICKER N_COTAS PRECO COMPRAR|VENDER")
    parser.add_argument("--killswitch", choices=["on", "off"],
                        help="Ativa ou desativa o kill-switch")

    args = parser.parse_args()

    # Importações lazy para não sobrecarregar em chamadas simples
    from core.orchestrator import Orchestrator
    orch = Orchestrator()

    if args.setup:
        orch.run_setup(
            capital_total=args.setup,
            meta_renda_mensal=args.meta,
            aporte_mensal=args.aporte,
        )

    elif args.daily:
        orch.run_daily()

    elif args.weekly:
        orch.run_weekly()

    elif args.monthly:
        orch.run_monthly()

    elif args.status:
        _show_status(orch)

    elif args.project:
        _show_projection(
            capital=float(args.project[0]),
            aporte=float(args.project[1]),
            anos=int(args.project[2]),
        )

    elif args.confirm:
        ticker, cotas, preco, acao = args.confirm
        orch.confirm_trade(
            ticker=ticker.upper(),
            n_cotas=float(cotas),
            preco=float(preco),
            acao=acao.upper(),
        )

    elif args.killswitch:
        from risk.kill_switch import RiskManager
        rm = RiskManager()
        if args.killswitch == "on":
            rm.activate_kill_switch("ativado manualmente")
            print("🛑 Kill-switch ATIVADO. Bot parado.")
        else:
            rm.deactivate_kill_switch()
            print("✅ Kill-switch DESATIVADO. Bot reativado.")

    else:
        parser.print_help()


def _show_status(orch) -> None:
    import json
    from income.dividend_calendar import DividendCalendar
    from risk.kill_switch import RiskManager
    from datetime import date

    portfolio = orch._load_portfolio()
    risk = RiskManager().get_status()
    cal = DividendCalendar()
    today = date.today()
    mi = cal.get_monthly_income(today.year, today.month)

    print("\n" + "="*50)
    print("📊 STATUS DO QUANTBOT")
    print("="*50)

    bot_p = portfolio.get("bot", {})
    if bot_p:
        print(f"\n🤖 PORTFÓLIO DO BOT ({len(bot_p)} ativos):")
        for ticker, info in bot_p.items():
            cotas = info.get("n_cotas", 0)
            preco = info.get("preco_medio", 0)
            print(f"  {ticker}: {cotas:.0f} cotas | PM R${preco:.2f}")
    else:
        print("\n🤖 Portfólio do bot: vazio (execute --weekly para recomendações)")

    print(f"\n💰 RENDA PASSIVA - {today.strftime('%m/%Y')}:")
    print(f"  Recebida: R${mi.total_bruto:.2f}")
    print(f"  Prevista: R${mi.total_previsto:.2f}")

    print(f"\n⚠️ RISCO:")
    print(f"  Kill-switch: {'ATIVO ⛔' if risk['kill_switch_ativo'] else 'OK ✅'}")
    print(f"  Drawdown total: {risk['drawdown_total_pct']:.1f}%")
    print(f"  Drawdown diário: {risk['drawdown_diario_pct']:.1f}%")

    alerts = cal.alert_messages(7)
    if alerts:
        print("\n⏰ DIVIDENDOS NOS PRÓXIMOS 7 DIAS:")
        for a in alerts:
            print(f"  {a}")

    print("="*50)


def _show_projection(capital: float, aporte: float, anos: int) -> None:
    from income.income_projector import IncomePortfolio, AssetPosition, IncomeProjector

    portfolio = IncomePortfolio()
    portfolio.adicionar(AssetPosition(
        ticker="CARTEIRA",
        nome="Portfólio atual",
        tipo="FII",
        valor_atual=capital,
        dy_anual=11.5,
        pagamento="mensal",
        isento_ir=True,
    ))

    projector = IncomeProjector()
    print("\n" + projector.resumo_formatado(portfolio, aporte_mensal=aporte))


if __name__ == "__main__":
    main()
