"""
Linha de comando do motor de cripto.

    python -m cripto.cli precos                    preço vivo e spread real
    python -m cripto.cli custo                     a conta do alvo de centavos
    python -m cripto.cli backtest BTCUSDT --tempo 1h --dias 400
    python -m cripto.cli validar BTCUSDT --tempo 1h --dias 400
    python -m cripto.cli estudo --dias 400         bateria completa
    python -m cripto.cli papel --simbolos BTCUSDT,ETHUSDT
    python -m cripto.cli sinal BTCUSDT             score da barra mais recente
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import carrego as mod_carrego
from . import estudo as mod_estudo
from . import mercado, papel, relatorio, sinais, walkforward
from .backtest import ConfigBacktest, executar
from .custos import ModeloCusto
from .exchanges import cliente_padrao
from .sinais import ConfigSinal


def _log(verboso: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verboso else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def _saida(dados) -> None:
    print(json.dumps(dados, indent=2, ensure_ascii=False, default=str))


def cmd_precos(args) -> int:
    cliente = cliente_padrao()
    linhas = []
    for simbolo in args.simbolos.split(","):
        simbolo = simbolo.strip().upper()
        try:
            book = cliente.book(simbolo, profundidade=20)
            linhas.append(
                {
                    "simbolo": simbolo,
                    "fonte": cliente.ultima_fonte,
                    "melhor_bid": book.melhor_bid,
                    "melhor_ask": book.melhor_ask,
                    "spread_pct": round(book.spread_pct, 4),
                    "desequilibrio_book": round(book.desequilibrio(), 3),
                }
            )
        except Exception as exc:
            linhas.append({"simbolo": simbolo, "erro": str(exc)})
    _saida(linhas)
    return 0


def cmd_custo(args) -> int:
    modelo = ModeloCusto.da_exchange(args.exchange)
    print(relatorio.relatorio_alvo_centavos(modelo))
    return 0


def cmd_backtest(args) -> int:
    df = mercado.historico(args.simbolo, args.tempo, dias=args.dias)
    mercado.valida_serie(df, f"{args.simbolo} {args.tempo}")

    config = ConfigBacktest(
        capital_inicial=args.capital,
        intervalo=args.tempo,
        modelo_custo=ModeloCusto.da_exchange(args.exchange),
        config_sinal=ConfigSinal(permitir_venda=args.permitir_venda),
    )
    res = executar(df, args.simbolo, config)

    if args.markdown:
        print(relatorio.relatorio_backtest(res, df["fechamento"]))
    else:
        _saida(res.resumo(df["fechamento"]))
    return 0


def cmd_validar(args) -> int:
    df = mercado.historico(args.simbolo, args.tempo, dias=args.dias)
    mercado.valida_serie(df, f"{args.simbolo} {args.tempo}")

    base = ConfigBacktest(
        capital_inicial=args.capital,
        intervalo=args.tempo,
        modelo_custo=ModeloCusto.da_exchange(args.exchange),
        config_sinal=ConfigSinal(permitir_venda=args.permitir_venda),
    )
    treino, teste = mod_estudo.TEMPOS_PADRAO.get(args.tempo, (2_000, 500))
    janelas, _, resumo = walkforward.walk_forward(
        df, args.simbolo, base=base, barras_treino=treino, barras_teste=teste
    )
    _saida(
        {
            "resumo": resumo,
            "veredito": walkforward.veredito(resumo),
            "janelas": [
                {
                    "teste": [str(j.teste_inicio)[:10], str(j.teste_fim)[:10]],
                    "sharpe_teste": j.sharpe_teste,
                    "retorno_pct": j.retorno_teste_pct,
                    "operacoes": j.operacoes_teste,
                }
                for j in janelas
            ],
        }
    )
    return 0


def cmd_estudo(args) -> int:
    simbolos = tuple(s.strip().upper() for s in args.simbolos.split(","))
    rel = mod_estudo.rodar(simbolos=simbolos, dias=args.dias)
    caminho = mod_estudo.salvar(rel)
    if args.markdown:
        print(relatorio.relatorio_estudo(rel))
    else:
        _saida({"arquivo": str(caminho), "conclusao": rel["conclusao"], "aprovados": rel["aprovados"]})
    return 0


def cmd_sinal(args) -> int:
    df = mercado.historico(args.simbolo, args.tempo, dias=max(args.dias, 60))
    dados = sinais.gerar_sinais(df, ConfigSinal(permitir_venda=args.permitir_venda))
    ultima = dados.index[-1]
    detalhe = sinais.explica_barra(dados, ultima)
    detalhe["preco_fechamento"] = float(dados["fechamento"].iloc[-1])
    detalhe["posicao_alvo"] = int(dados["posicao_alvo"].iloc[-1])
    detalhe["stop_sugerido"] = float(dados["stop_long"].iloc[-1])
    detalhe["alvo_sugerido"] = float(dados["alvo_long"].iloc[-1])
    detalhe["aviso"] = (
        "Score é leitura de indicador, não previsão. Só opere depois que a "
        "validação fora da amostra aprovar a configuração."
    )
    _saida(detalhe)
    return 0


def cmd_carrego(args) -> int:
    """Carrego de funding: o retorno que não depende de adivinhar direção."""
    simbolos = tuple(s.strip().upper() for s in args.simbolos.split(","))
    tabela = mod_carrego.comparar(simbolos, capital_usd=args.capital)
    if tabela.empty:
        print("nenhum par respondeu", file=sys.stderr)
        return 1

    print(tabela.to_string(index=False))
    print()
    print(
        "liquido_ano_pct já desconta as quatro pernas de montagem e desmontagem.\n"
        "Compare sempre com a renda fixa em reais antes de decidir: se o carrego\n"
        "líquido em dólar não bate o CDI, a estrutura está pagando para dar trabalho."
    )
    detalhe = mod_carrego.analisar(simbolos[0], capital_usd=args.capital)
    print()
    for aviso in detalhe["avisos"]:
        print(f"  - {aviso}")
    return 0


def cmd_papel(args) -> int:
    operador = papel.OperadorPapel(
        book=papel.carregar(capital_inicial=args.capital),
        modelo_custo=ModeloCusto.da_exchange(args.exchange),
    )

    fechadas = operador.revisar_posicoes()
    for f in fechadas:
        print(f"fechada {f.simbolo} ({f.motivo}): US$ {f.pnl_liquido_usd:+.4f}", file=sys.stderr)

    for simbolo in (s.strip().upper() for s in args.simbolos.split(",")):
        try:
            df = mercado.historico(simbolo, args.tempo, dias=60)
            dados = sinais.gerar_sinais(df, ConfigSinal(permitir_venda=args.permitir_venda))
            alvo = int(dados["posicao_alvo"].iloc[-1])
            score = float(dados["score"].iloc[-1])
            if alvo != 0:
                operador.abrir(
                    simbolo,
                    alvo,
                    stop=float(dados["stop_long" if alvo == 1 else "stop_short"].iloc[-1]),
                    alvo=float(dados["alvo_long" if alvo == 1 else "alvo_short"].iloc[-1]),
                    score=score,
                )
        except Exception as exc:
            print(f"{simbolo}: {exc}", file=sys.stderr)

    operador.marcar_valor()
    _saida(operador.resumo())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="cripto",
        description="Motor de cripto do QuantBot: dado real, custo real, validação honesta.",
    )
    p.add_argument("-v", "--verboso", action="store_true")
    sub = p.add_subparsers(dest="comando", required=True)

    def comuns(sp, com_simbolo=True):
        if com_simbolo:
            sp.add_argument("simbolo", nargs="?", default="BTCUSDT")
        sp.add_argument("--tempo", default="1h", choices=list(mercado.DURACAO_MS))
        sp.add_argument("--dias", type=int, default=400)
        sp.add_argument("--capital", type=float, default=1_000.0)
        sp.add_argument("--exchange", default="binance")
        sp.add_argument("--permitir-venda", action="store_true", dest="permitir_venda")
        return sp

    sp = sub.add_parser("precos", help="preço vivo, spread e desequilíbrio do book")
    sp.add_argument("--simbolos", default="BTCUSDT,ETHUSDT,SOLUSDT")
    sp.set_defaults(func=cmd_precos)

    sp = sub.add_parser("custo", help="a conta do alvo de US$ 0,10")
    sp.add_argument("--exchange", default="binance")
    sp.set_defaults(func=cmd_custo)

    sp = comuns(sub.add_parser("backtest", help="backtest com custo em toda ponta"))
    sp.add_argument("--markdown", action="store_true")
    sp.set_defaults(func=cmd_backtest)

    sp = comuns(sub.add_parser("validar", help="walk-forward + Deflated Sharpe + PBO"))
    sp.set_defaults(func=cmd_validar)

    sp = sub.add_parser("estudo", help="bateria completa em vários pares e tempos")
    sp.add_argument("--simbolos", default=",".join(mod_estudo.SIMBOLOS_PADRAO))
    sp.add_argument("--dias", type=int, default=400)
    sp.add_argument("--markdown", action="store_true")
    sp.set_defaults(func=cmd_estudo)

    sp = comuns(sub.add_parser("sinal", help="score da barra mais recente, explicado"))
    sp.set_defaults(func=cmd_sinal)

    sp = sub.add_parser("carrego", help="carrego de funding, líquido de custo")
    sp.add_argument("--simbolos", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    sp.add_argument("--capital", type=float, default=1_000.0)
    sp.set_defaults(func=cmd_carrego)

    sp = comuns(sub.add_parser("papel", help="um ciclo de paper trading"), com_simbolo=False)
    sp.add_argument("--simbolos", default="BTCUSDT,ETHUSDT")
    sp.set_defaults(func=cmd_papel)

    args = p.parse_args(argv)
    _log(args.verboso)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
