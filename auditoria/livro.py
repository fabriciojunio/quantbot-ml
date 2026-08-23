"""
O livro em si: escrita encadeada, leitura, e nada de UPDATE.

Todo evento entra com o hash do anterior embutido no próprio hash. Isso não
impede ninguém de abrir o arquivo e mudar uma linha — impede que a mudança
passe despercebida, que é o que dá para garantir num arquivo local.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("quantbot.auditoria")

CAMINHO_PADRAO = Path("data/auditoria/livro.sqlite3")

# Marca o começo da cadeia. Qualquer livro íntegro tem exatamente um evento
# cujo anterior é isto.
GENESE = "0" * 64

ESQUEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS evento (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    quando         TEXT    NOT NULL,
    tipo           TEXT    NOT NULL,
    origem         TEXT    NOT NULL,
    resumo         TEXT    NOT NULL,
    dados          TEXT    NOT NULL,
    hash_anterior  TEXT    NOT NULL,
    hash           TEXT    NOT NULL UNIQUE,
    corrige        INTEGER REFERENCES evento(id)
);

CREATE INDEX IF NOT EXISTS ix_evento_quando ON evento(quando);
CREATE INDEX IF NOT EXISTS ix_evento_tipo   ON evento(tipo, quando);

-- Estado da carteira em cada fechamento. Derivado dos eventos, gravado
-- separado porque consultar 20 anos de evento para desenhar um gráfico é
-- caro e desnecessário.
CREATE TABLE IF NOT EXISTS posicao_diaria (
    data           TEXT    NOT NULL,
    ativo          TEXT    NOT NULL,
    quantidade     REAL    NOT NULL,
    preco          REAL    NOT NULL,
    valor          REAL    NOT NULL,
    custo_medio    REAL    NOT NULL,
    evento_id      INTEGER NOT NULL REFERENCES evento(id),
    PRIMARY KEY (data, ativo)
);

CREATE TABLE IF NOT EXISTS patrimonio_diario (
    data           TEXT    PRIMARY KEY,
    caixa          REAL    NOT NULL,
    investido      REAL    NOT NULL,
    total          REAL    NOT NULL,
    aportado       REAL    NOT NULL,
    cota           REAL    NOT NULL,
    cdi_acumulado  REAL    NOT NULL,
    evento_id      INTEGER NOT NULL REFERENCES evento(id)
);

-- Cada versão de parâmetro que a calibração produziu, com a janela em que foi
-- ajustada e a janela em que foi aplicada. É o histórico do aprendizado.
CREATE TABLE IF NOT EXISTS parametro (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nome            TEXT    NOT NULL,
    valor           TEXT    NOT NULL,
    ajustado_de     TEXT    NOT NULL,
    ajustado_ate    TEXT    NOT NULL,
    vale_a_partir   TEXT    NOT NULL,
    metrica_dentro  REAL,
    metrica_fora    REAL,
    evento_id       INTEGER NOT NULL REFERENCES evento(id)
);

CREATE INDEX IF NOT EXISTS ix_parametro_nome ON parametro(nome, vale_a_partir);
"""


class TipoEvento(str, Enum):
    """O que aconteceu. Fechado de propósito: tipo livre vira lixo em dois anos."""

    ABERTURA = "abertura"                # livro criado
    COLETA = "coleta"                    # dado de mercado recebido
    DECISAO = "decisao"                  # o robô decidiu algo
    ORDEM = "ordem"                      # ordem simulada emitida
    APORTE = "aporte"                    # dinheiro novo entrou
    FECHAMENTO = "fechamento"            # patrimônio do dia registrado
    CALIBRACAO = "calibracao"            # parâmetro reajustado
    VALIDACAO = "validacao"              # estratégia avaliada
    ALERTA = "alerta"                    # algo saiu do esperado
    CORRECAO = "correcao"                # conserta um evento anterior
    NOTICIA = "noticia"                  # informação externa relevante


@dataclass(frozen=True)
class Evento:
    id: int
    quando: str
    tipo: TipoEvento
    origem: str
    resumo: str
    dados: dict[str, Any]
    hash_anterior: str
    hash: str
    corrige: int | None = None


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonico(dados: Any) -> str:
    """JSON estável: mesma entrada, mesmo texto, sempre.

    `sort_keys` e separadores fixos são obrigatórios — sem isso o hash muda
    conforme a ordem em que o dicionário foi montado, e a cadeia quebra por
    motivo nenhum.
    """
    return json.dumps(dados, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def calcular_hash(
    quando: str,
    tipo: str,
    origem: str,
    resumo: str,
    dados: Any,
    hash_anterior: str,
) -> str:
    corpo = "|".join([quando, tipo, origem, resumo, _canonico(dados), hash_anterior])
    return hashlib.sha256(corpo.encode("utf-8")).hexdigest()


@dataclass
class Livro:
    """Um livro-razão aberto. Use `abrir()` em vez de instanciar direto."""

    caminho: Path
    conexao: sqlite3.Connection = field(repr=False)
    _trava: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── Escrita ──────────────────────────────────────────────────────────

    def registrar(
        self,
        tipo: TipoEvento,
        origem: str,
        resumo: str,
        dados: dict[str, Any] | None = None,
        corrige: int | None = None,
        quando: str | None = None,
    ) -> Evento:
        """Acrescenta um evento ao fim da cadeia.

        `corrige` aponta para o evento que este conserta. O anterior continua
        no livro: quem quiser o histórico verdadeiro vê os dois, e quem quiser
        o valor corrente usa `estado_corrigido()`.
        """
        if not resumo.strip():
            raise ValueError("todo evento precisa de um resumo legível")

        dados = dados or {}
        with self._trava:
            anterior = self._ultimo_hash()
            quando = quando or _agora()
            h = calcular_hash(quando, tipo.value, origem, resumo, dados, anterior)

            cur = self.conexao.execute(
                "INSERT INTO evento (quando, tipo, origem, resumo, dados, "
                "hash_anterior, hash, corrige) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (quando, tipo.value, origem, resumo, _canonico(dados), anterior, h, corrige),
            )
            self.conexao.commit()
            return Evento(cur.lastrowid, quando, tipo, origem, resumo, dados, anterior, h, corrige)

    def registrar_fechamento(
        self,
        data: str,
        caixa: float,
        posicoes: dict[str, dict[str, float]],
        aportado: float,
        cota: float,
        cdi_acumulado: float,
        origem: str = "robo",
    ) -> Evento:
        """Grava o retrato do dia: patrimônio, posição por ativo e cota.

        Reescrever o mesmo dia é permitido e é o único lugar onde há UPDATE —
        um dia pode ser recalculado quando a fonte corrige um preço. O evento
        que gerou a linha continua no livro, então a substituição fica
        rastreável mesmo assim.
        """
        investido = sum(p["valor"] for p in posicoes.values())
        total = caixa + investido

        evento = self.registrar(
            TipoEvento.FECHAMENTO, origem,
            f"Fechamento de {data}: patrimônio {total:.2f}, aportado {aportado:.2f}",
            {
                "data": data, "caixa": round(caixa, 2), "investido": round(investido, 2),
                "total": round(total, 2), "aportado": round(aportado, 2),
                "cota": round(cota, 8), "ativos": sorted(posicoes),
            },
        )

        with self._trava:
            self.conexao.execute(
                "INSERT OR REPLACE INTO patrimonio_diario "
                "(data, caixa, investido, total, aportado, cota, cdi_acumulado, evento_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (data, caixa, investido, total, aportado, cota, cdi_acumulado, evento.id),
            )
            for ativo, p in posicoes.items():
                self.conexao.execute(
                    "INSERT OR REPLACE INTO posicao_diaria "
                    "(data, ativo, quantidade, preco, valor, custo_medio, evento_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (data, ativo, p["quantidade"], p["preco"], p["valor"],
                     p.get("custo_medio", 0.0), evento.id),
                )
            self.conexao.commit()
        return evento

    def registrar_parametro(
        self,
        nome: str,
        valor: Any,
        ajustado_de: str,
        ajustado_ate: str,
        vale_a_partir: str,
        metrica_dentro: float | None = None,
        metrica_fora: float | None = None,
        origem: str = "calibracao",
    ) -> Evento:
        """Registra um parâmetro aprendido, com a janela em que foi aprendido.

        As três datas são o que separa aprendizado de invenção. Um parâmetro
        ajustado de 2010 a 2015 e aplicado a partir de 2015 pode ser avaliado
        fora da amostra; o mesmo parâmetro ajustado até 2026 e aplicado a 2015
        seria vazamento de futuro, e o `vale_a_partir` deixa isso visível.
        """
        if vale_a_partir < ajustado_ate:
            raise ValueError(
                f"parâmetro '{nome}' valeria a partir de {vale_a_partir} mas foi ajustado "
                f"com dados até {ajustado_ate}: isso é usar informação do futuro"
            )

        evento = self.registrar(
            TipoEvento.CALIBRACAO, origem,
            f"{nome} = {valor} (ajustado de {ajustado_de} a {ajustado_ate})",
            {"nome": nome, "valor": valor, "ajustado_de": ajustado_de,
             "ajustado_ate": ajustado_ate, "vale_a_partir": vale_a_partir,
             "metrica_dentro": metrica_dentro, "metrica_fora": metrica_fora},
        )
        with self._trava:
            self.conexao.execute(
                "INSERT INTO parametro (nome, valor, ajustado_de, ajustado_ate, "
                "vale_a_partir, metrica_dentro, metrica_fora, evento_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (nome, _canonico(valor), ajustado_de, ajustado_ate, vale_a_partir,
                 metrica_dentro, metrica_fora, evento.id),
            )
            self.conexao.commit()
        return evento

    # ── Leitura ──────────────────────────────────────────────────────────

    def _ultimo_hash(self) -> str:
        linha = self.conexao.execute(
            "SELECT hash FROM evento ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return linha["hash"] if linha else GENESE

    def total_de_eventos(self) -> int:
        return self.conexao.execute("SELECT COUNT(*) c FROM evento").fetchone()["c"]

    def evento(self, id_: int) -> Evento | None:
        linha = self.conexao.execute("SELECT * FROM evento WHERE id = ?", (id_,)).fetchone()
        return _para_evento(linha) if linha else None

    def eventos(
        self,
        tipo: TipoEvento | None = None,
        desde: str | None = None,
        ate: str | None = None,
        limite: int | None = None,
    ) -> list[Evento]:
        sql = "SELECT * FROM evento WHERE 1=1"
        args: list[Any] = []
        if tipo:
            sql += " AND tipo = ?"
            args.append(tipo.value)
        if desde:
            sql += " AND quando >= ?"
            args.append(desde)
        if ate:
            sql += " AND quando <= ?"
            args.append(ate)
        sql += " ORDER BY id"
        if limite:
            sql += f" LIMIT {int(limite)}"
        return [_para_evento(x) for x in self.conexao.execute(sql, args)]

    def percorrer(self) -> Iterator[Evento]:
        """Percorre a cadeia inteira sem carregar tudo na memória."""
        for linha in self.conexao.execute("SELECT * FROM evento ORDER BY id"):
            yield _para_evento(linha)

    def corrigidos(self) -> set[int]:
        """Ids de eventos que foram substituídos por uma correção posterior."""
        return {
            r["corrige"]
            for r in self.conexao.execute("SELECT DISTINCT corrige FROM evento WHERE corrige IS NOT NULL")
        }

    def patrimonio(self, desde: str | None = None) -> list[dict]:
        sql = "SELECT * FROM patrimonio_diario"
        args: list[Any] = []
        if desde:
            sql += " WHERE data >= ?"
            args.append(desde)
        sql += " ORDER BY data"
        return [dict(r) for r in self.conexao.execute(sql, args)]

    def parametros_vigentes(self, em: str) -> dict[str, Any]:
        """Os parâmetros que valiam numa data, sem olhar para frente.

        A cláusula `vale_a_partir <= ?` é o que impede o robô de usar hoje um
        parâmetro que só foi aprendido depois. É a diferença entre backtest e
        ficção.
        """
        linhas = self.conexao.execute(
            "SELECT nome, valor, MAX(vale_a_partir) AS v FROM parametro "
            "WHERE vale_a_partir <= ? GROUP BY nome",
            (em,),
        )
        return {r["nome"]: json.loads(r["valor"]) for r in linhas}

    def historico_do_parametro(self, nome: str) -> list[dict]:
        return [
            dict(r)
            for r in self.conexao.execute(
                "SELECT * FROM parametro WHERE nome = ? ORDER BY vale_a_partir", (nome,)
            )
        ]

    # ── Manutenção ───────────────────────────────────────────────────────

    def exportar(self, destino: Path) -> Path:
        """Despeja o livro em JSON, para quem não tiver SQLite daqui a 40 anos."""
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        conteudo = {
            "gerado_em": _agora(),
            "origem": str(self.caminho),
            "total_de_eventos": self.total_de_eventos(),
            "eventos": [
                {
                    "id": e.id, "quando": e.quando, "tipo": e.tipo.value, "origem": e.origem,
                    "resumo": e.resumo, "dados": e.dados, "hash_anterior": e.hash_anterior,
                    "hash": e.hash, "corrige": e.corrige,
                }
                for e in self.percorrer()
            ],
        }
        destino.write_text(json.dumps(conteudo, indent=2, ensure_ascii=False), encoding="utf-8")
        return destino

    def fechar(self) -> None:
        self.conexao.close()

    def __enter__(self) -> Livro:
        return self

    def __exit__(self, *_) -> None:
        self.fechar()


def _para_evento(linha: sqlite3.Row) -> Evento:
    return Evento(
        id=linha["id"],
        quando=linha["quando"],
        tipo=TipoEvento(linha["tipo"]),
        origem=linha["origem"],
        resumo=linha["resumo"],
        dados=json.loads(linha["dados"]),
        hash_anterior=linha["hash_anterior"],
        hash=linha["hash"],
        corrige=linha["corrige"],
    )


def abrir(caminho: Path | str = CAMINHO_PADRAO, origem: str = "sistema") -> Livro:
    """Abre (ou cria) o livro. Livro novo já nasce com o evento de abertura."""
    caminho = Path(caminho)
    if caminho != Path(":memory:"):
        caminho.parent.mkdir(parents=True, exist_ok=True)

    conexao = sqlite3.connect(str(caminho), check_same_thread=False)
    conexao.row_factory = sqlite3.Row
    conexao.executescript(ESQUEMA)
    conexao.commit()

    livro = Livro(caminho=caminho, conexao=conexao)
    if livro.total_de_eventos() == 0:
        livro.registrar(
            TipoEvento.ABERTURA, origem,
            "Livro-razão aberto",
            {"versao_do_esquema": 1, "caminho": str(caminho)},
        )
        logger.info("Livro criado em %s", caminho)
    return livro


@contextmanager
def aberto(caminho: Path | str = CAMINHO_PADRAO, origem: str = "sistema") -> Iterator[Livro]:
    livro = abrir(caminho, origem)
    try:
        yield livro
    finally:
        livro.fechar()
