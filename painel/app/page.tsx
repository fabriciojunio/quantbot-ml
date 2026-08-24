import { readFileSync } from "node:fs";
import { join } from "node:path";

import { Curva } from "@/components/Curva";
import type { Painel } from "@/lib/painel";
import { formatarBRL, formatarData, formatarPct, nomeDe } from "@/lib/painel";

/**
 * A página é gerada no build, a partir do retrato que o robô produziu depois
 * de verificar a cadeia de hash. A tela não fala com o banco e não roda nada
 * em servidor: é HTML.
 */
function carregar(): Painel {
  const caminho = join(process.cwd(), "dados", "painel.json");
  return JSON.parse(readFileSync(caminho, "utf-8")) as Painel;
}

export default function Pagina() {
  const p = carregar();
  const venceu = p.referencia_cdi !== null && p.patrimonio > p.referencia_cdi;
  const positivo = p.resultado >= 0;

  return (
    <>
      <header className="cabecalho">
        <div>
          <h1>Livro do Robô</h1>
          <p className="subtitulo">
            Retrato de {formatarData(p.gerado_em)} · preços reais, dinheiro simulado
          </p>
        </div>
        <span className="selo">
          {p.dinheiro_real ? "dinheiro real" : "simulação"}
        </span>
      </header>

      <div className="faixa">
        <div>
          <div className="rotulo">Patrimônio</div>
          <div className="numero">{formatarBRL(p.patrimonio)}</div>
          <div className="nota">
            {formatarBRL(p.aportado)} depositados
          </div>
        </div>

        <div>
          <div className="rotulo">Resultado</div>
          <div className={`numero ${positivo ? "alta" : "baixa"}`}>
            {formatarBRL(p.resultado)}
          </div>
          <div className={`nota ${positivo ? "alta" : "baixa"}`}>
            {formatarPct(p.resultado_pct)} sobre o depositado
          </div>
        </div>

        <div>
          <div className="rotulo">Retorno ao ano</div>
          <div className="numero">
            {p.tir_pct !== null ? `${p.tir_pct.toFixed(2).replace(".", ",")}%` : "—"}
          </div>
          <div className="nota">
            {p.tir_cdi_pct !== null
              ? `CDI no mesmo período: ${p.tir_cdi_pct.toFixed(2).replace(".", ",")}%`
              : "sem referência ainda"}
          </div>
        </div>

        <div>
          <div className="rotulo">Contra o CDI</div>
          <div className={`numero ${venceu ? "alta" : "baixa"}`}>
            {p.referencia_cdi !== null
              ? formatarBRL(p.patrimonio - p.referencia_cdi)
              : "—"}
          </div>
          <div className="nota">
            {p.referencia_cdi !== null
              ? `mesmo dinheiro em CDI daria ${formatarBRL(p.referencia_cdi)}`
              : "sem aportes registrados"}
          </div>
        </div>
      </div>

      <section>
        <h2>Patrimônio e depósitos</h2>
        <Curva serie={p.serie} />
        <div className="aviso-caixa">
          A linha tracejada é o dinheiro que entrou. Ela sobe todo mês porque o
          aporte é mensal, e não tem nada a ver com desempenho. <strong>O
          resultado é a distância entre as duas linhas</strong> — é o único
          número deste painel que mede o trabalho do robô.
        </div>
      </section>

      <section>
        <h2>Onde está o dinheiro</h2>
        <div className="rolagem">
          <table>
            <thead>
              <tr>
                <th>Ativo</th>
                <th>Peso</th>
                <th>Valor</th>
                <th>Preço médio</th>
                <th>Preço hoje</th>
                <th>Resultado</th>
              </tr>
            </thead>
            <tbody>
              {p.posicoes.map((pos) => (
                <tr key={pos.ticker}>
                  <td>
                    <span className="ativo-nome">{nomeDe(pos.ticker)}</span>
                    <span className="ativo-ticker">{pos.ticker}</span>
                  </td>
                  <td
                    className="barra"
                    style={{ ["--largura" as string]: `${pos.peso_pct}%` }}
                  >
                    <span>{pos.peso_pct.toFixed(1).replace(".", ",")}%</span>
                  </td>
                  <td>{formatarBRL(pos.valor)}</td>
                  <td>{formatarBRL(pos.custo_medio)}</td>
                  <td>{formatarBRL(pos.preco)}</td>
                  <td className={pos.resultado_pct >= 0 ? "alta" : "baixa"}>
                    {formatarPct(pos.resultado_pct)}
                  </td>
                </tr>
              ))}
              <tr>
                <td>
                  <span className="ativo-nome">Caixa</span>
                  <span className="ativo-ticker">rendendo CDI</span>
                </td>
                <td
                  className="barra"
                  style={{
                    ["--largura" as string]: `${
                      p.patrimonio > 0 ? (p.caixa / p.patrimonio) * 100 : 0
                    }%`,
                  }}
                >
                  <span>
                    {(p.patrimonio > 0 ? (p.caixa / p.patrimonio) * 100 : 0)
                      .toFixed(1)
                      .replace(".", ",")}
                    %
                  </span>
                </td>
                <td>{formatarBRL(p.caixa)}</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="legenda" style={{ marginTop: 20 }}>
          <span>
            Exposição a moeda estrangeira ·{" "}
            <strong>
              {p.exposicao_cambial.estrangeira_pct.toFixed(1).replace(".", ",")}%
            </strong>
          </span>
          <span>Custo pago até hoje · {formatarBRL(p.custo_pago)}</span>
          <span>Imposto pago até hoje · {formatarBRL(p.imposto_pago)}</span>
        </div>
      </section>

      {p.ultimas_ordens.length > 0 && (
        <section>
          <h2>Últimas ordens, com o motivo de cada uma</h2>
          <div className="rolagem">
            <table>
              <thead>
                <tr>
                  <th>Data</th>
                  <th>Ativo</th>
                  <th>Operação</th>
                  <th>Valor</th>
                  <th style={{ textAlign: "left" }}>Por quê</th>
                </tr>
              </thead>
              <tbody>
                {[...p.ultimas_ordens].reverse().map((o, i) => (
                  <tr key={`${o.quando}-${i}`}>
                    <td style={{ textAlign: "left" }}>
                      {o.data ? formatarData(o.data) : "—"}
                    </td>
                    <td style={{ textAlign: "left" }}>{o.ticker ?? "—"}</td>
                    <td>
                      <span
                        className="marca"
                        style={{
                          color:
                            o.tipo === "compra" ? "var(--alta)" : "var(--baixa)",
                        }}
                      >
                        {o.tipo ?? "—"}
                      </span>
                    </td>
                    <td>
                      {o.valor_bruto !== undefined
                        ? formatarBRL(o.valor_bruto)
                        : "—"}
                    </td>
                    <td className="ordem-motivo">{o.motivo || o.resumo}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section>
        <h2>Integridade do livro</h2>
        <div className="auditoria">
          <div>
            <div className="rotulo">Cadeia</div>
            <div
              className={`numero ${
                p.auditoria.cadeia_intacta ? "alta" : "baixa"
              }`}
              style={{ fontSize: 20 }}
            >
              {p.auditoria.cadeia_intacta ? "íntegra" : "QUEBRADA"}
            </div>
            <div className="nota">
              {p.auditoria.quebras === 0
                ? "nenhum evento alterado"
                : `${p.auditoria.quebras} quebra(s) encontrada(s)`}
            </div>
          </div>

          <div>
            <div className="rotulo">Eventos registrados</div>
            <div className="numero" style={{ fontSize: 20 }}>
              {p.auditoria.eventos.toLocaleString("pt-BR")}
            </div>
            <div className="nota">
              desde{" "}
              {p.auditoria.primeiro_evento
                ? formatarData(p.auditoria.primeiro_evento)
                : "—"}
            </div>
          </div>

          <div style={{ gridColumn: "1 / -1" }}>
            <div className="rotulo">Hash final da cadeia</div>
            <div className="hash">{p.auditoria.hash_final ?? "—"}</div>
          </div>
        </div>

        <div className="aviso-caixa">
          Cada evento carrega o hash do anterior. Alterar um dia antigo quebra
          todos os dias seguintes, e a verificação aponta onde.{" "}
          <strong>
            Guarde o hash acima em outro lugar — um e-mail para você mesmo
            serve
          </strong>
          : com ele fora do arquivo, até uma reescrita completa da cadeia
          aparece.
        </div>
      </section>

      {p.satelite && (
        <section>
          <h2>Satélite</h2>
          <p style={{ margin: 0, fontSize: 14 }}>
            <span
              className="marca"
              style={{
                color: p.satelite.liberado ? "var(--alta)" : "var(--atencao)",
                marginRight: 12,
              }}
            >
              {p.satelite.liberado ? "liberado" : "fechado"}
            </span>
            {p.satelite.motivo}
          </p>
        </section>
      )}

      <footer>
        <p>
          <strong style={{ color: "var(--tinta)" }}>{p.aviso}</strong>
        </p>
        <p>
          Este painel não é recomendação de investimento. Os números vêm de
          cotação pública e de regras escritas no código, e passado não garante
          futuro — as quedas que aparecem no gráfico vão acontecer de novo, em
          datas que ninguém sabe.
        </p>
        <p>Gerado em {formatarData(p.gerado_em)} a partir do livro-razão.</p>
      </footer>
    </>
  );
}
