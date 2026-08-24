import type { PontoDaSerie } from "@/lib/painel";
import { formatarBRL } from "@/lib/painel";

/**
 * Patrimônio contra o que foi depositado.
 *
 * As duas linhas juntas são o gráfico honesto. Só o patrimônio sobe quase
 * sempre — porque entra dinheiro novo todo mês — e olhar só para ele faz
 * qualquer carteira parecer boa. O que interessa é a distância entre a linha
 * cheia e a tracejada: essa distância é o resultado, e só ela.
 *
 * SVG na mão em vez de biblioteca de gráfico. São duas linhas: qualquer
 * biblioteca aqui seria 200 KB para desenhar dois `path`, e uma dependência a
 * mais para manter viva por décadas.
 */
export function Curva({ serie }: { serie: PontoDaSerie[] }) {
  if (serie.length < 2) {
    return (
      <p style={{ color: "var(--tinta-fraca)", fontSize: 13 }}>
        Ainda não há dias suficientes para desenhar a curva.
      </p>
    );
  }

  const L = 32;
  const R = 12;
  const T = 14;
  const B = 26;
  const W = 1000;
  const H = 300;

  const valores = serie.flatMap((p) => [p.total, p.aportado]);
  const max = Math.max(...valores);
  const min = Math.min(...valores, 0);
  const faixa = max - min || 1;

  const x = (i: number) => L + (i / (serie.length - 1)) * (W - L - R);
  const y = (v: number) => T + (1 - (v - min) / faixa) * (H - T - B);

  const linha = (campo: "total" | "aportado") =>
    serie.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p[campo]).toFixed(1)}`).join(" ");

  const area =
    `M${x(0).toFixed(1)},${y(serie[0]!.aportado).toFixed(1)} ` +
    serie.map((p, i) => `L${x(i).toFixed(1)},${y(p.aportado).toFixed(1)}`).join(" ") +
    " " +
    serie
      .map((p, i) => [i, p] as const)
      .reverse()
      .map(([i, p]) => `L${x(i).toFixed(1)},${y(p.total).toFixed(1)}`)
      .join(" ") +
    " Z";

  const ultimo = serie[serie.length - 1]!;
  const ganhando = ultimo.total >= ultimo.aportado;

  // Quatro marcas no eixo, nos anos redondos que existirem na série.
  const marcas = [0, 0.33, 0.66, 1].map((f) => {
    const i = Math.round(f * (serie.length - 1));
    return { i, texto: serie[i]!.data.slice(0, 7) };
  });

  return (
    <>
      <svg
        className="grafico"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`Patrimônio de ${formatarBRL(ultimo.total)} contra ${formatarBRL(
          ultimo.aportado
        )} depositados`}
      >
        <defs>
          <linearGradient id="preenchimento" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              stopColor={ganhando ? "var(--alta)" : "var(--baixa)"}
              stopOpacity="0.20"
            />
            <stop
              offset="100%"
              stopColor={ganhando ? "var(--alta)" : "var(--baixa)"}
              stopOpacity="0.02"
            />
          </linearGradient>
        </defs>

        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <line
            key={f}
            x1={L}
            x2={W - R}
            y1={T + f * (H - T - B)}
            y2={T + f * (H - T - B)}
            stroke="var(--linha)"
            strokeWidth="1"
          />
        ))}

        <path d={area} fill="url(#preenchimento)" />
        <path
          d={linha("aportado")}
          fill="none"
          stroke="var(--tinta-tenue)"
          strokeWidth="1.5"
          strokeDasharray="5 4"
        />
        <path
          d={linha("total")}
          fill="none"
          stroke={ganhando ? "var(--alta)" : "var(--baixa)"}
          strokeWidth="2"
          strokeLinejoin="round"
        />
        <circle
          cx={x(serie.length - 1)}
          cy={y(ultimo.total)}
          r="3.5"
          fill={ganhando ? "var(--alta)" : "var(--baixa)"}
        />

        {marcas.map((m) => (
          <text
            key={m.i}
            x={x(m.i)}
            y={H - 8}
            fontSize="11"
            fill="var(--tinta-tenue)"
            fontFamily="var(--mono)"
            textAnchor={m.i === 0 ? "start" : m.i === serie.length - 1 ? "end" : "middle"}
          >
            {m.texto}
          </text>
        ))}
      </svg>

      <div className="legenda">
        <span>
          <i style={{ background: ganhando ? "var(--alta)" : "var(--baixa)" }} />
          Patrimônio · {formatarBRL(ultimo.total)}
        </span>
        <span>
          <i
            style={{
              background: "var(--tinta-tenue)",
              height: 2,
              backgroundImage:
                "repeating-linear-gradient(90deg, var(--tinta-tenue) 0 4px, transparent 4px 8px)",
            }}
          />
          Depositado · {formatarBRL(ultimo.aportado)}
        </span>
        <span style={{ color: ganhando ? "var(--alta)" : "var(--baixa)" }}>
          Diferença · {formatarBRL(ultimo.total - ultimo.aportado)}
        </span>
      </div>
    </>
  );
}
