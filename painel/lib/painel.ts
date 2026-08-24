/**
 * O contrato entre o robô e a tela.
 *
 * Os tipos aqui são a única definição do formato. Se o robô mudar o JSON sem
 * mudar isto, o `tsc --noEmit` do build reprova — que é exatamente o que se
 * quer: uma tela que mostra campo faltando como "undefined" é pior que uma
 * tela que não sobe.
 */

export type Posicao = {
  ticker: string;
  cotas: number;
  preco: number;
  valor: number;
  custo_medio: number;
  resultado_pct: number;
  peso_pct: number;
};

export type PontoDaSerie = {
  data: string;
  total: number;
  aportado: number;
  cota: number;
};

export type Ordem = {
  quando: string;
  resumo: string;
  data?: string;
  ticker?: string;
  tipo?: string;
  valor_bruto?: number;
  custo?: number;
  imposto?: number;
  motivo?: string;
};

export type Auditoria = {
  eventos: number;
  cadeia_intacta: boolean;
  quebras: number;
  hash_final: string | null;
  primeiro_evento: string | null;
};

export type ExposicaoCambial = {
  por_moeda: Record<string, number>;
  estrangeira_pct: number;
  local_pct: number;
};

export type Painel = {
  gerado_em: string;
  dinheiro_real: boolean;
  aviso: string;
  patrimonio: number;
  aportado: number;
  resultado: number;
  resultado_pct: number;
  caixa: number;
  custo_pago: number;
  imposto_pago: number;
  referencia_cdi: number | null;
  tir_pct: number | null;
  tir_cdi_pct: number | null;
  posicoes: Posicao[];
  exposicao_cambial: ExposicaoCambial;
  serie: PontoDaSerie[];
  ultimas_ordens: Ordem[];
  auditoria: Auditoria;
  satelite?: { liberado: boolean; motivo: string };
  configuracao?: Record<string, unknown>;
};

export function formatarBRL(valor: number, casas = 2): string {
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "BRL",
    minimumFractionDigits: casas,
    maximumFractionDigits: casas,
  }).format(valor);
}

export function formatarPct(valor: number, casas = 2): string {
  const sinal = valor > 0 ? "+" : "";
  return `${sinal}${valor.toFixed(casas).replace(".", ",")}%`;
}

export function formatarData(iso: string): string {
  const d = new Date(iso.length <= 10 ? `${iso}T12:00:00Z` : iso);
  return new Intl.DateTimeFormat("pt-BR", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).format(d);
}

/** Nome legível de cada ticker. A tela fala a língua de quem lê, não a do sistema. */
export const NOMES: Record<string, string> = {
  "BOVA11.SA": "Ibovespa",
  "SMAL11.SA": "Small caps Brasil",
  "IVVB11.SA": "S&P 500 em reais",
  "IMAB11.SA": "Tesouro IPCA+",
  "B5P211.SA": "Tesouro IPCA+ curto",
  "GOLD11.SA": "Ouro",
  "XFIX11.SA": "Fundos imobiliários",
  "DIVO11.SA": "Ações de dividendo",
  "FIXA11.SA": "Prefixado",
  "BTC-BRL": "Bitcoin",
  CDI: "Caixa (CDI)",
};

export function nomeDe(ticker: string): string {
  return NOMES[ticker] ?? ticker;
}
