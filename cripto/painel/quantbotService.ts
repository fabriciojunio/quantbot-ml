/**
 * Ponte entre o painel Sentinela e o motor `cripto/` do QuantBot.
 *
 * Substitui as fontes sintéticas do painel por dados reais:
 *
 *   orderFlowDataService.generateLiveOrderFlowData  ->  buscarBook
 *   cryptoTapeDbService.*                           ->  buscarTape
 *   hftConfluenceService.generateLocal...           ->  buscarSinal
 *   tradingExecutionService (localStorage)          ->  buscarCarteiraPapel
 *
 * Toda resposta traz `fonte` (qual exchange respondeu) e `sintetico: false`.
 * Se o motor estiver fora do ar, as funções devolvem `null` — o painel deve
 * mostrar "sem dados" em vez de preencher com número inventado. Era esse o
 * defeito central da versão anterior: o dado faltava e a tela mentia.
 */

const BASE = import.meta.env?.VITE_QUANTBOT_URL ?? 'http://localhost:8100';

export interface PrecoReal {
  simbolo: string;
  melhor_bid: number;
  melhor_ask: number;
  meio: number;
  spread_pct: number;
  desequilibrio_book: number;
  fonte: string;
  sintetico: false;
}

export interface NivelBook {
  preco: number;
  quantidade: number;
}

export interface BookReal {
  simbolo: string;
  fonte: string;
  melhor_bid: number;
  melhor_ask: number;
  spread_pct: number;
  desequilibrio: number;
  maior_parede_compra: { preco: number; quantidade: number; notional_usd: number } | null;
  maior_parede_venda: { preco: number; quantidade: number; notional_usd: number } | null;
  niveis_bid: NivelBook[];
  niveis_ask: NivelBook[];
  sintetico: false;
}

export interface TapeReal {
  simbolo: string;
  fonte: string;
  negocios_analisados: number;
  volume_agressao_compra: number;
  volume_agressao_venda: number;
  delta: number;
  delta_pct: number;
  corte_bloco_usd: number | null;
  blocos_grandes: Array<{
    quando_ms: number;
    preco: number;
    quantidade: number;
    notional_usd: number;
    lado: 'compra' | 'venda';
  }>;
  sintetico: false;
}

export interface ContribuicaoIndicador {
  voto: number | null;
  peso_pct: number;
  contribuicao: number | null;
}

export interface SinalReal {
  quando: string;
  simbolo: string;
  tempo: string;
  score: number;
  fechamento: number;
  posicao_alvo: -1 | 0 | 1;
  stop_sugerido: number;
  alvo_sugerido: number;
  indicadores: Record<string, ContribuicaoIndicador>;
  calculado_de: string;
  fonte: string;
  aviso: string;
}

export interface CarteiraPapel {
  capital_inicial_usd: number;
  valor_atual_usd: number;
  retorno_pct: number;
  caixa_usd: number;
  posicoes_abertas: Array<{
    simbolo: string;
    lado: 'COMPRA' | 'VENDA';
    preco_entrada: number;
    notional_usd: number;
    stop: number;
    alvo: number;
    score_entrada: number;
    aberta_em: string;
  }>;
  operacoes_fechadas: number;
  taxa_acerto_pct: number;
  custo_total_pago_usd: number;
  pnl_liquido_usd: number;
  modo: string;
}

export interface ContaDeCusto {
  exchange: string;
  taxa_taker_pct: number;
  taxa_maker_pct: number;
  custo_ida_volta_pct: number;
  teste_do_alvo: {
    notional_usd: number;
    alvo_usd: number;
    custo_usd: number;
    resultado_liquido_usd: number;
    viavel: boolean;
    quantas_vezes_o_alvo_o_custo_come: number;
  };
}

/** GET com timeout. Devolve null em qualquer falha — nunca dado inventado. */
async function buscar<T>(rota: string, timeoutMs = 8000): Promise<T | null> {
  const controlador = new AbortController();
  const alarme = setTimeout(() => controlador.abort(), timeoutMs);
  try {
    const resposta = await fetch(`${BASE}${rota}`, {
      signal: controlador.signal,
      headers: { Accept: 'application/json' },
      cache: 'no-store',
    });
    if (!resposta.ok) {
      console.warn(`[QuantBot] ${rota} respondeu ${resposta.status}`);
      return null;
    }
    return (await resposta.json()) as T;
  } catch (erro) {
    console.warn(`[QuantBot] ${rota} indisponível:`, erro);
    return null;
  } finally {
    clearTimeout(alarme);
  }
}

export async function buscarPrecos(simbolos: string[]): Promise<PrecoReal[] | null> {
  const dados = await buscar<{ precos: PrecoReal[] }>(
    `/api/cripto/precos?simbolos=${encodeURIComponent(simbolos.join(','))}`
  );
  return dados?.precos ?? null;
}

export async function buscarBook(simbolo: string, profundidade = 50): Promise<BookReal | null> {
  return buscar<BookReal>(`/api/cripto/book/${simbolo}?profundidade=${profundidade}`);
}

export async function buscarTape(simbolo: string, limite = 300): Promise<TapeReal | null> {
  return buscar<TapeReal>(`/api/cripto/tape/${simbolo}?limite=${limite}`);
}

export async function buscarSinal(simbolo: string, tempo = '1h'): Promise<SinalReal | null> {
  return buscar<SinalReal>(`/api/cripto/sinal/${simbolo}?tempo=${tempo}`, 20000);
}

export async function buscarCarteiraPapel(): Promise<CarteiraPapel | null> {
  return buscar<CarteiraPapel>('/api/cripto/papel');
}

export async function buscarContaDeCusto(
  notionalUsd: number,
  alvoUsd: number,
  exchange = 'binance'
): Promise<ContaDeCusto | null> {
  return buscar<ContaDeCusto>(
    `/api/cripto/custo?notional_usd=${notionalUsd}&alvo_usd=${alvoUsd}&exchange=${exchange}`
  );
}

export async function motorNoAr(): Promise<boolean> {
  const s = await buscar<{ status: string }>('/api/cripto/saude', 3000);
  return s?.status === 'ok';
}

/**
 * Formata o score em rótulo. Mantém o vocabulário do painel, mas com uma zona
 * neutra larga: "não sei" é resposta legítima e evita a entrada compulsória
 * que fazia a versão anterior operar o tempo todo.
 */
export function rotuloDoScore(score: number): string {
  if (score >= 75) return 'COMPRA FORTE';
  if (score >= 62) return 'COMPRA';
  if (score > 38) return 'NEUTRO';
  if (score > 25) return 'VENDA';
  return 'VENDA FORTE';
}
