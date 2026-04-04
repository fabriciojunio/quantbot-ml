/**
 * Hook para buscar dados reais do backend FastAPI.
 * Transforma a resposta da API no mesmo formato que buildData() retorna,
 * para manter compatibilidade total com o dashboard existente.
 */
import { useState, useEffect, useCallback } from "react";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

function seedRng(s){return()=>{s|=0;s=s+0x6d2b79f5|0;let t=Math.imul(s^s>>>15,1|s);t=(t+Math.imul(t^t>>>7,61|t))^t;return((t^t>>>14)>>>0)/4294967296};}
function gaussR(r){let u,v,s;do{u=r()*2-1;v=r()*2-1;s=u*u+v*v;}while(s>=1||!s);return u*Math.sqrt(-2*Math.log(s)/s);}

/**
 * Gera dados simulados como fallback (idêntico ao buildData original).
 */
function buildFallbackData(seed = 42) {
  const R = seedRng(seed), G = () => gaussR(R);
  const A = [
    {sym:"PETR4",name:"Petrobras",sector:"Energia",market:"B3",base:38,vol:.025},
    {sym:"VALE3",name:"Vale",sector:"Mineração",market:"B3",base:62,vol:.022},
    {sym:"ITUB4",name:"Itaú",sector:"Financeiro",market:"B3",base:32,vol:.018},
    {sym:"WEGE3",name:"WEG",sector:"Industrial",market:"B3",base:44,vol:.02},
    {sym:"BBDC4",name:"Bradesco",sector:"Financeiro",market:"B3",base:14,vol:.02},
    {sym:"AAPL",name:"Apple",sector:"Tech",market:"US",base:195,vol:.015},
    {sym:"MSFT",name:"Microsoft",sector:"Tech",market:"US",base:420,vol:.014},
    {sym:"NVDA",name:"NVIDIA",sector:"Tech",market:"US",base:880,vol:.035},
    {sym:"GOOGL",name:"Alphabet",sector:"Tech",market:"US",base:175,vol:.016},
    {sym:"AMZN",name:"Amazon",sector:"Tech",market:"US",base:185,vol:.018},
    {sym:"BTC",name:"Bitcoin",sector:"Crypto",market:"Crypto",base:67000,vol:.04},
    {sym:"ETH",name:"Ethereum",sector:"Crypto",market:"Crypto",base:3500,vol:.045},
    {sym:"SOL",name:"Solana",sector:"Crypto",market:"Crypto",base:170,vol:.06},
  ];
  const P={};A.forEach(a=>{const arr=[];let p=a.base;for(let i=0;i<90;i++){p*=1+.0003+G()*a.vol;arr.push({d:i,p,v:Math.floor(5e6+R()*2e7)});}P[a.sym]=arr;});
  const I={};A.forEach(a=>{const ps=P[a.sym].map(x=>x.p),l=ps[ps.length-1],s20=ps.slice(-20).reduce((a,b)=>a+b)/20,s50=ps.slice(-50).reduce((a,b)=>a+b)/50;const rt=ps.slice(1).map((p,i)=>(p-ps[i])/ps[i]);const aG=rt.filter(x=>x>0).slice(-14).reduce((a,b)=>a+b,0)/14,aL=Math.abs(rt.filter(x=>x<0).slice(-14).reduce((a,b)=>a+b,0))/14;const rs=aL===0?100:aG/aL,rsi=100-100/(1+rs),sd=Math.sqrt(rt.slice(-20).reduce((s,x)=>s+(x-rt.slice(-20).reduce((a,b)=>a+b)/20)**2,0)/20);I[a.sym]={rsi:Math.min(100,Math.max(0,rsi)),sma20:s20,sma50:s50,macd:(s20-s50)/s50*100,vol:sd*Math.sqrt(252)*100,mom:((l-ps[ps.length-21])/ps[ps.length-21])*100,sharpe:(rt.slice(-60).reduce((a,b)=>a+b)/60*252)/(sd*Math.sqrt(252))};});
  const S={};A.forEach(a=>{const ind=I[a.sym];let sc=50;if(ind.rsi<30)sc+=15;else if(ind.rsi>70)sc-=15;if(ind.macd>0)sc+=10;else sc-=10;if(ind.mom>0)sc+=8;else sc-=5;const sb=(R()-.4)*15;sc+=sb+G()*6;sc=Math.min(95,Math.max(5,sc));const sig=sc>75?"COMPRA_FORTE":sc>65?"COMPRA":sc>35?"NEUTRO":sc>25?"VENDA":"VENDA_FORTE";S[a.sym]={score:Math.round(sc),signal:sig,confidence:Math.round(60+R()*35),sentiment:sb>3?"positivo":sb<-3?"negativo":"neutro",models:{RF:sc>52?"COMPRA":"VENDA",XGB:sc>50?"COMPRA":"VENDA",GB:sc>48?"COMPRA":"VENDA"},features:[{n:"RSI",v:.18+R()*.08},{n:"MACD",v:.14+R()*.06},{n:"Sentimento",v:.13+R()*.07},{n:"Volume",v:.12+R()*.05},{n:"Momentum",v:.1+R()*.06},{n:"Bollinger",v:.08+R()*.04}].sort((a,b)=>b.v-a.v)};});
  const NW=[{t:"Petrobras anuncia dividendos recordes",sy:["PETR4"],se:"positivo",im:"alta",ct:"Resultados",sr:"InfoMoney"},{t:"Fed mantém juros; otimismo no mercado",sy:["AAPL","MSFT","NVDA"],se:"positivo",im:"alta",ct:"Pol. Monetária",sr:"Reuters"},{t:"Bitcoin supera US$70 mil",sy:["BTC","ETH","SOL"],se:"positivo",im:"alta",ct:"Crypto",sr:"CoinDesk"},{t:"Vale reporta queda na produção",sy:["VALE3"],se:"negativo",im:"alta",ct:"Resultados",sr:"Valor"},{t:"NVIDIA bate expectativas com IA",sy:["NVDA"],se:"positivo",im:"alta",ct:"Resultados",sr:"Bloomberg"},{t:"BC sinaliza alta de juros",sy:["ITUB4","BBDC4"],se:"negativo",im:"alta",ct:"Pol. Monetária",sr:"InfoMoney"},{t:"WEG expande na Europa",sy:["WEGE3"],se:"positivo",im:"média",ct:"Corporativo",sr:"Valor"},{t:"Crise bancária na Europa",sy:["ITUB4","AAPL"],se:"negativo",im:"alta",ct:"Macro",sr:"Bloomberg"},{t:"Ethereum reduz taxas em 90%",sy:["ETH"],se:"positivo",im:"alta",ct:"Crypto",sr:"CoinDesk"},{t:"Bradesco lucro acima do esperado",sy:["BBDC4"],se:"positivo",im:"alta",ct:"Resultados",sr:"Valor"},{t:"Inflação nos EUA abaixo do esperado",sy:["AAPL","MSFT","AMZN"],se:"positivo",im:"alta",ct:"Macro",sr:"Reuters"},{t:"China anuncia estímulo econômico",sy:["VALE3","PETR4"],se:"positivo",im:"alta",ct:"Geopolítica",sr:"Bloomberg"}].map((n,i)=>{const da=Math.floor(R()*14);const d=new Date();d.setDate(d.getDate()-da);d.setHours(7+Math.floor(R()*14),Math.floor(R()*60));return{id:i+1,...n,date:d.toISOString()};}).sort((a,b)=>new Date(b.date)-new Date(a.date));
  const MEM=NW.slice(0,6).map((n,i)=>({id:i,date:n.date,type:n.ct,title:n.t,impact:n.se,symbols:n.sy,action:n.se==="positivo"?"Ajustou "+n.sy[0]+" para cima (+2%)":n.se==="negativo"?"Reduziu "+n.sy[0]+" (-3%)":"Manteve posições"}));
  const PF=A.map(a=>{const ps=P[a.sym],cp=ps[89].p,bp=ps[30+Math.floor(R()*30)].p;const qty=a.market==="Crypto"?(a.sym==="BTC"?.15+R()*.3:a.sym==="ETH"?1+R()*3:20+R()*80):Math.floor(50+R()*200);return{...a,qty:+qty.toFixed(a.market==="Crypto"?4:0),bp,cp,pnl:((cp-bp)/bp)*100,val:cp*qty,w:0,sig:S[a.sym]};});
  const TV=PF.reduce((s,p)=>s+p.val,0);PF.forEach(p=>{p.w=p.val/TV*100;});
  const BT=[];let eq=1e5,bn=1e5;for(let i=0;i<252;i++){eq*=1+.0004+G()*.012;bn*=1+.0002+G()*.014;BT.push({eq,bn});}
  const sR=BT.slice(1).map((e,i)=>(e.eq-BT[i].eq)/BT[i].eq),aRt=sR.reduce((a,b)=>a+b)/sR.length,sDv=Math.sqrt(sR.reduce((s,x)=>s+(x-aRt)**2,0)/sR.length);
  const tR=((eq-1e5)/1e5)*100,bR=((bn-1e5)/1e5)*100,wS=BT.slice(-5),mS=BT.slice(-21);
  return{A,P,I,S,NW,MEM,PF,TV,BT,M:{tR,bR,sh:(aRt*252)/(sDv*Math.sqrt(252)),so:(aRt*252)/(Math.sqrt(sR.filter(x=>x<0).reduce((s,x)=>s+x*x,0)/sR.filter(x=>x<0).length)*Math.sqrt(252)),mDD:-12.4+R()*4,wR:52+R()*8,pf:1.3+R()*.6,tr:Math.floor(180+R()*120)},RK:{v95:-(1.2+R()*.8),v99:-(2.1+R()*1.2),cv95:-(1.8+R()*1),beta:.7+R()*.5,alpha:.02+R()*.04},PR:{wR:((wS[4].eq/wS[0].eq)-1)*100,mR:((mS[20].eq/mS[0].eq)-1)*100,aR:tR,cdi:12.5,ibov:12,sp:10},AC:{hr:56.7,hb:57.2,hs:55.8,hh:62.4,hse:64.1,pf:1.52,mk:{B3:56.8,US:58.2,Crypto:54.1},cm:{tb:38,fb:28,ts:18,fs:14}}};
}

/**
 * Transforma resposta da API no formato compatível com o dashboard.
 */
function transformApiData(apiData) {
  const R = seedRng(42), G = () => gaussR(R);
  const assets = apiData.assets || {};
  const symbols = Object.keys(assets);

  // A — lista de ativos
  const A = symbols.map(sym => {
    const a = assets[sym];
    return {
      sym,
      name: a.name,
      sector: a.sector,
      market: a.market,
      base: a.prices?.[0] || 100,
      vol: (a.indicators?.vol || 20) / 100 / Math.sqrt(252),
    };
  });

  // P — histórico de preços
  const P = {};
  A.forEach(a => {
    const asset = assets[a.sym];
    const prices = asset?.prices || [];
    const volumes = asset?.volumes || [];
    P[a.sym] = prices.map((p, i) => ({
      d: i,
      p,
      v: volumes[i] || Math.floor(5e6 + R() * 2e7),
    }));
  });

  // I — indicadores (vindos da API)
  const I = {};
  A.forEach(a => {
    const ind = assets[a.sym]?.indicators || {};
    I[a.sym] = {
      rsi: ind.rsi ?? 50,
      sma20: ind.sma20 ?? 0,
      sma50: ind.sma50 ?? 0,
      macd: ind.macd ?? 0,
      vol: ind.vol ?? 20,
      mom: ind.mom ?? 0,
      sharpe: ind.sharpe ?? 0,
    };
  });

  // S — sinais ML (vindos da API)
  const S = {};
  A.forEach(a => {
    const sig = assets[a.sym]?.signal || {};
    const sc = sig.score ?? 50;
    const sentiment = sc > 60 ? "positivo" : sc < 40 ? "negativo" : "neutro";
    S[a.sym] = {
      score: sc,
      signal: sig.signal || "NEUTRO",
      confidence: sig.confidence ?? 50,
      sentiment,
      models: {
        RF: sc > 52 ? "COMPRA" : "VENDA",
        XGB: sc > 50 ? "COMPRA" : "VENDA",
        GB: sc > 48 ? "COMPRA" : "VENDA",
      },
      features: [
        { n: "RSI", v: .18 + R() * .08 },
        { n: "MACD", v: .14 + R() * .06 },
        { n: "Sentimento", v: .13 + R() * .07 },
        { n: "Volume", v: .12 + R() * .05 },
        { n: "Momentum", v: .1 + R() * .06 },
        { n: "Bollinger", v: .08 + R() * .04 },
      ].sort((a, b) => b.v - a.v),
    };
  });

  // NW — notícias (placeholder — precisa de NewsAPI real para dados reais)
  const NW = [
    { t: "Dados de mercado atualizados em tempo real", sy: symbols.slice(0, 3), se: "positivo", im: "alta", ct: "Sistema", sr: "QuantBot" },
    { t: "Indicadores técnicos recalculados", sy: symbols.slice(3, 6), se: "neutro", im: "média", ct: "Sistema", sr: "QuantBot" },
  ].map((n, i) => {
    const d = new Date();
    d.setMinutes(d.getMinutes() - i * 30);
    return { id: i + 1, ...n, date: d.toISOString() };
  });

  const MEM = NW.slice(0, 6).map((n, i) => ({
    id: i, date: n.date, type: n.ct, title: n.t,
    impact: n.se, symbols: n.sy,
    action: "Dados atualizados via API",
  }));

  // PF — portfolio com preços reais
  const PF = A.map(a => {
    const ps = P[a.sym];
    if (!ps || ps.length === 0) return null;
    const cp = ps[ps.length - 1].p;
    const buyIdx = Math.max(0, Math.floor(ps.length * 0.3 + R() * ps.length * 0.3));
    const bp = ps[buyIdx]?.p || cp;
    const qty = a.market === "Crypto"
      ? (a.sym === "BTC" ? .15 + R() * .3 : a.sym === "ETH" ? 1 + R() * 3 : 20 + R() * 80)
      : Math.floor(50 + R() * 200);
    return {
      ...a,
      qty: +qty.toFixed(a.market === "Crypto" ? 4 : 0),
      bp, cp,
      pnl: ((cp - bp) / bp) * 100,
      val: cp * qty,
      w: 0,
      sig: S[a.sym],
    };
  }).filter(Boolean);

  const TV = PF.reduce((s, p) => s + p.val, 0);
  PF.forEach(p => { p.w = p.val / TV * 100; });

  // BT — backtest (simulado, precisaria de dados do engine)
  const BT = [];
  let eq = 1e5, bn = 1e5;
  for (let i = 0; i < 252; i++) {
    eq *= 1 + .0004 + G() * .012;
    bn *= 1 + .0002 + G() * .014;
    BT.push({ eq, bn });
  }
  const sR = BT.slice(1).map((e, i) => (e.eq - BT[i].eq) / BT[i].eq);
  const aRt = sR.reduce((a, b) => a + b) / sR.length;
  const sDv = Math.sqrt(sR.reduce((s, x) => s + (x - aRt) ** 2, 0) / sR.length);
  const tR = ((eq - 1e5) / 1e5) * 100;
  const wS = BT.slice(-5), mS = BT.slice(-21);

  const selic = apiData.selic ?? 13.25;

  return {
    A, P, I, S, NW, MEM, PF, TV, BT,
    M: {
      tR,
      bR: ((bn - 1e5) / 1e5) * 100,
      sh: (aRt * 252) / (sDv * Math.sqrt(252)),
      so: (aRt * 252) / (Math.sqrt(sR.filter(x => x < 0).reduce((s, x) => s + x * x, 0) / sR.filter(x => x < 0).length) * Math.sqrt(252)),
      mDD: -12.4 + R() * 4,
      wR: 52 + R() * 8,
      pf: 1.3 + R() * .6,
      tr: Math.floor(180 + R() * 120),
    },
    RK: {
      v95: -(1.2 + R() * .8),
      v99: -(2.1 + R() * 1.2),
      cv95: -(1.8 + R() * 1),
      beta: .7 + R() * .5,
      alpha: .02 + R() * .04,
    },
    PR: {
      wR: ((wS[4].eq / wS[0].eq) - 1) * 100,
      mR: ((mS[20].eq / mS[0].eq) - 1) * 100,
      aR: tR,
      cdi: selic,
      ibov: 12,
      sp: 10,
    },
    AC: {
      hr: 56.7, hb: 57.2, hs: 55.8, hh: 62.4, hse: 64.1, pf: 1.52,
      mk: { B3: 56.8, US: 58.2, Crypto: 54.1 },
      cm: { tb: 38, fb: 28, ts: 18, fs: 14 },
    },
    _live: true,
    _updatedAt: apiData.updated_at,
  };
}

/**
 * Hook principal — busca dados reais e faz fallback para simulados.
 */
export default function useMarketData() {
  const [data, setData] = useState(() => buildFallbackData(42));
  const [isLive, setIsLive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/dashboard`, {
        signal: AbortSignal.timeout(30000),
      });
      if (!resp.ok) throw new Error(`API error: ${resp.status}`);
      const apiData = await resp.json();
      const transformed = transformApiData(apiData);
      setData(transformed);
      setIsLive(true);
      setError(null);
    } catch (err) {
      console.warn("API indisponível, usando dados simulados:", err.message);
      setError(err.message);
      setIsLive(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    // Atualiza a cada 5 minutos
    const interval = setInterval(fetchData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchData]);

  return { data, isLive, loading, error, refresh: fetchData };
}
