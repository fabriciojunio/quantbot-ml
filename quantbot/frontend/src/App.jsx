import { useState, useEffect, useRef, useCallback } from "react";
import useMarketData from "./useMarketData";

const fm=(n,d=2)=>Number(n).toFixed(d);
const fP=n=>(n>=0?"+":"")+fm(n)+"%";
const fC=n=>"$"+Number(n).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
const fK=n=>n>=1e6?"$"+fm(n/1e6,2)+"M":n>=1000?"$"+fm(n/1000,1)+"k":fC(n);
const tA=d=>{const m=Math.floor((Date.now()-new Date(d))/6e4);return m<60?m+"min":m<1440?Math.floor(m/60)+"h":Math.floor(m/1440)+"d";};
const C={bg:"#0a0a0a",c1:"#0e0e0e",c2:"#141414",bd:"#1e1e1e",bd2:"#2a2a2a",gold:"#D4A843",gd:"#A07830",grn:"#2ECC71",red:"#E74C3C",cy:"#C9B458",pu:"#F0D68A",tx:"#e8e0d0",tx2:"#8a7d6b",tx3:"#5a5248"};
const TS=()=>new Date().toLocaleTimeString("pt-BR",{hour:"2-digit",minute:"2-digit",second:"2-digit"});

function Spark({data,w=100,h=28,color=C.gold}){if(!data||!data.length)return null;const mn=Math.min(...data),mx=Math.max(...data),rg=mx-mn||1;const pts=data.map((v,i)=>((i/(data.length-1))*w)+","+( h-((v-mn)/rg)*h)).join(" ");return(<svg width={w} height={h}><defs><linearGradient id={"sp"+color.replace("#","")} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity="0.2"/><stop offset="100%" stopColor={color} stopOpacity="0"/></linearGradient></defs><polygon points={"0,"+h+" "+pts+" "+w+","+h} fill={"url(#sp"+color.replace("#","")+")"}/><polyline points={pts} fill="none" stroke={color} strokeWidth="1.5"/></svg>);}

function EqCurve({data,w=680,h=200}){if(!data||!data.length)return null;const pL=50,pT=10,pR=10,iW=w-pL-pR,iH=h-pT-20;const all=data.flatMap(d=>[d.eq,d.bn]),mn=Math.min(...all)*.98,mx=Math.max(...all)*1.02,rg=mx-mn||1;const tX=i=>pL+(i/(data.length-1))*iW,tY=v=>pT+iH-((v-mn)/rg)*iH;const ep=data.map((d,i)=>(i===0?"M":"L")+tX(i)+","+tY(d.eq)).join(" ");const bp=data.map((d,i)=>(i===0?"M":"L")+tX(i)+","+tY(d.bn)).join(" ");const gL=[0,1,2,3].map(i=>{const v=mn+(rg/3)*i;return(<g key={i}><line x1={pL} y1={tY(v)} x2={w-pR} y2={tY(v)} stroke={C.bd} strokeWidth="1"/><text x={pL-6} y={tY(v)+4} textAnchor="end" fill={C.tx3} fontSize="9" fontFamily="sans-serif">{fK(v)}</text></g>);});return(<svg width="100%" viewBox={"0 0 "+w+" "+h}><defs><linearGradient id="eqg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={C.gold} stopOpacity="0.12"/><stop offset="100%" stopColor={C.gold} stopOpacity="0"/></linearGradient></defs>{gL}<path d={ep+" L"+tX(data.length-1)+","+tY(mn)+" L"+tX(0)+","+tY(mn)+" Z"} fill="url(#eqg)"/><path d={bp} fill="none" stroke={C.tx3} strokeWidth="1.2" strokeDasharray="5 4"/><path d={ep} fill="none" stroke={C.gold} strokeWidth="2"/></svg>);}

function Gauge({score,sz=54}){const clr=score>65?C.grn:score>45?C.gold:C.red,ci=Math.PI*(sz-8),dOff=ci-(score/100)*ci;return(<div style={{position:"relative",width:sz,height:sz/2+6}}><svg width={sz} height={sz/2+6}><path d={"M 4,"+(sz/2)+" A "+(sz/2-4)+","+(sz/2-4)+" 0 0,1 "+(sz-4)+","+(sz/2)} fill="none" stroke={C.bd} strokeWidth="5" strokeLinecap="round"/><path d={"M 4,"+(sz/2)+" A "+(sz/2-4)+","+(sz/2-4)+" 0 0,1 "+(sz-4)+","+(sz/2)} fill="none" stroke={clr} strokeWidth="5" strokeLinecap="round" strokeDasharray={ci/2+""} strokeDashoffset={dOff/2+""} style={{transition:"stroke-dashoffset .8s"}}/></svg><div style={{position:"absolute",bottom:0,left:"50%",transform:"translateX(-50%)",fontSize:14,fontWeight:700,color:clr}}>{score}</div></div>);}

function SigBadge({signal}){const mp={COMPRA_FORTE:{c:C.grn,l:"COMPRA FORTE"},COMPRA:{c:"#45D98A",l:"COMPRA"},NEUTRO:{c:C.gold,l:"NEUTRO"},VENDA:{c:"#F07B6E",l:"VENDA"},VENDA_FORTE:{c:C.red,l:"VENDA FORTE"}};const it=mp[signal]||{c:C.tx3,l:signal};return(<span style={{display:"inline-flex",alignItems:"center",gap:4,padding:"3px 10px",borderRadius:4,fontSize:10,fontWeight:700,background:it.c+"15",color:it.c,border:"1px solid "+it.c+"30"}}><span style={{width:5,height:5,borderRadius:"50%",background:it.c}}/>{it.l}</span>);}
function SentBadge({sentiment}){const mp={positivo:{c:C.grn,i:"▲"},negativo:{c:C.red,i:"▼"},neutro:{c:C.gold,i:"●"}};const it=mp[sentiment]||mp.neutro;return(<span style={{fontSize:11,fontWeight:600,color:it.c}}>{it.i+" "+sentiment.charAt(0).toUpperCase()+sentiment.slice(1)}</span>);}
function Donut({segments,sz=140}){const tot=segments.reduce((s,x)=>s+x.v,0),rad=sz/2-12,ci=2*Math.PI*rad,cols=["#D4A843","#2ECC71","#C9B458","#A07830"];let acc=0;const circles=segments.map((seg,i)=>{const pct=seg.v/tot,da=pct*ci+" "+ci,doff=-acc*ci;acc+=pct;return(<circle key={i} cx={sz/2} cy={sz/2} r={rad} fill="none" stroke={cols[i%cols.length]} strokeWidth="14" strokeDasharray={da} strokeDashoffset={doff} transform={"rotate(-90 "+sz/2+" "+sz/2+")"}/>);});return(<svg width={sz} height={sz}>{circles}<circle cx={sz/2} cy={sz/2} r={rad-12} fill={C.bg}/></svg>);}

const TABS=[{id:"overview",l:"Visão Geral"},{id:"perf",l:"Performance"},{id:"news",l:"Notícias"},{id:"signals",l:"Sinais ML"},{id:"trade",l:"Paper Trading"},{id:"accuracy",l:"Acurácia"},{id:"memory",l:"Memória"},{id:"risk",l:"Risco"}];

export default function App(){
  const[tab,setTab]=useState("overview");
  const{data:D,isLive,loading,error:apiError,refresh}=useMarketData();
  const[selA,setSelA]=useState(null);
  const[nF,setNF]=useState("todas");
  const[tSym,setTSym]=useState("PETR4");
  const[tQty,setTQty]=useState("");
  const[tMsg,setTMsg]=useState("");
  const[cash,setCash]=useState(100000);
  const[hold,setHold]=useState({});
  const[ords,setOrds]=useState([]);
  const[alerts,setAlerts]=useState([]);
  const oc=useRef(0);

  // ══ BOT AUTOMÁTICO ══
  const[botOn,setBotOn]=useState(false);
  const[botLog,setBotLog]=useState([]);
  const[showConfirm,setShowConfirm]=useState(null); // "on" | "off" | null
  const botInterval=useRef(null);
  const holdRef=useRef(hold);
  const cashRef=useRef(cash);
  useEffect(()=>{holdRef.current=hold;},[hold]);
  useEffect(()=>{cashRef.current=cash;},[cash]);

  const addLog=(type,msg,sym,detail)=>{setBotLog(prev=>[{id:Date.now()+Math.random(),time:TS(),type,msg,sym,detail},...prev].slice(0,50));};

  const botCycle=useCallback(()=>{
    addLog("info","Iniciando ciclo de análise...",null,"Analisando "+D.A.length+" ativos");
    setTimeout(()=>{
      addLog("news","Notícias coletadas: "+D.NW.length+" artigos",null,"Fontes: InfoMoney, Reuters, Bloomberg, CoinDesk, Valor");
      setTimeout(()=>{
        const buySignals=D.PF.filter(p=>p.sig.score>65&&p.sig.confidence>65).sort((a,b)=>b.sig.score-a.sig.score);
        const sellSignals=D.PF.filter(p=>p.sig.score<25&&p.sig.confidence>60);
        buySignals.slice(0,3).forEach(p=>{
          const price=D.P[p.sym][89].p;
          const currentHold=holdRef.current;
          const currentCash=cashRef.current;
          if(currentHold[p.sym]){
            addLog("hold","Já possui "+p.sym+", mantendo posição",p.sym,"Score: "+p.sig.score+" | Sentimento: "+p.sig.sentiment);
            return;
          }
          const allocPct=Math.min(0.08,p.sig.score/100*0.1);
          const totalVal=currentCash+Object.entries(currentHold).reduce((s,[k,h])=>s+h.q*(D.P[k]?.[89]?.p||h.ap),0);
          const amount=totalVal*allocPct;
          const qty=Math.floor(amount/price);
          if(qty<=0||qty*price*1.001>currentCash){
            addLog("skip","Capital insuficiente para "+p.sym,p.sym,"Necessário: "+fC(qty*price));
            return;
          }
          const cost=qty*price*1.001;
          setCash(c=>c-cost);
          setHold(h=>{const prev=h[p.sym];if(prev){const nq=prev.q+qty;return{...h,[p.sym]:{...prev,q:nq,ap:(prev.ap*prev.q+price*qty)/nq}};}return{...h,[p.sym]:{s:p.sym,nm:p.name,q:qty,ap:price}};});
          oc.current++;
          setOrds(o=>[{id:oc.current,sym:p.sym,ty:"COMPRA",q:qty,pr:price,tot:qty*price,auto:true},...o]);
          addLog("buy","COMPRA: "+qty+"x "+p.sym+" @ "+fC(price),p.sym,"Score: "+p.sig.score+" | Confiança: "+p.sig.confidence+"% | Sentimento: "+p.sig.sentiment+" | Alocação: "+fm(allocPct*100,1)+"%");
        });
        sellSignals.forEach(p=>{
          const currentHold=holdRef.current;
          if(!currentHold[p.sym])return;
          const price=D.P[p.sym][89].p;
          const qty=currentHold[p.sym].q;
          const revenue=qty*price*.999;
          const pnl=(price-currentHold[p.sym].ap)*qty;
          setCash(c=>c+revenue);
          setHold(prev=>{const cp={...prev};delete cp[p.sym];return cp;});
          oc.current++;
          setOrds(o=>[{id:oc.current,sym:p.sym,ty:"VENDA",q:qty,pr:price,tot:qty*price,pnl,auto:true},...o]);
          addLog("sell","VENDA: "+qty+"x "+p.sym+" @ "+fC(price),p.sym,"Score: "+p.sig.score+" (abaixo de 35) | P&L: "+fC(pnl));
        });
        addLog("done","Ciclo concluído. Próximo em 30s.",null,"Posições: "+Object.keys(holdRef.current).length+" | Cash: "+fC(cashRef.current));
      },1500);
    },800);
  },[D]);

  const startBot=()=>{
    setBotOn(true);
    setShowConfirm(null);
    addLog("start","🤖 Bot Automático LIGADO",null,"Modo: Paper Trading | Análise contínua a cada 30s");
    botCycle();
    botInterval.current=setInterval(botCycle,30000);
  };

  const stopBot=()=>{
    setBotOn(false);
    setShowConfirm(null);
    if(botInterval.current){clearInterval(botInterval.current);botInterval.current=null;}
    addLog("stop","🛑 Bot Automático DESLIGADO",null,"Posições mantidas: "+Object.keys(holdRef.current).length);
  };

  const handleToggle=()=>{
    if(botOn){
      setShowConfirm("off");
    } else {
      setShowConfirm("on");
    }
  };

  useEffect(()=>{return()=>{if(botInterval.current)clearInterval(botInterval.current);};},[]);

  const doBuy=(sym,qty)=>{const pr=D.P[sym]?.[89]?.p;if(!pr)return "Ativo não encontrado";const cost=qty*pr*1.001;if(cost>cash)return "Saldo insuficiente";setCash(c=>c-cost);setHold(h=>{const prev=h[sym];if(prev){const nq=prev.q+qty;return{...h,[sym]:{...prev,q:nq,ap:(prev.ap*prev.q+pr*qty)/nq}};}return{...h,[sym]:{s:sym,nm:D.A.find(a=>a.sym===sym)?.name||sym,q:qty,ap:pr}};});oc.current++;setOrds(o=>[{id:oc.current,sym,ty:"COMPRA",q:qty,pr,tot:qty*pr},...o]);return null;};
  const doSell=(sym,qty)=>{const hh=hold[sym];if(!hh||hh.q<qty)return "Posição insuficiente";const pr=D.P[sym]?.[89]?.p;if(!pr)return "Erro";setCash(c=>c+qty*pr*.999);setHold(prev=>{const nq=hh.q-qty;if(nq<=0){const cp={...prev};delete cp[sym];return cp;}return{...prev,[sym]:{...hh,q:nq}};});oc.current++;setOrds(o=>[{id:oc.current,sym,ty:"VENDA",q:qty,pr,tot:qty*pr,pnl:(pr-hh.ap)*qty},...o]);return null;};
  const ptTot=cash+Object.entries(hold).reduce((s,[k,h])=>s+h.q*(D.P[k]?.[89]?.p||h.ap),0);
  const ptPnl=((ptTot-100000)/100000)*100;
  const filtN=nF==="todas"?D.NW:D.NW.filter(n=>n.se===nF);

  const card={background:C.c1,border:"1px solid "+C.bd,borderRadius:10,padding:18};
  const lbl={fontSize:11,color:C.tx3,textTransform:"uppercase",letterSpacing:1,marginBottom:6,fontWeight:600};
  const bigV={fontSize:22,fontWeight:700};
  const btnS=(a,clr=C.gold)=>({padding:"8px 16px",fontSize:11,fontWeight:a?700:500,color:a?C.bg:C.tx2,background:a?clr:"transparent",border:"1px solid "+(a?clr:C.bd),borderRadius:6,cursor:"pointer",fontFamily:"inherit"});
  const inp={background:C.c1,border:"1px solid "+C.bd2,borderRadius:6,color:C.tx,padding:"9px 12px",fontSize:12,fontFamily:"inherit",outline:"none",width:"100%"};
  const MC=({label,value,color=C.tx})=>(<div style={card}><div style={lbl}>{label}</div><div style={{...bigV,color}}>{value}</div></div>);
  const logColors={start:C.grn,stop:C.red,buy:C.grn,sell:C.red,news:C.cy,info:C.gold,hold:C.tx2,skip:C.tx3,done:C.pu};

  return(
<div style={{fontFamily:"'Nunito Sans','Segoe UI',system-ui,sans-serif",background:C.bg,color:C.tx,minHeight:"100vh"}}>
<link href="https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;500;600;700;800;900&display=swap" rel="stylesheet"/>

{showConfirm&&(<div style={{position:"fixed",top:0,left:0,right:0,bottom:0,background:"rgba(0,0,0,0.8)",zIndex:2000,display:"flex",alignItems:"center",justifyContent:"center"}}><div style={{background:C.c1,border:"1px solid "+C.bd2,borderRadius:14,padding:28,maxWidth:450,width:"90%",textAlign:"center"}}>
{showConfirm==="on"?(<><div style={{fontSize:28,marginBottom:12}}>🤖</div><div style={{fontSize:16,fontWeight:700,marginBottom:8}}>Ligar o Bot Automático?</div><div style={{fontSize:12,color:C.tx2,marginBottom:6,lineHeight:1.6}}>O bot vai analisar {D.A.length} ativos, buscar notícias de 5 fontes, gerar sinais ML e executar operações automaticamente com seu capital simulado de {fC(cash)}.</div><div style={{fontSize:11,color:C.gold,marginBottom:16,padding:"8px 12px",background:C.gold+"10",borderRadius:6}}>Modo Paper Trading — dinheiro fictício, sem risco real</div><div style={{display:"flex",gap:10,justifyContent:"center"}}><button onClick={startBot} style={{...btnS(true,C.grn),padding:"10px 28px",fontSize:13}}>Ligar Bot</button><button onClick={()=>setShowConfirm(null)} style={{...btnS(false),padding:"10px 28px",fontSize:13}}>Cancelar</button></div></>)
:(<><div style={{fontSize:28,marginBottom:12}}>⚠️</div><div style={{fontSize:16,fontWeight:700,marginBottom:8,color:C.red}}>Desligar o Bot Automático?</div>{Object.keys(hold).length>0?(<><div style={{fontSize:12,color:C.tx2,marginBottom:8,lineHeight:1.6}}>Você tem <strong style={{color:C.gold}}>{Object.keys(hold).length} posições abertas</strong>:</div><div style={{background:C.c2,borderRadius:8,padding:12,marginBottom:12,textAlign:"left"}}>{Object.entries(hold).map(([sym,h])=>{const cp=D.P[sym]?.[89]?.p||h.ap;const pnl=((cp-h.ap)/h.ap)*100;return(<div key={sym} style={{display:"flex",justifyContent:"space-between",padding:"4px 0",fontSize:12}}><span style={{fontWeight:600}}>{sym} ({h.q.toFixed(1)} un)</span><span style={{color:pnl>=0?C.grn:C.red,fontWeight:600}}>{fP(pnl)} | {fC(h.q*cp)}</span></div>);})}</div><div style={{fontSize:11,color:C.tx3,marginBottom:12}}>As posições serão mantidas. Você pode vendê-las manualmente depois.</div></>):(<div style={{fontSize:12,color:C.tx2,marginBottom:16}}>Nenhuma posição aberta. O bot será desligado.</div>)}<div style={{display:"flex",gap:10,justifyContent:"center"}}><button onClick={stopBot} style={{...btnS(true,C.red),padding:"10px 28px",fontSize:13}}>Desligar Bot</button><button onClick={()=>setShowConfirm(null)} style={{...btnS(false),padding:"10px 28px",fontSize:13}}>Cancelar</button></div></>)}
</div></div>)}

<div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"14px 24px",borderBottom:"1px solid "+C.bd,background:"#060606",position:"sticky",top:0,zIndex:100}}>
<div style={{display:"flex",alignItems:"center",gap:10}}><div style={{width:32,height:32,borderRadius:8,background:"linear-gradient(135deg,"+C.gold+","+C.gd+")",display:"flex",alignItems:"center",justifyContent:"center",fontSize:15,fontWeight:900,color:C.bg}}>Q</div><div><div style={{fontSize:16,fontWeight:800,color:C.gold}}>QuantBot ML</div><div style={{fontSize:9,color:C.tx3,letterSpacing:1.5,textTransform:"uppercase"}}>Portfolio Intelligence</div></div></div>
<div style={{display:"flex",alignItems:"center",gap:12}}>
<span style={{display:"inline-flex",alignItems:"center",gap:5,padding:"4px 10px",borderRadius:12,fontSize:10,fontWeight:700,background:isLive?C.grn+"18":"#F39C12"+"18",color:isLive?C.grn:"#F39C12",border:"1px solid "+(isLive?C.grn+"40":"#F39C12"+"40")}}><span style={{width:6,height:6,borderRadius:"50%",background:isLive?C.grn:"#F39C12"}}/>{isLive?"LIVE":"SIMULADO"}</span>
<button onClick={handleToggle} style={{display:"flex",alignItems:"center",gap:6,padding:"6px 14px",borderRadius:20,border:"1px solid "+(botOn?C.grn+"40":C.bd2),background:botOn?C.grn+"15":"transparent",cursor:"pointer",fontFamily:"inherit"}}><span style={{width:8,height:8,borderRadius:"50%",background:botOn?C.grn:C.tx3,animation:botOn?"pulse 1.5s infinite":"none"}}/><span style={{fontSize:11,fontWeight:600,color:botOn?C.grn:C.tx3}}>{botOn?"Bot Ativo":"Bot Inativo"}</span></button>
</div></div>

<div style={{display:"flex",gap:0,borderBottom:"1px solid "+C.bd,overflowX:"auto",background:"#060606",padding:"0 8px"}}>{TABS.map(t=>(<button key={t.id} onClick={()=>setTab(t.id)} style={{padding:"12px 18px",fontSize:12,fontWeight:tab===t.id?700:500,color:tab===t.id?C.gold:C.tx3,background:"none",border:"none",borderBottom:tab===t.id?"2px solid "+C.gold:"2px solid transparent",cursor:"pointer",fontFamily:"inherit",whiteSpace:"nowrap"}}>{t.l}</button>))}</div>

<div style={{padding:"20px 24px",maxWidth:1280,margin:"0 auto"}}>

{tab==="overview"&&(<div style={{display:"flex",flexDirection:"column",gap:14}}>
<div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(170px,1fr))",gap:12}}>
<MC label="Valor Total" value={fC(D.TV)}/><MC label="Retorno Anual" value={fP(D.PR.aR)} color={D.PR.aR>=0?C.gold:C.red}/><MC label="Sharpe" value={fm(D.M.sh)} color={C.cy}/><MC label="Acerto" value={D.AC.hr+"%"} color={C.gold}/><MC label="Win Rate" value={fm(D.M.wR,1)+"%"} color={C.grn}/><MC label="Max Drawdown" value={fP(D.M.mDD)} color={C.red}/></div>
<div style={{display:"grid",gridTemplateColumns:"5fr 3fr",gap:12}}>
<div style={card}><div style={{...lbl,marginBottom:10}}>Equity Curve — Bot vs Benchmark (1 ano)</div><EqCurve data={D.BT}/><div style={{display:"flex",gap:16,marginTop:8,justifyContent:"center"}}><span style={{fontSize:11,color:C.gold}}>━ Bot: {fP(D.M.tR)}</span><span style={{fontSize:11,color:C.tx3}}>╌ Bench: {fP(D.M.bR)}</span></div></div>
<div style={card}><div style={{...lbl,marginBottom:10}}>Alocação por Mercado</div><div style={{display:"flex",justifyContent:"center",margin:"12px 0"}}><Donut segments={[{v:D.PF.filter(p=>p.market==="B3").reduce((s,p)=>s+p.val,0)},{v:D.PF.filter(p=>p.market==="US").reduce((s,p)=>s+p.val,0)},{v:D.PF.filter(p=>p.market==="Crypto").reduce((s,p)=>s+p.val,0)}]}/></div>{[{l:"B3",c:C.gold,v:D.PF.filter(p=>p.market==="B3").reduce((s,p)=>s+p.w,0)},{l:"US Stocks",c:C.grn,v:D.PF.filter(p=>p.market==="US").reduce((s,p)=>s+p.w,0)},{l:"Crypto",c:C.cy,v:D.PF.filter(p=>p.market==="Crypto").reduce((s,p)=>s+p.w,0)}].map(x=>(<div key={x.l} style={{display:"flex",alignItems:"center",gap:8,fontSize:13,marginBottom:6}}><span style={{width:8,height:8,borderRadius:3,background:x.c}}/><span style={{color:C.tx2,flex:1}}>{x.l}</span><span style={{fontWeight:700}}>{fm(x.v)}%</span></div>))}</div></div></div>)}

{tab==="perf"&&(<div style={{display:"flex",flexDirection:"column",gap:14}}>
<div style={{...card,border:"1px solid "+C.gold+"25"}}><div style={{display:"flex",borderBottom:"1px solid "+C.bd2,padding:"10px 0",marginBottom:6}}><div style={{width:"32%",fontSize:10,color:C.gold,textTransform:"uppercase",fontWeight:700}}>Métrica</div>{["Semana","Mês","Ano"].map(p=>(<div key={p} style={{width:"22.6%",fontSize:10,color:C.gold,textAlign:"right",textTransform:"uppercase",fontWeight:700}}>{p}</div>))}</div>
{[{h:true,l:"RETORNOS"},{l:"Retorno",w:D.PR.wR,m:D.PR.mR,a:D.PR.aR},{l:"Anualizado",w:D.PR.wR*52,m:D.PR.mR*12,a:D.PR.aR},{h:true,l:"BENCHMARKS"},{l:"CDI",w:D.PR.cdi/52,m:D.PR.cdi/12,a:D.PR.cdi},{l:"Ibovespa",w:D.PR.ibov/52,m:D.PR.ibov/12,a:D.PR.ibov},{l:"S&P 500",w:D.PR.sp/52,m:D.PR.sp/12,a:D.PR.sp},{h:true,l:"ALPHA"},{l:"vs CDI",w:D.PR.wR-D.PR.cdi/52,m:D.PR.mR-D.PR.cdi/12,a:D.PR.aR-D.PR.cdi},{l:"vs Ibovespa",w:D.PR.wR-D.PR.ibov/52,m:D.PR.mR-D.PR.ibov/12,a:D.PR.aR-D.PR.ibov},{l:"vs S&P 500",w:D.PR.wR-D.PR.sp/52,m:D.PR.mR-D.PR.sp/12,a:D.PR.aR-D.PR.sp}].map((row,i)=>row.h?(<div key={i} style={{fontSize:11,color:C.gold,margin:"12px 0 4px",fontWeight:700}}>{row.l}</div>):(<div key={i} style={{display:"flex",borderBottom:"1px solid "+C.bd,padding:"8px 0"}}><div style={{width:"32%",fontSize:12,color:C.tx2}}>{row.l}</div>{[row.w,row.m,row.a].map((v,j)=>(<div key={j} style={{width:"22.6%",fontSize:13,fontWeight:600,textAlign:"right",color:v>=0?C.grn:C.red}}>{fP(v)}</div>))}</div>))}
</div><div style={card}><div style={{...lbl,marginBottom:8}}>Quanto $100.000 virariam em 1 ano</div><div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(150px,1fr))",gap:10}}>{[{n:"🤖 QuantBot",v:1e5*(1+D.PR.aR/100),c:C.gold,hl:true},{n:"CDI",v:1e5*(1+D.PR.cdi/100),c:C.tx2},{n:"Ibovespa",v:1e5*(1+D.PR.ibov/100),c:C.tx2},{n:"S&P 500",v:1e5*(1+D.PR.sp/100),c:C.tx2}].map(x=>(<div key={x.n} style={{padding:14,background:C.c2,borderRadius:8,textAlign:"center",border:x.hl?"1px solid "+C.gold+"30":"1px solid "+C.bd}}><div style={{fontSize:11,color:C.tx3,marginBottom:4}}>{x.n}</div><div style={{fontSize:20,fontWeight:800,color:x.c}}>{fC(x.v)}</div></div>))}</div></div></div>)}

{tab==="news"&&(<div style={{display:"flex",flexDirection:"column",gap:10}}><div style={{display:"flex",gap:6}}>{["todas","positivo","negativo","neutro"].map(f=>(<button key={f} style={btnS(nF===f)} onClick={()=>setNF(f)}>{f==="todas"?"Todas":f.charAt(0).toUpperCase()+f.slice(1)}</button>))}</div>{filtN.map(n=>(<div key={n.id} style={{...card,borderLeft:"3px solid "+(n.se==="positivo"?C.grn:n.se==="negativo"?C.red:C.gold),cursor:"pointer"}} onClick={()=>{setSelA(n.sy[0]);setTab("signals");}}><div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}><span style={{fontSize:10,padding:"2px 8px",borderRadius:4,background:C.c2,color:C.tx3}}>{n.sr}</span><span style={{fontSize:10,color:C.tx3}}>{n.ct}</span><span style={{fontSize:10,color:C.tx3,marginLeft:"auto"}}>{tA(n.date)}</span></div><div style={{fontSize:14,fontWeight:600,marginBottom:6}}>{n.t}</div><div style={{display:"flex",alignItems:"center",gap:8,flexWrap:"wrap"}}><SentBadge sentiment={n.se}/><span style={{fontSize:10,color:C.tx3}}>Impacto: {n.im}</span>{n.sy.map(s=>(<span key={s} style={{fontSize:9,padding:"2px 6px",borderRadius:3,background:C.gold+"12",color:C.gold,fontWeight:600}}>{s}</span>))}</div></div>))}</div>)}

{tab==="signals"&&(()=>{const ast=selA?D.PF.find(p=>p.sym===selA):D.PF.sort((a,b)=>b.sig.score-a.sig.score)[0];const ind=D.I[ast.sym],sig=D.S[ast.sym];return(<div style={{display:"flex",flexDirection:"column",gap:12}}><div style={{...card,display:"flex",alignItems:"center",gap:12,flexWrap:"wrap"}}><select value={ast.sym} onChange={e=>setSelA(e.target.value)} style={{...inp,width:"auto",minWidth:180}}>{D.A.map(a=>(<option key={a.sym} value={a.sym}>{a.sym+" — "+a.name}</option>))}</select><div style={{flex:1}}><span style={{fontSize:20,fontWeight:800}}>{ast.sym}</span><span style={{fontSize:13,color:C.tx2,marginLeft:8}}>{ast.name}</span></div><div style={{textAlign:"right"}}><div style={{fontSize:20,fontWeight:800}}>{fC(ast.cp)}</div><div style={{color:ast.pnl>=0?C.grn:C.red,fontSize:14,fontWeight:600}}>{fP(ast.pnl)}</div></div></div>
<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
<div style={card}><div style={{...lbl,marginBottom:8}}>Score ML + Sentimento</div><div style={{display:"flex",alignItems:"center",gap:14,marginBottom:10}}><Gauge score={sig.score}/><div><SigBadge signal={sig.signal}/><div style={{fontSize:11,color:C.tx3,marginTop:4}}>Confiança: {sig.confidence}%</div><div style={{marginTop:4}}><SentBadge sentiment={sig.sentiment}/></div></div></div><div style={{...lbl,marginBottom:4}}>Modelos</div>{Object.entries(sig.models).map(([k,v])=>(<div key={k} style={{display:"flex",alignItems:"center",gap:5,marginBottom:3}}><span style={{width:6,height:6,borderRadius:"50%",background:v==="COMPRA"?C.grn:C.red}}/><span style={{fontSize:11,color:C.tx2}}>{k+": "+v}</span></div>))}</div>
<div style={card}><div style={{...lbl,marginBottom:8}}>Features</div>{sig.features.map(ft=>(<div key={ft.n} style={{display:"flex",alignItems:"center",gap:6,marginBottom:5}}><span style={{fontSize:11,color:C.tx2,width:80,textAlign:"right"}}>{ft.n}</span><div style={{flex:1,height:6,background:C.bd,borderRadius:3,overflow:"hidden"}}><div style={{width:(ft.v/sig.features[0].v)*100+"%",height:"100%",background:"linear-gradient(90deg,"+C.gold+","+C.gd+")",borderRadius:3}}/></div><span style={{fontSize:10,color:C.gold,width:30}}>{fm(ft.v*100,0)+"%"}</span></div>))}</div>
<div style={card}><div style={{...lbl,marginBottom:8}}>Indicadores</div><div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>{[{l:"RSI",v:fm(ind.rsi),c:ind.rsi>70?C.red:ind.rsi<30?C.grn:C.gold},{l:"MACD",v:fm(ind.macd,3),c:ind.macd>0?C.grn:C.red},{l:"Volatilidade",v:fm(ind.vol)+"%",c:C.gold},{l:"Momentum",v:fP(ind.mom),c:ind.mom>0?C.grn:C.red},{l:"SMA 20",v:fK(ind.sma20),c:C.cy},{l:"Sharpe",v:fm(ind.sharpe),c:C.pu}].map((x,i)=>(<div key={i} style={{padding:8,background:C.c2,borderRadius:6}}><div style={{fontSize:9,color:C.tx3,textTransform:"uppercase",marginBottom:2}}>{x.l}</div><div style={{fontSize:15,fontWeight:700,color:x.c}}>{x.v}</div></div>))}</div></div>
<div style={card}><div style={{...lbl,marginBottom:8}}>Preço 90D</div><Spark data={D.P[ast.sym].map(d=>d.p)} color={ast.pnl>=0?C.gold:C.red} w={300} h={80}/></div>
</div></div>);})()}

{tab==="trade"&&(<div style={{display:"flex",flexDirection:"column",gap:12}}>
<div style={{...card,display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:12,border:"1px solid "+(botOn?C.grn+"30":C.bd)}}>
<div><div style={{fontSize:14,fontWeight:700}}>Bot Automático</div><div style={{fontSize:11,color:C.tx2,marginTop:2}}>{botOn?"Analisando mercado e operando automaticamente":"Clique para ativar o bot"}</div></div>
<button onClick={handleToggle} style={{padding:"10px 24px",borderRadius:24,border:"none",background:botOn?C.red:C.grn,color:C.bg,fontSize:13,fontWeight:700,cursor:"pointer",fontFamily:"inherit",minWidth:140}}>{botOn?"⏹ Desligar Bot":"▶ Ligar Bot"}</button>
</div>

{botLog.length>0&&(<div style={{...card,maxHeight:280,overflowY:"auto"}}><div style={{...lbl,marginBottom:8}}>Log do Bot (tempo real)</div>{botLog.map(log=>(<div key={log.id} style={{display:"flex",gap:8,padding:"6px 0",borderBottom:"1px solid "+C.bd,fontSize:11}}><span style={{color:C.tx3,minWidth:60,fontFamily:"monospace"}}>{log.time}</span><span style={{width:8,height:8,borderRadius:"50%",background:logColors[log.type]||C.tx3,marginTop:4,flexShrink:0}}/><div style={{flex:1}}><div style={{color:logColors[log.type]||C.tx2,fontWeight:600}}>{log.msg}</div>{log.detail&&(<div style={{fontSize:10,color:C.tx3,marginTop:2}}>{log.detail}</div>)}</div></div>))}</div>)}

<div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(170px,1fr))",gap:12}}>
<MC label="Capital Disponível" value={fC(cash)} color={C.gold}/><MC label="Valor Total" value={fC(ptTot)}/><MC label="P&L Simulado" value={fP(ptPnl)} color={ptPnl>=0?C.grn:C.red}/><MC label="Operações" value={ords.length}/></div>

{!botOn&&(<div style={{...card,display:"flex",gap:10,alignItems:"flex-end",flexWrap:"wrap"}}><div style={{flex:1,minWidth:140}}><div style={lbl}>Ativo</div><select value={tSym} onChange={e=>setTSym(e.target.value)} style={inp}>{D.A.map(a=>(<option key={a.sym} value={a.sym}>{a.sym+" — "+a.name}</option>))}</select></div><div style={{width:100}}><div style={lbl}>Quantidade</div><input type="number" value={tQty} onChange={e=>setTQty(e.target.value)} placeholder="0" style={inp}/></div><button onClick={()=>{const q=parseFloat(tQty);if(!q)return setTMsg("Quantidade inválida");const err=doBuy(tSym,q);setTMsg(err||("✅ Compra: "+q+"x "+tSym));setTQty("");}} style={btnS(true,C.grn)}>Comprar</button><button onClick={()=>{const q=parseFloat(tQty);if(!q)return setTMsg("Quantidade inválida");const err=doSell(tSym,q);setTMsg(err||("✅ Venda: "+q+"x "+tSym));setTQty("");}} style={{...btnS(true,C.red),background:C.red}}>Vender</button></div>)}
{tMsg&&(<div style={{padding:"10px 14px",borderRadius:6,background:tMsg.includes("✅")?C.grn+"12":C.red+"12",fontSize:12,color:tMsg.includes("✅")?C.grn:C.red}}>{tMsg}</div>)}

{Object.keys(hold).length>0&&(<div style={card}><div style={{...lbl,marginBottom:8}}>Posições Abertas</div>{Object.entries(hold).map(([sym,h])=>{const cp=D.P[sym]?.[89]?.p||h.ap;const pnl=((cp-h.ap)/h.ap)*100;return(<div key={sym} style={{display:"flex",alignItems:"center",gap:12,padding:"10px 0",borderBottom:"1px solid "+C.bd,fontSize:12}}><span style={{fontWeight:700,width:60}}>{sym}</span><span style={{color:C.tx2}}>{h.q.toFixed(2)+" un"}</span><span style={{color:C.tx2}}>{"PM: "+fC(h.ap)}</span><span style={{color:pnl>=0?C.grn:C.red,fontWeight:700}}>{fP(pnl)}</span><span style={{marginLeft:"auto",fontWeight:700}}>{fC(h.q*cp)}</span></div>);})}</div>)}
{ords.length>0&&(<div style={card}><div style={{...lbl,marginBottom:8}}>Histórico</div>{ords.slice(0,15).map(o=>(<div key={o.id} style={{display:"flex",alignItems:"center",gap:8,padding:"8px 0",borderBottom:"1px solid "+C.bd,fontSize:11}}><span style={{color:o.ty==="COMPRA"?C.grn:C.red,fontWeight:700,width:65}}>{o.ty}</span><span style={{fontWeight:600,width:55}}>{o.sym}</span><span style={{color:C.tx2}}>{o.q.toFixed(2)+" x "+fC(o.pr)}</span>{o.auto&&(<span style={{fontSize:8,padding:"1px 5px",borderRadius:3,background:C.pu+"15",color:C.pu}}>AUTO</span>)}<span style={{marginLeft:"auto",fontWeight:700}}>{fC(o.tot)}</span></div>))}</div>)}
</div>)}

{tab==="accuracy"&&(<div style={{display:"flex",flexDirection:"column",gap:14}}><div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(160px,1fr))",gap:12}}><MC label="Taxa de Acerto" value={D.AC.hr+"%"} color={C.gold}/><MC label="Compras" value={D.AC.hb+"%"} color={C.grn}/><MC label="Vendas" value={D.AC.hs+"%"} color={C.red}/><MC label="Alta Confiança" value={D.AC.hh+"%"} color={C.cy}/><MC label="Com Sentimento" value={D.AC.hse+"%"} color={C.pu}/><MC label="Profit Factor" value={fm(D.AC.pf)} color={C.gold}/></div><div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}><div style={card}><div style={{...lbl,marginBottom:8}}>Matriz de Confusão</div><div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>{[{l:"Compra Certa",v:D.AC.cm.tb,c:C.grn},{l:"Compra Errada",v:D.AC.cm.fb,c:C.red},{l:"Venda Certa",v:D.AC.cm.ts,c:C.grn},{l:"Venda Errada",v:D.AC.cm.fs,c:C.red}].map((x,i)=>(<div key={i} style={{padding:12,background:x.c+"0a",borderRadius:6,textAlign:"center",border:"1px solid "+x.c+"20"}}><div style={{fontSize:10,color:x.c,marginBottom:4}}>{x.l}</div><div style={{fontSize:24,fontWeight:800,color:x.c}}>{x.v}</div></div>))}</div></div><div style={card}><div style={{...lbl,marginBottom:8}}>Por Mercado</div>{Object.entries(D.AC.mk).map(([mk,rate])=>(<div key={mk} style={{marginBottom:12}}><div style={{display:"flex",justifyContent:"space-between",fontSize:12,marginBottom:4}}><span style={{fontWeight:600}}>{mk}</span><span style={{color:rate>55?C.grn:C.gold,fontWeight:700}}>{fm(rate,1)+"%"}</span></div><div style={{height:7,background:C.bd,borderRadius:4,overflow:"hidden"}}><div style={{width:rate+"%",height:"100%",background:rate>55?C.grn:C.gold,borderRadius:4}}/></div></div>))}</div></div></div>)}

{tab==="memory"&&(<div style={{display:"flex",flexDirection:"column",gap:12}}><div style={{...card,borderLeft:"3px solid "+C.gold}}><div style={{fontSize:13,color:C.tx2,lineHeight:1.6}}>O bot registra eventos que impactaram decisões e compara com o histórico para identificar padrões recorrentes.</div></div><div style={{position:"relative",paddingLeft:20}}><div style={{position:"absolute",left:8,top:0,bottom:0,width:2,background:C.bd}}/>{D.MEM.map(m=>(<div key={m.id} style={{position:"relative",marginBottom:16}}><div style={{position:"absolute",left:-15,top:5,width:12,height:12,borderRadius:"50%",background:m.impact==="positivo"?C.grn:m.impact==="negativo"?C.red:C.gold,border:"2px solid "+C.bg}}/><div style={{...card,marginLeft:8}}><div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}><span style={{fontSize:10,color:C.tx3}}>{new Date(m.date).toLocaleDateString("pt-BR")}</span><span style={{fontSize:10,padding:"2px 6px",borderRadius:3,background:C.c2,color:C.tx3}}>{m.type}</span><SentBadge sentiment={m.impact}/></div><div style={{fontSize:13,fontWeight:600,marginBottom:6}}>{m.title}</div><div style={{fontSize:12,color:C.gold,background:C.gold+"10",padding:"8px 10px",borderRadius:6,borderLeft:"3px solid "+C.gold}}>{"🤖 "+m.action}</div></div></div>))}</div></div>)}

{tab==="risk"&&(<div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(170px,1fr))",gap:12}}>{[{l:"VaR 95%",v:fP(D.RK.v95),d:"Perda máxima em 95% dos dias",c:C.gold},{l:"VaR 99%",v:fP(D.RK.v99),d:"Perda máxima em 99% dos dias",c:C.red},{l:"CVaR 95%",v:fP(D.RK.cv95),d:"Perda média nos piores 5%",c:C.red},{l:"Beta",v:fm(D.RK.beta),d:"Sensibilidade ao mercado",c:C.cy},{l:"Alpha",v:fP(D.RK.alpha*100),d:"Retorno acima do benchmark",c:C.grn},{l:"Max Drawdown",v:fP(D.M.mDD),d:"Maior queda do pico",c:C.red},{l:"Sortino",v:fm(D.M.so),d:"Retorno / risco downside",c:C.pu},{l:"Profit Factor",v:fm(D.M.pf),d:"Ganho total / Perda total",c:C.gold}].map((x,i)=>(<div key={i} style={card}><div style={lbl}>{x.l}</div><div style={{fontSize:22,fontWeight:800,color:x.c,margin:"4px 0"}}>{x.v}</div><div style={{fontSize:10,color:C.tx3}}>{x.d}</div></div>))}</div>)}

</div>
<style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:#000}::-webkit-scrollbar-thumb{background:#1c1c1c;border-radius:4px}select option{background:#0c0c0c;color:#eaecef}input:focus,select:focus{border-color:#f0b90b50}*{box-sizing:border-box}strong{color:#f0b90b}`}</style>
</div>);
}
