"""
Build a single-file, offline interactive dashboard
==================================================
Reads output/results/*.json and emits dashboard/index.html with the result data
embedded (no server, no external dependencies) and lightweight inline-SVG charts.

Run after the experiments + compare. Usage:
    python -m visualization.build_dashboard
"""

import json
import os

from experiments import config

ROOT = config.ROOT
RESULTS = config.RESULTS_DIR
OUT_DIR = os.path.join(ROOT, "dashboard")


def _load(name, default=None):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def collect():
    return {
        "p2_exact": _load("p2_exact.json"),
        "p2_meta": _load("p2_metaheuristics.json"),
        "p2_sweep": _load("p2_alpha_sweep.json"),
        "p2_solution": _load("p2_solution.json"),
        "p1_reduced": _load("p1_reduced.json"),
        "p1_full": _load("p1_full_metaheuristics.json"),
        "p1_solution": _load("p1_solution.json"),
        "p3_results": _load("p3_results.json"),
        "p3_solution": _load("p3_solution.json"),
        "comparison": _load("comparison.json"),
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OR Optimization Dashboard</title>
<style>
  :root{ --bg:#0f172a; --card:#1e293b; --ink:#e2e8f0; --mut:#94a3b8;
         --accent:#38bdf8; --green:#34d399; --red:#f87171; --amber:#fbbf24; }
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
  header{padding:28px 32px;border-bottom:1px solid #334155}
  h1{margin:0;font-size:22px} .sub{color:var(--mut);margin-top:6px;font-size:14px}
  .tabs{display:flex;gap:8px;padding:16px 32px 0}
  .tab{padding:10px 18px;border-radius:10px 10px 0 0;background:#172033;color:var(--mut);
       cursor:pointer;border:1px solid transparent;border-bottom:none;font-weight:600}
  .tab.active{background:var(--card);color:var(--ink);border-color:#334155}
  main{padding:24px 32px;max-width:1100px}
  .panel{display:none} .panel.active{display:block}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin:18px 0}
  .card{background:var(--card);border:1px solid #334155;border-radius:14px;padding:18px}
  .kpi{font-size:26px;font-weight:700} .kpi.green{color:var(--green)} .kpi.red{color:var(--red)}
  .kpi.accent{color:var(--accent)}
  .lbl{color:var(--mut);font-size:13px;margin-top:4px}
  table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}
  th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #334155}
  th{color:var(--mut);font-weight:600} .num{text-align:right;font-variant-numeric:tabular-nums}
  h2{font-size:17px;margin:26px 0 6px} .desc{color:var(--mut);font-size:13px;max-width:760px}
  svg{background:#0b1220;border-radius:12px;border:1px solid #334155}
  .foot{color:var(--mut);font-size:12px;padding:18px 32px;border-top:1px solid #334155}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}
  .pill.ok{background:#064e3b;color:#6ee7b7}.pill.warn{background:#78350f;color:#fcd34d}
</style>
</head>
<body>
<header>
  <h1>Facility &amp; Infrastructure Optimization &mdash; Renewable-Energy Company</h1>
  <div class="sub">Exact MILP vs metaheuristics across three OR problems &middot; offline dashboard</div>
</header>
<div class="tabs" id="tabs"></div>
<main id="main"></main>
<div class="foot">Generated from <code>output/results/*.json</code> &middot; all figures reproducible via the seeded runners.</div>

<script>
const DATA = __DATA__;
const fmt = (x,d=0)=> (x==null||isNaN(x))?'&mdash;':Number(x).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d});

function el(tag, attrs={}, html=''){ const e=document.createElement(tag);
  for(const k in attrs) e.setAttribute(k,attrs[k]); if(html) e.innerHTML=html; return e; }

// --- tiny inline-SVG charts ---
function barChart(labels, values, opts={}){
  const W=opts.w||520,H=opts.h||240,P=40, n=values.length;
  const max=Math.max(...values, opts.ref||0)*1.1, bw=(W-2*P)/n*0.6, gap=(W-2*P)/n;
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  if(opts.ref){ const y=H-P-(opts.ref/max)*(H-2*P);
    s+=`<line x1="${P}" y1="${y}" x2="${W-P}" y2="${y}" stroke="#f87171" stroke-dasharray="5 4"/>`;
    s+=`<text x="${W-P}" y="${y-5}" fill="#f87171" font-size="11" text-anchor="end">optimum</text>`; }
  values.forEach((v,i)=>{ const h=(v/max)*(H-2*P), x=P+gap*i+(gap-bw)/2, y=H-P-h;
    const c=opts.colors?opts.colors[i]:'#38bdf8';
    s+=`<rect x="${x}" y="${y}" width="${bw}" height="${h}" rx="4" fill="${c}"/>`;
    s+=`<text x="${x+bw/2}" y="${H-P+16}" fill="#94a3b8" font-size="11" text-anchor="middle">${labels[i]}</text>`;
    s+=`<text x="${x+bw/2}" y="${y-6}" fill="#e2e8f0" font-size="11" text-anchor="middle">${fmt(v)}</text>`; });
  return s+`</svg>`;
}
function lineChart(series, opts={}){
  const W=opts.w||520,H=opts.h||240,P=42;
  const all=series.flatMap(s=>s.y); const ymin=Math.min(...all),ymax=Math.max(...all);
  const n=Math.max(...series.map(s=>s.y.length));
  const sx=i=>P+(i/(n-1))*(W-2*P), sy=v=>H-P-((v-ymin)/(ymax-ymin||1))*(H-2*P);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  if(opts.ref!=null){const y=sy(opts.ref);s+=`<line x1="${P}" y1="${y}" x2="${W-P}" y2="${y}" stroke="#f87171" stroke-dasharray="5 4"/>`;}
  series.forEach(se=>{ let d=se.y.map((v,i)=>`${i?'L':'M'}${sx(i*(n-1)/(se.y.length-1)).toFixed(1)} ${sy(v).toFixed(1)}`).join(' ');
    s+=`<path d="${d}" fill="none" stroke="${se.c}" stroke-width="2"/>`; });
  // legend
  series.forEach((se,i)=>{ s+=`<rect x="${P+i*120}" y="12" width="12" height="12" fill="${se.c}"/>
    <text x="${P+i*120+18}" y="22" fill="#94a3b8" font-size="11">${se.name}</text>`; });
  s+=`<text x="${P}" y="${H-8}" fill="#94a3b8" font-size="11">search progress &rarr;</text>`;
  return s+`</svg>`;
}

// --- panels ---
function p2Panel(){
  const d=DATA.p2_exact, m=DATA.p2_meta, sw=DATA.p2_sweep;
  let h=`<div class="grid">
    <div class="card"><div class="kpi accent">${fmt(d&&d.objective)}</div><div class="lbl">exact optimum (CBC)</div></div>
    <div class="card"><div class="kpi">${d?d.n_open:'&mdash;'}</div><div class="lbl">substations built</div></div>
    <div class="card"><div class="kpi green">+${m?fmt(m.GA.gap_to_exact_pct,2):'&mdash;'}%</div><div class="lbl">GA gap to optimum</div></div>
    <div class="card"><div class="kpi red">+${m?fmt(m.PSO.gap_to_exact_pct,2):'&mdash;'}%</div><div class="lbl">PSO gap to optimum</div></div></div>`;
  if(m){ const labels=['exact','GA','SA','PSO'];
    const vals=[d.objective,m.GA.best_cost,m.SA.best_cost,m.PSO.best_cost];
    h+=`<h2>Best objective by method</h2>`+barChart(labels,vals,{ref:d.objective,
      colors:['#1e293b','#38bdf8','#34d399','#fbbf24']});
    const series=['GA','SA','PSO'].map((k,i)=>({name:k,y:m[k].best_history,c:['#38bdf8','#34d399','#fbbf24'][i]}));
    h+=`<h2>Convergence (best seed)</h2>`+lineChart(series,{ref:d.objective}); }
  if(sw){ h+=`<h2>Transmission-rate sweep</h2><div class="desc">As the transmission rate &alpha; rises, the optimal number of substations grows &mdash; the classic facility-location trade-off.</div>`;
    h+=barChart(sw.map(r=>r.alpha_mult+'&times;'), sw.map(r=>r.n_open), {colors:sw.map(()=> '#38bdf8')});
    h+=`<table><tr><th>&alpha; mult</th><th class="num">substations</th><th class="num">construction</th><th class="num">transmission</th><th>status</th></tr>`;
    sw.forEach(r=>{h+=`<tr><td>${r.alpha_mult}&times;</td><td class="num">${r.n_open}</td><td class="num">${fmt(r.construction)}</td><td class="num">${fmt(r.transmission)}</td><td><span class="pill ${r.feasible?'ok':'warn'}">${r.status}</span></td></tr>`;});
    h+=`</table>`; }
  return h;
}
function p1Panel(){
  const r=DATA.p1_reduced, f=DATA.p1_full, s=DATA.p1_solution;
  let h=`<div class="grid">
    <div class="card"><div class="kpi accent">${r?fmt(r.exact.objective):'&mdash;'}</div><div class="lbl">reduced ${r?r.K+'&times;'+r.K:''} exact optimum</div></div>
    <div class="card"><div class="kpi">${s?fmt(s.objective):'&mdash;'}</div><div class="lbl">best full layout (${s?s.method:''})</div></div>
    <div class="card"><div class="kpi green">${f?fmt(f.SA.best_cost):'&mdash;'}</div><div class="lbl">SA best (full)</div></div>
    <div class="card"><div class="kpi amber">${f?fmt(f.GA.best_cost):'&mdash;'}</div><div class="lbl">GA best (full)</div></div></div>`;
  if(f){ const series=['GA','SA'].map((k,i)=>({name:k,y:f[k].best_history,c:['#fbbf24','#34d399'][i]}));
    h+=`<h2>Full QAP convergence (30&rarr;197)</h2><div class="desc">Simulated annealing typically outperforms the GA on QAP.</div>`+lineChart(series); }
  if(r){ h+=`<h2>Reduced instance: exact vs heuristic</h2>`+barChart(['exact','GA','SA'],
      [r.exact.objective,r.metaheuristics.GA.best_cost,r.metaheuristics.SA.best_cost],
      {ref:r.exact.objective,colors:['#1e293b','#38bdf8','#34d399']}); }
  return h;
}
function p3Panel(){
  const r=DATA.p3_results, s=DATA.p3_solution;
  let h=`<div class="grid">
    <div class="card"><div class="kpi green">${r?fmt(r.coi_optimal):'&mdash;'}</div><div class="lbl">COI optimal travel</div></div>
    <div class="card"><div class="kpi accent">${r?fmt(100*(1-r.coi_optimal/r.random_mean),1):'&mdash;'}%</div><div class="lbl">saved vs random</div></div>
    <div class="card"><div class="kpi">${s?fmt(s.total_slots):'&mdash;'}</div><div class="lbl">storage slots</div></div>
    <div class="card"><div class="kpi">${r&&r.exact_lp.matches?'&#10003;':'&mdash;'}</div><div class="lbl">LP verifies COI optimal</div></div></div>`;
  if(r){ h+=`<h2>COI vs random layout</h2>`+barChart(['random (mean)','COI (optimal)'],
      [r.random_mean,r.coi_optimal],{colors:['#94a3b8','#34d399']}); }
  if(s){ const ord=s.order; h+=`<h2>Optimal storage ordering (nearest first)</h2><table>
    <tr><th>rank</th><th>product</th><th class="num">throughput</th><th class="num">storage</th><th class="num">COI</th></tr>`;
    ord.forEach((p,i)=>{ h+=`<tr><td>${i+1}</td><td>${s.product_names[p]}</td><td class="num">${fmt(s.throughput[p])}</td><td class="num">${fmt(s.storage[p])}</td><td class="num">${fmt(s.coi[p],3)}</td></tr>`; });
    h+=`</table>`; }
  return h;
}

const PANELS=[['P2 &middot; Substations',p2Panel],['P1 &middot; Layout',p1Panel],['P3 &middot; Storage',p3Panel]];
const tabs=document.getElementById('tabs'), main=document.getElementById('main');
PANELS.forEach(([name,fn],i)=>{
  const t=el('div',{class:'tab'+(i===0?' active':'')},name); tabs.appendChild(t);
  const p=el('div',{class:'panel'+(i===0?' active':'')}); p.innerHTML=fn(); main.appendChild(p);
  t.onclick=()=>{ document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
    t.classList.add('active'); main.children[i].classList.add('active'); };
});
</script>
</body>
</html>
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = collect()
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    out = os.path.join(OUT_DIR, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[dashboard] wrote {out} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
