#!/usr/bin/env python3
"""
build_config_site.py — mini-site statique d'exploration PAR CONFIG.

Généralise scripts/build_umap_site.sh (UMAP seule, arbo V5.4.1 câblée en dur) :
découvre les configs sous une racine quelconque et produit, pour chacune, une
page à onglets :

  * Ranking  — cross_seed_gene_ranking.tsv en table triable / filtrable
               (virtualisée, ~13k gènes OK), sélecteur de colonnes, détail par
               gène, export CSV de la vue courante.
  * UMAP     — umap_interactive.html en iframe si présente, sinon les PNG
               umap_*.png + la commande pour générer l'interactive.
  * Figures  — galerie de toutes les PNG de la config (analysis/, s*/figure/…).
  * Infos    — run_config.json, SUMMARY.md, liens vers les TSV bruts.

Plus deux pages transverses :
  * index.html  — liste des configs (+ figures globales de la version).
  * genes.html  — un gène → son rang / driver_score dans TOUTES les configs
                  (lecture directe de la sensibilité au graphe = circularité).

Aucune donnée n'est copiée : les PNG / TSV sont référencés en chemin relatif
depuis --out. Seul le ranking principal est converti en payload JS compact
(colonnaire + dictionnaires de chaînes) pour rester utilisable en file://.

Usage
-----
    # site de toute une version
    python scripts/build_config_site.py --root output/gnn_vgae/V6.1.3

    # + servir en local
    python scripts/build_config_site.py --root output/gnn_vgae/V6.1.3 --serve

    # arbo V5.4.1 (cross_seed/<cfg>/) — même commande, découverte auto
    python scripts/build_config_site.py \\
        --root output/interpretation/V5.4.1/cross_seed --out /tmp/site_v541
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RANKING_NAME = "cross_seed_gene_ranking.tsv"
# Sous-dossiers candidats pour le ranking « principal » d'une config, par ordre
# de préférence. Le premier trouvé gagne ; les autres deviennent des variantes.
PRIMARY_SUBDIRS = ("analysis", "xseed", "cross_seed_report", "report_axisV4", "")

# Colonnes affichées par défaut (les autres restent chargées, masquées, et
# réactivables via le sélecteur de colonnes).
DEFAULT_COLS = [
    "target", "driver_score", "discovery_score", "validation_score",
    "evidence_tier", "direction", "canon_diff", "canon_cosine",
    "canon_amplitude", "n_modes_present", "mean_robustness", "sign_consistent",
    "is_hub_inflated", "target_ppi_degree", "senescence_specificity",
    "vgae_rank", "is_de_significant", "de_log2fc_p4_vs_p16", "n_aging_dbs",
    "is_tf", "marker_driver_conflict",
]
# Colonnes longues → jamais dans la table, uniquement dans le panneau détail.
DETAIL_ONLY = ("interpretation", "member_of_strong_pathways")


# --------------------------------------------------------------------------- #
# Découverte des configs
# --------------------------------------------------------------------------- #
def discover_configs(root: Path, max_depth: int = 4) -> list[dict]:
    """Trouve les configs = dossiers contenant (à ≤max_depth) un ranking.

    Retourne une liste de dicts {name, dir, ranking, variants}. Le `dir` de la
    config est le parent du dossier de rapport (analysis/, xseed/…), c.-à-d. la
    racine du run — celle qui porte aussi build/, s1/, logs/.
    """
    hits: dict[Path, Path] = {}          # config_dir -> ranking principal
    for rk in sorted(root.rglob(RANKING_NAME)):
        try:
            depth = len(rk.relative_to(root).parts)
        except ValueError:
            continue
        if depth > max_depth:
            continue
        report_dir = rk.parent
        # On remonte d'un cran si le rapport est dans un sous-dossier connu.
        cfg_dir = (report_dir.parent
                   if report_dir.name in PRIMARY_SUBDIRS[:-1]
                   or report_dir.name.startswith("xseed")
                   else report_dir)
        prev = hits.get(cfg_dir)
        if prev is None or _rank_priority(rk) < _rank_priority(prev):
            hits[cfg_dir] = rk

    configs = []
    for cfg_dir, rk in sorted(hits.items()):
        rel = cfg_dir.relative_to(root) if cfg_dir != root else Path(cfg_dir.name)
        name = str(rel).replace(os.sep, "/")
        variants = sorted(
            p for p in rk.parent.glob("cross_seed_gene_ranking*.tsv") if p != rk)
        variants += sorted(rk.parent.glob("*/cross_seed_gene_ranking__*.tsv"))
        configs.append({"name": name, "dir": cfg_dir, "ranking": rk,
                        "variants": variants})
    return configs


def _rank_priority(p: Path) -> int:
    """Plus petit = préféré (analysis/ avant xseed/ avant le reste)."""
    try:
        return PRIMARY_SUBDIRS.index(p.parent.name)
    except ValueError:
        return len(PRIMARY_SUBDIRS)


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", name)


def collect_assets(cfg: dict, root: Path) -> dict:
    """PNG, UMAP interactive, configs et rapports d'une config."""
    d: Path = cfg["dir"]
    pngs = sorted(p for p in d.rglob("*.png") if "_umap_site" not in p.parts)
    umap_html = next(iter(sorted(d.rglob("umap_interactive.html"))), None)
    run_cfg = next(iter(sorted(d.rglob("run_config.json"))), None)
    summary = next(iter(sorted(d.rglob("SUMMARY.md"))), None)
    tsvs = sorted(p for p in d.rglob("*.tsv")
                  if p.parent == cfg["ranking"].parent)
    return {"pngs": pngs, "umap_html": umap_html, "run_config": run_cfg,
            "summary": summary, "tsvs": tsvs}


# --------------------------------------------------------------------------- #
# Payload JS compact (colonnaire + interning des chaînes)
# --------------------------------------------------------------------------- #
def encode_table(df: pd.DataFrame) -> dict:
    """DataFrame → dict colonnaire JSON-able, chaînes internées.

    Format : {"cols":[...], "n":int, "data":[<col>,...]} où <col> est soit une
    liste de nombres/null, soit {"d":[valeurs uniques], "c":[codes]}.
    """
    cols, data = [], []
    for c in df.columns:
        s = df[c]
        cols.append(str(c))
        if pd.api.types.is_bool_dtype(s):
            data.append([None if pd.isna(v) else bool(v) for v in s])
        elif pd.api.types.is_numeric_dtype(s):
            vals = []
            for v in s:
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    vals.append(None)
                else:
                    f = float(v)
                    vals.append(int(f) if f.is_integer() and abs(f) < 1e15
                                else round(f, 4))
            data.append(vals)
        else:
            uniq: dict[str, int] = {}
            codes = []
            for v in s:
                v = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
                if v not in uniq:
                    uniq[v] = len(uniq)
                codes.append(uniq[v])
            data.append({"d": list(uniq.keys()), "c": codes})
    return {"cols": cols, "n": int(len(df)), "data": data}


def load_ranking(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # sign_consistent est écrit "" quand indéterminé → on garde en chaîne.
    return df


# --------------------------------------------------------------------------- #
# Gabarits
# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#fff;--fg:#1b1b1b;--mut:#6b6b6b;--line:#e2e2e2;--acc:#2166ac;
 --acc-bg:#eaf2fa;--head:#f6f7f9;--warn:#b2182b;}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--mut:#9aa0a6;
 --line:#2c3038;--acc:#7fb3e3;--acc-bg:#1d2a38;--head:#1e2126;--warn:#e58a95;}}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;margin:0;background:var(--bg);
 color:var(--fg);font-size:14px}
a{color:var(--acc)}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;
 gap:14px;align-items:baseline;flex-wrap:wrap;position:sticky;top:0;
 background:var(--bg);z-index:20}
header h1{font-size:16px;margin:0;font-weight:600}
header .sub{color:var(--mut);font-size:12px}
nav.tabs{display:flex;gap:2px;padding:0 16px;border-bottom:1px solid var(--line);
 background:var(--bg);position:sticky;top:41px;z-index:19;flex-wrap:wrap}
nav.tabs button{background:none;border:0;border-bottom:2px solid transparent;
 padding:8px 12px;font:inherit;color:var(--mut);cursor:pointer}
nav.tabs button.on{color:var(--acc);border-bottom-color:var(--acc);font-weight:600}
main{padding:12px 16px}
.panel{display:none}.panel.on{display:block}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
input,select,button.btn{padding:5px 8px;font:inherit;border:1px solid var(--line);
 border-radius:5px;background:var(--bg);color:var(--fg)}
button.btn{cursor:pointer}
button.btn:hover{background:var(--acc-bg)}
.muted{color:var(--mut);font-size:12px}
#tw{border:1px solid var(--line);border-radius:6px;overflow:auto;max-height:72vh;
 position:relative}
table.grid{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%}
table.grid th{position:sticky;top:0;background:var(--head);z-index:2;
 border-bottom:1px solid var(--line);padding:6px 9px;text-align:left;
 white-space:nowrap;cursor:pointer;font-weight:600;font-size:12px}
table.grid th:hover{color:var(--acc)}
table.grid td{padding:4px 9px;border-bottom:1px solid var(--line);
 white-space:nowrap;font-variant-numeric:tabular-nums}
table.grid tr:hover td{background:var(--acc-bg)}
table.grid td.g{font-weight:600}
.tier-A_confirmed{color:#1a7f37}.tier-B_discovery{color:#2166ac}
.tier-D_hub{color:var(--warn)}.tier-E_noise{color:var(--mut)}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.gal figure{margin:0;border:1px solid var(--line);border-radius:6px;padding:8px}
.gal figcaption{font-size:12px;color:var(--mut);margin-bottom:6px;word-break:break-all}
.gal img{width:100%;height:auto;cursor:zoom-in;background:#fff}
iframe.umap{width:100%;height:80vh;border:1px solid var(--line);border-radius:6px}
pre{background:var(--head);padding:10px;border-radius:6px;overflow:auto;
 font-size:12px;max-height:60vh}
#detail{position:fixed;right:0;top:0;bottom:0;width:min(420px,92vw);
 background:var(--bg);border-left:1px solid var(--line);padding:14px;
 overflow:auto;transform:translateX(100%);transition:transform .15s;z-index:40}
#detail.on{transform:none;box-shadow:-8px 0 24px rgba(0,0,0,.12)}
#detail dl{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:12px}
#detail dt{color:var(--mut)}#detail dd{margin:0;word-break:break-word}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;z-index:50;
 align-items:center;justify-content:center;cursor:zoom-out}
#lb img{max-width:96vw;max-height:96vh}
ul.cfgs{columns:2;list-style:none;padding:0}
ul.cfgs li{margin:3px 0;break-inside:avoid}
.cols{display:none;border:1px solid var(--line);border-radius:6px;padding:8px;
 margin-bottom:8px;max-height:180px;overflow:auto;columns:4}
.cols.on{display:block}
.cols label{display:block;font-size:12px;break-inside:avoid}
"""

APP_JS = r"""
/* Table virtualisée : décodage colonnaire, tri, filtres, détail, export. */
(function(){
var ROW_H=25, PAD=8;
window.SiteTable = function(mount, payload, opts){
  opts = opts || {};
  var cols = payload.cols, N = payload.n;
  var get = payload.data.map(function(col){
    if(col && col.d){ return function(i){ var v=col.d[col.c[i]]; return v===""?null:v; }; }
    return function(i){ return col[i]; };
  });
  var idx = new Int32Array(N); for(var i=0;i<N;i++) idx[i]=i;
  var view = idx, sortCol=-1, sortDir=-1, visible = (opts.visible||cols).filter(function(c){return cols.indexOf(c)>=0;});
  var detailCols = opts.detailCols||[];
  var wrap=document.createElement('div'); wrap.id='tw';
  var spacerTop=document.createElement('div'), spacerBot=document.createElement('div');
  var tbl=document.createElement('table'); tbl.className='grid';
  var thead=document.createElement('thead'), tbody=document.createElement('tbody');
  tbl.appendChild(thead); tbl.appendChild(tbody);
  wrap.appendChild(spacerTop); wrap.appendChild(tbl); wrap.appendChild(spacerBot);
  mount.appendChild(wrap);

  function ci(c){ return cols.indexOf(c); }
  function val(c,i){ var k=ci(c); return k<0?null:get[k](i); }
  function fmt(v){
    if(v===null||v===undefined) return '';
    if(typeof v==='number') return Number.isInteger(v)?v:v.toFixed(Math.abs(v)<1?3:2);
    return String(v);
  }
  function head(){
    var tr=document.createElement('tr');
    visible.forEach(function(c){
      var th=document.createElement('th');
      th.textContent=c+(sortCol===ci(c)?(sortDir<0?' ▼':' ▲'):'');
      th.onclick=function(){ sort(ci(c)); };
      tr.appendChild(th);
    });
    thead.innerHTML=''; thead.appendChild(tr);
  }
  function sort(k){
    if(k<0) return;
    sortDir = (k===sortCol) ? -sortDir : -1;
    sortCol = k;
    var f=get[k], arr=Array.prototype.slice.call(view);
    arr.sort(function(a,b){
      var x=f(a), y=f(b);
      var xe=(x===null||x===undefined||x===''), ye=(y===null||y===undefined||y==='');
      if(xe&&ye) return 0;
      if(xe) return 1;              /* vides toujours en bas */
      if(ye) return -1;
      if(typeof x==='string'||typeof y==='string'){ x=String(x); y=String(y); }
      /* sortDir=-1 → décroissant (1er clic : on veut les gros scores en haut) */
      return x<y?-sortDir:(x>y?sortDir:0);
    });
    view=arr; head(); render();
  }
  function render(){
    var st=wrap.scrollTop, h=wrap.clientHeight;
    var first=Math.max(0,Math.floor(st/ROW_H)-PAD);
    var last=Math.min(view.length,Math.ceil((st+h)/ROW_H)+PAD);
    spacerTop.style.height=(first*ROW_H)+'px';
    spacerBot.style.height=Math.max(0,(view.length-last)*ROW_H)+'px';
    var html='';
    for(var r=first;r<last;r++){
      var i=view[r];
      html+='<tr data-i="'+i+'">';
      for(var c=0;c<visible.length;c++){
        var v=val(visible[c],i), cls='';
        if(visible[c]==='target') cls=' class="g"';
        else if(visible[c]==='evidence_tier'&&v) cls=' class="tier-'+v+'"';
        html+='<td'+cls+'>'+fmt(v)+'</td>';
      }
      html+='</tr>';
    }
    tbody.innerHTML=html;
    if(opts.onCount) opts.onCount(view.length, N);
  }
  tbody.onclick=function(e){
    var tr=e.target.closest('tr'); if(!tr) return; showDetail(+tr.dataset.i);
  };
  function showDetail(i){
    var d=document.getElementById('detail'); if(!d) return;
    var h='<div class="bar"><b>'+fmt(val('target',i))+'</b>'+
      '<a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene='+
      encodeURIComponent(fmt(val('target',i)))+'" target="_blank" rel="noopener">GeneCards</a>'+
      '<button class="btn" onclick="document.getElementById(\'detail\').classList.remove(\'on\')">Fermer</button></div><dl>';
    cols.forEach(function(c){
      var v=get[ci(c)](i);
      if(v===null||v===''||v===undefined) return;
      h+='<dt>'+c+'</dt><dd>'+fmt(v)+'</dd>';
    });
    d.innerHTML=h+'</dl>'; d.classList.add('on');
  }
  wrap.addEventListener('scroll',render);

  var api={
    filter:function(q, extra){
      q=(q||'').trim().toLowerCase();
      var terms=q?q.split(/[\s,;]+/).filter(Boolean):[];
      var tcol=ci('target'), icol=ci('interpretation');
      var out=[];
      for(var r=0;r<idx.length;r++){
        var i=idx[r];
        if(terms.length){
          var g=String(get[tcol](i)||'').toLowerCase();
          var it=icol>=0?String(get[icol](i)||'').toLowerCase():'';
          var ok=false;
          for(var t=0;t<terms.length;t++){
            if(g.indexOf(terms[t])>=0||it.indexOf(terms[t])>=0){ok=true;break;}
          }
          if(!ok) continue;
        }
        if(extra && !extra(function(c){return val(c,i);})) continue;
        out.push(i);
      }
      view=out;
      if(sortCol>=0){ var k=sortCol; sortCol=-1; sortDir=-1; sort(k); }
      else { wrap.scrollTop=0; render(); }
    },
    setVisible:function(list){ visible=list; head(); render(); },
    cols:cols, visible:function(){return visible;},
    exportCSV:function(name){
      var lines=[visible.join(',')];
      for(var r=0;r<view.length;r++){
        var i=view[r];
        lines.push(visible.map(function(c){
          var v=val(c,i); v=(v===null||v===undefined)?'':String(v);
          return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;
        }).join(','));
      }
      var b=new Blob([lines.join('\n')],{type:'text/csv'});
      var a=document.createElement('a'); a.href=URL.createObjectURL(b);
      a.download=name||'view.csv'; a.click();
    },
    sortBy:function(c){ sort(ci(c)); }
  };
  head(); render();
  return api;
};

window.initTabs=function(){
  var btns=document.querySelectorAll('nav.tabs button');
  btns.forEach(function(b){
    b.onclick=function(){
      btns.forEach(function(x){x.classList.remove('on');});
      document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('on');});
      b.classList.add('on');
      var p=document.getElementById('panel-'+b.dataset.tab);
      if(p) p.classList.add('on');
      if(location.hash.slice(1)!==b.dataset.tab) history.replaceState(null,'','#'+b.dataset.tab);
    };
  });
  var want=location.hash.slice(1);
  var start=document.querySelector('nav.tabs button[data-tab="'+want+'"]')||btns[0];
  if(start) start.click();
  var lb=document.getElementById('lb');
  document.querySelectorAll('.gal img').forEach(function(im){
    im.onclick=function(){ lb.style.display='flex';
      lb.querySelector('img').src=im.src; };
  });
  if(lb) lb.onclick=function(){ lb.style.display='none'; };
};
})();
"""


def page(title: str, body: str, depth: int = 1, extra_head: str = "") -> str:
    up = "../" * depth
    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{up}assets/style.css">
<script src="{up}assets/app.js"></script>{extra_head}
</head><body>
{body}
<div id="detail"></div><div id="lb"><img alt=""></div>
</body></html>"""


def rel(target: Path, from_dir: Path) -> str:
    return os.path.relpath(target, from_dir).replace(os.sep, "/")


# --------------------------------------------------------------------------- #
# Génération
# --------------------------------------------------------------------------- #
def build_config_page(cfg: dict, assets: dict, out: Path, version: str,
                      all_cols: bool) -> dict:
    """Écrit cfg/<slug>.html + data/<slug>.js. Retourne un résumé pour l'index."""
    name = cfg["name"]
    sl = slug(name)
    cfg_dir_out = out / "cfg"
    data_dir = out / "data"
    cfg_dir_out.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    df = load_ranking(cfg["ranking"])
    # Le payload porte TOUTES les colonnes (le panneau détail les affiche) ;
    # DETAIL_ONLY est seulement exclu du sélecteur de colonnes de la table.
    payload = encode_table(df)
    (data_dir / f"{sl}.js").write_text(
        "window.PAYLOAD=" + json.dumps(payload, separators=(",", ":")) + ";",
        encoding="utf-8")

    visible = ([c for c in df.columns if c not in DETAIL_ONLY] if all_cols
               else [c for c in DEFAULT_COLS if c in df.columns])
    pickable = [c for c in df.columns if c not in DETAIL_ONLY]

    # --- onglet Ranking --------------------------------------------------- #
    checks = "".join(
        f'<label><input type="checkbox" value="{c}"'
        f'{" checked" if c in visible else ""}> {c}</label>'
        for c in pickable)
    tiers = sorted(str(v) for v in df.get("evidence_tier", pd.Series(dtype=str)
                                          ).dropna().unique())
    tier_opts = "".join(f'<option value="{t}">{t}</option>' for t in tiers)
    rank_rel = rel(cfg["ranking"], cfg_dir_out)
    variants = "".join(
        f'<li><a href="{rel(v, cfg_dir_out)}">{v.name}</a></li>'
        for v in cfg["variants"][:40])

    panel_rank = f"""<div class="bar">
  <input id="q" placeholder="gène(s) — ex. HMGB2, SYNJ2" size="26">
  <select id="ftier"><option value="">tous tiers</option>{tier_opts}</select>
  <select id="fdir"><option value="">toute direction</option>
    <option value="pro">pro-senescence</option>
    <option value="anti">anti-senescence</option></select>
  <label class="muted"><input type="checkbox" id="fnohub"> exclure hub-inflated</label>
  <label class="muted">driver_score &ge; <input id="fds" type="number" step="0.05"
      min="0" max="1" style="width:70px"></label>
  <button class="btn" onclick="document.querySelector('.cols').classList.toggle('on')">Colonnes</button>
  <button class="btn" onclick="T.exportCSV('{sl}_view.csv')">Export CSV</button>
  <span class="muted" id="cnt"></span>
</div>
<div class="cols">{checks}</div>
<div id="mount"></div>
<p class="muted">Clic sur une ligne = détail complet (toutes colonnes + GeneCards).
Tri par clic sur l'en-tête. Source :
<a href="{rank_rel}">{cfg['ranking'].name}</a>
{'<br>Variantes : <ul>' + variants + '</ul>' if variants else ''}</p>"""

    # --- onglet UMAP ------------------------------------------------------- #
    if assets["umap_html"]:
        u = rel(assets["umap_html"], cfg_dir_out)
        panel_umap = (f'<p class="muted"><a href="{u}" target="_blank">Ouvrir '
                      f'en plein écran</a></p><iframe class="umap" src="{u}"></iframe>')
    else:
        umap_pngs = [p for p in assets["pngs"] if p.name.startswith("umap_")]
        gal = "".join(
            f'<figure><figcaption>{p.name}</figcaption>'
            f'<img loading="lazy" src="{rel(p, cfg_dir_out)}" alt="{p.name}"></figure>'
            for p in umap_pngs)
        rundir = rel(cfg["dir"], Path.cwd()) if cfg["dir"].is_absolute() else str(cfg["dir"])
        panel_umap = (
            f'<p class="muted">Pas d\'UMAP interactive pour cette config. '
            f'Pour la générer :</p><pre>python src/validation/viz/interpret_embedding.py \\\n'
            f'    --run-dir {rundir}/s1 \\\n'
            f'    --ranking {cfg["ranking"]} \\\n'
            f'    --out-dir {cfg["ranking"].parent}/interpret \\\n'
            f'    --umap-only --plotly-cdn --reuse-umap</pre>'
            f'<div class="gal">{gal}</div>' if umap_pngs else
            f'<p class="muted">Ni UMAP interactive ni PNG umap_* trouvés.</p>')

    # --- onglet Figures ---------------------------------------------------- #
    others = [p for p in assets["pngs"] if not p.name.startswith("umap_")]
    groups: dict[str, list[Path]] = {}
    for p in others:
        groups.setdefault(str(p.parent.relative_to(cfg["dir"])), []).append(p)
    figs = ""
    for grp, ps in sorted(groups.items()):
        gal = "".join(
            f'<figure><figcaption>{p.name}</figcaption>'
            f'<img loading="lazy" src="{rel(p, cfg_dir_out)}" alt="{p.name}"></figure>'
            for p in ps)
        figs += f'<h3>{grp or "."} <span class="muted">({len(ps)})</span></h3><div class="gal">{gal}</div>'
    panel_figs = figs or '<p class="muted">Aucune figure.</p>'

    # --- onglet Infos ------------------------------------------------------ #
    infos = ""
    if assets["run_config"]:
        try:
            j = json.loads(assets["run_config"].read_text())
            infos += f"<h3>run_config.json</h3><pre>{json.dumps(j, indent=2)[:20000]}</pre>"
        except Exception:
            pass
    if assets["summary"]:
        txt = assets["summary"].read_text(errors="replace")[:40000]
        infos += (f'<h3>SUMMARY.md <a class="muted" href="'
                  f'{rel(assets["summary"], cfg_dir_out)}">(brut)</a></h3><pre>{_esc(txt)}</pre>')
    tsv_links = "".join(f'<li><a href="{rel(t, cfg_dir_out)}">{t.name}</a></li>'
                        for t in assets["tsvs"])
    infos += f"<h3>TSV du dossier de rapport</h3><ul>{tsv_links}</ul>"

    body = f"""<header>
  <h1><a href="../index.html">&larr; {version}</a> &middot; {name}</h1>
  <span class="sub">{len(df)} gènes &middot; {len(df.columns)} colonnes</span>
  <span class="sub"><a href="../genes.html">Comparer les configs par gène</a></span>
</header>
<nav class="tabs">
  <button data-tab="rank">Ranking</button>
  <button data-tab="umap">UMAP</button>
  <button data-tab="figs">Figures ({len(others)})</button>
  <button data-tab="info">Infos</button>
</nav>
<main>
  <div class="panel" id="panel-rank">{panel_rank}</div>
  <div class="panel" id="panel-umap">{panel_umap}</div>
  <div class="panel" id="panel-figs">{panel_figs}</div>
  <div class="panel" id="panel-info">{infos}</div>
</main>
<script src="../data/{sl}.js"></script>
<script>
var T=SiteTable(document.getElementById('mount'), PAYLOAD, {{
  visible:{json.dumps(visible)},
  onCount:function(n,tot){{document.getElementById('cnt').textContent=n+' / '+tot+' gènes';}}
}});
function applyFilters(){{
  var tier=document.getElementById('ftier').value;
  var dir=document.getElementById('fdir').value;
  var nohub=document.getElementById('fnohub').checked;
  var ds=parseFloat(document.getElementById('fds').value);
  T.filter(document.getElementById('q').value, function(v){{
    if(tier && v('evidence_tier')!==tier) return false;
    if(dir){{ var d=String(v('direction')||''); if(d.indexOf(dir)!==0) return false; }}
    if(nohub && (v('is_hub_inflated')===true||v('is_hub_inflated')==='True')) return false;
    if(!isNaN(ds) && !(v('driver_score')>=ds)) return false;
    return true;
  }});
}}
['q','ftier','fdir','fnohub','fds'].forEach(function(id){{
  var e=document.getElementById(id);
  e.addEventListener(e.tagName==='INPUT'&&e.type!=='checkbox'?'input':'change', applyFilters);
}});
document.querySelector('.cols').addEventListener('change', function(){{
  var on=[].slice.call(document.querySelectorAll('.cols input:checked')).map(function(i){{return i.value;}});
  T.setVisible(on);
}});
T.sortBy('driver_score');
initTabs();
</script>"""
    (cfg_dir_out / f"{sl}.html").write_text(page(f"{name} — {version}", body, depth=1),
                                            encoding="utf-8")

    top = df.nlargest(5, "driver_score")["target"].astype(str).tolist() \
        if "driver_score" in df.columns else []
    return {"name": name, "slug": sl, "n_genes": len(df),
            "n_figs": len(assets["pngs"]), "umap": bool(assets["umap_html"]),
            "top": top}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_genes_page(configs: list[dict], out: Path, version: str) -> None:
    """Page transverse : un gène → rang + driver_score dans chaque config."""
    per_cfg: dict[str, pd.DataFrame] = {}
    for cfg in configs:
        try:
            d = pd.read_csv(cfg["ranking"], sep="\t",
                            usecols=["target", "driver_score"], low_memory=False)
        except Exception:
            continue
        d = d.dropna(subset=["target"]).copy()
        d["target"] = d["target"].astype(str)
        d = d.sort_values("driver_score", ascending=False)
        d["rank"] = np.arange(1, len(d) + 1)
        per_cfg[cfg["name"]] = d.set_index("target")

    if not per_cfg:
        return
    genes = sorted(set().union(*[set(d.index) for d in per_cfg.values()]))
    gidx = {g: i for i, g in enumerate(genes)}
    names = list(per_cfg.keys())
    ranks, scores = [], []
    for nm in names:
        d = per_cfg[nm]
        r = [None] * len(genes)
        s = [None] * len(genes)
        for g, row in zip(d.index, d.itertuples(index=False)):
            i = gidx.get(g)
            if i is None:
                continue
            r[i] = int(row.rank)
            v = float(row.driver_score)
            s[i] = round(v, 3) if np.isfinite(v) else None
        ranks.append(r)
        scores.append(s)

    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / "_genes.js").write_text(
        "window.GENES=" + json.dumps(
            {"genes": genes, "configs": names, "ranks": ranks, "scores": scores,
             "slugs": [slug(n) for n in names]},
            separators=(",", ":")) + ";", encoding="utf-8")

    body = f"""<header>
  <h1><a href="index.html">&larr; {version}</a> &middot; Un gène, toutes les configs</h1>
  <span class="sub">{len(genes)} gènes &middot; {len(names)} configs</span>
</header>
<main>
<div class="bar">
  <input id="g" placeholder="gène — ex. SYNJ2" list="gl" size="24" autocomplete="off">
  <datalist id="gl"></datalist>
  <button class="btn" onclick="show()">Voir</button>
  <span class="muted">Écart de rang entre configs = sensibilité au graphe (circularité).</span>
</div>
<div id="out"></div>
</main>
<script src="data/_genes.js"></script>
<script>
var D=GENES, I={{}};
D.genes.forEach(function(g,i){{I[g]=i;}});
var dl=document.getElementById('gl');
D.genes.slice(0,4000).forEach(function(g){{
  var o=document.createElement('option'); o.value=g; dl.appendChild(o);}});
function show(){{
  var g=document.getElementById('g').value.trim().toUpperCase();
  var i=I[g];
  var o=document.getElementById('out');
  if(i===undefined){{o.innerHTML='<p class="muted">Gène absent de toutes les configs.</p>';return;}}
  var rows=D.configs.map(function(c,k){{
    return {{c:c, s:D.slugs[k], r:D.ranks[k][i], v:D.scores[k][i]}};
  }}).filter(function(x){{return x.r!==null;}});
  rows.sort(function(a,b){{return a.r-b.r;}});
  var best=rows.length?rows[0].r:0, worst=rows.length?rows[rows.length-1].r:0;
  var h='<p><b>'+g+'</b> — rang '+best+' &rarr; '+worst+' selon la config '+
        '(<a target="_blank" rel="noopener" href="https://www.genecards.org/cgi-bin/carddisp.pl?gene='+
        encodeURIComponent(g)+'">GeneCards</a>)</p>';
  h+='<table class="grid"><thead><tr><th>config</th><th>rang</th><th>driver_score</th></tr></thead><tbody>';
  rows.forEach(function(x){{
    h+='<tr><td><a href="cfg/'+x.s+'.html#rank">'+x.c+'</a></td><td>'+x.r+
       '</td><td>'+(x.v===null?'':x.v.toFixed(3))+'</td></tr>';
  }});
  o.innerHTML=h+'</tbody></table>';
}}
document.getElementById('g').addEventListener('change',show);
</script>"""
    (out / "genes.html").write_text(page(f"Gènes — {version}", body, depth=0),
                                    encoding="utf-8")


def build_index(summaries: list[dict], out: Path, root: Path,
                version: str) -> None:
    items = "".join(
        f'<li><a href="cfg/{s["slug"]}.html">{s["name"]}</a> '
        f'<span class="muted">— {s["n_genes"]} gènes, {s["n_figs"]} fig.'
        f'{", UMAP" if s["umap"] else ""}</span></li>'
        for s in summaries)
    # Figures globales de la version (hors configs).
    cfg_dirs = {Path(s["name"]).parts[0] for s in summaries}
    glob_pngs = [p for p in sorted(root.rglob("*.png"))
                 if p.relative_to(root).parts[0] not in cfg_dirs
                 and "_site" not in p.parts][:60]
    gal = "".join(
        f'<figure><figcaption>{rel(p, out)}</figcaption>'
        f'<img loading="lazy" src="{rel(p, out)}" alt="{p.name}"></figure>'
        for p in glob_pngs)
    body = f"""<header><h1>{version} — explorateur par config</h1>
<span class="sub">{len(summaries)} configs &middot; généré le {datetime.now():%Y-%m-%d %H:%M}</span>
<span class="sub"><a href="genes.html">Un gène, toutes les configs &rarr;</a></span>
</header>
<main>
<p class="muted">Chaque config : ranking interactif (tri / filtres / export),
UMAP du latent, galerie de figures, run_config. Comparer les configs = lire la
part du signal portée par le graphe plutôt que par le DE.</p>
<ul class="cfgs">{items}</ul>
{'<h3>Figures globales de la version</h3><div class="gal">' + gal + '</div>' if gal else ''}
</main>
<script>initTabs();</script>"""
    (out / "index.html").write_text(page(f"{version} — configs", body, depth=0),
                                    encoding="utf-8")


def serve(site: Path, root: Path, port: int, auth: str | None) -> None:
    """Sert le site en local.

    On sert l'ancêtre commun de --out et --root, PAS `site` : les figures et
    TSV sont référencés en relatif hors du dossier du site, et
    SimpleHTTPRequestHandler refuse toute remontée au-dessus de sa racine.
    """
    import base64
    import functools
    import http.server
    import socketserver

    serve_dir = Path(os.path.commonpath([site, root]))
    entry = rel(site / "index.html", serve_dir)
    base = http.server.SimpleHTTPRequestHandler
    if auth:
        user, _, pw = auth.partition(":")
        tok = "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()

        class Auth(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.headers.get("Authorization") != tok:
                    self.send_response(401)
                    self.send_header("WWW-Authenticate", 'Basic realm="site"')
                    self.end_headers()
                    self.wfile.write(b"auth requise")
                    return
                super().do_GET()
        base = Auth
    handler = functools.partial(base, directory=str(serve_dir))
    print(f"### racine servie : {serve_dir}")
    print(f"### http://127.0.0.1:{port}/{entry}  (Ctrl-C pour arrêter)")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="Racine à explorer (ex. output/gnn_vgae/V6.1.3).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Dossier du site (défaut <root>/_site).")
    ap.add_argument("--version", default=None, help="Titre (défaut = nom de --root).")
    ap.add_argument("--all-columns", action="store_true",
                    help="Affiche toutes les colonnes par défaut (sinon subset).")
    ap.add_argument("--max-depth", type=int, default=4,
                    help="Profondeur max de recherche du ranking (défaut 4).")
    ap.add_argument("--only", default=None,
                    help="Regex : ne garder que les configs dont le nom matche.")
    ap.add_argument("--serve", action="store_true", help="Sert le site en local.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--auth", default=None, help="user:pass (Basic Auth).")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Racine absente : {root}")
    out = (args.out or root / "_site").resolve()
    version = args.version or root.name

    print(f"[scan] {root}")
    configs = discover_configs(root, args.max_depth)
    if args.only:
        pat = re.compile(args.only)
        configs = [c for c in configs if pat.search(c["name"])]
    if not configs:
        raise SystemExit(f"Aucun {RANKING_NAME} trouvé sous {root}.")
    print(f"[scan] {len(configs)} config(s)")

    out.mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    (out / "assets" / "style.css").write_text(CSS, encoding="utf-8")
    (out / "assets" / "app.js").write_text(APP_JS, encoding="utf-8")

    summaries = []
    for cfg in configs:
        assets = collect_assets(cfg, root)
        try:
            s = build_config_page(cfg, assets, out, version, args.all_columns)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERREUR] {cfg['name']} : {e}")
            continue
        summaries.append(s)
        print(f"  [ok] {cfg['name']} — {s['n_genes']} gènes, "
              f"{s['n_figs']} fig., UMAP={'oui' if s['umap'] else 'non'}")

    build_genes_page(configs, out, version)
    build_index(summaries, out, root, version)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\n[SUCCESS] {out}/index.html  ({size/1e6:.1f} Mo générés)")

    if args.serve:
        serve(out, root, args.port, args.auth)


if __name__ == "__main__":
    main()
