#!/usr/bin/env python3
"""
build_config_site.py — static per-config exploration site.

Generalises scripts/build_umap_site.sh (UMAP only, V5.4.1 layout hard-coded):
discovers configs under an arbitrary root and emits, for each one, a tabbed
page:

  * Ranking  — any TSV of the config in a sortable / filterable virtual table
               (~13k rows fine), column picker with inline documentation, per
               gene detail panel, CSV export of the current view.
  * UMAP     — umap_interactive.html in an iframe when present, otherwise the
               umap_*.png figures + the command to generate the interactive one.
  * Figures  — gallery of every PNG of the config (analysis/, s*/figure/…).
  * Colonnes — glossary: what each column of the main ranking means.
  * Infos    — run_config.json, SUMMARY.md, links to the raw TSVs.

Plus two cross-cutting pages:
  * index.html  — config list (+ version-level global figures).
  * genes.html  — one gene → its rank / driver_score in EVERY config, i.e. a
                  direct read of graph sensitivity (circularity).

No data is copied: PNGs and TSVs are referenced by relative path from --out.
Only the main ranking is converted to a compact JS payload (columnar + string
interning) so that the headline table also works over file://. Every other TSV
is fetched and parsed client-side, which requires --serve (see SERVE below).

Usage
-----
    # whole version
    python scripts/build_config_site.py --root output/gnn_vgae/V6.1.3

    # + local server (recommended: unlocks the alternative tables)
    python scripts/build_config_site.py --root output/gnn_vgae/V6.1.3 --serve

    # V5.4.1 layout (cross_seed/<cfg>/) — same command, auto-discovery
    python scripts/build_config_site.py \\
        --root output/interpretation/V5.4.1/cross_seed --out /tmp/site_v541
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RANKING_NAME = "cross_seed_gene_ranking.tsv"
# Candidate sub-directories for the "primary" ranking of a config, by
# decreasing preference. First hit wins; the others become alternative tables.
PRIMARY_SUBDIRS = ("analysis", "xseed", "cross_seed_report", "report_axisV4", "")

# Columns shown by default. Every other column stays loaded and can be toggled
# on from the column picker.
DEFAULT_COLS = [
    "target", "driver_score", "discovery_score", "validation_score",
    "evidence_tier", "direction", "canon_diff", "canon_cosine",
    "canon_amplitude", "n_modes_present", "mean_robustness", "sign_consistent",
    "is_hub_inflated", "target_ppi_degree", "senescence_specificity",
    "vgae_rank", "is_de_significant", "de_log2fc_p4_vs_p16", "n_aging_dbs",
    "is_tf", "marker_driver_conflict",
]
# Long free-text columns: pickable, but off by default (they blow up row width).
WIDE_COLS = ("interpretation", "member_of_strong_pathways")


# --------------------------------------------------------------------------- #
# Column glossary
# --------------------------------------------------------------------------- #
# Ordered families -> (title, [(column, description)]). Descriptions are user
# facing and stay in French, like the rest of the site. Dynamic column names
# (per-axis, signed readout) are resolved by col_doc() below.
COL_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Identité et scores de tête", [
        ("target",
         "Gène perturbé. Une ligne = un gène, les 3 modes (KO/KD/OE) agrégés."),
        ("driver_score",
         "Colonne de tri. ∈[0,1], 100 % graphe : 0.35·amplitude + 0.30·pureté "
         "(|cos|) + 0.15·couverture (n_modes/3) + 0.10·cohérence de signe + "
         "0.10·centralité (rang VGAE) ; ×0.9 si hub-inflated. N'utilise AUCUNE "
         "information de littérature — c'est ce qui la rend comparable entre "
         "configs."),
        ("discovery_score",
         "Signal graphe fort MAIS littérature absente (non DE-significatif, "
         "0 gene-set). Candidat exploratoire, à valider expérimentalement."),
        ("validation_score",
         "Corroboration par la littérature seule (DE-significativité, "
         "gene-sets), pondérée par robustesse/stabilité. Indépendant du "
         "driver_score : un score de driver élevé ne s'y propage pas."),
        ("evidence_tier",
         "A_confirmed = driver pur + littérature · B_discovery = driver pur "
         "sans littérature (prioritaire à valider) · C_effector = littérature "
         "mais driver impur (marqueur probable) · D_hub = artefact de "
         "connectivité · E_noise = ni l'un ni l'autre. Priorité : D testé en "
         "premier. « Driver pur » = driver_score ≥ 0.33 ET |cos| ≥ 0.4 "
         "(seuil 0.5 avant le 2026-08-13 : le score a perdu ses points de "
         "base, 0.33 est le même quantile)."),
        ("direction",
         "pro-senescence / anti-senescence / neutral d'après le signe "
         "canonicalisé. Suffixe « (mixed) » si OE et perte-de-fonction ne "
         "s'opposent pas."),
        ("interpretation",
         "Phrase générée + tags qualité : [unreliable] (robustesse/stabilité "
         "sous le seuil), [hub-inflated], [incoherent], [low-purity signal]."),
    ]),
    ("Métriques de perturbation", [
        ("canon_diff",
         "FORCE de l'effet : déplacement projeté sur l'axe de sénescence, "
         "canonicalisé (signé par l'OE si présent, sinon par −perte ; en mode "
         "`aligned` KO/KD sont réalignés puis moyennés avec l'OE)."),
        ("canon_cosine",
         "PURETÉ directionnelle : cosinus entre le déplacement induit et l'axe "
         "de sénescence. |cos| < 0.4 = effet diffus, non dirigé (→ tier D)."),
        ("canon_amplitude",
         "Fraction directionnelle corrigée du degré ∈[−1,1]. Pureté sans la "
         "force ; diagnostic, ne nourrit PAS le driver_score."),
        ("max_abs_diff",
         "|déplacement| max sur les 3 modes, sans canonicalisation de signe. "
         "Sert de tie-break au tri par driver_score."),
        ("max_abs_cosine", "Idem pour le cosinus : |cos| max sur les modes."),
        ("mean_abs_extent",
         "Étendue du shift (variante normalisée), moyennée sur les modes."),
        ("mean_abs_degree",
         "Shift normalisé par le degré, moyenné sur les modes (diagnostic)."),
        ("n_modes_present",
         "Nombre de modes disponibles pour ce gène (1 à 3 : KO, KD, OE)."),
        ("KO_diff", "Déplacement projeté brut du knockout (non canonicalisé)."),
        ("KD_diff", "Déplacement projeté brut du knockdown (siRNA-like, ×0.1-0.2)."),
        ("OE_diff", "Déplacement projeté brut de la surexpression."),
        ("KO_cos", "Cosinus brut du knockout."),
        ("KD_cos", "Cosinus brut du knockdown."),
        ("OE_cos", "Cosinus brut de la surexpression."),
    ]),
    ("Qualité et artefacts", [
        ("mean_robustness",
         "Fraction des seeds où le gène apparaît (moyenne sur les modes). "
         "1.0 = présent dans toutes. La table n'est PAS filtrée dessus."),
        ("mean_stability",
         "Stabilité du signe de l'effet entre seeds. < 1 = le sens de l'effet "
         "change d'une seed à l'autre."),
        ("sign_consistent",
         "OE et perte-de-fonction poussent en sens opposés = signature d'un "
         "vrai driver causal. Vide si un seul mode disponible."),
        ("is_hub_inflated",
         "|diff| > 50 ET |cos| < 0.3 ET degré PPI > 200 : amplitude "
         "explicable par la connectivité. Force le tier D. N'atténue plus le "
         "driver_score : le facteur ×0.9 a été retiré le 2026-08-07 (mesuré "
         "inerte — aucun gène ne porte le flag dans les configs V6.1.3)."),
        ("is_low_purity_signal",
         "Même profil mais degré ≤ 200 : l'amplitude n'est PAS explicable par "
         "un effet de hub (cas borderline type ASNS). Ne pénalise pas le score."),
        ("target_ppi_degree", "Degré du gène dans la couche PPI. > 200 = candidat hub."),
        ("coexpr_degree",
         "Degré dans la couche de coexpression (si --coexpr-degree-file a été "
         "fourni au rapport). Les hubs chromatine sont des hubs de coexpr."),
    ]),
    ("Spécificité sénescence (clusters)", [
        ("cosine_quiescent_like",
         "Cosinus moyen sur (P4, P16_cluster_0) — c0 est quiescent-like, "
         "proche de P4. Nom historique : c0 est en fait prolifératif-persistant."),
        ("cosine_senescent",
         "Cosinus moyen sur (c1, c2, c3), les clusters réellement sénescents."),
        ("senescence_specificity",
         "cosine_senescent − cosine_quiescent_like. > 0 = le gène bouge les "
         "clusters sénescents SANS bouger les quiescents = driver spécifique."),
    ]),
    ("Croisements externes (baselines)", [
        ("vgae_importance",
         "Importance du gène dans la baseline VGAE (centralité du latent), "
         "indépendante de toute perturbation."),
        ("vgae_rank",
         "Rang associé. Un rang élevé + un driver_score élevé = trouvaille "
         "propre au readout de perturbation, pas de la simple centralité."),
        ("is_de_significant",
         "Gène différentiellement exprimé P4 vs P16. Mode `magnitude-rank` "
         "(défaut : rang |ΔExpr| ≤ N) ou `pvalue` (padj < 0.05 ET "
         "|log2FC| ≥ 0.5). Vide = gène absent de la table DE."),
        ("de_log2fc_p4_vs_p16",
         "log2FC MAST signé (> 0 = up en P16). Découplé de is_de_significant."),
        ("de_neglog10_padj", "−log10(padj) MAST, clippé."),
        ("n_aging_dbs",
         "Nombre de gene-sets du registre contenant le gène (EndoSEN, aging "
         "DBs…). ≥ 2 compte comme corroboration littérature."),
        ("is_tf",
         "Facteur de transcription (pySCENIC ∪ CollecTRI). Cascade pléiotrope "
         "attendue → seuil de cosinus assoupli dans l'interprétation."),
        ("member_of_strong_pathways",
         "Pathways Reactome forts/modérés (issus du pathway ranking) "
         "contenant le gène."),
    ]),
    ("Diagnostics de dé-biaisage du degré", [
        ("ds_diffcos",
         "force × pureté (normalisée p99). Aggrave le biais de degré PPI. "
         "Diagnostic uniquement : aucune colonne ds_* ne pilote le tri."),
        ("ds_amp", "|canon_amplitude| : pureté seule, sans effet de degré."),
        ("ds_ppideg",
         "(force / degré PPI) × pureté : dé-biaise le hub PPI, laisse le hub "
         "coexpr."),
        ("ds_totdeg",
         "(force / (degré PPI + coexpr)) × pureté. Peut sur-corriger."),
    ]),
    ("Conflit marqueur / driver", [
        ("marker_driver_conflict",
         "Le signe DE contredit l'effet causal (cosine_senescent) : le gène "
         "est un marqueur, pas un driver du sens qu'on lui prête (cas FHL2, "
         "up en P16 mais anti-sénescent)."),
    ]),
]

# Dynamic column families, matched by regex when a name is not in COL_GROUPS.
COL_PATTERNS: list[tuple[str, str]] = [
    (r"^driver_score_(.+)$",
     "driver_score recalculé sur l'axe {0} (même formule). Non-rankant : le "
     "tri de tête reste le driver_score global."),
    (r"^canon_diff_(.+)$", "Force de l'effet projetée sur l'axe {0}."),
    (r"^canon_cos_(.+)$", "Pureté directionnelle sur l'axe {0}."),
    (r"^signed_readout_(pert|de|latent)$",
     "Readout signé du fan-out 1-hop, rôle de la cible pris via « {0} » "
     "(pert = effet causal, aveugle aux effecteurs ; de = signe DE, aveugle "
     "aux compensatoires ; latent = position au repos, biaisée). Aucune des "
     "trois n'est headline : leur désaccord EST le signal."),
    (r"^signed_coherence_(pert|de|latent)$",
     "Cohérence du fan-out pour le rôle « {0} » : |Σ sgn(rôle)·sign_pred| / N "
     "∈[0,1]. Degree-free, contrairement au readout."),
    (r"^signed_n_role_(pert|de)$",
     "Nombre de cibles du fan-out ayant un rôle « {0} » défini."),
    (r"^signed_fanout_n$", "Nombre de cibles dans le fan-out signé 1-hop."),
    (r"^signed_fanout_conflict_frac$",
     "Fraction des cibles où role_de et role_pert se contredisent = signal "
     "marqueur/driver au niveau de la source."),
    (r"^signed_pred_known_agree$",
     "Accord entre le signe prédit (bilinéaire) et le signe curé connu."),
    (r"^target_total_degree$", "Degré toutes couches confondues."),
    (r"^target_ppi_degree_only$", "Degré PPI strict (hors autres couches)."),
    (r"^n_seeds_present$", "Nombre de seeds où la ligne est présente."),
    (r"^robustness_score$", "Fraction des seeds où la ligne est présente."),
    (r"^direction_stability$", "Stabilité du signe entre seeds."),
    (r"^in_(.+)$", "Appartenance au gene-set « {0} » du registre."),
]

# Axis labels appearing in dynamic column names -> readable form.
AXIS_LABELS = {
    "P16_cluster_0": "P4→c0 (prolifératif persistant)",
    "P16_cluster_1": "P4→c1 (sénescent ECM-mild)",
    "P16_cluster_2": "P4→c2 (sénescent OIS)",
    "P16_cluster_3": "P4→c3 (sénescent SASP)",
}

COL_DOC: dict[str, str] = {c: d for _, cols in COL_GROUPS for c, d in cols}


def col_doc(name: str) -> str:
    """Description of a column, resolving dynamic families. '' when unknown."""
    if name in COL_DOC:
        return COL_DOC[name]
    for pat, tpl in COL_PATTERNS:
        m = re.match(pat, name)
        if m:
            arg = m.group(1) if m.groups() else ""
            return tpl.format(AXIS_LABELS.get(arg, arg))
    return ""


# --------------------------------------------------------------------------- #
# Config discovery
# --------------------------------------------------------------------------- #
def discover_configs(root: Path, max_depth: int = 4) -> list[dict]:
    """Find configs = directories holding a ranking at depth <= max_depth.

    Returns dicts {name, dir, ranking, tables}. `dir` is the run root (the one
    also holding build/, s1/, logs/), i.e. the parent of the report directory.
    """
    hits: dict[Path, Path] = {}          # config dir -> primary ranking
    for rk in sorted(root.rglob(RANKING_NAME)):
        try:
            depth = len(rk.relative_to(root).parts)
        except ValueError:
            continue
        if depth > max_depth:
            continue
        report_dir = rk.parent
        cfg_dir = (report_dir.parent
                   if report_dir.name in PRIMARY_SUBDIRS[:-1]
                   or report_dir.name.startswith("xseed")
                   else report_dir)
        prev = hits.get(cfg_dir)
        if prev is None or _rank_priority(rk) < _rank_priority(prev):
            hits[cfg_dir] = rk

    configs = []
    for cfg_dir, rk in sorted(hits.items()):
        rel_ = cfg_dir.relative_to(root) if cfg_dir != root else Path(cfg_dir.name)
        configs.append({"name": str(rel_).replace(os.sep, "/"),
                        "dir": cfg_dir, "ranking": rk})
    return configs


def _rank_priority(p: Path) -> int:
    """Lower = preferred (analysis/ before xseed/ before the rest)."""
    try:
        return PRIMARY_SUBDIRS.index(p.parent.name)
    except ValueError:
        return len(PRIMARY_SUBDIRS)


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", name)


def collect_assets(cfg: dict) -> dict:
    """PNGs, interactive UMAP, run config, report and all loadable TSVs."""
    d: Path = cfg["dir"]
    report_dir: Path = cfg["ranking"].parent
    pngs = sorted(p for p in d.rglob("*.png") if "_site" not in p.parts)
    umap_html = next(iter(sorted(d.rglob("umap_interactive.html"))), None)
    run_cfg = next(iter(sorted(d.rglob("run_config.json"))), None)
    summary = next(iter(sorted(d.rglob("SUMMARY.md"))), None)
    # Every TSV of the report dir and of its immediate sub-directories becomes
    # a selectable table (axis_methods/, reproject_ahn/, interpret/, …).
    tsvs = sorted(set(report_dir.glob("*.tsv")) | set(report_dir.glob("*/*.tsv")))
    return {"pngs": pngs, "umap_html": umap_html, "run_config": run_cfg,
            "summary": summary, "tsvs": tsvs, "report_dir": report_dir}


# --------------------------------------------------------------------------- #
# Compact JS payload (columnar + string interning)
# --------------------------------------------------------------------------- #
def encode_table(df: pd.DataFrame) -> dict:
    """DataFrame -> JSON-able columnar dict, strings interned.

    Format: {"cols":[...], "n":int, "data":[<col>,...]} where <col> is either a
    list of numbers/booleans/null, or {"d":[unique values], "c":[codes]}.
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
                v = ("" if v is None or (isinstance(v, float) and pd.isna(v))
                     else str(v))
                if v not in uniq:
                    uniq[v] = len(uniq)
                codes.append(uniq[v])
            data.append({"d": list(uniq.keys()), "c": codes})
    return {"cols": cols, "n": int(len(df)), "data": data}


# --------------------------------------------------------------------------- #
# Static assets
# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#fff;--fg:#1b1b1b;--mut:#6b6b6b;--line:#e2e2e2;--acc:#2166ac;
 --acc-bg:#eaf2fa;--head:#f6f7f9;--warn:#b2182b;--ok:#1a7f37;}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--mut:#9aa0a6;
 --line:#2c3038;--acc:#7fb3e3;--acc-bg:#1d2a38;--head:#1e2126;--warn:#e58a95;
 --ok:#5fc27e;}}
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
select{max-width:min(46vw,430px)}
button.btn{cursor:pointer}
button.btn:hover{background:var(--acc-bg)}
.muted{color:var(--mut);font-size:12px}
.warn{color:var(--warn)}
/* Virtual scroller. `#sizer` alone carries the full height and is resized only
   when the row set changes; scrolling moves the table with `top`, so nothing
   above the viewport ever changes size. That is what keeps the browser's
   scroll anchoring from nudging scrollTop and re-firing `scroll` in a loop
   (overflow-anchor:none is the belt to that braces). */
#tw{border:1px solid var(--line);border-radius:6px;overflow:auto;max-height:70vh;
 position:relative;overflow-anchor:none}
#tw .sizer{position:relative;width:100%;overflow-anchor:none}
#tw table.grid{position:absolute;top:0;left:0}
table.grid{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%}
table.grid th{position:sticky;top:0;background:var(--head);z-index:2;
 border-bottom:1px solid var(--line);padding:6px 9px;text-align:left;
 white-space:nowrap;cursor:help;font-weight:600;font-size:12px}
table.grid th:hover{color:var(--acc)}
/* Fixed row height: the virtual scroller assumes exactly ROW_H px per row.
   Any mismatch makes the total height drift on every render, which fires
   another scroll event — the table then scrolls on its own. */
table.grid td{height:26px;padding:0 9px;line-height:25px;
 border-bottom:1px solid var(--line);
 white-space:nowrap;font-variant-numeric:tabular-nums;
 max-width:46ch;overflow:hidden;text-overflow:ellipsis}
table.grid tr:hover td{background:var(--acc-bg)}
table.grid td.g{font-weight:600}
table.grid th.rk,table.grid td.rk{text-align:right;color:var(--mut);
 padding-right:12px;max-width:7ch}
table.grid td.rk{font-size:12px}
.tier-A_confirmed{color:var(--ok)}.tier-B_discovery{color:var(--acc)}
.tier-D_hub{color:var(--warn)}.tier-E_noise{color:var(--mut)}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.gal figure{margin:0;border:1px solid var(--line);border-radius:6px;padding:8px}
.gal figcaption{font-size:12px;color:var(--mut);margin-bottom:6px;word-break:break-all}
.gal img{width:100%;height:auto;cursor:zoom-in;background:#fff}
iframe.umap{width:100%;height:80vh;border:1px solid var(--line);border-radius:6px}
pre{background:var(--head);padding:10px;border-radius:6px;overflow:auto;
 font-size:12px;max-height:60vh}
#detail{position:fixed;right:0;top:0;bottom:0;width:min(460px,94vw);
 background:var(--bg);border-left:1px solid var(--line);padding:14px;
 overflow:auto;transform:translateX(100%);transition:transform .15s;z-index:40}
#detail.on{transform:none;box-shadow:-8px 0 24px rgba(0,0,0,.12)}
#detail dl{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-size:12px}
#detail dt{color:var(--mut);cursor:help}
#detail dd{margin:0;word-break:break-word}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;z-index:50;
 flex-direction:column}
#lb.on{display:flex}
#lbbar{flex:0 0 auto;display:flex;gap:14px;align-items:center;padding:8px 12px;
 color:#eee;font-size:13px;background:rgba(0,0,0,.55)}
#lbbar a{color:#8ec2ef}
#lbbar .name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
#lbwrap{flex:1;min-height:0;overflow:auto;display:flex;align-items:center;
 justify-content:center;padding:8px}
#lbwrap img{max-width:100%;max-height:100%;cursor:zoom-in;background:#fff}
#lbwrap.zoom{display:block}
#lbwrap.zoom img{max-width:none;max-height:none;cursor:zoom-out}
ul.cfgs{columns:2;list-style:none;padding:0}
ul.cfgs li{margin:3px 0;break-inside:avoid}
/* Column picker: CSS grid, NOT multi-column — `columns` + overflow pushes the
   extra columns off-screen horizontally instead of scrolling them. */
.cols{display:none;border:1px solid var(--line);border-radius:6px;padding:10px;
 margin-bottom:8px;max-height:46vh;overflow-y:auto}
.cols.on{display:block}
.cols .grid2{display:grid;gap:2px 16px;
 grid-template-columns:repeat(auto-fill,minmax(min(100%,300px),1fr))}
.cols label{display:block;font-size:12px;padding:2px 0;cursor:help}
.cols label input{margin-right:6px}
.cols h4{margin:10px 0 4px;font-size:11px;text-transform:uppercase;
 letter-spacing:.04em;color:var(--mut);grid-column:1/-1}
.gloss h3{font-size:14px;margin:18px 0 6px;border-bottom:1px solid var(--line);
 padding-bottom:4px}
.gloss dl{display:grid;grid-template-columns:minmax(140px,auto) 1fr;gap:6px 16px;
 margin:0}
.gloss dt{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
 font-weight:600;word-break:break-all}
.gloss dd{margin:0;font-size:13px;color:var(--fg)}
.gloss .absent{opacity:.45}
"""

APP_JS = r"""
/* Virtual table: columnar decoding, sort, filters, detail panel, CSV export. */
(function(){
/* ROW_H MUST equal the rendered row height set in style.css
   (table.grid td{height:26px}) — any mismatch makes the scroller drift. */
var ROW_H=26, PAD=8;

/* TSV text -> same payload shape as the Python encoder. Numeric columns are
   detected by trying to parse every non-empty cell. */
window.parseTSV = function(text){
  var lines=text.split(/\r?\n/);
  while(lines.length && lines[lines.length-1]==='') lines.pop();
  if(!lines.length) return {cols:[],n:0,data:[]};
  var cols=lines[0].split('\t'), n=lines.length-1;
  var raw=cols.map(function(){return new Array(n);});
  for(var r=0;r<n;r++){
    var f=lines[r+1].split('\t');
    for(var c=0;c<cols.length;c++) raw[c][r]=f[c]===undefined?'':f[c];
  }
  var data=raw.map(function(col){
    var numeric=true;
    for(var i=0;i<col.length;i++){
      var v=col[i];
      if(v===''||v==='NA'||v==='nan'||v==='None') continue;
      if(isNaN(Number(v))){ numeric=false; break; }
    }
    if(numeric){
      return col.map(function(v){
        if(v===''||v==='NA'||v==='nan'||v==='None') return null;
        var f=Number(v); return Number.isInteger(f)?f:Math.round(f*1e4)/1e4;
      });
    }
    var d=[], idx={}, codes=new Array(col.length);
    for(var i=0;i<col.length;i++){
      var v=col[i];
      if(v==='NA'||v==='nan'||v==='None') v='';
      if(!(v in idx)){ idx[v]=d.length; d.push(v); }
      codes[i]=idx[v];
    }
    return {d:d,c:codes};
  });
  return {cols:cols,n:n,data:data};
};

window.SiteTable = function(mount, payload, opts){
  opts = opts || {};
  mount.innerHTML='';
  var cols = payload.cols, N = payload.n;
  var doc = window.COL_DOC_FN || function(){return '';};
  var get = payload.data.map(function(col){
    if(col && col.d){ return function(i){ var v=col.d[col.c[i]]; return v===''?null:v; }; }
    return function(i){ return col[i]; };
  });
  /* `order` = every row in the current sort, `rankOf` = 1-based position in
     `order` (so a filtered view still shows each row's rank in the FULL
     table), `view` = the filtered subset in the same order. */
  var order = new Array(N); for(var i=0;i<N;i++) order[i]=i;
  var rankOf = new Int32Array(N);
  var view = order, sortCol=-1, sortDir=-1;
  var lastQ='', lastExtra=null;
  var visible = (opts.visible||cols).filter(function(c){return cols.indexOf(c)>=0;});
  if(!visible.length) visible = cols.slice(0, 12);
  var keyCol = cols.indexOf('target')>=0 ? 'target'
             : (cols.indexOf('gene')>=0 ? 'gene' : cols[0]);
  var wrap=document.createElement('div'); wrap.id='tw';
  var sizer=document.createElement('div'); sizer.className='sizer';
  var tbl=document.createElement('table'); tbl.className='grid';
  var thead=document.createElement('thead'), tbody=document.createElement('tbody');
  tbl.appendChild(thead); tbl.appendChild(tbody);
  sizer.appendChild(tbl); wrap.appendChild(sizer);
  mount.appendChild(wrap);

  function ci(c){ return cols.indexOf(c); }
  function val(c,i){ var k=ci(c); return k<0?null:get[k](i); }
  function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
  function fmt(v){
    if(v===null||v===undefined) return '';
    if(typeof v==='number') return Number.isInteger(v)?v:v.toFixed(Math.abs(v)<1?3:2);
    return String(v);
  }
  function head(){
    var tr=document.createElement('tr');
    var rk=document.createElement('th');
    rk.className='rk'; rk.textContent='#';
    rk.title='Rang dans le tri courant, sur la table entière — il ne change '
            +'pas quand vous filtrez.';
    tr.appendChild(rk);
    visible.forEach(function(c){
      var th=document.createElement('th');
      th.textContent=c+(sortCol===ci(c)?(sortDir<0?' ▼':' ▲'):'');
      var d=doc(c); if(d) th.title=c+' — '+d;
      th.onclick=function(){ sort(ci(c)); };
      tr.appendChild(th);
    });
    thead.innerHTML=''; thead.appendChild(tr);
  }
  function sort(k){
    if(k<0) return;
    sortDir = (k===sortCol) ? -sortDir : -1;
    sortCol = k;
    var f=get[k];
    /* Sort ALL rows, not just the visible subset: ranks must stay absolute. */
    order.sort(function(a,b){
      var x=f(a), y=f(b);
      var xe=(x===null||x===undefined||x===''), ye=(y===null||y===undefined||y==='');
      if(xe&&ye) return 0;
      if(xe) return 1;                      /* empties always last */
      if(ye) return -1;
      if(typeof x==='string'||typeof y==='string'){ x=String(x); y=String(y); }
      /* sortDir=-1 -> descending (first click puts the big scores on top) */
      return x<y?-sortDir:(x>y?sortDir:0);
    });
    for(var r=0;r<order.length;r++) rankOf[order[r]]=r+1;
    head(); refilter(true);
  }
  function refilter(keepScroll){
    var terms=lastQ?lastQ.split(/[\s,;]+/).filter(Boolean):[];
    var kcol=ci(keyCol), icol=ci('interpretation');
    var out=[];
    for(var r=0;r<order.length;r++){
      var i=order[r];
      if(terms.length){
        var g=String(get[kcol](i)||'').toLowerCase();
        var it=icol>=0?String(get[icol](i)||'').toLowerCase():'';
        var ok=false;
        for(var t=0;t<terms.length;t++){
          if(g.indexOf(terms[t])>=0||it.indexOf(terms[t])>=0){ok=true;break;}
        }
        if(!ok) continue;
      }
      if(lastExtra && !lastExtra(function(c){return val(c,i);})) continue;
      out.push(i);
    }
    view=out;
    if(!keepScroll) wrap.scrollTop=0;
    render(true);
  }
  /* Virtual window. `#sizer` holds the full height; the table is absolutely
     positioned and only its `top` moves while scrolling — no layout above the
     viewport ever changes, so the scroller cannot feed itself. The sticky
     <thead> sits at the top of the table and takes layout space above the
     rows, hence the headH offset in the index maths. */
  var lastFirst=-1, lastLast=-1, pending=false, headH=0;
  function syncSizer(){
    headH=thead.offsetHeight||headH;
    sizer.style.height=(view.length*ROW_H+headH)+'px';
  }
  function render(force){
    if(force||!headH) syncSizer();
    var st=Math.max(0, wrap.scrollTop-headH), h=wrap.clientHeight;
    var first=Math.max(0,Math.floor(st/ROW_H)-PAD);
    var last=Math.min(view.length,Math.ceil((st+h)/ROW_H)+PAD);
    if(!force && first===lastFirst && last===lastLast) return;
    lastFirst=first; lastLast=last;
    tbl.style.top=(first*ROW_H)+'px';
    var html='';
    for(var r=first;r<last;r++){
      var i=view[r];
      html+='<tr data-i="'+i+'"><td class="rk">'+(rankOf[i]||(r+1))+'</td>';
      for(var c=0;c<visible.length;c++){
        var v=val(visible[c],i), cls='', txt=esc(fmt(v));
        if(visible[c]===keyCol) cls=' class="g"';
        else if(visible[c]==='evidence_tier'&&v) cls=' class="tier-'+esc(v)+'"';
        html+='<td'+cls+(txt.length>40?' title="'+txt+'"':'')+'>'+txt+'</td>';
      }
      html+='</tr>';
    }
    tbody.innerHTML=html;
    /* thead height is only measurable once rows exist; re-sync if it moved. */
    if(thead.offsetHeight && thead.offsetHeight!==headH) syncSizer();
    if(opts.onCount) opts.onCount(view.length, N, summary());
  }
  /* Short status line: when the filter isolates a few rows, spell out their
     rank so a gene search answers "where does it sit?" directly. */
  function summary(){
    if(!view.length || view.length>3 || view.length===N) return '';
    return view.map(function(i){
      var s = sortCol>=0 ? ' ('+cols[sortCol]+' '+fmt(get[sortCol](i))+')' : '';
      return fmt(val(keyCol,i))+' → rang '+rankOf[i]+' / '+N+s;
    }).join(' · ');
  }
  tbody.onclick=function(e){
    var tr=e.target.closest('tr'); if(!tr) return; showDetail(+tr.dataset.i);
  };
  function showDetail(i){
    var d=document.getElementById('detail'); if(!d) return;
    var key=fmt(val(keyCol,i));
    var gc=/^[A-Z0-9-]{2,20}$/i.test(key)
      ? '<a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene='+
        encodeURIComponent(key)+'" target="_blank" rel="noopener">GeneCards</a>' : '';
    var h='<div class="bar"><b>'+esc(key)+'</b>'+gc+
      '<button class="btn" onclick="document.getElementById(\'detail\')'+
      '.classList.remove(\'on\')">Fermer</button></div><dl>';
    cols.forEach(function(c){
      var v=get[ci(c)](i);
      if(v===null||v===''||v===undefined) return;
      var t=doc(c);
      h+='<dt'+(t?' title="'+esc(t)+'"':'')+'>'+esc(c)+'</dt><dd>'+esc(fmt(v))+'</dd>';
    });
    d.innerHTML=h+'</dl>'; d.classList.add('on');
  }
  /* rAF-throttled: one render per frame at most, never re-entrant. */
  wrap.addEventListener('scroll', function(){
    if(pending) return;
    pending=true;
    requestAnimationFrame(function(){ pending=false; render(false); });
  });

  var api={
    filter:function(q, extra){
      lastQ=(q||'').trim().toLowerCase();
      lastExtra=extra||null;
      refilter(false);
    },
    setVisible:function(list){ visible=list.slice(); head(); render(true); },
    cols:cols, keyCol:keyCol, visible:function(){return visible.slice();},
    has:function(c){ return cols.indexOf(c)>=0; },
    exportCSV:function(name){
      var lines=['rang,'+visible.join(',')];
      for(var r=0;r<view.length;r++){
        var i=view[r];
        lines.push(rankOf[i]+','+visible.map(function(c){
          var v=val(c,i); v=(v===null||v===undefined)?'':String(v);
          return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;
        }).join(','));
      }
      var b=new Blob([lines.join('\n')],{type:'text/csv'});
      var a=document.createElement('a'); a.href=URL.createObjectURL(b);
      a.download=name||'view.csv'; a.click();
    },
    sortBy:function(c){ sort(ci(c)); },
    rankOf:function(i){ return rankOf[i]; }
  };
  head();
  for(var r=0;r<order.length;r++) rankOf[order[r]]=r+1;
  render(true);
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
      if(location.hash.slice(1)!==b.dataset.tab)
        history.replaceState(null,'','#'+b.dataset.tab);
    };
  });
  var want=location.hash.slice(1);
  var start=document.querySelector('nav.tabs button[data-tab="'+want+'"]')||btns[0];
  if(start) start.click();
  initLightbox();
};

/* Lightbox. Event delegation on document, so figures added later (or living
   in a hidden panel at init time) still open. Click the image to toggle a
   true 1:1 zoom inside a scrollable area; click the backdrop or Escape to
   close; the toolbar links to the original file. */
window.initLightbox=function(){
  var lb=document.getElementById('lb');
  /* Defer instead of giving up: the overlay markup may not be parsed yet
     depending on where the caller's <script> sits in the document. */
  if(!lb){
    if(document.readyState==='loading')
      document.addEventListener('DOMContentLoaded', window.initLightbox);
    return;
  }
  if(lb.dataset.ready) return;
  lb.dataset.ready='1';
  var wrap=document.getElementById('lbwrap');
  var img=wrap.querySelector('img');
  var name=document.getElementById('lbname');
  var open=document.getElementById('lbopen');
  function close(){ lb.classList.remove('on'); wrap.classList.remove('zoom'); }
  window.__lbClose=close;
  document.addEventListener('click', function(e){
    var t=e.target;
    if(t.tagName==='IMG' && t.closest('.gal')){
      img.src=t.src;
      name.textContent=decodeURIComponent(t.getAttribute('src').split('/').pop());
      open.href=t.src;
      wrap.classList.remove('zoom');
      lb.classList.add('on');
    }
  });
  img.addEventListener('click', function(e){
    e.stopPropagation();
    wrap.classList.toggle('zoom');
  });
  wrap.addEventListener('click', function(e){ if(e.target===wrap) close(); });
  document.addEventListener('keydown', function(e){
    if(e.key==='Escape') close();
  });
};
})();
"""

# Column documentation shipped to the browser: exact names + dynamic patterns,
# resolved by the same rules as the Python col_doc().
COLDOC_JS_TMPL = """
window.COL_DOC = ${doc};
window.COL_PATTERNS = ${patterns};
window.AXIS_LABELS = ${axes};
window.COL_DOC_FN = function(name){
  if(window.COL_DOC[name]) return window.COL_DOC[name];
  for(var i=0;i<window.COL_PATTERNS.length;i++){
    var p=window.COL_PATTERNS[i], m=new RegExp(p[0]).exec(name);
    if(m){ var a=m[1]||''; return p[1].replace('{0}', window.AXIS_LABELS[a]||a); }
  }
  return '';
};
"""


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def page(title: str, body: str, depth: int = 1) -> str:
    up = "../" * depth
    html = f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{up}assets/style.css">
<script src="{up}assets/app.js"></script>
<script src="{up}assets/coldoc.js"></script>
</head><body>
<div id="detail"></div>
<div id="lb">
  <div id="lbbar">
    <span class="name" id="lbname"></span>
    <span class="muted" style="color:#aaa">clic sur l'image = zoom 1:1 &middot;
      Échap = fermer</span>
    <a id="lbopen" href="#" target="_blank" rel="noopener">Ouvrir l'original</a>
    <button class="btn" onclick="__lbClose()">Fermer</button>
  </div>
  <div id="lbwrap"><img alt=""></div>
</div>
{body}
</body></html>"""
    check_overlay_order(html, title)
    return html


def check_overlay_order(html: str, title: str = "") -> None:
    """Guard the ordering invariant that already broke once.

    The overlay markup must be parsed BEFORE the inline script that wires it,
    otherwise getElementById returns null at init time and clicking a figure
    silently does nothing.
    """
    lb_at = html.find('id="lb"')
    if lb_at < 0:
        raise RuntimeError(f"page({title!r}): #lb markup missing.")
    inits = [i for i in (html.find("initLightbox()"), html.find("initTabs()"))
             if i != -1]
    if inits and lb_at > min(inits):
        raise RuntimeError(
            f"page({title!r}): #lb markup at {lb_at} comes after the init call "
            f"at {min(inits)} — the lightbox would never bind.")


def rel(target: Path, from_dir: Path) -> str:
    return os.path.relpath(target, from_dir).replace(os.sep, "/")


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def table_label(p: Path, report_dir: Path) -> str:
    """Readable label for a selectable TSV."""
    try:
        r = p.relative_to(report_dir)
    except ValueError:
        r = Path(p.name)
    return str(r).replace(os.sep, "/")


def table_group(p: Path, report_dir: Path) -> str:
    """optgroup a TSV belongs to."""
    name = p.name
    parent = p.parent.name if p.parent != report_dir else ""
    if name.startswith("cross_seed_gene_ranking"):
        return "Rankings dérivés (axes, sous-ensembles)"
    if parent:
        return f"Tables de {parent}/"
    return "Autres tables du rapport"


# --------------------------------------------------------------------------- #
# Page builders
# --------------------------------------------------------------------------- #
def build_config_page(cfg: dict, assets: dict, out: Path, version: str,
                      all_cols: bool) -> dict:
    """Write cfg/<slug>.html + data/<slug>.js. Returns an index summary."""
    name = cfg["name"]
    sl = slug(name)
    cfg_out = out / "cfg"
    data_dir = out / "data"
    cfg_out.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["ranking"], sep="\t", low_memory=False)
    # The payload carries EVERY column (the detail panel shows them all).
    (data_dir / f"{sl}.js").write_text(
        "window.PAYLOAD=" + json.dumps(encode_table(df), separators=(",", ":")) + ";",
        encoding="utf-8")

    all_columns = [str(c) for c in df.columns]
    visible = ([c for c in all_columns if c not in WIDE_COLS] if all_cols
               else [c for c in DEFAULT_COLS if c in all_columns])

    # --- column picker: every column, grouped, with its description ------- #
    seen: set[str] = set()
    picker = ""
    for title, entries in COL_GROUPS:
        present = [c for c, _ in entries if c in all_columns]
        if not present:
            continue
        seen.update(present)
        picker += f'<h4>{esc(title)}</h4>'
        for c in present:
            picker += (f'<label title="{esc(col_doc(c))}">'
                       f'<input type="checkbox" value="{esc(c)}"'
                       f'{" checked" if c in visible else ""}>{esc(c)}</label>')
    rest = [c for c in all_columns if c not in seen]
    if rest:
        picker += '<h4>Colonnes par axe / readout signé / autres</h4>'
        for c in rest:
            picker += (f'<label title="{esc(col_doc(c))}">'
                       f'<input type="checkbox" value="{esc(c)}"'
                       f'{" checked" if c in visible else ""}>{esc(c)}</label>')

    tiers = sorted(str(v) for v in
                   df.get("evidence_tier", pd.Series(dtype=str)).dropna().unique())
    tier_opts = "".join(f'<option value="{esc(t)}">{esc(t)}</option>' for t in tiers)

    # --- table source selector: main ranking + every other TSV ------------ #
    report_dir = assets["report_dir"]
    main_rel = rel(cfg["ranking"], cfg_out)
    opts = (f'<optgroup label="Ranking principal (embarqué, marche hors serveur)">'
            f'<option value="" data-href="{main_rel}">{cfg["ranking"].name}</option>'
            f'</optgroup>')
    groups: dict[str, list[Path]] = {}
    for t in assets["tsvs"]:
        if t == cfg["ranking"]:
            continue
        groups.setdefault(table_group(t, report_dir), []).append(t)
    for grp in sorted(groups):
        opts += f'<optgroup label="{esc(grp)}">'
        for t in sorted(groups[grp]):
            lbl = table_label(t, report_dir)
            opts += (f'<option value="{esc(rel(t, cfg_out))}" '
                     f'data-href="{esc(rel(t, cfg_out))}">{esc(lbl)}</option>')
        opts += "</optgroup>"

    panel_rank = f"""<div class="bar">
  <label>Table : <select id="src">{opts}</select></label>
  <a id="rawlink" href="{main_rel}" class="muted">TSV brut</a>
  <span class="muted" id="loadmsg"></span>
</div>
<div class="bar">
  <input id="q" placeholder="recherche — ex. HMGB2, SYNJ2" size="26">
  <select id="ftier"><option value="">tous tiers</option>{tier_opts}</select>
  <select id="fdir"><option value="">toute direction</option>
    <option value="pro">pro-senescence</option>
    <option value="anti">anti-senescence</option></select>
  <label class="muted"><input type="checkbox" id="fnohub"> exclure hub-inflated</label>
  <label class="muted">driver_score &ge; <input id="fds" type="number" step="0.05"
      min="0" max="1" style="width:70px"></label>
  <button class="btn" id="colsbtn">Colonnes</button>
  <button class="btn" id="csvbtn">Export CSV</button>
  <span class="muted" id="cnt"></span>
  <span id="hit" style="font-weight:600;color:var(--acc)"></span>
</div>
<div class="cols">
  <div class="bar"><button class="btn" id="calla">Tout</button>
    <button class="btn" id="cnone">Rien</button>
    <button class="btn" id="cdef">Défaut</button>
    <span class="muted">Survolez un nom pour sa définition &middot;
      onglet <b>Colonnes</b> pour le glossaire complet.</span></div>
  <div class="grid2" id="colgrid">{picker}</div>
</div>
<div id="mount"></div>
<p class="muted">Clic sur une ligne = détail complet (toutes colonnes, avec
définitions au survol) + GeneCards. Tri par clic sur l'en-tête.</p>"""

    # --- UMAP tab --------------------------------------------------------- #
    if assets["umap_html"]:
        u = rel(assets["umap_html"], cfg_out)
        panel_umap = (f'<p class="muted"><a href="{u}" target="_blank">Ouvrir en '
                      f'plein écran</a></p><iframe class="umap" src="{u}"></iframe>')
    else:
        umap_pngs = [p for p in assets["pngs"] if p.name.startswith("umap_")]
        gal = "".join(
            f'<figure><figcaption>{esc(p.name)}</figcaption>'
            f'<img loading="lazy" src="{rel(p, cfg_out)}" alt="{esc(p.name)}"></figure>'
            for p in umap_pngs)
        cmd = (f'python src/validation/viz/interpret_embedding.py \\\n'
               f'    --run-dir {cfg["dir"]}/s1 \\\n'
               f'    --ranking {cfg["ranking"]} \\\n'
               f'    --out-dir {report_dir}/interpret \\\n'
               f'    --umap-only --plotly-cdn --reuse-umap')
        panel_umap = (
            f'<p class="muted">Pas d\'UMAP interactive pour cette config. '
            f'Pour la générer (le site la récupère au rebuild suivant) :</p>'
            f'<pre>{esc(cmd)}</pre>'
            + (f'<div class="gal">{gal}</div>' if umap_pngs
               else '<p class="muted">Aucun PNG umap_* non plus.</p>'))

    # --- Figures tab ------------------------------------------------------ #
    others = [p for p in assets["pngs"] if not p.name.startswith("umap_")]
    fgroups: dict[str, list[Path]] = {}
    for p in others:
        fgroups.setdefault(str(p.parent.relative_to(cfg["dir"])), []).append(p)
    figs = ""
    for grp, ps in sorted(fgroups.items()):
        gal = "".join(
            f'<figure><figcaption>{esc(p.name)}</figcaption>'
            f'<img loading="lazy" src="{rel(p, cfg_out)}" alt="{esc(p.name)}"></figure>'
            for p in ps)
        figs += (f'<h3>{esc(grp or ".")} <span class="muted">({len(ps)})</span></h3>'
                 f'<div class="gal">{gal}</div>')
    panel_figs = figs or '<p class="muted">Aucune figure.</p>'

    # --- Glossary tab ----------------------------------------------------- #
    gloss = ('<p class="muted">Définitions des colonnes du ranking principal. '
             'Les colonnes absentes de cette config sont grisées. Les mêmes '
             'définitions apparaissent au survol des en-têtes et du panneau '
             'détail.</p>')
    for title, entries in COL_GROUPS:
        gloss += f"<h3>{esc(title)}</h3><dl>"
        for c, d in entries:
            cls = "" if c in all_columns else ' class="absent"'
            gloss += f'<dt{cls}>{esc(c)}</dt><dd{cls}>{esc(d)}</dd>'
        gloss += "</dl>"
    if rest:
        gloss += "<h3>Colonnes par axe / readout signé</h3><dl>"
        for c in rest:
            gloss += f'<dt>{esc(c)}</dt><dd>{esc(col_doc(c) or "—")}</dd>'
        gloss += "</dl>"
    panel_gloss = f'<div class="gloss">{gloss}</div>'

    # --- Infos tab -------------------------------------------------------- #
    infos = ""
    if assets["run_config"]:
        try:
            j = json.loads(assets["run_config"].read_text())
            infos += (f"<h3>run_config.json</h3>"
                      f"<pre>{esc(json.dumps(j, indent=2)[:20000])}</pre>")
        except Exception:  # noqa: BLE001 - a malformed config must not kill the page
            pass
    if assets["summary"]:
        txt = assets["summary"].read_text(errors="replace")[:40000]
        infos += (f'<h3>SUMMARY.md <a class="muted" href='
                  f'"{rel(assets["summary"], cfg_out)}">(brut)</a></h3>'
                  f'<pre>{esc(txt)}</pre>')
    infos += ("<h3>TSV de la config</h3><ul>" + "".join(
        f'<li><a href="{rel(t, cfg_out)}">{esc(table_label(t, report_dir))}</a></li>'
        for t in assets["tsvs"]) + "</ul>")

    body = f"""<header>
  <h1><a href="../index.html">&larr; {esc(version)}</a> &middot; {esc(name)}</h1>
  <span class="sub">{len(df)} gènes &middot; {len(all_columns)} colonnes
    &middot; {len(assets["tsvs"])} tables</span>
  <span class="sub"><a href="../genes.html">Comparer les configs par gène</a></span>
</header>
<nav class="tabs">
  <button data-tab="rank">Ranking</button>
  <button data-tab="umap">UMAP</button>
  <button data-tab="figs">Figures ({len(others)})</button>
  <button data-tab="cols">Colonnes</button>
  <button data-tab="info">Infos</button>
</nav>
<main>
  <div class="panel" id="panel-rank">{panel_rank}</div>
  <div class="panel" id="panel-umap">{panel_umap}</div>
  <div class="panel" id="panel-figs">{panel_figs}</div>
  <div class="panel" id="panel-cols">{panel_gloss}</div>
  <div class="panel" id="panel-info">{infos}</div>
</main>
<script src="../data/{sl}.js"></script>
<script>
var DEFAULT_COLS={json.dumps(visible)}, T=null, CURRENT='{esc(cfg["ranking"].name)}';
var $=function(id){{return document.getElementById(id);}};

function pickerFor(cols, checked){{
  /* Rebuild the picker for an arbitrary table (alternative TSVs have their own
     column sets). The main ranking keeps its grouped, documented picker. */
  var h='', set={{}};
  checked.forEach(function(c){{set[c]=1;}});
  cols.forEach(function(c){{
    var d=window.COL_DOC_FN(c);
    h+='<label title="'+(d||c).replace(/"/g,'&quot;')+'"><input type="checkbox" value="'+
       c+'"'+(set[c]?' checked':'')+'>'+c+'</label>';
  }});
  $('colgrid').innerHTML=h;
}}

function mount(payload, visible, keepPicker){{
  T=SiteTable($('mount'), payload, {{
    visible:visible,
    onCount:function(n,tot,detail){{
      $('cnt').textContent=n+' / '+tot+' lignes';
      $('hit').textContent=detail||'';
    }}
  }});
  if(!keepPicker) pickerFor(payload.cols, T.visible());
  if(T.has('driver_score')) T.sortBy('driver_score');
  applyFilters();
}}

function applyFilters(){{
  if(!T) return;
  var tier=$('ftier').value, dir=$('fdir').value;
  var nohub=$('fnohub').checked, ds=parseFloat($('fds').value);
  T.filter($('q').value, function(v){{
    if(tier && v('evidence_tier')!==tier) return false;
    if(dir){{ var d=String(v('direction')||''); if(d.indexOf(dir)!==0) return false; }}
    if(nohub && (v('is_hub_inflated')===true||v('is_hub_inflated')==='True')) return false;
    if(!isNaN(ds) && !(v('driver_score')>=ds)) return false;
    return true;
  }});
}}

/* Alternative tables are fetched and parsed client-side: no duplicated payload
   on disk, but file:// blocks the request -> explicit message. */
function loadTable(href, label){{
  $('loadmsg').textContent='Chargement de '+label+' …';
  fetch(href).then(function(r){{
    if(!r.ok) throw new Error('HTTP '+r.status);
    return r.text();
  }}).then(function(txt){{
    var p=window.parseTSV(txt);
    CURRENT=label;
    var vis=p.cols.filter(function(c){{return DEFAULT_COLS.indexOf(c)>=0;}});
    mount(p, vis.length?vis:p.cols.slice(0,14));
    $('loadmsg').textContent=label+' — '+p.n+' lignes, '+p.cols.length+' colonnes';
  }}).catch(function(e){{
    $('loadmsg').innerHTML='<span class="warn">Impossible de charger '+label+
      ' ('+e.message+'). Les tables alternatives nécessitent un serveur : '+
      'relancez build_config_site.py avec <b>--serve</b>, ou ouvrez le TSV brut.</span>';
  }});
}}

$('src').addEventListener('change', function(){{
  var o=this.options[this.selectedIndex];
  $('rawlink').href=o.dataset.href;
  if(!this.value){{ CURRENT='{esc(cfg["ranking"].name)}';
    mount(PAYLOAD, DEFAULT_COLS); $('loadmsg').textContent=''; }}
  else loadTable(this.value, o.textContent);
}});
['q','ftier','fdir','fnohub','fds'].forEach(function(id){{
  var e=$(id);
  e.addEventListener(e.tagName==='INPUT'&&e.type!=='checkbox'?'input':'change',
                     applyFilters);
}});
$('colsbtn').onclick=function(){{document.querySelector('.cols').classList.toggle('on');}};
$('csvbtn').onclick=function(){{T.exportCSV('{sl}_'+CURRENT.replace(/\\.tsv$/,'')+'.csv');}};
$('colgrid').addEventListener('change', function(){{
  T.setVisible([].slice.call(document.querySelectorAll('#colgrid input:checked'))
    .map(function(i){{return i.value;}}));
}});
$('calla').onclick=function(){{ T.setVisible(T.cols);
  pickerFor(T.cols, T.cols); }};
$('cnone').onclick=function(){{ var k=[T.keyCol]; T.setVisible(k);
  pickerFor(T.cols, k); }};
$('cdef').onclick=function(){{
  var v=T.cols.filter(function(c){{return DEFAULT_COLS.indexOf(c)>=0;}});
  T.setVisible(v.length?v:T.cols.slice(0,14)); pickerFor(T.cols, v); }};

mount(PAYLOAD, DEFAULT_COLS, true);
initTabs();
</script>"""
    (cfg_out / f"{sl}.html").write_text(page(f"{name} — {version}", body, depth=1),
                                        encoding="utf-8")

    return {"name": name, "slug": sl, "n_genes": len(df),
            "n_figs": len(assets["pngs"]), "n_tables": len(assets["tsvs"]),
            "umap": bool(assets["umap_html"])}


def build_genes_page(configs: list[dict], out: Path, version: str) -> None:
    """Cross-cutting page: one gene -> rank + driver_score in every config."""
    per_cfg: dict[str, pd.DataFrame] = {}
    for cfg in configs:
        try:
            d = pd.read_csv(cfg["ranking"], sep="\t",
                            usecols=["target", "driver_score"], low_memory=False)
        except Exception:  # noqa: BLE001 - a config without driver_score is skipped
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
        r: list[int | None] = [None] * len(genes)
        s: list[float | None] = [None] * len(genes)
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
  <h1><a href="index.html">&larr; {esc(version)}</a> &middot; Un gène, toutes les configs</h1>
  <span class="sub">{len(genes)} gènes &middot; {len(names)} configs</span>
</header>
<main>
<div class="bar">
  <input id="g" placeholder="gène — ex. SYNJ2" list="gl" size="24" autocomplete="off">
  <datalist id="gl"></datalist>
  <button class="btn" onclick="show()">Voir</button>
  <span class="muted">L'écart de rang entre configs mesure la sensibilité au
    graphe : un driver qui ne tient que sur une config est un artefact de cette
    config.</span>
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
  var i=I[g], o=document.getElementById('out');
  if(i===undefined){{o.innerHTML='<p class="muted">Gène absent de toutes les configs.</p>';return;}}
  var rows=D.configs.map(function(c,k){{
    return {{c:c, s:D.slugs[k], r:D.ranks[k][i], v:D.scores[k][i]}};
  }}).filter(function(x){{return x.r!==null;}});
  rows.sort(function(a,b){{return a.r-b.r;}});
  var best=rows.length?rows[0].r:0, worst=rows.length?rows[rows.length-1].r:0;
  var h='<p><b>'+g+'</b> — rang '+best+' &rarr; '+worst+' selon la config '+
        '(<a target="_blank" rel="noopener" href="https://www.genecards.org/cgi-bin/carddisp.pl?gene='+
        encodeURIComponent(g)+'">GeneCards</a>)</p>';
  h+='<table class="grid"><thead><tr><th>config</th><th>rang</th>'+
     '<th>driver_score</th></tr></thead><tbody>';
  rows.forEach(function(x){{
    h+='<tr><td><a href="cfg/'+x.s+'.html#rank">'+x.c+'</a></td><td>'+x.r+
       '</td><td>'+(x.v===null?'':x.v.toFixed(3))+'</td></tr>';
  }});
  o.innerHTML=h+'</tbody></table>';
}}
document.getElementById('g').addEventListener('change',show);
initLightbox();
</script>"""
    (out / "genes.html").write_text(page(f"Gènes — {version}", body, depth=0),
                                    encoding="utf-8")


def build_index(summaries: list[dict], out: Path, root: Path, version: str) -> None:
    items = "".join(
        f'<li><a href="cfg/{s["slug"]}.html">{esc(s["name"])}</a> '
        f'<span class="muted">— {s["n_genes"]} gènes, {s["n_tables"]} tables, '
        f'{s["n_figs"]} fig.{", UMAP" if s["umap"] else ""}</span></li>'
        for s in summaries)
    cfg_dirs = {Path(s["name"]).parts[0] for s in summaries}
    glob_pngs = [p for p in sorted(root.rglob("*.png"))
                 if p.relative_to(root).parts[0] not in cfg_dirs
                 and "_site" not in p.parts][:60]
    gal = "".join(
        f'<figure><figcaption>{esc(rel(p, out))}</figcaption>'
        f'<img loading="lazy" src="{rel(p, out)}" alt="{esc(p.name)}"></figure>'
        for p in glob_pngs)
    body = f"""<header><h1>{esc(version)} — explorateur par config</h1>
<span class="sub">{len(summaries)} configs &middot; généré le {datetime.now():%Y-%m-%d %H:%M}</span>
<span class="sub"><a href="genes.html">Un gène, toutes les configs &rarr;</a></span>
</header>
<main>
<p class="muted">Chaque config : ranking interactif (tri / filtres / colonnes
documentées / export), toutes les tables dérivées, UMAP du latent, galerie de
figures, run_config. Comparer les configs = lire la part du signal portée par le
graphe plutôt que par le DE.</p>
<ul class="cfgs">{items}</ul>
{'<h3>Figures globales de la version</h3><div class="gal">' + gal + '</div>' if gal else ''}
</main>
<script>initTabs();</script>"""
    (out / "index.html").write_text(page(f"{version} — configs", body, depth=0),
                                    encoding="utf-8")


def write_assets(out: Path) -> None:
    from string import Template
    # The virtual scroller positions rows arithmetically: a JS/CSS row-height
    # mismatch makes the table drift and scroll on its own.
    js_h = re.search(r"var ROW_H=(\d+)", APP_JS)
    css_h = re.search(r"table\.grid td\{height:(\d+)px", CSS)
    if not js_h or not css_h or js_h.group(1) != css_h.group(1):
        raise RuntimeError(
            f"ROW_H (JS {js_h and js_h.group(1)}) must equal the CSS row height "
            f"({css_h and css_h.group(1)}).")
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / "assets" / "style.css").write_text(CSS, encoding="utf-8")
    (out / "assets" / "app.js").write_text(APP_JS, encoding="utf-8")
    (out / "assets" / "coldoc.js").write_text(
        Template(COLDOC_JS_TMPL).substitute(
            doc=json.dumps(COL_DOC, ensure_ascii=False),
            patterns=json.dumps(COL_PATTERNS, ensure_ascii=False),
            axes=json.dumps(AXIS_LABELS, ensure_ascii=False)),
        encoding="utf-8")


def serve(site: Path, root: Path, port: int, auth: str | None) -> None:
    """Serve the site locally.

    Serves the common ancestor of --out and --root, NOT `site`: figures and
    TSVs are referenced outside the site directory, and SimpleHTTPRequestHandler
    refuses any path above its own root.
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--serve", action="store_true",
                    help="Sert le site en local (requis pour les tables alternatives).")
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
    write_assets(out)

    summaries = []
    for cfg in configs:
        assets = collect_assets(cfg)
        try:
            s = build_config_page(cfg, assets, out, version, args.all_columns)
        except Exception as e:  # noqa: BLE001 - one broken config must not stop the build
            print(f"  [ERREUR] {cfg['name']} : {e}")
            continue
        summaries.append(s)
        print(f"  [ok] {cfg['name']} — {s['n_genes']} gènes, {s['n_tables']} tables, "
              f"{s['n_figs']} fig., UMAP={'oui' if s['umap'] else 'non'}")

    build_genes_page(configs, out, version)
    build_index(summaries, out, root, version)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\n[SUCCESS] {out}/index.html  ({size/1e6:.1f} Mo générés)")

    if args.serve:
        serve(out, root, args.port, args.auth)


if __name__ == "__main__":
    main()
