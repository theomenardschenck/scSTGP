#!/usr/bin/env python3
"""
interpret_embedding.py — Analyse interprétative du latent VGAE.

Deux blocs (gnn_futur §9.3) :

  BLOC EMBEDDING (toujours, AXE-LIBRE, indépendant des perturbations)
    - Communautés Louvain sur le graphe kNN du latent `z`
      (gene_embeddings_vgae.csv).
    - UMAP 2D coloré par communauté.
    - ORA hypergéométrique (Reactome + aging DBs) PAR communauté
      (réutilise src/validation/ora/ora_consensus.py).
    - Export ShinyGO (listes de gènes par communauté + background) pour
      cross-check externe.
    - Résumé : community_summary.tsv (taille, top pathways, aging, gènes
      représentatifs, flag `is_novel` = aucun pathway significatif → piste
      découverte hors-littérature).

  BLOC PERTURBATION (OPTIONNEL — si --ranking fourni OU auto-détecté)
    - Annote chaque communauté du driver_score (moy/max, # top-N drivers).
    - Extraction de cibles : par communauté, top gènes par
      centralité-intra-module × driver_score [× confidence si présente].
    - community_drivers.tsv + colonne `community` ajoutée au ranking.

Le bloc embedding ne dépend PAS du cluster (tourne sur les embeddings déjà
calculés). Le bloc perturbation s'active seulement si un ranking existe.

Usage
-----
    # Embedding seul (axe-libre)
    python src/validation/viz/interpret_embedding.py \\
        --run-dir output/gnn_vgae/V5.4.1/v5.4.baseline.s1 --shinygo

    # + croisement post-perturbation
    python src/validation/viz/interpret_embedding.py \\
        --run-dir output/gnn_vgae/V5.4.1/v5.4.baseline.s1 \\
        --ranking output/gnn_vgae/V5.4.1/_TEMP_aggcheck/aligned/cross_seed_gene_ranking.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- imports projet (ORA réutilisé) --------------------------------------- #
_THIS = Path(__file__).resolve()
_ORA_DIR = _THIS.parent.parent / "ora"
if str(_ORA_DIR) not in sys.path:
    sys.path.insert(0, str(_ORA_DIR))
import ora_consensus as ora  # noqa: E402


def _project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "data" / "databases").is_dir():
            return p
    return start.parents[3]


ROOT = _project_root(_THIS)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_embeddings(run_dir: Path) -> pd.DataFrame:
    """genes × latent (index = symbole)."""
    return pd.read_csv(run_dir / "gene_embeddings_vgae.csv", index_col=0)


def autodetect_ranking(run_dir: Path) -> Path | None:
    """Cherche un cross_seed_gene_ranking.tsv plausible (run + parents)."""
    cands = [
        run_dir / "cross_seed_gene_ranking.tsv",
        run_dir / "report_axisV4" / "cross_seed_gene_ranking.tsv",
        run_dir / "cross_seed_report" / "cross_seed_gene_ranking.tsv",
    ]
    par = run_dir.parent
    if par.is_dir():
        for sib in par.iterdir():
            if sib.is_dir() and sib.name.startswith("cross_seed"):
                cands.append(sib / "cross_seed_gene_ranking.tsv")
    for c in cands:
        if c.exists():
            return c
    return None


# --------------------------------------------------------------------------- #
# Bloc EMBEDDING : communautés + UMAP + ORA
# --------------------------------------------------------------------------- #
def build_communities(emb: pd.DataFrame, n_neighbors: int, resolution: float,
                      seed: int):
    """kNN(z) → graphe similarité → communautés Louvain. Renvoie (labels:Series,
    intra_degree:Series, graph:nx.Graph)."""
    import networkx as nx
    from sklearn.neighbors import kneighbors_graph

    X = emb.to_numpy(dtype=np.float32)
    genes = list(emb.index.astype(str))
    # graphe kNN pondéré par similarité (distance → exp(-d²/median²))
    A = kneighbors_graph(X, n_neighbors=n_neighbors, mode="distance",
                         include_self=False)
    A = A.maximum(A.T)            # symétrise
    d = A.data
    scale = np.median(d) ** 2 + 1e-9
    A.data = np.exp(-(d ** 2) / scale)
    G = nx.from_scipy_sparse_array(A)
    parts = nx.community.louvain_communities(G, weight="weight",
                                             resolution=resolution, seed=seed)
    labels = np.empty(len(genes), dtype=int)
    for cid, nodes in enumerate(parts):
        for n in nodes:
            labels[n] = cid
    labels = pd.Series(labels, index=genes, name="community")

    # centralité intra-module = somme des poids vers les voisins de même comm.
    intra = np.zeros(len(genes))
    lab = labels.to_numpy()
    coo = A.tocoo()
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if lab[i] == lab[j]:
            intra[i] += w
    intra = pd.Series(intra, index=genes, name="intra_degree")
    print(f"[communities] {len(parts)} communautés Louvain "
          f"(résolution={resolution}, kNN={n_neighbors}) sur {len(genes)} gènes.")
    return labels, intra, G


def run_umap(emb: pd.DataFrame, n_neighbors: int, seed: int) -> pd.DataFrame:
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.2,
                        random_state=seed, metric="euclidean")
    xy = reducer.fit_transform(emb.to_numpy(dtype=np.float32))
    return pd.DataFrame(xy, index=emb.index.astype(str), columns=["umap_x", "umap_y"])


def per_community_ora(labels: pd.Series, intra: pd.Series, min_size: int,
                      score: pd.Series | None = None):
    """ORA Reactome + aging par communauté. Renvoie un DataFrame résumé."""
    reactome = ora.load_reactome_gmt()
    aging = ora.load_aging_databases()
    background = set(labels.index)  # univers = gènes du graphe
    rows = []
    for cid, genes in labels.groupby(labels):
        members = set(genes.index)
        if len(members) < min_size:
            continue
        re_rows = ora.run_ora(members, background, reactome,
                              min_overlap=3, min_pw_size=5, max_pw_size=500)
        ag_rows = ora.run_ora(members, background, aging,
                              min_overlap=2, min_pw_size=2, max_pw_size=100000)
        sig = [r for r in re_rows if r.p_adj < 0.05]
        top3 = "; ".join(f"{r.pathway.replace('REACTOME_', '')[:38]} (q={r.p_adj:.1e})"
                         for r in re_rows[:3])
        top_aging = (f"{ag_rows[0].pathway} (q={ag_rows[0].p_adj:.1e})"
                     if ag_rows and ag_rows[0].p_adj < 0.05 else "—")
        reps = intra[list(members)].sort_values(ascending=False).head(8).index.tolist()
        rec = {
            "community": cid, "size": len(members),
            "n_sig_pathways": len(sig),
            "is_novel": len(sig) == 0,                # aucun pathway → découverte ?
            "top1_pathway": re_rows[0].pathway.replace("REACTOME_", "") if re_rows else "—",
            "top1_padj": round(re_rows[0].p_adj, 6) if re_rows else None,
            "top3_pathways": top3 or "—",
            "top_aging_db": top_aging,
            "representative_genes": ",".join(reps),
        }
        if score is not None:
            sc = score.reindex(list(members)).dropna()
            rec["driver_mean"] = round(float(sc.mean()), 3) if len(sc) else None
            rec["driver_max"] = round(float(sc.max()), 3) if len(sc) else None
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    return df


# Pathways sénescence-pertinents (fallback si aucune liste ni summary fournis).
# Liste élargie : 1er match dans l'ordre = couleur attribuée.
SENESCENCE_PATHWAYS = [
    "REACTOME_CELL_CYCLE_MITOTIC",
    "REACTOME_DNA_REPAIR",
    "REACTOME_CHROMATIN_MODIFYING_ENZYMES",
    "REACTOME_RESPIRATORY_ELECTRON_TRANSPORT",
    "REACTOME_GLYCOLYSIS",
    "REACTOME_SIGNALING_BY_INTERLEUKINS",
    "REACTOME_SIGNALING_BY_RECEPTOR_TYROSINE_KINASES",
    "REACTOME_SIGNALING_BY_GPCR",
    "REACTOME_RHO_GTPASE_CYCLE",
    "REACTOME_METABOLISM_OF_CARBOHYDRATES",
    "REACTOME_PHOSPHOLIPID_METABOLISM",
    "REACTOME_EPIGENETIC_REGULATION_OF_GENE_EXPRESSION",
    "REACTOME_MRNA_SPLICING",
    "REACTOME_EUKARYOTIC_TRANSLATION_INITIATION",
    "REACTOME_MITOCHONDRIAL_TRANSLATION",
    "REACTOME_NEDDYLATION",
    "REACTOME_PROTEIN_UBIQUITINATION",
    "REACTOME_INTRA_GOLGI_AND_RETROGRADE_GOLGI_TO_ER_TRAFFIC",
    "REACTOME_CLATHRIN_MEDIATED_ENDOCYTOSIS",
    "REACTOME_TRANSCRIPTIONAL_REGULATION_BY_TP53",
]

# Marqueurs cyclés pour distinguer 2 couleurs voisines (par blocs de palette).
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]


def _distinct_colors(n: int):
    """n couleurs catégorielles distinctes. tab20+tab20b+tab20c = 60 teintes
    bien séparées ; au-delà, échantillonnage HSV régulier."""
    import matplotlib.pyplot as plt
    import colorsys
    base = []
    for name in ("tab20", "tab20b", "tab20c"):
        base.extend(list(plt.get_cmap(name).colors))      # 20 chacune → 60
    if n <= len(base):
        return base[:n]
    return [colorsys.hsv_to_rgb(i / n, 0.65, 0.95) for i in range(n)]


def pathways_from_summary(summary: pd.DataFrame, reactome: dict,
                          n: int) -> list[str]:
    """Construit la liste de pathways à afficher à partir du top1 enrichi de
    chaque communauté (dédupliqué, trié par significativité). Lie les deux
    UMAP : 1 couleur ≈ 1 module dominant. Fallback = liste curée."""
    if summary is None or "top1_pathway" not in summary.columns:
        return [p for p in SENESCENCE_PATHWAYS if p in reactome][:n]
    sig = summary[(summary.get("n_sig_pathways", 0) > 0)].copy()
    if "top1_padj" in sig.columns:
        sig = sig.sort_values("top1_padj")
    out, seen = [], set()
    for name in sig["top1_pathway"].astype(str):
        full = name if name.startswith("REACTOME_") else f"REACTOME_{name}"
        if full in reactome and full not in seen:
            out.append(full); seen.add(full)
    # complète avec la liste curée si on n'a pas atteint n
    for p in SENESCENCE_PATHWAYS:
        if len(out) >= n:
            break
        if p in reactome and p not in seen:
            out.append(p); seen.add(p)
    return out[:n] if out else [p for p in SENESCENCE_PATHWAYS if p in reactome][:n]


def pathways_from_top_drivers(score: pd.Series, reactome: dict,
                              n_drivers: int, n_pathways: int,
                              min_size: int = 5, max_size: int = 250,
                              min_hits: int = 3) -> list[str]:
    """Pathways REACTOME **sur-représentés** parmi les meilleurs drivers.
    Pour éviter le biais de taille (les gros pathways génériques contiennent
    mécaniquement plus de top-drivers), on classe par **fold-enrichment** =
    observé / attendu, où attendu = n_drivers · |pathway∩univers| / |univers|.
    Bornes : pathways de `min_size`–`max_size` gènes (∩ univers), ≥ `min_hits`
    top-drivers. Univers = gènes scorés. Dédupliqué, cappé à `n_pathways`."""
    universe = set(score.index.astype(str))
    n_univ = len(universe)
    top_set = set(score.sort_values(ascending=False).head(n_drivers).index.astype(str))
    rows = []
    for pw, members in reactome.items():
        mem = members & universe
        if not (min_size <= len(mem) <= max_size):
            continue
        obs = len(top_set & mem)
        if obs < min_hits:
            continue
        expected = n_drivers * len(mem) / n_univ
        rows.append((pw, obs / expected if expected else 0.0))
    rows.sort(key=lambda t: t[1], reverse=True)
    return [pw for pw, _ in rows[:n_pathways]]


def fig_umap_pathways(xy: pd.DataFrame, reactome: dict, out: Path,
                      pathways: list[str] | None = None,
                      annotate_genes: list[str] | None = None,
                      title: str | None = None):
    """UMAP coloré par appartenance à un pathway REACTOME (1er match dans la
    liste ; gris si aucun). Montre si le latent organise spatialement les
    programmes connus — révélateur sur `no-reactome`. `annotate_genes` =
    noms à étiqueter sur la carte (ex. top drivers)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pw_list = [p for p in (pathways or SENESCENCE_PATHWAYS) if p in reactome]
    genes = list(xy.index)
    gset = set(genes)
    # assigne à chaque gène le 1er pathway de la liste auquel il appartient
    assign = {}
    for pw in pw_list:
        for g in (reactome[pw] & gset):
            assign.setdefault(g, pw)
    colors = _distinct_colors(len(pw_list))
    fig, ax = plt.subplots(figsize=(13, 12))
    none_mask = [g not in assign for g in genes]
    ax.scatter(xy.loc[none_mask, "umap_x"], xy.loc[none_mask, "umap_y"],
               s=3, alpha=0.2, color="lightgrey", label="(autre)")
    for k, pw in enumerate(pw_list):
        gg = [g for g in genes if assign.get(g) == pw]
        if not gg:
            continue
        ax.scatter(xy.loc[gg, "umap_x"], xy.loc[gg, "umap_y"], s=11, alpha=0.85,
                   color=colors[k], marker=_MARKERS[(k // 20) % len(_MARKERS)],
                   label=f"{pw.replace('REACTOME_', '')[:34]} (n={len(gg)})")
    # étiquettes des top drivers (positionnées sur leur point UMAP)
    for g in (annotate_genes or []):
        if g in xy.index:
            ax.annotate(g, (xy.at[g, "umap_x"], xy.at[g, "umap_y"]),
                        fontsize=7, fontweight="bold", color="black",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="black", lw=0.4, alpha=0.8))
    ax.set_title(title or f"UMAP latent VGAE — coloré par pathway REACTOME ({len(pw_list)})")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ncol = 3 if len(pw_list) > 8 else 2
    ax.legend(markerscale=2.5, fontsize=8, ncol=ncol, loc="upper center",
              bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[umap] wrote {out} ({len(pw_list)} pathways)")


def fig_umap_communities(xy: pd.DataFrame, labels: pd.Series, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 8))
    lab = labels.reindex(xy.index)
    cids = sorted(lab.unique())
    colors = _distinct_colors(len(cids))   # ≥60 teintes distinctes (vs tab20)
    for k, cid in enumerate(cids):
        m = lab == cid
        ax.scatter(xy.loc[m, "umap_x"], xy.loc[m, "umap_y"], s=5, alpha=0.65,
                   color=colors[k], marker=_MARKERS[(k // 20) % len(_MARKERS)],
                   label=f"C{cid} (n={int(m.sum())})")
    ax.set_title(f"UMAP latent VGAE — communautés Louvain ({len(cids)}, axe-libre)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ncol = 2 if len(cids) > 18 else 1
    ax.legend(markerscale=3, fontsize=6, ncol=ncol, loc="center left",
              bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[umap] wrote {out} ({len(cids)} communautés)")


def fig_umap_continuous(xy: pd.DataFrame, values: pd.Series, out: Path,
                        title: str, cbar_label: str, cmap: str,
                        diverging: bool = False, clip_pct: float = 0.0,
                        scale: str = "quantile", n_bins: int = 12,
                        gamma: float = 0.5):
    """UMAP coloré par une valeur continue (driver_score, cosine_senescent…).
    Gènes sans valeur → gris. `diverging` centre l'échelle sur 0 (anti/pro).

    `clip_pct` (défaut 0 = AUCUN écrêtage) borne l'échelle aux percentiles
    [clip_pct, 100-clip_pct]. ⚠️ Le défaut historique était 2, ce qui saturait
    les 2 % de tête — c'est-à-dire précisément les drivers : sur V6.1.3
    `driver_score` va de 0.00 à 0.80 mais p98 = 0.30-0.43, donc ~260 gènes
    (dont toutes les cibles) recevaient la même couleur et près de la moitié de
    l'étendue du score était invisible. Sur un score asymétrique dont l'intérêt est la queue haute,
    écrêter détruit le signal ; on montre donc toute l'étendue par défaut, et
    l'intervalle affiché est écrit sur la barre de couleur.

    `scale` gouverne la RÉPARTITION des couleurs sur l'intervalle (branche non
    divergente uniquement) :
      * "quantile" (défaut) — paliers aux quantiles, donc **une couleur par
        tranche d'effectif égal**. C'est la bonne transformation ici : le
        problème de `driver_score` n'est pas sa dynamique mais sa DENSITÉ — la
        moitié des gènes tient sous 0.09 (médiane 0.06-0.09 depuis le
        2026-08-13, contre 0.32 quand le score portait encore ses points de
        base), donc une échelle linéaire les peint tous pareil.
        Les étiquettes de la barre restent les vraies valeurs du score.
      * "power"  — PowerNorm d'exposant `gamma` (<1 étale le bas de l'échelle) ;
        continu, à préférer si les paliers gênent.
      * "linear" — comportement brut.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    vals = values.reindex(xy.index).astype(float)
    have = vals.notna()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.scatter(xy.loc[~have, "umap_x"], xy.loc[~have, "umap_y"],
               s=3, alpha=0.2, color="lightgrey", label="(non noté)")
    if diverging:
        vmax = float(np.nanmax(np.abs(vals[have]))) if have.any() else 1.0
        vmin = -vmax
    elif not have.any():
        vmin, vmax = 0.0, 1.0
    elif clip_pct > 0:
        vmin = float(np.nanpercentile(vals[have], clip_pct))
        vmax = float(np.nanpercentile(vals[have], 100 - clip_pct))
    else:
        vmin = float(np.nanmin(vals[have]))
        vmax = float(np.nanmax(vals[have]))
    kw, spacing, note = dict(cmap=cmap, vmin=vmin, vmax=vmax), None, ""
    if not diverging and have.any() and scale == "quantile":
        bounds = np.unique(np.nanpercentile(
            vals[have], np.linspace(0, 100, n_bins + 1)))
        if len(bounds) > 2:                       # dégénéré → repli linéaire
            cm = plt.get_cmap(cmap, len(bounds) - 1)
            kw = dict(cmap=cm, norm=mcolors.BoundaryNorm(bounds, cm.N))
            spacing = "uniform"                   # 1 palier = 1 tranche égale
            note = f", {len(bounds) - 1} paliers d'effectif égal"
    elif not diverging and scale == "power":
        kw = dict(cmap=cmap,
                  norm=mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax))
        note = f", PowerNorm γ={gamma}"
    sc = ax.scatter(xy.loc[have, "umap_x"], xy.loc[have, "umap_y"],
                    c=vals[have], s=7, alpha=0.85, **kw)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8,
                        **({"spacing": spacing} if spacing else {}))
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f"{cbar_label}  [{vmin:.3g} — {vmax:.3g}]" + note
                   + (f", écrêté à ±{clip_pct} %" if clip_pct > 0 else ""))
    ax.set_title(title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[umap] wrote {out}")


def _rgb_str(c) -> str:
    r, g, b = (int(255 * x) for x in c[:3])
    return f"rgb({r},{g},{b})"


# palettes catégorielles fixes (sémantique stable entre runs)
_TIER_COLORS = {"A_confirmed": "#1b9e77", "B_discovery": "#7570b3",
                "C_effector": "#d95f02", "D_hub": "#999999",
                "E_noise": "#dddddd"}
_DIR_COLORS = {"anti-senescence": "#2166ac", "anti-senescence (mixed)": "#92c5de",
               "pro-senescence": "#b2182b", "pro-senescence (mixed)": "#f4a582",
               "neutral": "#bbbbbb"}

# Catalogue de cibles (docs/target.md §7) → catégorie pour coloration UMAP.
TARGET_CATEGORY = {
    **{g: "1·PI-mTOR/sphingo (valider)" for g in ("OCRL", "SYNJ2", "SMPD1")},
    **{g: "2·séno-druggable" for g in ("UGCG", "GCLC", "GCLM", "NAMPT")},
    **{g: "3·ancrage publi" for g in ("HMGB2", "CYCS")},
    **{g: "noyau anti-sén (DE)" for g in ("H2AFZ", "HMGB1", "ENO1", "PKM",
                                          "FHL2", "CD59", "RAN", "DNMT1")},
    **{g: "multi-source DDR" for g in ("BRCA1", "TP53BP1", "MED1")},
    **{g: "4·pharmaco/épigénome" for g in ("CDK1", "CDK2", "CDK4", "CDK6",
                                           "PLK1", "AURKA", "AURKB", "MTOR",
                                           "AKT1", "EHMT1", "EHMT2", "DOT1L",
                                           "KMT2A", "KMT2C", "KMT2D", "RRM2")},
    **{g: "⚠ ambigu/effecteur" for g in ("ASNS", "TP53", "MYC", "CDKN2A",
                                         "CDKN1A", "CEBPB", "RELA")},
}
_TARGET_COLORS = {"1·PI-mTOR/sphingo (valider)": "#e41a1c",
                  "2·séno-druggable": "#ff7f00", "3·ancrage publi": "#984ea3",
                  "noyau anti-sén (DE)": "#377eb8", "multi-source DDR": "#4daf4a",
                  "4·pharmaco/épigénome": "#a65628", "⚠ ambigu/effecteur": "#f781bf",
                  "(autre)": "#e8e8e8"}

# Gabarit HTML de l'UMAP interactive (placeholders ${...}, substitués par
# string.Template ; le JS n'utilise PAS de '$' pour éviter les collisions).
_INTERACTIVE_TEMPLATE = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>UMAP interactive VGAE</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;padding:8px;}
 #bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px;}
 #bar input{padding:4px 6px;font-size:14px;width:170px;}
 #bar button{padding:5px 10px;font-size:13px;cursor:pointer;}
 #bar button.on{background:#2166ac;color:#fff;}
 #msg{color:#666;font-size:12px;}
 kbd{background:#eee;border:1px solid #ccc;border-radius:3px;padding:0 4px;font-size:11px;}
</style></head><body>
<div id="bar">
  <label>&#128269; Gène : <input id="genebox" list="genelist" placeholder="ex. HMGB2" autocomplete="off"></label>
  <datalist id="genelist">${options}</datalist>
  <button onclick="searchGene()">Localiser</button>
  <button onclick="clearSearch()">Effacer</button>
  <a id="gclink" href="#" target="_blank" rel="noopener" style="display:none;font-size:13px;">&#128279; GeneCards</a>
  <button id="namesbtn" onclick="toggleNames()">Afficher noms (vue)</button>
  <span id="msg">Maintenir <kbd>A</kbd> = noms de la vue &middot; molette = zoom (points s'agrandissent) &middot; clic sur un point = lien GeneCards &middot; ${n_genes} gènes</span>
</div>
${div_html}
<script>
(function(){
  var GENE_XY=${gene_xy}, MARKER=${marker_idx}, NAMES=${names_idx}, SEARCH=${search_idx};
  var BASE=4, NAMES_CAP=600, GD=document.getElementById('umap');
  var keys=Object.keys(GENE_XY), XS=[], YS=[];
  for(var i=0;i<keys.length;i++){var p=GENE_XY[keys[i]];XS.push(p[0]);YS.push(p[1]);}
  var XMIN=Math.min.apply(null,XS),XMAX=Math.max.apply(null,XS);
  var YMIN=Math.min.apply(null,YS),YMAX=Math.max.apply(null,YS);
  var XFULL=XMAX-XMIN||1, YFULL=YMAX-YMIN||1, namesOn=false, held=false;
  function rng(ax){var a=GD.layout[ax];return (a&&a.range)?a.range:null;}
  function curSpan(){
    var xa=rng('xaxis'),ya=rng('yaxis');
    if(!xa||!ya)return[XFULL,YFULL];
    return[Math.abs(xa[1]-xa[0]),Math.abs(ya[1]-ya[0])];
  }
  function rescale(){
    var s=curSpan(),r=Math.max(XFULL/s[0],YFULL/s[1]);
    var size=Math.min(16,Math.max(3,BASE*Math.sqrt(r)));
    Plotly.restyle(GD,{'marker.size':size},MARKER);
  }
  function updateGC(g){
    var a=document.getElementById('gclink');
    a.href='https://www.genecards.org/cgi-bin/carddisp.pl?gene='+encodeURIComponent(g);
    a.textContent='\\uD83D\\uDD17 GeneCards: '+g;a.style.display='inline';
  }
  function highlightGene(g,recenter){
    if(!(g in GENE_XY)){document.getElementById('msg').textContent='Gène introuvable : '+g;return;}
    var x=GENE_XY[g][0],y=GENE_XY[g][1];
    Plotly.restyle(GD,{x:[[x]],y:[[y]],text:[[g]]},[SEARCH]);
    if(recenter){var s=curSpan(),w=Math.max(2.5,s[0]/2),h=Math.max(2.5,s[1]/2);
      Plotly.relayout(GD,{'xaxis.range':[x-w,x+w],'yaxis.range':[y-h,y+h]});}
    updateGC(g);
    document.getElementById('msg').textContent=g+' (x='+x+', y='+y+')';
  }
  function searchGene(){highlightGene(document.getElementById('genebox').value.trim(),true);}
  function clearSearch(){Plotly.restyle(GD,{x:[[]],y:[[]],text:[[]]},[SEARCH]);
    document.getElementById('gclink').style.display='none';}
  function drawNames(){
    if(!namesOn){Plotly.restyle(GD,{x:[[]],y:[[]],text:[[]]},[NAMES]);return;}
    var xa=rng('xaxis'),ya=rng('yaxis'),gx=[],gy=[],gt=[];
    var x0=xa?Math.min(xa[0],xa[1]):-1e9,x1=xa?Math.max(xa[0],xa[1]):1e9;
    var y0=ya?Math.min(ya[0],ya[1]):-1e9,y1=ya?Math.max(ya[0],ya[1]):1e9;
    for(var i=0;i<keys.length;i++){var p=GENE_XY[keys[i]];
      if(p[0]>=x0&&p[0]<=x1&&p[1]>=y0&&p[1]<=y1){gx.push(p[0]);gy.push(p[1]);gt.push(keys[i]);}
    }
    if(gt.length>NAMES_CAP){document.getElementById('msg').textContent='Trop de gènes ('+gt.length+') — zoomez pour afficher les noms (max '+NAMES_CAP+').';
      Plotly.restyle(GD,{x:[[]],y:[[]],text:[[]]},[NAMES]);return;}
    document.getElementById('msg').textContent=gt.length+' noms affichés.';
    Plotly.restyle(GD,{x:[gx],y:[gy],text:[gt]},[NAMES]);
  }
  function toggleNames(){namesOn=!namesOn;
    document.getElementById('namesbtn').className=namesOn?'on':'';drawNames();}
  window.searchGene=searchGene;window.clearSearch=clearSearch;window.toggleNames=toggleNames;
  document.getElementById('genebox').addEventListener('change',searchGene);
  document.addEventListener('keydown',function(e){
    if((e.key==='a'||e.key==='A')&&!held&&e.target.tagName!=='INPUT'){held=true;namesOn=true;drawNames();}});
  document.addEventListener('keyup',function(e){
    if((e.key==='a'||e.key==='A')&&held){held=false;namesOn=false;drawNames();}});
  function init(){
    if(!window.Plotly||!GD||!GD.on){setTimeout(init,200);return;}
    GD.on('plotly_relayout',function(e){
      rescale();
      if(namesOn&&('xaxis.range[0]' in e||'xaxis.autorange' in e))drawNames();
    });
    GD.on('plotly_click',function(d){
      if(!d.points||!d.points.length)return;
      var cd=d.points[0].customdata;
      if(cd&&cd[0]){document.getElementById('genebox').value=cd[0];highlightGene(cd[0],false);}
    });
    rescale();
  }
  init();
})();
</script>
</body></html>
"""


def _assign_first_pathway(genes_index, reactome: dict, pw_list: list[str]):
    """gène → 1er pathway de pw_list auquel il appartient ('(autre)' sinon)."""
    gset = set(genes_index)
    assign = {}
    for pw in pw_list:
        for g in (reactome.get(pw, set()) & gset):
            assign.setdefault(g, pw)
    return [assign.get(g, "(autre)") for g in genes_index]


def fig_umap_interactive(xy: pd.DataFrame, labels: pd.Series,
                         score: pd.Series | None, signed: pd.Series | None,
                         reactome: dict, pw_list: list[str], out: Path,
                         pw_list_drivers: list[str] | None = None,
                         ranking: pd.DataFrame | None = None,
                         n_annotate: int = 15,
                         ablations: dict | None = None,
                         plotly_cdn: bool = False):
    """UMAP interactive (Plotly HTML autonome). Menu déroulant de coloration :
    communautés / pathways (top1 communauté) / pathways top-drivers / cibles /
    driver_score / anti-pro / evidence_tier / direction / Δrang vs ablation(s).
    Hover = gène + métriques + driver_score/rang dans chaque ablation. Recherche
    gène (auto-complétion), lien GeneCards, taille-au-zoom, noms-à-la-vue.
    `ablations` = {nom: Series(driver_score complète)} pour le croisement."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[interactive] plotly absent → HTML sauté (pip install plotly)")
        return
    df = xy.copy()
    df["community"] = labels.reindex(df.index)
    df["driver"] = (score.reindex(df.index) if score is not None
                    else pd.Series(np.nan, index=df.index))
    df["cos"] = (signed.reindex(df.index) if signed is not None
                 else pd.Series(np.nan, index=df.index))
    df["pathway"] = _assign_first_pathway(df.index, reactome, pw_list)
    if pw_list_drivers:
        df["pathway_drv"] = _assign_first_pathway(df.index, reactome, pw_list_drivers)
    # colonnes optionnelles du ranking pour hover + colorations catégorielles
    for col in ("evidence_tier", "direction", "n_aging_dbs"):
        df[col] = (ranking[col].reindex(df.index)
                   if ranking is not None and col in ranking.columns else None)
    df["target_category"] = [TARGET_CATEGORY.get(g, "(autre)") for g in df.index]

    # --- croisement ablations : rang base + par ablation + Δrang ------------- #
    abl_names = list(ablations) if ablations else []
    base_rank = (score.rank(ascending=False, method="min")
                 if score is not None else None)
    if base_rank is not None:
        df["rank_base"] = base_rank.reindex(df.index)
    for name in abl_names:
        s_full = ablations[name].astype(float)
        r_full = s_full.rank(ascending=False, method="min")
        df[f"drv__{name}"] = s_full.reindex(df.index)
        df[f"rank__{name}"] = r_full.reindex(df.index)
        drank = df[f"rank__{name}"] - df["rank_base"]
        df[f"drank__{name}"] = drank
        # échelle de couleur compressée (signed-log) : +↑ = chute sans la source
        df[f"dlr__{name}"] = np.sign(drank) * np.log10(1.0 + np.abs(drank))
        # ligne de hover formatée par gène
        def _fmt(g, nm=name):
            d, r, dd = df.at[g, f"drv__{nm}"], df.at[g, f"rank__{nm}"], df.at[g, f"drank__{nm}"]
            if pd.isna(d) or pd.isna(r):
                return f"{nm}: absent"
            s = f"{nm}: drv {d:.3f} · rang {int(r)}"
            if pd.isna(dd):              # gène absent du ranking de base → pas de Δ
                return s
            return s + f" (Δ{'+' if dd >= 0 else ''}{int(dd)})"
        df[f"hov__{name}"] = [_fmt(g) for g in df.index]

    base_cols = ["__gene__", "community", "driver", "cos",
                 "evidence_tier", "direction", "n_aging_dbs", "pathway"]

    def cd(sub):
        cols = [sub.index.astype(str),
                sub["community"].astype("Int64").astype(str),
                sub["driver"].round(3).astype(str),
                sub["cos"].round(3).astype(str),
                sub["evidence_tier"].astype(str),
                sub["direction"].astype(str),
                sub["n_aging_dbs"].astype(str),
                [str(p).replace("REACTOME_", "")[:40] for p in sub["pathway"]]]
        for name in abl_names:
            cols.append(sub[f"hov__{name}"].astype(str))
        if base_rank is not None:
            cols.append(sub["rank_base"].astype("Int64").astype(str))
        return np.column_stack(cols)

    _i_rank = 8 + len(abl_names)   # index de rank_base dans customdata
    HT = ("<b>%{customdata[0]}</b>"
          + (" · rang %{customdata[" + str(_i_rank) + "]}" if base_rank is not None else "")
          + "<br>communauté C%{customdata[1]}"
          "<br>driver_score %{customdata[2]} · cos_sén %{customdata[3]}"
          "<br>tier %{customdata[4]} · %{customdata[5]} · aging %{customdata[6]}"
          "<br>%{customdata[7]}")
    for k in range(len(abl_names)):
        HT += "<br>%{customdata[" + str(8 + k) + "]}"
    HT += "<extra></extra>"

    traces, groups = [], []

    def add_categorical(group, col, color_map=None, name_fmt=None, order=None):
        cats = order or sorted(df[col].dropna().unique(), key=str)
        pal = _distinct_colors(len(cats)) if color_map is None else None
        for k, c in enumerate(cats):
            s = df[df[col] == c]
            if len(s) == 0:
                continue
            col_c = (color_map.get(str(c), "#cccccc") if color_map
                     else ("lightgrey" if c == "(autre)" else _rgb_str(pal[k])))
            nm = name_fmt(c, len(s)) if name_fmt else f"{c} (n={len(s)})"
            traces.append(go.Scattergl(
                x=s["umap_x"], y=s["umap_y"], mode="markers", name=str(nm)[:36],
                legendgroup=group, marker=dict(size=4, color=col_c),
                customdata=cd(s), hovertemplate=HT, visible=(group == "comm")))
            groups.append(group)

    def add_continuous(group, col, colorscale, cbar, **kw):
        traces.append(go.Scattergl(
            x=df["umap_x"], y=df["umap_y"], mode="markers", name=col,
            showlegend=False,
            marker=dict(size=4, color=df[col], colorscale=colorscale,
                        showscale=True, colorbar=dict(title=cbar), **kw),
            customdata=cd(df), hovertemplate=HT, visible=False))
        groups.append(group)

    # colorations catégorielles
    add_categorical("comm", "community",
                    name_fmt=lambda c, n: f"C{int(c)} (n={n})")
    add_categorical("pw", "pathway",
                    name_fmt=lambda c, n: str(c).replace("REACTOME_", "")[:34])
    if pw_list_drivers:
        add_categorical("pwdrv", "pathway_drv",
                        name_fmt=lambda c, n: str(c).replace("REACTOME_", "")[:34])
    if ranking is not None and "evidence_tier" in ranking.columns:
        add_categorical("tier", "evidence_tier", color_map=_TIER_COLORS)
    if ranking is not None and "direction" in ranking.columns:
        add_categorical("dir", "direction", color_map=_DIR_COLORS)
    # cibles target.md ('(autre)' en premier = en fond, cibles au-dessus)
    tcats = ["(autre)"] + [c for c in _TARGET_COLORS if c != "(autre)"
                           and (df["target_category"] == c).any()]
    add_categorical("cibles", "target_category", color_map=_TARGET_COLORS,
                    order=tcats)
    # colorations continues
    if score is not None:
        add_continuous("drv", "driver", "Viridis", "driver_score")
    if signed is not None:
        m = float(np.nanmax(np.abs(df["cos"]))) if df["cos"].notna().any() else 1.0
        add_continuous("cos", "cos", "RdBu", "cos_sén<br>(rouge=pro)",
                       reversescale=True, cmin=-m, cmax=m)
    # croisement ablations : Δrang base→ablation (rouge = chute sans la source)
    for name in abl_names:
        mm = float(np.nanmax(np.abs(df[f"dlr__{name}"]))) if df[f"dlr__{name}"].notna().any() else 1.0
        add_continuous(f"abl__{name}", f"dlr__{name}", "RdBu",
                       f"Δrang→{name}<br>(rouge=chute)",
                       reversescale=True, cmin=-mm, cmax=mm)
    # étiquettes des top-drivers — TOUJOURS visibles (au-dessus de tout)
    if score is not None and n_annotate > 0:
        top = score.sort_values(ascending=False).head(n_annotate)
        lab = df.reindex([g for g in top.index if g in df.index])
        traces.append(go.Scattergl(
            x=lab["umap_x"], y=lab["umap_y"], mode="text",
            text=list(lab.index.astype(str)), textposition="top center",
            textfont=dict(size=9, color="black"), name="top drivers",
            showlegend=False, hoverinfo="skip", visible=True))
        groups.append("labels")
    # trace "noms (vue)" — peuplée par JS au maintien d'une touche / bouton
    traces.append(go.Scattergl(
        x=[], y=[], mode="text", textfont=dict(size=8, color="#333"),
        name="noms", showlegend=False, hoverinfo="skip", visible=True))
    groups.append("names")
    names_idx = len(traces) - 1
    # trace "recherche" — peuplée par JS depuis le champ de recherche
    traces.append(go.Scattergl(
        x=[], y=[], mode="markers+text", textposition="top center",
        textfont=dict(size=13, color="#000"),
        marker=dict(size=20, color="rgba(0,0,0,0)", symbol="circle",
                    line=dict(width=3, color="black")),
        name="recherche", showlegend=False, hoverinfo="skip", visible=True))
    groups.append("search")
    search_idx = len(traces) - 1

    # indices des traces "marqueurs" (à redimensionner au zoom ; ⊄ textes)
    marker_idx = [i for i, g in enumerate(groups)
                  if g not in ("labels", "names", "search")]

    def vis(active):
        keep = ("labels", "names", "search")
        return [(True if g in keep else g == active) for g in groups]

    def btn(label, group, legend, title):
        return dict(label=label, method="update",
                    args=[{"visible": vis(group)},
                          {"showlegend": legend, "title": title}])

    buttons = [btn("Communautés", "comm", True,
                   "UMAP interactive — communautés Louvain"),
               btn("Pathways (communautés)", "pw", True,
                   "UMAP interactive — pathways top1/communauté")]
    if pw_list_drivers:
        buttons.append(btn("Pathways (top drivers)", "pwdrv", True,
                           "UMAP interactive — pathways des top drivers"))
    buttons.append(btn("Cibles (target.md)", "cibles", True,
                       "UMAP interactive — cibles catalogue target.md"))
    if score is not None:
        buttons.append(btn("driver_score", "drv", False,
                           "UMAP interactive — driver_score"))
    if signed is not None:
        buttons.append(btn("anti / pro sénescence", "cos", False,
                           "UMAP interactive — anti(bleu)/pro(rouge) sénescence"))
    if ranking is not None and "evidence_tier" in ranking.columns:
        buttons.append(btn("evidence_tier", "tier", True,
                           "UMAP interactive — evidence_tier (A→E)"))
    if ranking is not None and "direction" in ranking.columns:
        buttons.append(btn("direction (anti/pro)", "dir", True,
                           "UMAP interactive — direction catégorielle"))
    for name in abl_names:
        buttons.append(btn(f"Δrang → {name}", f"abl__{name}", False,
                           f"UMAP interactive — Δrang base→{name} "
                           f"(rouge = chute sans la source)"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="UMAP interactive — communautés Louvain",
        width=1100, height=850, template="plotly_white",
        legend=dict(font=dict(size=8), itemsizing="constant"),
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.12, yanchor="top")],
        xaxis_title="UMAP-1", yaxis_title="UMAP-2")

    # --- assemblage HTML : div Plotly + contrôles + JS custom --------------- #
    import json
    from string import Template
    div_html = fig.to_html(full_html=False,
                           include_plotlyjs=("cdn" if plotly_cdn else True),
                           div_id="umap")
    gene_xy = {str(g): [round(float(x), 3), round(float(y), 3)]
               for g, x, y in zip(df.index, df["umap_x"], df["umap_y"])}
    options = "".join(f"<option value='{g}'>" for g in sorted(gene_xy))
    page = Template(_INTERACTIVE_TEMPLATE).safe_substitute(
        div_html=div_html, options=options, gene_xy=json.dumps(gene_xy),
        marker_idx=json.dumps(marker_idx), names_idx=names_idx,
        search_idx=search_idx, n_genes=len(gene_xy))
    out.write_text(page, encoding="utf-8")
    print(f"[interactive] wrote {out} ({len(traces)} traces, {len(buttons)} "
          f"colorations, {len(df)} gènes ; recherche+cibles+zoom-size+noms)")


def rank_communities_by_driver(labels: pd.Series, score: pd.Series,
                               summary: pd.DataFrame | None, top_k: int = 8):
    """Classe les communautés par driver_score MOYEN. Top gènes = par
    driver_score BRUT (degree-free, ≠ target_priority hub-biaisé)."""
    df = pd.DataFrame({"community": labels})
    df = df.join(score.rename("driver_score"), how="left")
    df["driver_score"] = df["driver_score"].fillna(0.0)
    pw = (summary.set_index("community")["top1_pathway"].to_dict()
          if summary is not None and "top1_pathway" in summary.columns else {})
    ag = (summary.set_index("community")["top_aging_db"].to_dict()
          if summary is not None and "top_aging_db" in summary.columns else {})
    rows = []
    for cid, sub in df.groupby("community"):
        s = sub["driver_score"]
        top = s.sort_values(ascending=False).head(top_k)
        rows.append({
            "community": cid, "size": len(sub),
            "driver_mean": round(float(s.mean()), 3),
            "driver_median": round(float(s.median()), 3),
            "driver_max": round(float(s.max()), 3),
            "top1_pathway": pw.get(cid, "—"),
            "top_aging_db": ag.get(cid, "—"),
            "top_genes": ",".join(top.index.astype(str).tolist()),
        })
    return (pd.DataFrame(rows)
            .sort_values("driver_mean", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Bloc PERTURBATION (optionnel)
# --------------------------------------------------------------------------- #
def perturbation_cross(labels: pd.Series, intra: pd.Series,
                       ranking: pd.DataFrame, score_col: str, top_n: int):
    """Cibles par communauté = top (centralité-intra × driver_score)."""
    score = ranking[score_col].astype(float)
    # confidence optionnelle (decoy/coherence) si présente
    conf_col = next((c for c in ("decoy_confidence", "signed_coherence")
                     if c in ranking.columns), None)
    df = pd.DataFrame({"community": labels, "intra_degree": intra})
    df = df.join(score.rename("driver_score"), how="left")
    if conf_col:
        df = df.join(ranking[conf_col], how="left")
    df["driver_score"] = df["driver_score"].fillna(0.0)
    # priorité cible = centralité-intra normalisée × driver_score
    inorm = df["intra_degree"] / (df["intra_degree"].max() + 1e-9)
    df["target_priority"] = (inorm * df["driver_score"]).round(4)
    top_drivers = set(score.sort_values(ascending=False).head(top_n).index)
    rows = []
    for cid, sub in df.groupby("community"):
        if len(sub) < 3:
            continue
        tgt = sub.sort_values("target_priority", ascending=False).head(5)
        rows.append({
            "community": cid, "size": len(sub),
            "driver_mean": round(float(sub["driver_score"].mean()), 3),
            "driver_max": round(float(sub["driver_score"].max()), 3),
            f"n_top{top_n}_drivers": int(sub.index.isin(top_drivers).sum()),
            "target_shortlist": ",".join(tgt.index.tolist()),
        })
    out = pd.DataFrame(rows).sort_values("driver_max", ascending=False).reset_index(drop=True)
    return out, df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Run VGAE (contient gene_embeddings_vgae.csv).")
    ap.add_argument("--ranking", type=Path, default=None,
                    help="cross_seed_gene_ranking.tsv (déclenche le bloc "
                         "perturbation). Auto-détecté si absent.")
    ap.add_argument("--score-col", default="driver_score")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="Résolution Louvain (↑ = plus de communautés).")
    ap.add_argument("--min-community-size", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--top-n-genes", type=int, default=8,
                    help="Nb de top gènes (par driver_score brut) listés par "
                         "communauté dans community_driver_ranking.tsv.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shinygo", action="store_true",
                    help="Exporte les listes de gènes par communauté pour ShinyGO.")
    ap.add_argument("--no-umap", action="store_true", help="Saute l'UMAP (rapide).")
    ap.add_argument("--no-interactive", action="store_true",
                    help="Saute l'UMAP interactive Plotly (HTML).")
    ap.add_argument("--pathways", default=None,
                    help="Liste REACTOME_* séparée par des virgules à colorer "
                         "sur l'UMAP pathways (override). Défaut = top1 enrichi "
                         "par communauté (dérivé du summary).")
    ap.add_argument("--n-pathways", type=int, default=18,
                    help="Nombre max de pathways colorés sur l'UMAP pathways.")
    ap.add_argument("--n-top-drivers", type=int, default=200,
                    help="Taille du top-driver pour dériver les pathways de "
                         "umap_pathways_top_drivers.png.")
    ap.add_argument("--annotate-drivers", type=int, default=15,
                    help="Nb de top drivers étiquetés (nom) sur l'UMAP "
                         "pathways des drivers.")
    ap.add_argument("--ablation-configs", default="no-coexpr",
                    help="Configs d'ablation à croiser dans l'UMAP interactive "
                         "(noms séparés par virgule, résolus en sibling de "
                         "<cross_seed>/<config>/). Défaut 'no-coexpr'. Vide = aucun.")
    ap.add_argument("--plotly-cdn", action="store_true",
                    help="UMAP interactive : charger plotly.js depuis le CDN "
                         "(HTML ~300 Ko au lieu de ~14 Mo ; nécessite internet "
                         "à l'ouverture). Recommandé pour un site multi-configs.")
    ap.add_argument("--umap-only", action="store_true",
                    help="Génère uniquement l'UMAP interactive (saute ORA, "
                         "ShinyGO et les PNG statiques) → rapide pour bâtir un site.")
    ap.add_argument("--reuse-umap", action="store_true",
                    help="Réutilise communities.tsv existant (community + umap_x/y) "
                         "au lieu de recalculer Louvain+UMAP → build quasi-instantané.")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or (run_dir / "interpretation")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[interpret] run={run_dir.name}  out={out_dir}")

    emb = load_embeddings(run_dir)
    print(f"[interpret] embeddings : {emb.shape[0]} gènes × {emb.shape[1]} dims")

    # --- BLOC EMBEDDING (toujours) ---
    precomp_xy = None
    reuse_path = out_dir / "communities.tsv"
    if args.reuse_umap and reuse_path.exists():
        ct = pd.read_csv(reuse_path, sep="\t", index_col=0)
        if {"community", "umap_x", "umap_y"}.issubset(ct.columns):
            labels = ct["community"].astype(int); labels.name = "community"
            intra = (ct["intra_degree"] if "intra_degree" in ct.columns
                     else pd.Series(0.0, index=ct.index, name="intra_degree"))
            precomp_xy = ct[["umap_x", "umap_y"]].copy()
            print(f"[interpret] --reuse-umap : {len(labels)} gènes, "
                  f"{labels.nunique()} communautés (depuis communities.tsv)")
    if precomp_xy is None:
        labels, intra, _G = build_communities(emb, args.n_neighbors,
                                              args.resolution, args.seed)

    # score optionnel pour annoter les communautés (driver_score si dispo)
    ranking_path = args.ranking or autodetect_ranking(run_dir)
    ranking = None
    score = None
    if ranking_path and Path(ranking_path).exists():
        rk = pd.read_csv(ranking_path, sep="\t")
        key = "target" if "target" in rk.columns else rk.columns[0]
        ranking = rk.set_index(key)
        if args.score_col in ranking.columns:
            score = ranking[args.score_col].astype(float)
        print(f"[interpret] ranking trouvé : {ranking_path} ({len(ranking)} gènes) "
              f"→ bloc perturbation ACTIVÉ")
    else:
        print("[interpret] aucun ranking → bloc perturbation SAUTÉ (embedding seul)")

    # rankings d'ablation à croiser (sibling configs de <cross_seed>/<config>/)
    ablations = {}
    if ranking_path and args.ablation_configs.strip():
        base_cfg_dir = Path(ranking_path).parent           # .../<config>/
        cross_dir = base_cfg_dir.parent                    # .../cross_seed/
        for name in [c.strip() for c in args.ablation_configs.split(",") if c.strip()]:
            p = cross_dir / name / "cross_seed_gene_ranking.tsv"
            if not p.exists():
                print(f"[interpret] ablation '{name}' introuvable ({p}) → ignorée")
                continue
            rk_a = pd.read_csv(p, sep="\t")
            key_a = "target" if "target" in rk_a.columns else rk_a.columns[0]
            rk_a = rk_a.set_index(key_a)
            if args.score_col in rk_a.columns:
                ablations[name] = rk_a[args.score_col].astype(float)
                print(f"[interpret] ablation croisée : {name} ({len(rk_a)} gènes)")
    if ablations:
        print(f"[interpret] {len(ablations)} ablation(s) dans l'UMAP interactive : "
              f"{', '.join(ablations)}")

    if args.umap_only:
        summary = None
        print("[interpret] --umap-only : ORA/ShinyGO/PNG sautés")
    else:
        summary = per_community_ora(labels, intra, args.min_community_size, score=score)
        summary.to_csv(out_dir / "community_summary.tsv", sep="\t", index=False)
        print(f"[interpret] wrote community_summary.tsv ({len(summary)} communautés ; "
              f"{int(summary['is_novel'].sum())} sans pathway significatif)")

    comm_tbl = pd.DataFrame({"community": labels, "intra_degree": intra.round(3)})

    if not args.no_umap:
        xy = precomp_xy if precomp_xy is not None else run_umap(emb, args.n_neighbors, args.seed)
        comm_tbl = comm_tbl.join(xy)
        reactome = ora.load_reactome_gmt()
        # liste de pathways à colorer (summary si dispo, sinon top-drivers/curé)
        if args.pathways:
            pw_list = [p.strip() for p in args.pathways.split(",") if p.strip()]
        elif summary is not None:
            pw_list = pathways_from_summary(summary, reactome, args.n_pathways)
        elif score is not None:
            pw_list = pathways_from_top_drivers(score, reactome,
                                                args.n_top_drivers, args.n_pathways)
        else:
            pw_list = [p for p in SENESCENCE_PATHWAYS if p in reactome][:args.n_pathways]
        drv_pw = (pathways_from_top_drivers(score, reactome, args.n_top_drivers,
                                            args.n_pathways)
                  if score is not None else None)
        # PNG statiques (sautés en --umap-only)
        if not args.umap_only:
            fig_umap_communities(xy, labels, out_dir / "umap_communities.png")
            fig_umap_pathways(xy, reactome, out_dir / "umap_pathways.png",
                              pathways=pw_list)
            if score is not None:
                fig_umap_continuous(
                    xy, score, out_dir / "umap_driver_score.png",
                    title="UMAP latent VGAE — driver_score",
                    cbar_label="driver_score", cmap="viridis")
                top_lbl = score.sort_values(ascending=False).head(args.annotate_drivers)
                fig_umap_pathways(
                    xy, reactome, out_dir / "umap_pathways_top_drivers.png",
                    pathways=drv_pw, annotate_genes=list(top_lbl.index.astype(str)),
                    title=f"UMAP — pathways des top-{args.n_top_drivers} drivers "
                          f"(étiquettes = top-{args.annotate_drivers})")
            if ranking is not None and "cosine_senescent" in ranking.columns:
                fig_umap_continuous(
                    xy, ranking["cosine_senescent"].astype(float),
                    out_dir / "umap_senescence_direction.png",
                    title="UMAP latent VGAE — anti (<0) / pro (>0) sénescence",
                    cbar_label="cosine_senescent", cmap="coolwarm", diverging=True)
        # UMAP INTERACTIVE (Plotly HTML : sélecteur de coloration + hover + zoom)
        if not args.no_interactive:
            signed = (ranking["cosine_senescent"].astype(float)
                      if ranking is not None and "cosine_senescent" in ranking.columns
                      else None)
            fig_umap_interactive(xy, labels, score, signed, reactome, pw_list,
                                 out_dir / "umap_interactive.html",
                                 pw_list_drivers=drv_pw, ranking=ranking,
                                 n_annotate=args.annotate_drivers,
                                 ablations=ablations or None,
                                 plotly_cdn=args.plotly_cdn)

    if args.shinygo and not args.umap_only:
        comms = {int(c): set(g.index) for c, g in labels.groupby(labels)
                 if len(g) >= args.min_community_size}
        ora.export_for_shinygo(comms, out_dir / "shinygo", background=set(labels.index))
        print(f"[interpret] ShinyGO export → {out_dir/'shinygo'}")

    # --- BLOC PERTURBATION (optionnel) ---
    if not args.umap_only and ranking is not None and args.score_col in ranking.columns:
        drv, gene_tbl = perturbation_cross(labels, intra, ranking, args.score_col, args.top_n)
        drv.to_csv(out_dir / "community_drivers.tsv", sep="\t", index=False)
        comm_tbl = comm_tbl.join(gene_tbl[["driver_score", "target_priority"]])
        print(f"[interpret] wrote community_drivers.tsv ({len(drv)} communautés notées)")
        # Classement des communautés par driver_score MOYEN + top gènes bruts
        rank = rank_communities_by_driver(labels, score, summary, top_k=args.top_n_genes)
        rank.to_csv(out_dir / "community_driver_ranking.tsv", sep="\t", index=False)
        print(f"[interpret] wrote community_driver_ranking.tsv (tri par driver_mean)")
        print("\nTop-8 communautés par driver_score MOYEN :")
        print(rank.head(8)[["community", "size", "driver_mean", "driver_max",
                            "top1_pathway", "top_genes"]].to_string(index=False))

    comm_tbl.to_csv(out_dir / "communities.tsv", sep="\t")
    print(f"[interpret] wrote communities.tsv ({len(comm_tbl)} gènes)")
    if summary is not None:
        print("\nTop communautés (taille × enrichissement) :")
        cols = ["community", "size", "n_sig_pathways", "top1_pathway", "top_aging_db"]
        if "driver_max" in summary.columns:
            cols.append("driver_max")
        print(summary[cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
