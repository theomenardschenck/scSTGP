#!/usr/bin/env python3
"""
head_to_head_baselines.py — Un outil PLUS SIMPLE sort-il les mêmes cibles que
le GNN, à partir de la SEULE source HuMess ? (target.md §8 head-to-head + §9.4
leave-one-in + §16.1 « OCRL = goulot de réseau »).

Idée : le GNN consomme HuMess via DEUX canaux séparables (gnn_vgae.py:1455-1605) —
  (1) FEATURE  `imp_delta = imp_P16_z − imp_P4_z`  (importance métabolique
      différentielle Corner Sampling, z-scorée) — DÉJÀ la sortie d'un outil
      métabolique simple ; circulaire (dérivé de l'expression) ;
  (2) ARÊTES   cocatalyse GPR (gènes co-catalysant la même réaction), avec un
      attribut différentiel `[in_P4, in_P16]` (réaction perdue/gagnée en P16).

On reconstruit ces canaux SANS GNN et on demande : est-ce qu'un simple
classement / une simple centralité retrouve OCRL/SYNJ2/SMPD1 ? Si oui, le GNN
est redondant pour ces cibles ; si la cible n'émerge qu'après propagation
graphe (driver_score), le GNN gagne sa place.

Baselines implémentées (v1 — côté HuMess, 100 % local) :
  (b) rang par |imp_delta|            — magnitude de remodelage métabolique
      + rang par imp_delta SIGNÉ      — direction pro-/anti-sénescence
  (d) centralité du graphe cocatalyse — degré + betweenness (= « goulot » §16.1)
      [optionnel : nécessite les GPR + networkx]
  (e) connectivité différentielle     — bilan arêtes gagnées(P16)/perdues(P4)

Sortie : table par-gène `head_to_head_baselines.tsv` (rangs de chaque baseline
+ driver_score GNN + rang-shift par cible) + métriques (ρ Spearman, AUROC
univarié vs CellAge) + focus sur les cibles métaboliques du catalogue.

GÉNÉRALISATION à une source d'arêtes arbitraire (`--edge-source`, répétable) :
le canal feature `imp_delta` est spécifique à HuMess, mais TOUTE source-graphe
(reactome_fi, régulons pySCENIC, OmniPath, PPI…) peut être testée par centralité.
**Workflow type** : si une ABLATION désigne une source X comme « porteuse » du
signal (cf. leave-one-out), lancer `--edge-source X <fichier> <colSrc> <colDst>`
et vérifier qu'un simple classement par betweenness de X **ne bat pas** le
driver_score sur les cibles (sinon le GNN est redondant pour ce dataset).
HuMess est OPTIONNEL (`--no-humess` ou source absente → skip propre), et la liste
de cibles suivies est configurable (`--targets`) pour un jeu non-HUVEC.

Données HuMess (à rapatrier du cluster GLiCID si lancé en local) :
  <humess-dir>/models/<COND>/cs/cs_gene_to_importance_<COND>.tsv   (symbol,bigg,importance)
  <humess-dir>/models/<COND>/stats/carveme.gr-rules.tsv            (reaction <TAB> GPR)

Usage
-----
    # HUVEC complet (HuMess + combiné + coexpr)
    python src/validation/reports/head_to_head_baselines.py \\
        --ranking output/interpretation/V5.4.1/cross_seed/baseline/cross_seed_gene_ranking.tsv \\
        --humess-dir data/humess/output_huvec --combined \\
        --coexpr-file data/pyscenic/diff_coexpr/coexpr_diff.tsv \\
        --out output/interpretation/V5.4.1/head_to_head/head_to_head_baselines.tsv

    # source porteuse arbitraire (ex reactome_fi) sur un autre jeu, sans HuMess
    python src/validation/reports/head_to_head_baselines.py \\
        --ranking <ranking>.tsv --no-humess --targets GENE1,GENE2,GENE3 \\
        --edge-source reactome_fi data/reactome_fi/FIsInGene_with_annotations.txt Gene1 Gene2 \\
        --out <out>.tsv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_THIS = Path(__file__).resolve()
ROOT = next((p for p in _THIS.parents if (p / "data" / "databases").is_dir()),
            _THIS.parents[3])

# Conditions HuMess (= noms de sous-dossiers models/<COND>/). Cf. gnn_vgae.py.
HUMESS_CONDITIONS = ["P4", "P16"]

# Cibles du catalogue dont on veut suivre le rang explicitement (target.md §2).
METAB_TARGETS = ["OCRL", "SYNJ2", "SMPD1", "UGCG", "GCLC", "GCLM", "NAMPT"]
# Contraste : cœur chromatine / mito (cibles co-expression ou DE, PAS HuMess-portées).
CONTRAST_TARGETS = ["HMGB2", "H2AFZ", "HMGB1", "SDHB"]

# --- GPR parsing (réplique exacte de gnn_vgae.py:1462-1475) ------------------
_GENE_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\.]+")
_GPR_SKIP = {"and", "or", "AND", "OR", "And", "Or"}


def _parse_gpr(gpr_str: str) -> set[str]:
    cleaned = gpr_str.replace("(", " ").replace(")", " ")
    return {t for t in _GENE_TOKEN_RE.findall(cleaned) if t not in _GPR_SKIP}


def _auc(score, label) -> float:
    """AUROC via rang de Mann-Whitney (sans sklearn ; identique à driver_baselines)."""
    s = np.asarray(score, float); y = np.asarray(label, int)
    ok = ~np.isnan(s); s, y = s[ok], y[ok]
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    df = pd.DataFrame({"s": s, "r": r}); df["r"] = df.groupby("s")["r"].transform("mean")
    return float((df["r"][y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _log1p_zscore(s: pd.Series) -> pd.Series:
    """log1p + z-score sur les gènes PRÉSENTS (réplique _log1p_zscore gnn_vgae.py:1582).

    `s` = importance brute indexée par gène (genes présents dans la condition).
    Retourne le z-score (les gènes absents sont gérés en amont = 0)."""
    if len(s) < 2:
        return pd.Series(0.0, index=s.index)
    vals = np.log1p(s.astype(float))
    mu, sd = vals.mean(), vals.std() + 1e-8
    return (vals - mu) / sd


# --- Canal 1 : feature imp_delta --------------------------------------------
def load_humess_importance(humess_dir: Path,
                           conditions=HUMESS_CONDITIONS) -> pd.DataFrame:
    """Reconstruit imp_P4_z, imp_P16_z, imp_delta, has_humess par gène.

    Réplique gnn_vgae.py:1548-1605 : max d'importance par symbole (un gène est
    aussi important que sa réaction la plus importante), puis log1p + z-score
    par condition sur les gènes présents. imp_delta = imp_P16_z − imp_P4_z."""
    z = {}
    present = {}
    for cond in conditions:
        path = humess_dir / "models" / cond / "cs" / f"cs_gene_to_importance_{cond}.tsv"
        if not path.exists():
            raise FileNotFoundError(
                f"Corner Sampling introuvable : {path}\n"
                f"  → rapatrier depuis le cluster : "
                f"$LAB_DIR/humess/output_huvec/models/{cond}/cs/")
        df = pd.read_csv(path, sep="\t").dropna(subset=["symbol", "importance"])
        agg = df.groupby("symbol")["importance"].max()         # 1 valeur / gène
        z[cond] = _log1p_zscore(agg)                           # z-score sur présents
        present[cond] = set(agg.index)

    genes = sorted(set().union(*present.values()))
    out = pd.DataFrame(index=genes)
    for cond in conditions:
        out[f"imp_{cond}_z"] = z[cond].reindex(genes).fillna(0.0)  # absent → 0
    out["imp_delta"] = out["imp_P16_z"] - out["imp_P4_z"]
    out["has_humess"] = 1
    return out.reset_index(names="gene")


# --- Canal 2 : arêtes cocatalyse GPR ----------------------------------------
def load_cocatalysis(humess_dir: Path,
                     conditions=HUMESS_CONDITIONS) -> pd.DataFrame | None:
    """Paires de gènes co-catalytiques (même GPR rule) + flags [in_P4, in_P16].

    Réplique gnn_vgae.py:1478-1535 (graphe complet intra-réaction)."""
    pair_flags: dict[tuple[str, str], list[float]] = {}
    found = False
    for ci, cond in enumerate(conditions):
        path = humess_dir / "models" / cond / "stats" / "carveme.gr-rules.tsv"
        if not path.exists():
            print(f"    [warn] GPR introuvable : {path} (canal arêtes SKIP pour {cond})")
            continue
        found = True
        with open(path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                genes = sorted(_parse_gpr(parts[1]))
                for a in range(len(genes)):
                    for b in range(a + 1, len(genes)):
                        key = (genes[a], genes[b])
                        pair_flags.setdefault(key, [0.0, 0.0])[ci] = 1.0
    if not found:
        return None
    rows = [(i, j, f[0], f[1]) for (i, j), f in pair_flags.items()]
    return pd.DataFrame(rows, columns=["g1", "g2", "in_P4", "in_P16"])


def cocat_topology(edges: pd.DataFrame) -> pd.DataFrame:
    """Centralité du graphe cocatalyse : degré + betweenness (= goulot §16.1)
    + connectivité différentielle (arêtes gagnées en P16 − perdues en P4)."""
    try:
        import networkx as nx
    except ImportError:
        print("    [warn] networkx absent → betweenness SKIP (pip install networkx)")
        nx = None

    deg: dict[str, int] = {}
    diff: dict[str, float] = {}
    for _, e in edges.iterrows():
        for g in (e.g1, e.g2):
            deg[g] = deg.get(g, 0) + 1
            # in_P16=1,in_P4=0 → +1 (gain sénescent) ; in_P4=1,in_P16=0 → −1 (perte)
            diff[g] = diff.get(g, 0.0) + (e.in_P16 - e.in_P4)
    out = pd.DataFrame({"gene": list(deg.keys())})
    out["cocat_degree"] = out["gene"].map(deg)
    out["cocat_diff_conn"] = out["gene"].map(diff)

    if nx is not None:
        G = nx.Graph()
        G.add_edges_from(edges[["g1", "g2"]].itertuples(index=False, name=None))
        bet = nx.betweenness_centrality(G, normalized=True)
        out["cocat_betweenness"] = out["gene"].map(bet)
    else:
        out["cocat_betweenness"] = np.nan
    return out


# --- Graphes externes pour test (f) combiné + baseline coexpr ----------------
def load_ppi_edges(ppi_dir: Path, universe: set[str],
                   thresh: int = 900) -> pd.DataFrame:
    """Arêtes STRING (combined_score ≥ thresh) mappées ENSP→symbole, restreintes
    à `universe`. Réplique gnn_vgae.py:950-978 (filtre source Ensembl_HGNC)."""
    alias = pd.read_csv(ppi_dir / "9606.protein.aliases.v12.0.txt.gz",
                        sep="\t", compression="gzip")
    alias = alias[alias["alias"].isin(universe) &
                  alias["source"].str.contains("Ensembl_HGNC", na=False)]
    sym2string = dict(zip(alias["alias"], alias["#string_protein_id"]))
    string2sym = {v: k for k, v in sym2string.items()}
    sids = set(sym2string.values())
    raw = pd.read_csv(ppi_dir / "9606.protein.links.v12.0.txt.gz",
                      sep=" ", compression="gzip")
    hc = raw[raw["protein1"].isin(sids) & raw["protein2"].isin(sids) &
             (raw["combined_score"] >= thresh)]
    g1 = hc["protein1"].map(string2sym); g2 = hc["protein2"].map(string2sym)
    return pd.DataFrame({"g1": g1, "g2": g2}).dropna()


def load_signaling_edges(omni_dir: Path) -> pd.DataFrame:
    """SIGNOR + CollecTRI (déjà en symboles HGNC). source/target_symbol."""
    frames = []
    for fn in ("signed_ppi_signor.tsv.gz", "tf_collectri.tsv.gz"):
        p = omni_dir / fn
        if p.exists():
            df = pd.read_csv(p, sep="\t", compression="gzip")
            frames.append(df.rename(columns={"source_symbol": "g1",
                                              "target_symbol": "g2"})[["g1", "g2"]])
        else:
            print(f"    [warn] signaling introuvable : {p}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["g1", "g2"])


def load_coexpr_edges(coexpr_file: Path) -> pd.DataFrame:
    """Réseau co-expression GRNBoost2 (TF→target). Garde `delta` si présent
    (= importance_p16 − importance_p4, analogue par-arête d'imp_delta)."""
    df = pd.read_csv(coexpr_file, sep="\t")
    out = df.rename(columns={"TF": "g1", "target": "g2"})
    cols = ["g1", "g2"] + (["delta"] if "delta" in out.columns else [])
    return out[cols]


def load_edge_list(path: Path, src_col: str, dst_col: str) -> pd.DataFrame:
    """Loader GÉNÉRIQUE pour une source d'arêtes arbitraire (reactome_fi,
    régulons pySCENIC, OmniPath, n'importe quel TSV/CSV gène-gène). Séparateur
    déduit de l'extension (.csv → ',' sinon TAB). Retourne g1, g2 (symboles)."""
    path = Path(path)
    sep = "," if path.suffix == ".csv" else "\t"
    df = pd.read_csv(path, sep=sep, usecols=[src_col, dst_col])
    return (df.rename(columns={src_col: "g1", dst_col: "g2"})
              .dropna().astype(str))


def graph_centrality(edges: pd.DataFrame, label: str,
                     approx_k: int = 500) -> pd.DataFrame:
    """Degré + betweenness (exacte si petit graphe, sinon k-sampling) par gène.
    `edges` = colonnes g1, g2."""
    try:
        import networkx as nx
    except ImportError:
        print(f"    [warn] networkx absent → centralité {label} SKIP")
        return pd.DataFrame(columns=["gene"])
    G = nx.Graph()
    G.add_edges_from(edges[["g1", "g2"]].itertuples(index=False, name=None))
    n = G.number_of_nodes()
    deg = dict(G.degree())
    k = None if n <= approx_k else approx_k
    if k:
        print(f"    {label} : {n} nœuds / {G.number_of_edges()} arêtes "
              f"— betweenness APPROX (k={k})")
    bet = nx.betweenness_centrality(G, k=k, seed=0, normalized=True)
    out = pd.DataFrame({"gene": list(deg)})
    out[f"{label}_degree"] = out["gene"].map(deg)
    out[f"{label}_betweenness"] = out["gene"].map(bet)
    out[f"rank_{label}_betw"] = _rank_desc(out[f"{label}_betweenness"])
    out[f"rank_{label}_deg"] = _rank_desc(out[f"{label}_degree"].astype(float))
    return out


# --- Assemblage head-to-head -------------------------------------------------
def _rank_desc(s: pd.Series) -> pd.Series:
    """Rang 1 = plus grande valeur (NaN → NaN)."""
    return s.rank(ascending=False, method="min")


def _rho(a, b) -> float:
    return round(float(spearmanr(a, b, nan_policy="omit")[0]), 3)


def _rho_driver(df: pd.DataFrame, col: str) -> tuple[float, int]:
    sub = df.dropna(subset=[col, "driver_score"])
    return _rho(sub[col], sub["driver_score"]), len(sub)


def _auroc_pair(df: pd.DataFrame, col: str, pos: set) -> tuple[float, float, int]:
    """AUROC(baseline) vs AUROC(driver) sur le MÊME support (genes où col définie)."""
    sub = df.dropna(subset=[col, "driver_score"]).copy()
    sub["_y"] = sub["gene"].isin(pos).astype(int)
    return (round(_auc(sub[col], sub["_y"]), 3),
            round(_auc(sub["driver_score"], sub["_y"]), 3), len(sub))


# En-têtes lisibles par label de centralité (sinon générique).
_SOURCE_HEADER = {
    "cocat":    "betweenness COCATALYSE (HuMess) → GNN",
    "combined": "betweenness COMBINÉ (cocat∪PPI∪signaling) → GNN [test f]",
    "coexpr":   "betweenness COEXPR (≈WGCNA) → GNN [contrôle positif chromatine]",
}


def run(ranking: Path, out: Path, cellage: Path, humess_dir: Path | None = None,
        with_edges: bool = True, combined: bool = False,
        coexpr_file: Path | None = None, ppi_dir: Path | None = None,
        omni_dir: Path | None = None,
        edge_sources: list[tuple] | None = None,
        focus_targets: list[str] | None = None,
        betw_k: int = 500) -> pd.DataFrame:
    edge_sources = edge_sources or []
    focus = focus_targets or (METAB_TARGETS + CONTRAST_TARGETS)

    # 1. colonne vertébrale = ranking GNN (univers complet)
    gnn = pd.read_csv(ranking, sep="\t").sort_values("driver_score", ascending=False)
    gnn = gnn.reset_index(drop=True)
    gnn["rank_driver"] = gnn.index + 1
    keep = ["target", "driver_score", "rank_driver", "canon_cosine",
            "is_de_significant", "evidence_tier", "direction"]
    keep = [c for c in keep if c in gnn.columns]
    universe = set(gnn["target"].astype(str))
    merged = gnn[keep].rename(columns={"target": "gene"})

    centralities: list[str] = []
    n_humess = 0

    # 2. canal HuMess (feature imp_delta + cocatalyse) — OPTIONNEL :
    #    skip propre si la source n'existe pas (usage sur un autre jeu de données).
    cocat_edges = None
    have_humess = humess_dir and (Path(humess_dir) / "models").exists()
    if have_humess:
        hm = load_humess_importance(humess_dir)
        n_humess = int(hm["has_humess"].sum())
        hm["abs_imp_delta"] = hm["imp_delta"].abs()
        hm["rank_absdelta"] = _rank_desc(hm["abs_imp_delta"])
        hm["rank_signeddelta"] = _rank_desc(hm["imp_delta"])
        if with_edges:
            cocat_edges = load_cocatalysis(humess_dir)
            if cocat_edges is not None:
                print(f"    cocatalyse : {len(cocat_edges)} paires")
                topo = cocat_topology(cocat_edges)
                topo["rank_cocat_betw"] = _rank_desc(topo["cocat_betweenness"])
                hm = hm.merge(topo, on="gene", how="left")
                centralities.append("cocat")
        merged = merged.merge(hm, on="gene", how="left")
    elif humess_dir:
        print(f"  [info] HuMess absent ({humess_dir}) → canal imp_delta/cocatalyse SKIP")

    # 3. baselines de graphe
    #    (f) graphe COMBINÉ cocatalyse ∪ PPI ∪ signaling
    if combined:
        parts = []
        if cocat_edges is not None:
            parts.append(cocat_edges[["g1", "g2"]])
        if ppi_dir:
            print("  PPI STRING (chargement ~lent)...")
            parts.append(load_ppi_edges(ppi_dir, universe))
        if omni_dir:
            parts.append(load_signaling_edges(omni_dir))
        if parts:
            merged = merged.merge(
                graph_centrality(pd.concat(parts, ignore_index=True), "combined", betw_k),
                on="gene", how="left")
            centralities.append("combined")
    #    coexpr (raccourci built-in)
    if coexpr_file:
        merged = merged.merge(
            graph_centrality(load_coexpr_edges(coexpr_file), "coexpr", betw_k),
            on="gene", how="left")
        centralities.append("coexpr")
    #    sources d'arêtes GÉNÉRIQUES (reactome_fi, pySCENIC, OmniPath, …)
    for label, path, src_col, dst_col in edge_sources:
        print(f"  source '{label}' : {path}")
        edges = load_edge_list(Path(path), src_col, dst_col)
        merged = merged.merge(graph_centrality(edges, label, betw_k),
                              on="gene", how="left")
        centralities.append(label)

    # 4. métriques : ρ au driver + AUROC vs driver sur support commun
    metrics = {"n_humess_genes": n_humess}
    if "abs_imp_delta" in merged.columns:
        metrics["rho_absdelta_driver"] = _rho_driver(merged, "abs_imp_delta")[0]
    for lab in centralities:
        metrics[f"rho_{lab}betw_driver"] = _rho_driver(merged, f"{lab}_betweenness")[0]
    pos = (set(pd.read_csv(cellage, sep="\t")["Gene symbol"].astype(str))
           if Path(cellage).exists() else None)
    aurocs = {}
    if pos is not None:
        pairs = ([("abs_imp_delta", "imp_delta")] if "abs_imp_delta" in merged.columns else [])
        pairs += [(f"{l}_betweenness", f"{l}_betw") for l in centralities]
        for col, lab in pairs:
            if col in merged.columns:
                aurocs[lab] = _auroc_pair(merged, col, pos)

    # 5. sortie TSV
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["gene", "driver_score", "rank_driver"]
    if "imp_delta" in merged.columns:
        cols += ["imp_delta", "abs_imp_delta", "rank_absdelta", "rank_signeddelta"]
    for lab in centralities:
        cols += [f"{lab}_betweenness", f"rank_{lab}_betw"]
    cols += ["canon_cosine", "is_de_significant", "evidence_tier"]
    cols = [c for c in cols if c in merged.columns]
    merged.sort_values("rank_driver").to_csv(out, sep="\t", index=False, columns=cols)

    _print_summary(metrics, aurocs, merged, centralities, focus, out)
    return merged


def _focused_table(merged: pd.DataFrame, rank_col: str, targets: list[str],
                   label: str) -> None:
    present = set(merged["gene"].values)
    show = merged.set_index("gene").reindex(targets)
    for g, row in show.iterrows():
        if g not in present:
            print(f"    {g:8s} : absent du ranking GNN"); continue
        rb = (f"{label}#{int(row[rank_col])}"
              if rank_col in row and pd.notna(row.get(rank_col)) else f"{label}#—")
        rd = int(row["rank_driver"]) if pd.notna(row.get("rank_driver")) else "—"
        print(f"    {g:8s} : {rb:>16s}  →  driver#{rd}")


def _print_summary(metrics, aurocs, merged, centralities, focus, out) -> None:
    print("\n=== head-to-head : un outil simple reproduit-il les cibles du GNN ? ===")
    if "rho_absdelta_driver" in metrics:
        print(f"  gènes HuMess = {metrics['n_humess_genes']}")
        print(f"  ρ(|imp_delta|, driver)        = {metrics['rho_absdelta_driver']}")
    for lab in centralities:
        print(f"  ρ(betweenness {lab:10s}, driver) = {metrics.get(f'rho_{lab}betw_driver','—')}")
    if aurocs:
        print("  AUROC CellAge (baseline vs driver, même support) :")
        for lab, (ab, ad, n) in aurocs.items():
            print(f"    {lab:16s} = {ab} vs driver {ad}   (n={n})")

    if "rank_absdelta" in merged.columns:
        print("\n  --- |imp_delta| (feature HuMess) → GNN ---")
        _focused_table(merged, "rank_absdelta", focus, "|Δ|")
    for lab in centralities:
        header = _SOURCE_HEADER.get(lab, f"betweenness {lab.upper()} → GNN")
        print(f"\n  --- {header} ---")
        _focused_table(merged, f"rank_{lab}_betw", focus, f"{lab}-bet")
    print(f"\n[wrote] {out}")


def _parse_targets(s: str | None) -> list[str] | None:
    return [g.strip() for g in s.split(",") if g.strip()] if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranking", type=Path, required=True,
                    help="cross_seed_gene_ranking.tsv (colonnes target, driver_score)")
    ap.add_argument("--humess-dir", type=Path,
                    default=ROOT / "data/humess/output_huvec",
                    help="racine HuMess locale (models/<COND>/cs|stats/) ; skip si absente")
    ap.add_argument("--no-humess", action="store_true",
                    help="ignore complètement HuMess (usage sur un jeu non-HUVEC)")
    ap.add_argument("--cellage", type=Path, default=ROOT / "data/databases/cellage3.tsv")
    ap.add_argument("--no-edges", action="store_true",
                    help="skip cocatalyse (canal feature imp_delta seul)")
    ap.add_argument("--combined", action="store_true",
                    help="test (f) : centralité du graphe cocatalyse ∪ PPI ∪ signaling")
    ap.add_argument("--ppi-dir", type=Path, default=ROOT / "data/PPI")
    ap.add_argument("--omnipath-dir", type=Path, default=ROOT / "data/omnipath")
    ap.add_argument("--coexpr-file", type=Path, default=None,
                    help="réseau coexpr GRNBoost2 (TF,target) → baseline coexpr")
    ap.add_argument("--edge-source", nargs=4, action="append", default=[],
                    metavar=("LABEL", "PATH", "SRCCOL", "DSTCOL"),
                    help="source d'arêtes GÉNÉRIQUE (répétable). Ex : "
                         "--edge-source reactome_fi data/reactome_fi/FIsInGene_with_annotations.txt Gene1 Gene2")
    ap.add_argument("--targets", default=None,
                    help="liste de cibles à suivre (CSV), ex 'OCRL,SYNJ2,HMGB2' "
                         "(défaut : cibles HUVEC §1+§2)")
    ap.add_argument("--betweenness-k", type=int, default=500,
                    help="échantillon k pour la betweenness approximée (gros graphes)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(args.ranking, args.out, args.cellage,
        humess_dir=None if args.no_humess else args.humess_dir,
        with_edges=not args.no_edges, combined=args.combined,
        coexpr_file=args.coexpr_file,
        ppi_dir=args.ppi_dir if args.combined else None,
        omni_dir=args.omnipath_dir if args.combined else None,
        edge_sources=args.edge_source,
        focus_targets=_parse_targets(args.targets),
        betw_k=args.betweenness_k)


if __name__ == "__main__":
    main()
