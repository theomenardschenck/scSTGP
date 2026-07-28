#!/usr/bin/env python
# =============================================================================
# 03_compare.py — Cross-method comparison of co-expression modules
# -----------------------------------------------------------------------------
# Compares the modules produced by:
#   - WGCNA on bulk GSE98440 (6 samples)
#   - WGCNA on bulk GSE163251 (8 samples)
#   - hdWGCNA on scRNA HUVEC Drop-seq (metacells)
# along three axes requested by the user:
#   (A) WGCNA-bulk vs hdWGCNA-scRNA  -> module overlap (Jaccard + Fisher)
#   (B) vs GRNBoost2 (current GNN coexpr channel) -> co-module recovery of edges
#   (C) senescence relevance -> module-trait, aging-DB enrichment, Tier-1 drivers
#
# Run in the `gnn` conda env (pandas/scipy). Outputs TSV + a markdown summary
# under output/coexpr_benchmark/comparison/.
# =============================================================================
import os
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom

# Racine du dépôt dérivée du fichier (ce script vit dans src/coexpr_benchmark/),
# surchargeable par STGP_ROOT. Un chemin absolu en dur rendait ces scripts
# injouables hors de la machine d'origine.
ROOT = os.environ.get("STGP_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BM   = os.path.join(ROOT, "output/coexpr_benchmark")
OUT  = os.path.join(BM, "comparison"); os.makedirs(OUT, exist_ok=True)

TIER1 = ["HMGB1", "HMGB2", "H2AFZ", "ENO1", "FHL2", "ASNS", "CEBPB", "NFE2L2",
         "MYC", "MAFF", "CDKN2A", "CDKN1A", "LMNA", "IL6", "LDHA", "PKM", "TPI1", "CYCS", "CD59"]

METHODS = {
    "wgcna_GSE98440": "WGCNA/bulk GSE98440 (n=6)",
    "wgcna_GSE163251": "WGCNA/bulk GSE163251 (n=8)",
    "hdwgcna_scrna":  "hdWGCNA/scRNA metacells",
}

# ---------------------------------------------------------------------------
def load_modules(name):
    f = os.path.join(BM, name, "gene_module.tsv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, sep="\t")
    d = d[d["module"].astype(str).str.lower() != "grey"]      # drop unassigned
    d = d[d["module"].astype(str) != "0"]
    return d[["gene", "module"]].dropna()

def load_aging_dbs():
    dbs = {}
    ca = os.path.join(ROOT, "data/databases/cellage3.tsv")
    if os.path.exists(ca):
        dbs["CellAge"] = set(pd.read_csv(ca, sep="\t")["Gene symbol"].dropna())
    ga = os.path.join(ROOT, "data/databases/genage_human.csv")
    if os.path.exists(ga):
        dbs["GenAge"] = set(pd.read_csv(ga)["symbol"].dropna())
    # SenMayo from MSigDB hallmark file if present (SAASP); else skip
    return dbs

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0

# ===========================================================================
mods = {k: load_modules(k) for k in METHODS}
available = {k: v for k, v in mods.items() if v is not None}
print("Loaded methods:", list(available))

log = ["# Co-expression benchmark — WGCNA (bulk) vs hdWGCNA (scRNA) vs GRNBoost2\n"]

# --- per-method overview ----------------------------------------------------
log.append("## Method overview\n")
log.append("| method | n genes assigned | n modules | median module size |")
log.append("|---|---|---|---|")
for k, d in available.items():
    sizes = d.groupby("module").size()
    log.append(f"| {METHODS[k]} | {len(d)} | {d['module'].nunique()} | {int(sizes.median())} |")
log.append("")

# ===========================================================================
# (A) WGCNA-bulk vs hdWGCNA-scRNA : best-matching module overlap
# ===========================================================================
log.append("## (A) Module concordance: bulk WGCNA vs scRNA hdWGCNA\n")
def module_overlap(dA, dB, nameA, nameB):
    universe = set(dA["gene"]) & set(dB["gene"])
    gA = {m: set(g) & universe for m, g in dA.groupby("module")["gene"]}
    gB = {m: set(g) & universe for m, g in dB.groupby("module")["gene"]}
    N = len(universe)
    rows = []
    for (ma, sa), (mb, sb) in product(gA.items(), gB.items()):
        if not sa or not sb:
            continue
        inter = len(sa & sb)
        if inter == 0:
            continue
        # Fisher exact on 2x2 contingency in the shared universe
        table = [[inter, len(sa) - inter],
                 [len(sb) - inter, N - len(sa) - len(sb) + inter]]
        _, p = fisher_exact(table, alternative="greater")
        rows.append(dict(modA=ma, modB=mb, nA=len(sa), nB=len(sb),
                         overlap=inter, jaccard=round(jaccard(sa, sb), 3), fisher_p=p))
    r = pd.DataFrame(rows).sort_values("fisher_p")
    r["pairing"] = f"{nameA} vs {nameB}"
    return r, N

overlap_all = []
pairs = [("wgcna_GSE98440", "hdwgcna_scrna"),
         ("wgcna_GSE163251", "hdwgcna_scrna"),
         ("wgcna_GSE98440", "wgcna_GSE163251")]
for a, b in pairs:
    if a in available and b in available:
        r, N = module_overlap(available[a], available[b], a, b)
        overlap_all.append(r)
        sig = r[r["fisher_p"] < 0.01]
        log.append(f"**{METHODS[a]}**  vs  **{METHODS[b]}**  (shared universe = {N} genes)")
        log.append(f"- significant module pairs (Fisher p<0.01): {len(sig)} ; "
                   f"max Jaccard = {r['jaccard'].max():.3f}")
        for _, row in r.head(5).iterrows():
            log.append(f"  - {row.modA} ~ {row.modB}: overlap={row.overlap} "
                       f"(J={row.jaccard}, p={row.fisher_p:.1e})")
        log.append("")
if overlap_all:
    pd.concat(overlap_all).to_csv(os.path.join(OUT, "module_overlap.tsv"), sep="\t", index=False)

# ===========================================================================
# (B) vs GRNBoost2 : do co-module gene pairs recover the GRN coexpr edges?
# ===========================================================================
log.append("## (B) Recovery of the GRNBoost2 co-expression channel\n")
grn_f = os.path.join(ROOT, "data/pyscenic/diff_coexpr/coexpr_diff.arboreto.mr5.tsv")
if os.path.exists(grn_f):
    grn = pd.read_csv(grn_f, sep="\t")
    grn_edges = set(map(frozenset, grn[["TF", "target"]].itertuples(index=False, name=None)))
    grn_edges = {e for e in grn_edges if len(e) == 2}
    grn_genes = set(grn["TF"]) | set(grn["target"])
    log.append(f"GRNBoost2 reference: {len(grn_edges)} edges over {len(grn_genes)} genes "
               f"(`arboreto.mr5`).\n")
    log.append("For each method we ask: of the GRNBoost2 edges whose **both** endpoints are "
               "assigned to a module, what fraction land in the **same** module "
               "(co-module rate) vs the random expectation 1/n_modules-weighted?\n")
    log.append("| method | GRN edges scorable | same-module | co-module rate | random exp. | enrichment |")
    log.append("|---|---|---|---|---|---|")
    for k, d in available.items():
        g2m = dict(zip(d["gene"], d["module"]))
        scorable = [(a, b) for e in grn_edges for a, b in [tuple(e)]
                    if a in g2m and b in g2m]
        if not scorable:
            log.append(f"| {METHODS[k]} | 0 | – | – | – | – |")
            continue
        same = sum(1 for a, b in scorable if g2m[a] == g2m[b])
        rate = same / len(scorable)
        # random expectation: sum_m (size_m/N)^2 over genes in scorable universe
        sizes = d.groupby("module").size()
        p = sizes / sizes.sum()
        exp = float((p ** 2).sum())
        log.append(f"| {METHODS[k]} | {len(scorable)} | {same} | {rate:.3f} | "
                   f"{exp:.3f} | {rate/exp:.1f}x |")
    log.append("")
else:
    log.append("_GRNBoost2 reference file not found._\n")

# ===========================================================================
# (C) senescence relevance
# ===========================================================================
log.append("## (C) Senescence relevance\n")

# C1 module-trait (from each method's module_trait.tsv)
log.append("### C1. Top senescence-correlated module per method\n")
log.append("| method | top module | r(eigengene, senescence) | p |")
log.append("|---|---|---|---|")
for k in available:
    mt_f = os.path.join(BM, k, "module_trait.tsv")
    if os.path.exists(mt_f):
        mt = pd.read_csv(mt_f, sep="\t").iloc[0]
        log.append(f"| {METHODS[k]} | {mt['module']} | {mt['cor_senescence']:.3f} | {mt['p_value']:.2g} |")
log.append("")

# C2 aging-DB enrichment of each method's senescence-top module
dbs = load_aging_dbs()
log.append(f"### C2. Aging-DB enrichment of the senescence-associated module "
           f"(DBs: {', '.join(dbs) if dbs else 'none found'})\n")
if dbs:
    log.append("| method | module | n genes | DB | overlap | hypergeom p |")
    log.append("|---|---|---|---|---|---|")
    for k, d in available.items():
        mt_f = os.path.join(BM, k, "module_trait.tsv")
        if not os.path.exists(mt_f):
            continue
        topmod = str(pd.read_csv(mt_f, sep="\t").iloc[0]["module"])
        genes = set(d[d["module"].astype(str) == topmod]["gene"])
        universe = set(d["gene"])
        for dbname, dbset in dbs.items():
            db_u = dbset & universe
            ov = len(genes & db_u)
            p = hypergeom.sf(ov - 1, len(universe), len(db_u), len(genes))
            log.append(f"| {METHODS[k]} | {topmod} | {len(genes)} | {dbname} | {ov} | {p:.2g} |")
    log.append("")

# C3 Tier-1 GNN driver module membership
log.append("### C3. Where do the GNN Tier-1 drivers land? (module + is it the sen-module?)\n")
sen_mod = {}
for k in available:
    mt_f = os.path.join(BM, k, "module_trait.tsv")
    if os.path.exists(mt_f):
        sen_mod[k] = str(pd.read_csv(mt_f, sep="\t").iloc[0]["module"])
rows = []
for g in TIER1:
    row = {"gene": g}
    for k, d in available.items():
        hit = d[d["gene"] == g]
        if len(hit):
            m = str(hit.iloc[0]["module"])
            row[k] = m + ("*" if m == sen_mod.get(k) else "")
        else:
            row[k] = "—"   # not assigned / filtered out (low variance)
    rows.append(row)
t1 = pd.DataFrame(rows)
t1.to_csv(os.path.join(OUT, "tier1_module_membership.tsv"), sep="\t", index=False)
log.append("`*` = gene sits in that method's senescence-associated module; `—` = filtered out (not in variable genes).\n")
log.append(t1.to_markdown(index=False))
log.append("")

# ---------------------------------------------------------------------------
with open(os.path.join(OUT, "comparison_report.md"), "w") as fh:
    fh.write("\n".join(log))
print("Wrote", os.path.join(OUT, "comparison_report.md"))
print("\n".join(log[:40]))
