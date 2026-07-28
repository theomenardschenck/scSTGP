#!/usr/bin/env python
# =============================================================================
# 04_figures.py — summary figures for the WGCNA/hdWGCNA/GRNBoost2 benchmark
# Run in the `gnn` env. Reads output/coexpr_benchmark/*; writes figures/.
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Racine du dépôt dérivée du fichier (ce script vit dans src/coexpr_benchmark/),
# surchargeable par STGP_ROOT. Un chemin absolu en dur rendait ces scripts
# injouables hors de la machine d'origine.
ROOT = os.environ.get("STGP_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BM   = os.path.join(ROOT, "output/coexpr_benchmark")
FIG  = os.path.join(BM, "figures"); os.makedirs(FIG, exist_ok=True)
METHODS = {"wgcna_GSE98440": "WGCNA\nbulk GSE98440 (n=6)",
           "wgcna_GSE163251": "WGCNA\nbulk GSE163251 (n=8)",
           "hdwgcna_scrna": "hdWGCNA\nscRNA metacells"}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# --- panel 1: |top module-trait correlation| + n modules --------------------
rows = []
for k, lab in METHODS.items():
    s = dict(l.split("\t") for l in open(os.path.join(BM, k, "summary.txt")))
    mt = pd.read_csv(os.path.join(BM, k, "module_trait.tsv"), sep="\t")
    rows.append((lab, abs(mt.iloc[0]["cor_senescence"]), int(s["n_modules"])))
labs, cors, nmod = zip(*rows)
b = ax[0].bar(labs, cors, color=["#4C72B0", "#55A868", "#C44E52"])
ax[0].set_ylabel("|r| eigengene ~ senescence (top module)")
ax[0].set_ylim(0, 1.05); ax[0].set_title("(A) Senescence module strength")
for bar, n in zip(b, nmod):
    ax[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
               f"{n} mod", ha="center", fontsize=9)

# --- panel 2: GRNBoost2 co-module enrichment --------------------------------
rep = open(os.path.join(BM, "comparison/comparison_report.md")).read()
enr = {}
for line in rep.splitlines():
    if "x |" in line and "|" in line and "enrichment" not in line:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 6 and cells[-2].endswith("x"):
            enr[cells[1]] = float(cells[-2][:-1])
order = ["WGCNA/bulk GSE98440 (n=6)", "WGCNA/bulk GSE163251 (n=8)", "hdWGCNA/scRNA metacells"]
vals = [enr.get(o, np.nan) for o in order]
ax[1].bar([METHODS[k] for k in METHODS], vals, color=["#4C72B0", "#55A868", "#C44E52"])
ax[1].axhline(1.0, ls="--", c="grey", label="random (1x)")
ax[1].set_ylabel("co-module enrichment vs random")
ax[1].set_title("(B) Recovery of GRNBoost2 edges"); ax[1].legend()

# --- panel 3: Tier-1 driver placement (sen-module vs other) -----------------
t1 = pd.read_csv(os.path.join(BM, "comparison/tier1_module_membership.tsv"), sep="\t")
cols = [c for c in t1.columns if c != "gene"]
M = np.zeros((len(t1), len(cols)))
for i, (_, r) in enumerate(t1.iterrows()):
    for j, c in enumerate(cols):
        v = str(r[c])
        M[i, j] = 2 if v.endswith("*") else (0 if v == "—" else 1)
im = ax[2].imshow(M, aspect="auto", cmap=matplotlib.colors.ListedColormap(
    ["#dddddd", "#7BAFD4", "#C44E52"]), vmin=0, vmax=2)
ax[2].set_xticks(range(len(cols))); ax[2].set_xticklabels([c.replace("wgcna_", "").replace("hdwgcna_", "hd:") for c in cols], rotation=30, ha="right", fontsize=8)
ax[2].set_yticks(range(len(t1))); ax[2].set_yticklabels(t1["gene"], fontsize=8)
ax[2].set_title("(C) GNN Tier-1 drivers:\ngrey=filtered, blue=other module, red=senescence module")

plt.tight_layout()
plt.savefig(os.path.join(FIG, "benchmark_summary.png"), dpi=140)
print("wrote", os.path.join(FIG, "benchmark_summary.png"))
