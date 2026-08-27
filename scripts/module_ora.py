#!/usr/bin/env python3
"""Module-level ORA: do EXTERNAL gene sets corroborate the five modules,
view by view?

The modules were defined from our own rankings, so testing them against
themselves would be circular. We therefore pick, for each module, the external
gene sets (KEGG / REACTOME / HALLMARK) that best represent it, and test THOSE
sets for enrichment in the top-N of each of the four views. The module is the
hypothesis; the database set is the independent yardstick.

Background = the graph gene universe (never the genome), BH correction over all
tested sets of the collection, as in ora_consensus.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repository root, derived from this file's location (scripts/module_ora.py) so
# the script runs from any working directory. This used to be one machine's
# absolute path, which made it unrunnable anywhere else.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/validation/ora"))
import ora_consensus as ora  # noqa: E402

VIEWS = {
    "T-o": "output_fi/rfi2.pure-dir",
    "T-c": "output_fi/rfi2.pure-legacy",
    "S-o": "output_fi/rfi2.rich-dir",
    "S-c": "output_fi/rfi2.rich-legacy",
}

# module -> [(collection, set name, short label)]
MODULE_SETS = {
    "Écriture chromatinienne": [
        ("REACTOME", "REACTOME_PKMTS_METHYLATE_HISTONE_LYSINES", "PKMT méthylent H3K"),
        ("KEGG", "KEGG_LYSINE_DEGRADATION", "dégradation lysine (KMT)"),
        ("REACTOME", "REACTOME_CHROMATIN_MODIFYING_ENZYMES", "enzymes mod. chromatine"),
    ],
    "Architecture chromatinienne": [
        ("REACTOME", "REACTOME_CELLULAR_SENESCENCE", "sénescence (Reactome)"),
        ("KEGG", "KEGG_CELLULAR_SENESCENCE", "sénescence (KEGG)"),
    ],
    "Phospho-inositides": [
        ("KEGG", "KEGG_INOSITOL_PHOSPHATE_METABOLISM", "métab. inositol-P"),
        ("KEGG", "KEGG_PHOSPHATIDYLINOSITOL_SIGNALING_SYSTEM", "signalisation PI"),
        ("REACTOME", "REACTOME_PI_METABOLISM", "métabolisme PI"),
    ],
    "Sphingolipides": [
        ("KEGG", "KEGG_SPHINGOLIPID_METABOLISM", "métab. sphingolipides"),
        ("REACTOME", "REACTOME_SPHINGOLIPID_METABOLISM", "sphingolipides (Reactome)"),
    ],
    "Interface endothéliale": [
        ("KEGG", "KEGG_ECM_RECEPTOR_INTERACTION", "ECM-récepteur"),
        ("REACTOME", "REACTOME_INTEGRIN_CELL_SURFACE_INTERACTIONS", "intégrines surface"),
        ("REACTOME", "REACTOME_TNFR2_NON_CANONICAL_NF_KB_PATHWAY", "TNFR2 / NF-kB"),
    ],
}

COLL = {"KEGG": ora.load_gmt(ora.KEGG_GMT_PATH),
        "HALLMARK": ora.load_gmt(ora.HALLMARK_GMT_PATH),
        "REACTOME": ora.load_reactome_gmt()}

# Same legacy-symbol fix as build_ora_memoire: without it H2AFZ, SETD8, WHSC1
# and the HIST1* family match no set at all, which is precisely the chromatin
# story this script is meant to corroborate.
_UNIV = set(pd.read_csv(ROOT / "output/ora_memoire/gene_reference.tsv",
                        sep="\t")["target"].astype(str))
for _c in list(COLL):
    COLL[_c], _n = ora.expand_gmt_with_legacy_symbols(COLL[_c], _UNIV)
    print(f"[module_ora] {_c}: +{_n} symboles hérités reconnus")


def ranking(view: str) -> pd.DataFrame:
    p = ROOT / f"output/gnn_vgae/V6.1.3/{VIEWS[view]}/analysis/cross_seed_gene_ranking.tsv"
    d = pd.read_csv(p, sep="\t")
    return d.sort_values("driver_score", ascending=False)


def run(top_n: int = 300) -> pd.DataFrame:
    rows = []
    for view in VIEWS:
        d = ranking(view)
        bg = set(d["target"].astype(str))
        gl = set(d.head(top_n)["target"].astype(str))
        for cname, sets in COLL.items():
            res = ora.run_ora(gl, bg, sets, min_overlap=1, min_pw_size=5,
                              max_pw_size=500)
            q = {r.pathway: (r.p_adj, r.k, r.genes) for r in res}
            for mod, specs in MODULE_SETS.items():
                for c2, sname, label in specs:
                    if c2 != cname:
                        continue
                    padj, k, genes = q.get(sname, (1.0, 0, ""))
                    rows.append(dict(module=mod, set=sname, label=label,
                                     collection=cname, view=view, q=padj, k=k,
                                     genes=genes, top_n=top_n))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for N in (100, 300):
        df = run(N)
        print(f"\n{'='*94}\nTOP-{N} : q (BH) de chaque ensemble externe, par vue\n{'='*94}")
        piv = df.pivot_table(index=["module", "label"], columns="view",
                             values="q", aggfunc="first")[list(VIEWS)]
        kpiv = df.pivot_table(index=["module", "label"], columns="view",
                              values="k", aggfunc="first")[list(VIEWS)]
        for (mod, lab), r in piv.iterrows():
            krow = kpiv.loc[(mod, lab)]
            cells = " ".join(
                f"{v:>9s}" for v in
                [(f"{r[c]:.0e}({int(krow[c])})" if r[c] < 0.05 else "  ns") for c in VIEWS])
            print(f"{mod[:26]:28s} {lab[:26]:28s} {cells}")
        df.to_csv(ROOT / f"output/ora_memoire/module_ora_top{N}.tsv",
                  sep="\t", index=False)
        print(f"-> output/ora_memoire/module_ora_top{N}.tsv")
