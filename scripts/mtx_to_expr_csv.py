#!/usr/bin/env python3
"""mtx_to_expr_csv.py — V6 : convertit la sortie sparse de export_snrna_system.R
(<dir>/cells/{matrix.mtx.gz,genes.tsv,barcodes.tsv} + <dir>/cell_group.tsv) en
matrices expr_<groupe>.csv (cellules × genes) attendues par
build_diff_coexpr.py grnboost2-local, + expr_all.csv (poole).

Usage :
  python scripts/mtx_to_expr_csv.py --dir data/pyscenic/GSE252921_endo
"""
import argparse
import gzip
import os

import numpy as np
import pandas as pd
from scipy.io import mmread


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dossier de sortie export_snrna_system.R")
    ap.add_argument("--normalize", action="store_true",
                    help="log1p(CPM/100) par cellule (defaut : comptes bruts)")
    args = ap.parse_args()

    cells = os.path.join(args.dir, "cells")
    mtx_path = os.path.join(cells, "matrix.mtx.gz")
    opener = gzip.open if mtx_path.endswith(".gz") else open
    with opener(mtx_path, "rb") as fh:
        m = mmread(fh).tocsr()                       # genes x cells
    genes = [l.strip() for l in open(os.path.join(cells, "genes.tsv"))]
    barcodes = [l.strip() for l in open(os.path.join(cells, "barcodes.tsv"))]
    assert m.shape == (len(genes), len(barcodes)), (m.shape, len(genes), len(barcodes))

    X = m.T.toarray().astype(np.float32)             # cells x genes
    if args.normalize:
        libsize = X.sum(axis=1, keepdims=True)
        libsize[libsize == 0] = 1.0
        X = np.log1p(X / libsize * 1e4)

    df = pd.DataFrame(X, index=barcodes, columns=genes)
    cg = pd.read_csv(os.path.join(args.dir, "cell_group.tsv"), sep="\t",
                     header=None, names=["barcode", "group"])
    grp = dict(zip(cg["barcode"].astype(str), cg["group"].astype(str)))

    df.loc[[b for b in df.index if b in grp]].to_csv(
        os.path.join(args.dir, "expr_all.csv"))
    print(f"[conv] expr_all.csv : {df.shape[0]} cellules x {df.shape[1]} genes")
    for g in sorted(set(grp.values())):
        bc = [b for b in df.index if grp.get(b) == g]
        out = os.path.join(args.dir, f"expr_{g}.csv")
        df.loc[bc].to_csv(out)
        print(f"[conv] expr_{g}.csv : {len(bc)} cellules x {df.shape[1]} genes -> {out}")


if __name__ == "__main__":
    main()
