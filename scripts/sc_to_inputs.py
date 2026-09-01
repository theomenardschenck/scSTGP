#!/usr/bin/env python3
"""Turn any single-cell deposit into the inputs stateshift expects.

WHY THIS EXISTS
---------------
Two entry points already existed for a new dataset: a bulk matrix
(`build_diff_coexpr.py prep-matrices`) and Seurat RDS / GEO tar archives
(`workflow/preprocess/`, which needs R). The formats most GEO scRNA series
actually ship — 10x MatrixMarket triplets, 10x HDF5, or an `.h5ad` — had no
path in. This closes that gap, so a dataset unrelated to the reference system
can be run without touching the code or installing R.

FOUR INPUT SHAPES, ONE OUTPUT SHAPE
-----------------------------------
    --sample GROUP MTX BARCODES     10x MatrixMarket, one per condition
    --tenx-h5 GROUP FILE            10x HDF5 (filtered_feature_bc_matrix.h5)
    --csv GROUP FILE                dense matrix (orientation auto-detected)
    --h5ad FILE --group-col COL     AnnData; the conditions live in .obs

WHAT IT WRITES (into --out-dir)
-------------------------------
    expr_all.csv          cells x genes, first column = cell id   (dataset.expr_matrix)
    expr_<group>.csv      same, one per condition                 (GRNBoost2 input)
    cell_group.tsv        cell id <TAB> group, no header          (dataset.group_meta)
    cell_metadata.csv     cell,group with a header                (pySCENIC / AUCell)
    abundance_table.tsv   genes x pseudobulk samples              (HuMess input)
    samplesheet.tsv       pseudobulk sample <TAB> group           (HuMess input)
    DE_<b>_vs_<a>.tsv     gene / log_fc / pvalue / padj / stat    (readout only)

`cell_group.tsv` is the file to point `dataset.group_meta` at — NOT
`samplesheet.tsv`. The graph build joins the group on the FIRST COLUMN of the
expression matrix, which holds cell ids; a donor-level samplesheet silently
maps to nothing, every group ends up with zero cells and the per-group features
come out empty.

USAGE
-----
    python scripts/sc_to_inputs.py \
        --sample ctrl data/.../GSM2560248_2.1.mtx.gz data/.../GSM2560248_barcodes.tsv.gz \
        --sample stim data/.../GSM2560249_2.2.mtx.gz data/.../GSM2560249_barcodes.tsv.gz \
        --genes data/.../GSE96583_batch2.genes.tsv.gz \
        --cell-meta data/.../GSE96583_batch2.total.tsne.df.tsv.gz \
        --meta-group-col stim --meta-donor-col ind \
        --keep-col multiplets --keep-value singlet \
        --subsample 3000 --min-frac 0.02 \
        --out-dir data/pyscenic/GSE96583

    python scripts/sc_to_inputs.py --h5ad study.h5ad \
        --group-col condition --donor-col patient \
        --groups healthy,disease --out-dir data/pyscenic/study
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_genes(path: Path) -> list[str]:
    """genes.tsv -> gene symbols, made unique (last column wins, ENSEMBL first)."""
    df = pd.read_csv(path, sep="\t", header=None)
    symbols = df.iloc[:, -1].astype(str).str.strip()
    # duplicated symbols are common (readthrough loci); suffix them so the
    # columns stay addressable, then let the graph drop what it does not know.
    seen: dict[str, int] = {}
    out = []
    for s in symbols:
        if s in seen:
            seen[s] += 1
            out.append(f"{s}.{seen[s]}")
        else:
            seen[s] = 0
            out.append(s)
    return out


def read_sample(mtx: Path, barcodes: Path, n_genes: int):
    """Read one 10x sample -> (csr cells x genes, barcode list)."""
    m = scipy.io.mmread(str(mtx)).tocsr()
    bcs = [l.strip() for l in _open(barcodes)]
    if m.shape[0] == n_genes:                    # genes x cells (10x convention)
        m = m.T.tocsr()
    if m.shape[0] != len(bcs):
        raise SystemExit(f"{mtx}: {m.shape[0]} lignes vs {len(bcs)} barcodes")
    return m, bcs



def read_tenx_h5(path: Path):
    """10x HDF5 -> (csr cells x genes, gene symbols, barcodes)."""
    import scanpy as sc
    ad = sc.read_10x_h5(str(path))
    ad.var_names_make_unique()
    return sp.csr_matrix(ad.X), list(ad.var_names), list(ad.obs_names)


def _looks_like_barcodes(labels) -> bool:
    sample = [str(x) for x in list(labels)[:200]]
    if not sample:
        return False
    hits = sum(1 for x in sample
               if len(x) > 12 and x.count("-") <= 2
               and sum(c in "ACGT" for c in x.split("-")[0]) > 0.7 * len(x.split("-")[0]))
    return hits > 0.5 * len(sample)


def orient_frames(frames: list[pd.DataFrame], forced: str) -> list[pd.DataFrame]:
    """Return frames as cells x genes.

    `auto` uses the one signal that is always available with two or more
    conditions: **genes are shared between conditions, cell ids are not**. The
    barcode look-alike test is only the fallback for a single frame. Getting
    this wrong used to produce an empty intersection and a silently empty
    output, so an undecidable case now says so instead of guessing.
    """
    if forced == "cells-rows":
        return frames
    if forced == "genes-rows":
        return [f.T for f in frames]

    if len(frames) >= 2:
        def share(a, b):
            inter = len(set(a) & set(b))
            return inter / max(1, min(len(a), len(b)))
        idx = share(frames[0].index, frames[1].index)
        col = share(frames[0].columns, frames[1].columns)
        if idx > 0.5 and col <= 0.5:
            print(f"[prep] orientation : lignes partagées entre conditions "
                  f"({idx:.0%}) → lignes = gènes, transposition")
            return [f.T for f in frames]
        if col > 0.5 and idx <= 0.5:
            print(f"[prep] orientation : colonnes partagées ({col:.0%}) "
                  f"→ lignes = cellules")
            return frames

    if _looks_like_barcodes(frames[0].columns) and not _looks_like_barcodes(frames[0].index):
        print("[prep] orientation : colonnes façon codes-barres → transposition")
        return [f.T for f in frames]
    if _looks_like_barcodes(frames[0].index):
        print("[prep] orientation : lignes façon codes-barres → lignes = cellules")
        return frames

    raise SystemExit(
        "Orientation de la matrice indécidable : ni les codes-barres ni le "
        "recoupement des étiquettes entre conditions ne tranchent.\n"
        f"  lignes   : {list(frames[0].index)[:4]} …\n"
        f"  colonnes : {list(frames[0].columns)[:4]} …\n"
        "  → précisez --csv-orientation cells-rows | genes-rows")


def read_dense_csv(path: Path) -> pd.DataFrame:
    """Dense matrix, as stored (orientation resolved later, jointly)."""
    return pd.read_csv(path, sep=None, engine="python", index_col=0)


def read_h5ad(path: Path, group_col: str, donor_col: str | None,
              groups: list[str] | None, layer: str | None, use_raw: bool):
    """AnnData -> per-group (csr, ids, donors) plus the shared gene list."""
    import anndata
    ad = anndata.read_h5ad(str(path))
    if use_raw and ad.raw is not None:
        mat, var_names = sp.csr_matrix(ad.raw.X), list(ad.raw.var_names)
    elif layer:
        mat, var_names = sp.csr_matrix(ad.layers[layer]), list(ad.var_names)
    else:
        mat, var_names = sp.csr_matrix(ad.X), list(ad.var_names)
    if group_col not in ad.obs.columns:
        raise SystemExit(f"--group-col {group_col!r} absent de .obs "
                         f"(colonnes : {list(ad.obs.columns)[:15]})")
    labels = ad.obs[group_col].astype(str).to_numpy()
    donors = (ad.obs[donor_col].astype(str).to_numpy() if donor_col
              and donor_col in ad.obs.columns else np.array(["NA"] * ad.n_obs))
    wanted = groups or list(dict.fromkeys(labels))
    if len(wanted) < 2:
        raise SystemExit("il faut au moins deux groupes")
    out = []
    for g in wanted:
        idx = np.flatnonzero(labels == g)
        if idx.size == 0:
            raise SystemExit(f"groupe {g!r} absent de .obs[{group_col!r}] "
                             f"(valeurs : {sorted(set(labels))[:10]})")
        out.append((g, mat[idx], [str(x) for x in ad.obs_names[idx]], donors[idx]))
    return out, var_names


def lognormalize(mat: sp.csr_matrix, target: float = 1e4) -> sp.csr_matrix:
    """Seurat LogNormalize: counts per `target`, then log1p."""
    mat = mat.astype(np.float32)
    totals = np.asarray(mat.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    mat = sp.diags(target / totals, dtype=np.float32) @ mat
    mat.data = np.log1p(mat.data)
    return mat.tocsr()


def wilcoxon_de(mat: sp.csr_matrix, groups: np.ndarray, genes: list[str],
                pole_a: str, pole_b: str) -> pd.DataFrame:
    """Rank-sum test per gene, pole_b vs pole_a, on the normalised matrix.

    Implemented directly (ranks on the dense column) rather than through scanpy
    so the converter keeps a light dependency footprint; the statistic is the
    same one `rank_genes_groups(method="wilcoxon")` reports, tie-corrected.
    """
    from scipy.stats import mannwhitneyu

    a = np.flatnonzero(groups == pole_a)
    b = np.flatnonzero(groups == pole_b)
    dense = np.asarray(mat.todense(), dtype=np.float32)
    mean_a = dense[a].mean(axis=0)
    mean_b = dense[b].mean(axis=0)
    stat, pval = mannwhitneyu(dense[b], dense[a], axis=0, alternative="two-sided")
    # log fold change on the natural log1p scale, as Seurat/scanpy report it
    log_fc = np.log2((np.expm1(mean_b) + 1e-9) / (np.expm1(mean_a) + 1e-9))
    # Benjamini-Hochberg
    order = np.argsort(pval)
    ranked = pval[order]
    n = len(pval)
    padj_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    padj = np.empty(n)
    padj[order] = np.clip(padj_sorted, 0, 1)
    # z-like statistic, signed by the fold change: usable as `stat` anchor
    z = (stat - len(a) * len(b) / 2) / np.sqrt(len(a) * len(b) * (n + 1) / 12.0)
    return pd.DataFrame({
        "gene_symbol": genes, "log_fc": log_fc, "pvalue": pval,
        "padj": padj, "stat": z,
        f"mean_{pole_a}": mean_a, f"mean_{pole_b}": mean_b,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", nargs=3, action="append", metavar=("GROUP", "MTX", "BARCODES"),
                    help="10x MatrixMarket; repeat per condition, first = reference pole")
    ap.add_argument("--tenx-h5", nargs=2, action="append", metavar=("GROUP", "FILE"),
                    help="10x HDF5; repeat per condition")
    ap.add_argument("--csv", nargs=2, action="append", metavar=("GROUP", "FILE"),
                    help="dense matrix per condition (orientation auto-detected)")
    ap.add_argument("--csv-orientation", default="auto",
                    choices=["auto", "cells-rows", "genes-rows"],
                    help="--csv: which axis holds the cells (default: auto)")
    ap.add_argument("--h5ad", default=None, help="AnnData holding every condition")
    ap.add_argument("--group-col", default=None, help="--h5ad: .obs column with the condition")
    ap.add_argument("--donor-col", default=None, help="--h5ad: .obs column with the donor")
    ap.add_argument("--groups", default=None,
                    help="--h5ad: comma-separated groups, reference pole first")
    ap.add_argument("--layer", default=None, help="--h5ad: use this layer instead of .X")
    ap.add_argument("--use-raw", action="store_true", help="--h5ad: use .raw.X")
    ap.add_argument("--already-normalised", action="store_true",
                    help="values are already log-normalised: skip normalisation")
    ap.add_argument("--genes", default=None,
                    help="genes.tsv(.gz) for --sample (symbol in last column)")
    ap.add_argument("--cell-meta", default=None,
                    help="optional per-cell table (index = barcode) for filtering/donors")
    ap.add_argument("--meta-group-col", default=None,
                    help="column of --cell-meta holding the condition (disambiguates "
                         "barcodes reused across samples)")
    ap.add_argument("--meta-donor-col", default=None,
                    help="column holding the donor id (pseudobulk for HuMess)")
    ap.add_argument("--keep-col", default=None, help="filter column, e.g. multiplets")
    ap.add_argument("--keep-value", default=None, help="value to keep, e.g. singlet")
    ap.add_argument("--celltype-col", default=None, help="restrict to one cell type")
    ap.add_argument("--celltype", default=None)
    ap.add_argument("--subsample", type=int, default=3000, help="cells per condition")
    ap.add_argument("--min-frac", type=float, default=0.05,
                    help="keep a gene detected in >= this fraction of cells")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-de", action="store_true", help="skip the differential table")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_modes = sum(bool(x) for x in (args.sample, args.tenx_h5, args.csv, args.h5ad))
    if n_modes != 1:
        raise SystemExit("choisir UN mode : --sample / --tenx-h5 / --csv / --h5ad")

    # every reader returns the same thing: [(group, csr cells x genes, ids, donors)]
    if args.h5ad:
        if not args.group_col:
            raise SystemExit("--h5ad exige --group-col")
        blocks, genes = read_h5ad(
            Path(args.h5ad), args.group_col, args.donor_col,
            [g.strip() for g in args.groups.split(",")] if args.groups else None,
            args.layer, args.use_raw)
    elif args.csv:
        frames = [read_dense_csv(Path(path)) for _, path in args.csv]
        frames = orient_frames(frames, args.csv_orientation)
        genes = [g for g in frames[0].columns
                 if all(g in set(f.columns) for f in frames[1:])]
        if not genes:
            raise SystemExit(
                "Aucun gène commun aux conditions : les colonnes ne se recoupent "
                "pas. Orientation inversée ? → --csv-orientation genes-rows")
        blocks = [(group, sp.csr_matrix(f[genes].to_numpy(dtype=np.float32)),
                   [str(i) for i in f.index], np.array(["NA"] * f.shape[0]))
                  for (group, _), f in zip(args.csv, frames)]
    elif args.tenx_h5:
        blocks, genes = [], None
        for group, path in args.tenx_h5:
            mat, gs, ids = read_tenx_h5(Path(path))
            if genes is None:
                genes = gs
            elif gs != genes:
                common = [g for g in genes if g in set(gs)]
                if not common:
                    raise SystemExit("aucun gène commun entre les fichiers 10x")
                mat = mat[:, [gs.index(g) for g in common]]
                blocks = [(g, m[:, [genes.index(x) for x in common]], i, d)
                          for g, m, i, d in blocks]
                genes = common
                print(f"[prep] gènes communs entre conditions : {len(genes)}")
            blocks.append((group, mat, ids, np.array(["NA"] * mat.shape[0])))
    else:
        if not args.genes:
            raise SystemExit("--sample exige --genes")
        genes = read_genes(Path(args.genes))
        blocks = None
    if genes is None:
        raise SystemExit("aucun gène lu")
    print(f"[prep] {len(genes)} gènes déclarés")

    meta = None
    if args.cell_meta:
        meta = pd.read_csv(args.cell_meta, sep="\t", index_col=0, low_memory=False)
        print(f"[prep] metadata cellule : {meta.shape[0]} lignes, "
              f"colonnes {list(meta.columns)[:8]}")

    mats, ids, grp, donors = [], [], [], []
    iterable = (args.sample if blocks is None
                else [(g, m, i, d) for g, m, i, d in blocks])
    for entry in iterable:
        if blocks is None:
            group, mtx, bcs = entry
            mat, barcodes = read_sample(Path(mtx), Path(bcs), len(genes))
            preset_donor = None
        else:
            group, mat, barcodes, preset_donor = entry
        keep = np.arange(mat.shape[0])
        donor = np.array(["NA"] * mat.shape[0], dtype=object)

        if preset_donor is not None:
            donor = np.asarray(preset_donor, dtype=object)
        if meta is not None:
            sub = meta
            if args.meta_group_col:               # rows of THIS condition only
                sub = sub[sub[args.meta_group_col].astype(str) == group]
            sub = sub[~sub.index.duplicated(keep="first")]
            aligned = sub.reindex(barcodes)
            ok = aligned.notna().any(axis=1).to_numpy()
            if args.keep_col and args.keep_value:
                ok &= (aligned[args.keep_col].astype(str) == args.keep_value).to_numpy()
            if args.celltype_col and args.celltype:
                ok &= (aligned[args.celltype_col].astype(str) == args.celltype).to_numpy()
            keep = np.flatnonzero(ok)
            if args.meta_donor_col:
                donor = aligned[args.meta_donor_col].astype(str).to_numpy()
            print(f"[prep] {group}: {mat.shape[0]} cellules -> {len(keep)} après filtres")

        if 0 < args.subsample < len(keep):
            keep = rng.choice(keep, size=args.subsample, replace=False)
            keep.sort()
        mats.append(mat[keep])
        ids += [f"{group}_{barcodes[i]}" for i in keep]     # unique across samples
        grp += [group] * len(keep)
        donors += [f"{group}_{donor[i]}" for i in keep]
        print(f"[prep] {group}: {len(keep)} cellules retenues")

    mat = sp.vstack(mats).tocsr()
    groups = np.array(grp)
    print(f"[prep] matrice fusionnée : {mat.shape[0]} cellules × {mat.shape[1]} gènes")

    detected = np.asarray((mat > 0).sum(axis=0)).ravel() / mat.shape[0]
    gkeep = np.flatnonzero(detected >= args.min_frac)
    genes_kept = [genes[i] for i in gkeep]
    if not genes_kept:
        raise SystemExit(
            f"Aucun gène retenu au seuil --min-frac {args.min_frac}. Matrice vide, "
            "orientation inversée, ou seuil trop haut pour ces données.")
    counts = mat[:, gkeep]
    print(f"[prep] gènes détectés ≥ {args.min_frac:.0%} : {len(genes_kept)}")

    norm = counts.tocsr() if args.already_normalised else lognormalize(counts)
    if args.already_normalised:
        print("[prep] normalisation SAUTÉE (--already-normalised) ; le pseudobulk "
              "somme alors des valeurs déjà transformées — approximatif")
    dense = pd.DataFrame(np.asarray(norm.todense()), index=ids, columns=genes_kept)
    dense.to_csv(out / "expr_all.csv")
    print(f"[prep] écrit {out/'expr_all.csv'}")
    for group in dict.fromkeys(grp):
        dense.loc[groups == group].to_csv(out / f"expr_{group}.csv")

    pd.DataFrame({"cell": ids, "group": grp}).to_csv(
        out / "cell_group.tsv", sep="\t", header=False, index=False)
    # Same mapping, with a header and as CSV: that is what scenic_from_r.py
    # reads (STGP_SCENIC_META + STGP_SCENIC_CLUSTER_COL=group). Two files for
    # one mapping is redundant, but each consumer wants its own shape.
    pd.DataFrame({"group": grp}, index=pd.Index(ids, name="cell")).to_csv(
        out / "cell_metadata.csv")

    # pseudobulk per donor x condition: HuMess reasons in samples, not cells
    pseudo = pd.DataFrame(np.asarray(counts.todense()), index=donors,
                          columns=genes_kept).groupby(level=0).sum().T
    pseudo.index.name = "gene"
    pseudo.to_csv(out / "abundance_table.tsv", sep="\t")
    sheet = pd.DataFrame({"sample": pseudo.columns,
                          "group": [c.split("_")[0] for c in pseudo.columns]})
    sheet.to_csv(out / "samplesheet.tsv", sep="\t", header=False, index=False)
    print(f"[prep] pseudobulk : {pseudo.shape[1]} échantillons")

    if not args.no_de:
        # poles = first and last condition, in the order they were given
        ordered = list(dict.fromkeys(grp))
        pole_a, pole_b = ordered[0], ordered[-1]
        de = wilcoxon_de(norm, groups, genes_kept, pole_a, pole_b)
        de = de.sort_values("pvalue")
        path = out / f"DE_{pole_b}_vs_{pole_a}.tsv"
        de.to_csv(path, sep="\t", index=False)
        sig = (de.padj < 0.05).sum()
        print(f"[prep] DE {pole_b} vs {pole_a} : {sig} gènes padj<0.05 → {path}")
        print(de.head(10)[["gene_symbol", "log_fc", "padj"]].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
