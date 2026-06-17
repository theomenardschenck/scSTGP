#!/usr/bin/env python3
"""
Ranking comparison script.

Read several `cross_seed_gene_ranking.tsv` (one per config), turn each one
into a per-gene rank (1 = best driver_score), and write a single annotated
result TSV listing, for every gene, its rank in each selected config plus the
difference relative to a chosen baseline config.

Output layout (indexed by `gene`):
    dir / DE / tier         : annotation pulled from the baseline config
                              (direction, DE-significance, evidence tier)
    <config>                : rank of the gene in that config (1 = top driver)
    delta_<config>          : baseline_rank - config_rank
                              (> 0 = gene climbs vs baseline, < 0 = drops)
    mean / min / max / std  : summary of the rank across all configs

Config names are auto-shortened by stripping the common prefix/suffix shared
by every selected directory (e.g. `v5.4.no-coexpr.s1` -> `no-coexpr`). Use
`--rename full=short` for further custom abbreviations.

Usage
-----
    # explicit list, first one = baseline, keep this gene order
    rank_compare.py --seeds baseline/.../x.tsv no-coexpr/.../x.tsv \
                    --baseline baseline \
                    --genes H2AFZ HMGB1 ENO1 ...

    # every cross_seed_gene_ranking.tsv found under a directory tree
    rank_compare.py --dir output/gnn_vgae/V5.4.1/_TEMP_s1_driverscore \
                    --baseline v5.4.baseline.s1 --rename v5.4.baseline.s1=baseline
"""
from pathlib import Path
from os.path import commonprefix
import argparse
import pandas as pd

RANK_BY = "driver_score"   # column used to rank genes (descending)
GENE_COL = "target"        # gene identifier column

# annotation columns copied from the baseline ranking file (raw -> output name)
DIR_COL = "direction"
DE_COL = "is_de_significant"
TIER_COL = "evidence_tier"

# directory names that don't identify a config on their own: look one level up
_GENERIC_DIRS = {"cross_seed_report", "perturbation", "interpret"}


# ----------------------------------------------------------------------
# config labels
# ----------------------------------------------------------------------

def raw_label(path: Path) -> str:
    """Discriminating directory name for a ranking tsv."""
    if path.stem != "cross_seed_gene_ranking":
        return path.stem
    parent = path.parent
    if parent.name in _GENERIC_DIRS:
        return parent.parent.name
    return parent.name


def shorten(labels: list[str]) -> list[str]:
    """Strip the common leading/trailing dotted tokens shared by all labels."""
    if len(labels) < 2:
        return labels
    toks = [lab.split(".") for lab in labels]

    # common leading tokens
    lead = 0
    while all(len(t) > lead + 1 and t[lead] == toks[0][lead] for t in toks):
        lead += 1
    # common trailing tokens
    tail = 0
    while all(len(t) > lead + tail + 1 and t[-1 - tail] == toks[0][-1 - tail]
              for t in toks):
        tail += 1

    out = []
    for t in toks:
        kept = t[lead: len(t) - tail] if tail else t[lead:]
        out.append(".".join(kept) if kept else ".".join(t))
    return out


def unique(labels: list[str]) -> list[str]:
    """Guarantee uniqueness with a #n suffix on collision."""
    out, seen = [], {}
    for lab in labels:
        if lab in seen:
            seen[lab] += 1
            out.append(f"{lab}#{seen[lab]}")
        else:
            seen[lab] = 0
            out.append(lab)
    return out


def build_labels(files: list[Path], rename: dict[str, str]) -> list[str]:
    labels = shorten([raw_label(f) for f in files])
    labels = [rename.get(lab, lab) for lab in labels]
    return unique(labels)


# ----------------------------------------------------------------------
# ranks + annotation
# ----------------------------------------------------------------------

def read_ranking(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if RANK_BY in df.columns:
        df = df.sort_values(RANK_BY, ascending=False, kind="mergesort")
    if GENE_COL not in df.columns:
        df = df.rename(columns={df.columns[0]: GENE_COL})
    return df


def gene_ranks(df: pd.DataFrame, label: str) -> pd.Series:
    """Series gene -> rank (1 = best driver_score), best kept on duplicates."""
    ranks = pd.Series(
        range(1, len(df) + 1),
        index=pd.Index(df[GENE_COL], name="gene"),
        name=label,
    )
    return ranks[~ranks.index.duplicated(keep="first")]


def annotation(df: pd.DataFrame) -> pd.DataFrame:
    """Compact per-gene annotation block (dir / DE / tier)."""
    g = df.drop_duplicates(subset=GENE_COL).set_index(GENE_COL)
    g.index.name = "gene"
    ann = pd.DataFrame(index=g.index)

    if DIR_COL in g:
        ann["dir"] = (g[DIR_COL].astype(str)
                      .str.replace("anti-senescence", "anti-sen", regex=False)
                      .str.replace("pro-senescence", "pro-sen", regex=False))
    if DE_COL in g:
        ann["DE"] = g[DE_COL].map(lambda v: "DE" if bool(v) is True else "—")
    if TIER_COL in g:
        ann["tier"] = g[TIER_COL].astype(str).str.split("_").str[0]
    return ann


# ----------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", type=Path,
                     help="Directory searched recursively for "
                          "cross_seed_gene_ranking.tsv")
    src.add_argument("--seeds", type=Path, nargs="+",
                     help="Explicit list of ranking tsv files")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Config label (after shortening/rename) used as "
                         "baseline for the delta columns (default: 1st config)")
    ap.add_argument("--genes", type=str, nargs="+", default=None,
                    help="Subset of genes to keep, in the given order")
    ap.add_argument("--top", type=int, default=None,
                    help="Keep the union of the top-N genes of every config "
                         "(a gene is kept if it ranks <= N in any config, "
                         "even if absent from some configs)")
    ap.add_argument("--rename", type=str, nargs="+", default=None,
                    metavar="OLD=NEW",
                    help="Custom config renames applied after auto-shortening")
    ap.add_argument("--sort", action="store_true",
                    help="Sort output by baseline rank instead of keeping the "
                         "--genes order")
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("ranking_summary.tsv"),
                    help="Output tsv path (default: ranking_summary.tsv)")
    args = ap.parse_args()

    rename = {}
    for item in (args.rename or []):
        if "=" not in item:
            raise SystemExit(f"--rename expects OLD=NEW, got: {item}")
        k, v = item.split("=", 1)
        rename[k] = v

    if args.dir:
        files = sorted(args.dir.rglob("cross_seed_gene_ranking.tsv"))
        if not files:
            raise SystemExit(f"No cross_seed_gene_ranking.tsv under {args.dir}")
    else:
        files = [f for f in args.seeds if f.exists()]
        for f in args.seeds:
            if not f.exists():
                print(f"[skip] missing file: {f}")
    if not files:
        raise SystemExit("No usable ranking file found.")

    labels = build_labels(files, rename)
    frames = {lab: read_ranking(f) for lab, f in zip(labels, files)}

    # rank table: one column per config, aligned on the union of genes
    ranks = pd.concat([gene_ranks(frames[lab], lab) for lab in labels], axis=1)

    # baseline column
    base = args.baseline
    if base is not None and base not in labels:
        raise SystemExit(
            f"Baseline '{base}' not found. Available: {', '.join(labels)}")
    if base is None:
        base = labels[0]

    # put baseline rank column first
    ordered = [base] + [c for c in labels if c != base]
    ranks = ranks[ordered]

    # annotation from the baseline file
    ann = annotation(frames[base]).reindex(ranks.index)

    # deltas vs baseline
    deltas = pd.DataFrame(index=ranks.index)
    for c in labels:
        if c == base:
            continue
        deltas[f"delta_{c}"] = ranks[base] - ranks[c]

    # stats across configs
    stats = pd.DataFrame(index=ranks.index)
    stats["mean"] = ranks.mean(axis=1).round(1)
    stats["min"] = ranks.min(axis=1)
    stats["max"] = ranks.max(axis=1)
    stats["std"] = ranks.std(axis=1).round(1)

    # integer display (nullable) for rank-like columns; NaN = gene absent
    int_block = pd.concat([ranks, deltas], axis=1)
    int_block = int_block.astype("Float64").round().astype("Int64")
    stats[["min", "max"]] = stats[["min", "max"]].astype("Float64").astype("Int64")

    table = pd.concat([ann, int_block, stats], axis=1)

    # gene selection / ordering
    if args.genes:
        present = [g for g in args.genes if g in table.index]
        missing = [g for g in args.genes if g not in table.index]
        if missing:
            print(f"[warn] genes not found: {', '.join(missing)}")
        table = table.loc[present]
        if args.sort:
            table = table.sort_values(base, kind="mergesort")
    else:
        if args.top:
            # gene kept if it lands in the top-N of at least one config
            keep = (ranks <= args.top).any(axis=1)
            table = table.loc[keep]
            print(f"[ok] {int(keep.sum())} genes in top-{args.top} of >=1 config")
        # sort by mean rank so the strongest genes come first across configs
        table = table.sort_values("mean", kind="mergesort")

    table.to_csv(args.output, sep="\t")
    print(f"[ok] {len(table)} genes x {len(files)} configs -> {args.output}")
    print(f"[ok] baseline = {base} | configs: {', '.join(ordered)}")


if __name__ == "__main__":
    main()
