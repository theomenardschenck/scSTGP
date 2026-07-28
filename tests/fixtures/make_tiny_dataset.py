#!/usr/bin/env python
"""Generate a TINY but COMPLETE dataset so the pipeline can actually run.

WHY THIS EXISTS
---------------
`config.smoke.yaml` only proves the DAG *resolves*: every rule is skipped for
lack of inputs, because the real inputs are 17 GB and are not — and should not
be — versioned. So nobody outside this machine could execute a single stage.
That is a poor state for a repository meant to be read and reused by a lab.

This generator writes a self-contained dataset of a few dozen genes and eight
bulk samples, in the exact formats the pipeline expects. It runs the real code
path — graph build, VGAE training, in-silico perturbation, cross-seed
aggregation, validation, report — in seconds, on a laptop, offline.

WHAT IT IS NOT
--------------
It is NOT biology. Expression values are drawn from a generator, and the two
"states" differ by an injected effect on a handful of genes. Nothing about the
resulting ranking means anything. Its only claim is that the machinery is
wired: a driver score comes out at the far end, with the right columns.

Use it to check that a refactor did not break the pipeline, and to let a new
reader see the whole thing move before asking for the real data.

Usage
-----
    python tests/fixtures/make_tiny_dataset.py --out data_tiny
    bash workflow/run.sh --backend local --configfile workflow/config/config.tiny.yaml
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

# Real HGNC symbols, so that HGNC alias normalisation and the versioned
# OmniPath caches have something to bite on if those layers are switched on.
# The senescence names are here only so the fixture reads like the real thing;
# the numbers underneath are synthetic.
GENES = [
    # chromatine / réplication
    "HMGB1", "HMGB2", "H2AZ1", "HMGA1", "H1-5", "MKI67", "CDCA2", "ANLN",
    "RAD51AP1", "CKAP2L", "MIS18BP1", "MCM2", "PCNA", "TOP2A",
    # glycolyse / métabolisme
    "ENO1", "LDHA", "PKM", "TPI1", "GAPDH", "HK1", "PFKP", "ALDOA",
    "ASNS", "GCLC", "GCLM", "NAMPT", "IMPDH2",
    # effecteurs de sénescence / SASP
    "CDKN1A", "CDKN2A", "TP53", "IL6", "IL1B", "CXCL8", "SERPINE1", "LMNA",
    "GADD45A", "MDM2", "RB1", "E2F1",
    # facteurs de transcription
    "MYC", "CEBPB", "NFE2L2", "JUN", "FOS", "RELA", "NFKB1", "ATF4", "DDIT3",
    "MAFF", "KLF6", "STAT3", "SP1", "EGR1",
    # signalisation / divers
    "AKT1", "MTOR", "MAPK1", "MAPK3", "EGFR", "CTNNB1", "FHL2", "CD59",
    "CYCS", "SOD2", "CAT", "VIM",
]

# Genes given a real effect between the two groups. Recovering these is the
# fixture's only sanity signal — and even that is weak, since the effect is
# injected, not discovered.
PERTURBED = ["CDKN1A", "CDKN2A", "IL6", "SERPINE1", "HMGB2", "MKI67", "TOP2A"]


def write_expression(out: Path, genes, n_per_group=4, seed=0):
    """Bulk matrix: first column = sample, then one column per gene."""
    rng = np.random.default_rng(seed)
    groups = ["pro"] * n_per_group + ["sen"] * n_per_group
    samples = [f"{g}_{i}" for i, g in enumerate(groups)]

    base = rng.lognormal(mean=2.0, sigma=0.8, size=len(genes))
    rows = []
    for grp in groups:
        vals = base * rng.lognormal(0, 0.15, size=len(genes))
        if grp == "sen":
            for gene in PERTURBED:
                idx = genes.index(gene)
                # up for effectors, down for proliferation markers
                vals[idx] *= 4.0 if gene in ("CDKN1A", "CDKN2A", "IL6", "SERPINE1") else 0.25
        rows.append(vals)

    df = pd.DataFrame(rows, columns=genes)
    df.insert(0, "sample", samples)
    path = out / "gnn_data" / "expr_tiny.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    # samplesheet: NO header, col0 = sample, col1 = group (the pipeline reads it
    # with header=None and sniffs the separator).
    meta = pd.DataFrame({"sample": samples, "group": groups})
    meta.to_csv(out / "gnn_data" / "samplesheet_tiny.tsv", sep="\t",
                index=False, header=False)
    return df, groups


def write_ppi(out: Path, genes, seed=1, density=0.06):
    """STRING-format links + aliases, gzipped, exactly as the builder reads them."""
    rng = np.random.default_rng(seed)
    ppi_dir = out / "PPI"
    ppi_dir.mkdir(parents=True, exist_ok=True)

    sid = {g: f"9606.ENSP{i:011d}" for i, g in enumerate(genes)}
    with gzip.open(ppi_dir / "9606.protein.aliases.v12.0.txt.gz", "wt") as fh:
        fh.write("#string_protein_id\talias\tsource\n")
        for g, s in sid.items():
            fh.write(f"{s}\t{g}\tEnsembl_HGNC\n")

    # Space-separated with a header line — the builder passes sep=" ".
    edges = []
    for i, a in enumerate(genes):
        for b in genes[i + 1:]:
            if rng.random() < density:
                edges.append((sid[a], sid[b], int(rng.integers(700, 1000))))
    with gzip.open(ppi_dir / "9606.protein.links.v12.0.txt.gz", "wt") as fh:
        fh.write("protein1 protein2 combined_score\n")
        for a, b, s in edges:
            fh.write(f"{a} {b} {s}\n")
            fh.write(f"{b} {a} {s}\n")   # STRING lists each pair both ways
    return len(edges)


def write_reactome(out: Path, genes, seed=2, n_pathways=8):
    """MSigDB GMT: name <tab> url <tab> genes…  (2..MAX genes after intersection)."""
    rng = np.random.default_rng(seed)
    db = out / "databases"
    db.mkdir(parents=True, exist_ok=True)
    with open(db / "c2.cp.reactome.symbols.gmt", "w") as fh:
        for k in range(n_pathways):
            size = int(rng.integers(4, 12))
            members = rng.choice(genes, size=size, replace=False)
            fh.write(f"REACTOME_TINY_PATHWAY_{k}\thttps://example.invalid/{k}\t"
                     + "\t".join(members) + "\n")
    return n_pathways


def write_de(out: Path, genes, seed=3):
    """DE table in the canonical schema (Seurat/MAST naming, auto-detected)."""
    rng = np.random.default_rng(seed)
    lfc = rng.normal(0, 0.6, len(genes))
    for gene in PERTURBED:
        i = genes.index(gene)
        lfc[i] = 2.0 if gene in ("CDKN1A", "CDKN2A", "IL6", "SERPINE1") else -2.0
    pval = np.clip(np.abs(rng.normal(0, 0.2, len(genes))), 1e-8, 1.0)
    pval[[genes.index(g) for g in PERTURBED]] = 1e-6
    df = pd.DataFrame({
        "gene": genes,
        "avg_log2FC": lfc,
        "p_val": pval,
        "p_val_adj": np.clip(pval * 2, 0, 1),
        "stat": lfc / 0.3,
    })
    path = out / "gnn_data" / "DE_tiny_sen_vs_pro.csv"
    df.to_csv(path, index=False)
    return path


def write_aging_dbs(out: Path, genes):
    """The five post-hoc aging references, in the exact shape the scorer expects.

    Section 14 of the scorer reads GenAge, MSigDB Hallmarks, CellAge, AgeAnno and
    a local table — each with its own container, path and column name. They are
    reference sets used only for evaluation, never for training, so stand-ins
    are legitimate here; what matters is that the code path executes.
    """
    import zipfile

    db = out / "databases"
    db.mkdir(parents=True, exist_ok=True)
    ref = PERTURBED + genes[:12]

    # 1. GenAge — a ZIP holding a CSV with a lowercase `symbol` column. The
    #    loader always checks the ZIP, even when the CSV is already extracted.
    genage_csv = db / "genage_human.csv"
    pd.DataFrame({"symbol": ref, "name": ref}).to_csv(genage_csv, index=False)
    with zipfile.ZipFile(db / "genage_human.zip", "w") as z:
        z.write(genage_csv, arcname="genage_human.csv")

    # 2. MSigDB Hallmarks — GMT. At least one name must match AGING_KEYWORDS,
    #    otherwise the aging gene set comes out empty.
    with open(db / "h.all.symbols.gmt", "w") as fh:
        fh.write("HALLMARK_TINY_SENESCENCE\thttps://example.invalid/1\t"
                 + "\t".join(ref) + "\n")
        fh.write("HALLMARK_TINY_P53_PATHWAY\thttps://example.invalid/2\t"
                 + "\t".join(genes[:20]) + "\n")
        fh.write("HALLMARK_TINY_NEUTRAL\thttps://example.invalid/3\t"
                 + "\t".join(genes[20:30]) + "\n")

    # 3. CellAge — a ZIP holding a TSV; the loader picks the first column whose
    #    name contains symbol/gene/name.
    cellage_tsv = db / "cellage3.tsv"
    pd.DataFrame({"gene_symbol": ref, "senescence_effect": ["Induces"] * len(ref)}
                 ).to_csv(cellage_tsv, sep="\t", index=False)
    with zipfile.ZipFile(db / "cellAge.zip", "w") as z:
        z.write(cellage_tsv, arcname="cellage3.tsv")

    # 4. AgeAnno — comma-separated, latin-1, column `gene`.
    ageanno = db / "ageanno"
    ageanno.mkdir(exist_ok=True)
    pd.DataFrame({"gene": ref, "tissue": ["tiny"] * len(ref)}).to_csv(
        ageanno / "aging_DEGs.txt", index=False, encoding="latin-1")

    # 5. Local table — lives at DATA_DIR (not databases/) and uses a CAPITALISED
    #    `Symbol`. Both were wrong on the first attempt; the pipeline said so.
    pd.DataFrame({"Symbol": ref}).to_csv(
        out / "human_age_related_gene.csv", index=False)


def write_gene_set(out: Path):
    """Target list, so the tiny perturbation stays restricted and fast."""
    d = out / "gene_sets"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tiny_targets.txt").write_text("\n".join(PERTURBED) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_tiny", help="dossier de sortie")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples-per-group", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, groups = write_expression(out, GENES, args.samples_per_group, args.seed)
    n_ppi = write_ppi(out, GENES)
    n_pw = write_reactome(out, GENES)
    de_path = write_de(out, GENES)
    write_aging_dbs(out, GENES)
    write_gene_set(out)

    print(f"Jeu minuscule écrit dans {out.resolve()}")
    print(f"  gènes            : {len(GENES)}")
    print(f"  échantillons     : {len(groups)}  ({sorted(set(groups))})")
    print(f"  arêtes PPI       : {n_ppi} paires")
    print(f"  pathways Reactome: {n_pw}")
    print(f"  table DE         : {de_path.name}")
    print("\nLancer :")
    print("  bash workflow/run.sh --backend local "
          "--configfile workflow/config/config.tiny.yaml")


if __name__ == "__main__":
    main()
