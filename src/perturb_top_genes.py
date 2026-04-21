#!/usr/bin/env python3
"""
perturb_top_genes.py — Batch perturbation of the top-ranked GNN genes.

Runs knockdown / knockout / overexpression on the top-N genes by
`vgae_importance` from a trained VGAE run. Optionally also extracts each
gene's dominant REACTOME pathway (smallest REACTOME set containing the
gene, within size bounds) and re-runs the same three modes on the full
pathway.

Every perturbation is delegated to `gnn_perturbation.py` via subprocess,
so this script stays decoupled from the model internals.

Usage
-----
    # Gene-only perturbations (top-20, all three modes)
    python src/perturb_top_genes.py \\
        --run-dir output/gnn_vgae/V3_Run3 \\
        --top-n 20

    # Also perturb each top-gene's dominant REACTOME pathway
    python src/perturb_top_genes.py \\
        --run-dir output/gnn_vgae/V3_Run3 \\
        --top-n 20 --also-pathway

    # Custom gene list ONLY (no top-N), all three modes
    python src/perturb_top_genes.py \\
        --run-dir output/gnn_vgae/V3_Run3 \\
        --top-n 0 --genes-file my_genes.txt

    # Top-20 + a few extras (deduped, top-N first)
    python src/perturb_top_genes.py \\
        --run-dir output/gnn_vgae/V3_Run3 \\
        --top-n 20 --extra-genes ATF3,CEBPB,DDIT3

Outputs
-------
    <run-dir>/perturbation/<mode>_<gene>/                       — per-gene
    <run-dir>/perturbation/<mode>_pw_<pathway_slug>/            — per-pathway
    <run-dir>/perturbation/manifest.csv                         — index
    data/pathway_gene_list/<pathway_slug>.txt                   — pathway lists
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PERTURB_SCRIPT = Path(__file__).resolve().parent / "gnn_perturbation.py"
PATHWAY_LIST_DIR = ROOT / "data/pathway_gene_list"
GMT_PATH = ROOT / "data/databases/c2.cp.reactome.symbols.gmt"

DEFAULT_MODES = ("knockdown", "knockout", "overexpress")


def load_reactome_gmt() -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    with open(GMT_PATH) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, _url, *genes = parts
            sets[name] = {g.strip() for g in genes if g.strip()}
    return sets


def slugify_pathway(name: str, max_len: int = 60) -> str:
    slug = re.sub(r"^REACTOME_", "", name).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug[:max_len]


def find_dominant_pathway(gene: str,
                          reactome: dict[str, set[str]],
                          min_size: int = 5,
                          max_size: int = 500) -> str | None:
    """Smallest REACTOME pathway (within [min,max]) containing the gene."""
    candidates = [
        (name, len(members)) for name, members in reactome.items()
        if gene in members and min_size <= len(members) <= max_size
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], x[0]))
    return candidates[0][0]


def write_pathway_gene_list(pathway: str,
                            reactome: dict[str, set[str]],
                            out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify_pathway(pathway)
    path = out_dir / f"{slug}.txt"
    if path.exists():
        return path
    genes = sorted(reactome[pathway])
    with open(path, "w") as f:
        f.write(f"# Auto-generated from {pathway}\n")
        f.write(f"# Size: {len(genes)} genes\n")
        for g in genes:
            f.write(g + "\n")
    return path


def load_top_genes(run_dir: Path, top_n: int) -> list[str]:
    if top_n <= 0:
        return []
    rk = pd.read_csv(run_dir / "gene_ranking_vgae.csv")
    rk = rk.sort_values("vgae_importance", ascending=False)
    return rk["gene"].head(top_n).astype(str).tolist()


def load_extra_genes(genes_file: Path | None,
                     extra_genes: str | None) -> list[str]:
    """Combine genes from a file and/or a comma-separated string."""
    out: list[str] = []
    if genes_file is not None:
        with open(genes_file) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                out.append(line)
    if extra_genes:
        out.extend(g.strip() for g in extra_genes.split(",") if g.strip())
    return out


def dedupe_preserve_order(genes: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def run_perturbation(run_dir: Path,
                     mode: str,
                     out_dir: Path,
                     gene: str | None = None,
                     gene_list: Path | None = None,
                     factor: float | None = None,
                     skip_existing: bool = True) -> bool:
    """Return True if the run produced (or already had) a summary.json."""
    summary = out_dir / "summary.json"
    if skip_existing and summary.exists():
        print(f"  [skip] {out_dir.name} (summary.json exists)")
        return True

    cmd = [
        sys.executable, str(PERTURB_SCRIPT),
        "--run-dir", str(run_dir),
        "--mode", mode,
        "--out-dir", str(out_dir),
    ]
    if gene is not None:
        cmd += ["--genes", gene]
    if gene_list is not None:
        cmd += ["--gene-list", str(gene_list)]
    if factor is not None:
        cmd += ["--factor", str(factor)]

    print(f"  -> {mode} | out={out_dir.name}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [fail] rc={r.returncode}")
        print(r.stdout[-400:])
        print(r.stderr[-400:])
        return False
    return summary.exists()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Trained VGAE run directory (must contain "
                         "gene_ranking_vgae.csv + best_vgae.pt).")
    ap.add_argument("--top-n", type=int, default=20,
                    help="Number of top-importance genes to perturb (default 20). "
                         "Set to 0 to skip the top-N selection and use only "
                         "--genes-file / --extra-genes.")
    ap.add_argument("--genes-file", type=Path, default=None,
                    help="Optional file with one gene symbol per line "
                         "(# comments allowed). Combined with --top-n.")
    ap.add_argument("--extra-genes", type=str, default=None,
                    help="Optional comma-separated list of additional gene "
                         "symbols. Combined with --top-n and --genes-file.")
    ap.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES),
                    choices=list(DEFAULT_MODES),
                    help="Perturbation modes to run (default: all three).")
    ap.add_argument("--also-pathway", action="store_true",
                    help="For each top gene, also perturb its dominant "
                         "REACTOME pathway (smallest containing set).")
    ap.add_argument("--oe-factor", type=float, default=3.0,
                    help="Multiplier used for overexpress (default 3.0).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if summary.json already exists.")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    perturb_root = run_dir / "perturbation"
    perturb_root.mkdir(parents=True, exist_ok=True)

    top_genes = load_top_genes(run_dir, args.top_n)
    extra_genes = load_extra_genes(args.genes_file, args.extra_genes)
    all_genes = dedupe_preserve_order(top_genes + extra_genes)

    if not all_genes:
        ap.error("No genes to perturb. Provide --top-n > 0 and/or "
                 "--genes-file / --extra-genes.")

    if top_genes:
        print(f"Top-{args.top_n} genes by vgae_importance:")
        print("  " + ", ".join(top_genes))
    if extra_genes:
        deduped_extra = [g for g in extra_genes if g not in set(top_genes)]
        print(f"Extra genes ({len(deduped_extra)} new): "
              + ", ".join(deduped_extra))

    reactome = None
    if args.also_pathway:
        print("Loading REACTOME GMT ...")
        reactome = load_reactome_gmt()
        print(f"  {len(reactome)} pathways")

    manifest: list[dict] = []

    # ---- per-gene perturbations ----
    for gene in all_genes:
        print(f"\n=== gene: {gene} ===")
        for mode in args.modes:
            out = perturb_root / f"{mode}_{gene}"
            factor = args.oe_factor if mode == "overexpress" else None
            ok = run_perturbation(run_dir, mode, out, gene=gene,
                                  factor=factor,
                                  skip_existing=not args.force)
            manifest.append({
                "target_type": "gene",
                "target": gene,
                "pathway": "",
                "mode": mode,
                "factor": factor or "",
                "out_dir": str(out.relative_to(run_dir)),
                "status": "ok" if ok else "fail",
            })

    # ---- per-pathway perturbations (dominant pathway of each top gene) ----
    if args.also_pathway:
        seen_pathways: set[str] = set()
        for gene in all_genes:
            pw = find_dominant_pathway(gene, reactome)
            if pw is None:
                print(f"\n[warn] no suitable REACTOME pathway for {gene}")
                continue
            if pw in seen_pathways:
                print(f"\n=== pathway: {pw} (already done for another gene) ===")
                continue
            seen_pathways.add(pw)

            slug = slugify_pathway(pw)
            gene_list_path = write_pathway_gene_list(pw, reactome,
                                                     PATHWAY_LIST_DIR)
            print(f"\n=== pathway: {pw} "
                  f"({len(reactome[pw])} genes, seeded by {gene}) ===")

            for mode in args.modes:
                out = perturb_root / f"{mode}_pw_{slug}"
                factor = args.oe_factor if mode == "overexpress" else None
                ok = run_perturbation(run_dir, mode, out,
                                      gene_list=gene_list_path,
                                      factor=factor,
                                      skip_existing=not args.force)
                manifest.append({
                    "target_type": "pathway",
                    "target": gene,               # seed gene
                    "pathway": pw,
                    "mode": mode,
                    "factor": factor or "",
                    "out_dir": str(out.relative_to(run_dir)),
                    "status": "ok" if ok else "fail",
                })

    # ---- manifest ----
    manifest_path = perturb_root / "manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"\nWrote manifest: {manifest_path}  ({len(manifest)} runs)")


if __name__ == "__main__":
    main()
