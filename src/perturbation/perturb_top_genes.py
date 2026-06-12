#!/usr/bin/env python3
"""
perturb_top_genes.py — Batch perturbation of the top-ranked GNN genes.

Two modes:

(1) TOP-N mode (default, subprocess-based, one folder per target)
    Runs knockdown / knockout / overexpression on the top-N genes by
    `vgae_importance` from a trained VGAE run. Optionally also extracts
    each gene's dominant REACTOME pathway and re-runs the same three
    modes on the full pathway. Each perturbation is delegated to
    `gnn_perturbation.py` via subprocess and produces its own directory.

(2) ALL mode (--all-genes / --all-pathways, in-process, no per-target folder)
    Loads the VGAE run once and iterates over EVERY gene (or every
    REACTOME pathway within size bounds), running each mode in-process
    via `gnn_perturbation.run_perturbation_once()`. No per-target folder
    is created — one aggregated TSV per mode is written at the run_dir root:
        perturbation_all_genes_{knockout,knockdown,overexpress}.tsv
        perturbation_all_pathways_{knockout,knockdown,overexpress}.tsv
    One row per target, columns flattened from summary.json. Splitting by
    mode avoids overwriting a previous mode's output on re-runs.

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

    # ALL genes, single mode (fast-ish: ~n_genes forward passes)
    python src/perturb_top_genes.py \\
        --run-dir output/gnn_vgae/V3_Run3 \\
        --all-genes --modes knockdown

    # ALL REACTOME pathways (size in [5, 300]), all three modes
    python src/perturb_top_genes.py \\
        --run-dir output/gnn_vgae/V3_Run3 \\
        --all-pathways --pw-min-size 5 --pw-max-size 300

Outputs
-------
    TOP-N mode:
      <run-dir>/perturbation/<mode>_<gene>/                     — per-gene
      <run-dir>/perturbation/<mode>_pw_<pathway_slug>/          — per-pathway
      <run-dir>/perturbation/manifest.csv                       — index
      data/pathway_gene_list/<pathway_slug>.txt                 — pathway lists

    ALL mode:
      <run-dir>/perturbation_all_genes_{mode}.tsv               — flat TSV
      <run-dir>/perturbation_all_genes_signed_fanout.tsv        — readout signé
                  (long-format par arête S→T ; V5.5, gnn_futur §8.A/§9.2 ;
                   --de-file pour le rôle DE ; DE-free sinon)
      <run-dir>/perturbation_all_pathways_{mode}.tsv            — flat TSV
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Make gnn_perturbation importable under both layouts:
#   * nested (local repo)  : src/perturbation/this.py  +  src/gnn/gnn_perturbation.py
#   * flat   (cluster copy): src/this.py              +  src/gnn_perturbation.py
_THIS_DIR = Path(__file__).resolve().parent
_IMPORT_PATH = (_THIS_DIR if (_THIS_DIR / "gnn_perturbation.py").exists()
                else _THIS_DIR.parent / "gnn")
if str(_IMPORT_PATH) not in sys.path:
    sys.path.insert(0, str(_IMPORT_PATH))

def _find_project_root(start: Path, fallback_levels: int = 1) -> Path:
    """Trouve le projet en cherchant data/databases/ vers le haut.
    Robuste à src/perturbation/foo.py (local) et src/foo.py (cluster flat).
    Override env GNN_PROJECT_ROOT possible."""
    import os
    env = os.environ.get("GNN_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    s = start.resolve()
    for p in [s] + list(s.parents):
        if (p / "data" / "databases").is_dir():
            return p
    return s.parents[fallback_levels]


ROOT = _find_project_root(Path(__file__))
PERTURB_SCRIPT = Path(__file__).resolve().parent / "gnn_perturbation.py"
PATHWAY_LIST_DIR = ROOT / "data/pathway_gene_list"
GMT_PATH = ROOT / "data/databases/c2.cp.reactome.symbols.gmt"

DEFAULT_MODES = ("knockdown", "knockout", "overexpress")

# Colonnes attendues dans le TSV agrégé (--all-genes / --all-pathways).
# L'ordre fixe ici reflète l'ordre CELL_GROUPS dans gnn_perturbation.py.
CELL_GROUPS = ("P4", "P16_cluster_0", "P16_cluster_1",
               "P16_cluster_2", "P16_cluster_3")

# Track if run_perturbation_once supports include_details (for rétro-compatibility).
_SUPPORTS_INCLUDE_DETAILS = True


def _run_perturbation_compat(model, data, gene_symbols, gene_to_idx,
                            spec, base_imp, base_rank, z_cg_base, mu_base,
                            targets, mode, factor, top_k, fdr,
                            reactome, background, group_expr=None,
                            axis_global=None, axes_cluster=None,
                            out_dir=None, include_details=False,
                            ko_mode="mask", ko_soft_factor=0.1,
                            kd_factor=0.15, ko_edge_factor=1.0, legacy=False):
    """Wrapper to run_perturbation_once that handles rétro-compatibility.

    Tries to pass include_details + ko_mode (+ V5.5 kd_factor/ko_edge_factor/
    legacy) if supported; falls back to older signatures.
    """
    from gnn_perturbation import run_perturbation_once

    global _SUPPORTS_INCLUDE_DETAILS
    if not _SUPPORTS_INCLUDE_DETAILS:
        # Old signature: no include_details parameter (and no ko_mode).
        return run_perturbation_once(
            model, data, gene_symbols, gene_to_idx,
            spec, base_imp, base_rank, z_cg_base, mu_base,
            targets, mode, factor, top_k, fdr,
            reactome, background, group_expr,
            axis_global, axes_cluster, out_dir)

    try:
        return run_perturbation_once(
            model, data, gene_symbols, gene_to_idx,
            spec, base_imp, base_rank, z_cg_base, mu_base,
            targets, mode, factor, top_k, fdr,
            reactome, background, group_expr,
            axis_global, axes_cluster, out_dir,
            write_full=True, include_details=include_details,
            ko_mode=ko_mode, ko_soft_factor=ko_soft_factor,
            kd_factor=kd_factor, ko_edge_factor=ko_edge_factor, legacy=legacy)
    except TypeError as e:
        # V5.5 kd_factor/ko_edge_factor/legacy pas supportés → retry sans eux
        if "kd_factor" in str(e) or "ko_edge_factor" in str(e) or "legacy" in str(e):
            print("[compat] gnn_perturbation sans params V5.5 (kd_factor/"
                  "ko_edge_factor/legacy) ; retry avec ko_mode seul.")
            return run_perturbation_once(
                model, data, gene_symbols, gene_to_idx,
                spec, base_imp, base_rank, z_cg_base, mu_base,
                targets, mode, factor, top_k, fdr,
                reactome, background, group_expr,
                axis_global, axes_cluster, out_dir,
                write_full=True, include_details=include_details,
                ko_mode=ko_mode, ko_soft_factor=ko_soft_factor)
        # V5.4 ko_mode pas supporté → fallback sans, comportement = cut historique
        if "ko_mode" in str(e) or "ko_soft_factor" in str(e):
            print(f"[compat] gnn_perturbation sans ko_mode (V5.3 ou antérieur), "
                  f"fallback en mode 'cut' (statu quo)")
            return run_perturbation_once(
                model, data, gene_symbols, gene_to_idx,
                spec, base_imp, base_rank, z_cg_base, mu_base,
                targets, mode, factor, top_k, fdr,
                reactome, background, group_expr,
                axis_global, axes_cluster, out_dir,
                write_full=True, include_details=include_details)
        if "include_details" in str(e):
            _SUPPORTS_INCLUDE_DETAILS = False
            print(f"[compat] Falling back to old signature (include_details not supported)")
            return run_perturbation_once(
                model, data, gene_symbols, gene_to_idx,
                spec, base_imp, base_rank, z_cg_base, mu_base,
                targets, mode, factor, top_k, fdr,
                reactome, background, group_expr,
                axis_global, axes_cluster, out_dir)
        raise


def flatten_summary(target_type: str, target: str,
                    pathway: str, mode: str, factor,
                    summary: dict) -> dict:
    """Convert a nested summary dict to a flat row for the aggregated TSV.

    Flattens cell_group_shift_relative into per-group columns and joins
    list-valued fields (top5_delta_pathways, targets_missing) with ';'.
    """
    shift = summary.get("cell_group_shift_relative", {}) or {}
    shift_gw = summary.get("cell_group_shift_gene_weighted", {}) or {}
    shift_gw_norm = summary.get("cell_group_shift_gene_weighted_norm", {}) or {}
    shift_gw_diff = summary.get("cell_group_shift_gene_differential", {}) or {}
    shift_gw_direct = summary.get("cell_group_shift_gene_direct_target", {}) or {}
    proj_global = summary.get("cell_group_shift_projected_global", {}) or {}
    proj_cluster = summary.get("cell_group_shift_projected_cluster", {}) or {}
    row = {
        "target_type": target_type,
        "target": target,
        "pathway": pathway,
        "mode": mode,
        "factor": factor if factor is not None else "",
        "n_targets_in_graph": summary.get("n_targets_in_graph"),
        "n_targets_missing": len(summary.get("targets_missing", []) or []),
        "targets_missing": ";".join(summary.get("targets_missing", []) or []),
        "n_rising": summary.get("n_rising"),
        "n_falling": summary.get("n_falling"),
        "median_abs_delta_rank": summary.get("median_abs_delta_rank"),
        "max_up_gene": summary.get("max_up_gene"),
        "max_up_delta_rank": summary.get("max_up_delta_rank"),
        "max_down_gene": summary.get("max_down_gene"),
        "max_down_delta_rank": summary.get("max_down_delta_rank"),
        "n_sig_delta_pathways": summary.get("n_sig_delta_pathways"),
        "top5_delta_pathways": ";".join(
            summary.get("top5_delta_pathways", []) or []),
        # Ancienne méthode (cell_group hidden).
        "max_shift_group": summary.get("max_shift_group"),
        "max_shift_relative": summary.get("max_shift_relative"),
        # Nouvelle méthode (gene-weighted). L'indicateur principal est
        # max_shift_gene_differential* : c'est le groupe le plus
        # spécifiquement affecté par la perturbation, après retrait du biais
        # housekeeping.
        "max_shift_gene_differential_group":
            summary.get("max_shift_gene_differential_group"),
        "max_shift_gene_differential":
            summary.get("max_shift_gene_differential"),
        # Option 2 : shift SIGNÉ (projection sur axe sénescence).
        "max_proj_signed_diff_group": summary.get("max_proj_signed_diff_group"),
        "max_proj_signed_diff": summary.get("max_proj_signed_diff"),
        # Variantes normalisées (comparaison des stratégies de correction hub).
        "max_proj_signed_norm_group": summary.get("max_proj_signed_norm_group"),
        "max_proj_signed_norm": summary.get("max_proj_signed_norm"),
        "max_proj_signed_amplitude_group": summary.get("max_proj_signed_amplitude_group"),
        "max_proj_signed_amplitude": summary.get("max_proj_signed_amplitude"),
        "max_proj_signed_extent_group": summary.get("max_proj_signed_extent_group"),
        "max_proj_signed_extent": summary.get("max_proj_signed_extent"),
        "max_proj_signed_degree_group": summary.get("max_proj_signed_degree_group"),
        "max_proj_signed_degree": summary.get("max_proj_signed_degree"),
        "max_proj_signed_cosine_group": summary.get("max_proj_signed_cosine_group"),
        "max_proj_signed_cosine": summary.get("max_proj_signed_cosine"),
        "target_ppi_degree": summary.get("target_ppi_degree"),
        # Readout signé (fan-out 1-hop, gnn_futur §8.A / §9.2). APERÇU
        # role_latent/role_de seulement (role_pert + flag = report-side).
        "signed_fanout_n": summary.get("signed_fanout_n"),
        "signed_readout_latent": summary.get("signed_readout_latent"),
        "signed_coherence_latent": summary.get("signed_coherence_latent"),
        "signed_coherence_latent_knownsign": summary.get("signed_coherence_latent_knownsign"),
        "signed_pred_known_agree": summary.get("signed_pred_known_agree"),
        "signed_has_bilinear": summary.get("signed_has_bilinear"),
        "signed_readout_de": summary.get("signed_readout_de"),
        "signed_coherence_de": summary.get("signed_coherence_de"),
        "signed_role_latde_agree": summary.get("signed_role_latde_agree"),
    }
    for grp in CELL_GROUPS:
        row[f"shift_{grp}"] = shift.get(grp)
        row[f"shift_gene_weighted_{grp}"] = shift_gw.get(grp)
        row[f"shift_gene_weighted_norm_{grp}"] = shift_gw_norm.get(grp)
        row[f"shift_gene_differential_{grp}"] = shift_gw_diff.get(grp)
        row[f"shift_gene_direct_target_{grp}"] = shift_gw_direct.get(grp)
        # Projections globales / cluster : toutes les métriques.
        _metric_keys = ("proj_signed", "proj_signed_diff", "proj_signed_norm",
                        "proj_signed_amplitude", "proj_signed_extent",
                        "proj_signed_degree", "proj_signed_cosine")
        if grp in proj_global:
            proj_data = proj_global[grp]
            for k in _metric_keys:
                row[f"{k}_global_{grp}"] = proj_data.get(k)
        if grp in proj_cluster:
            proj_data = proj_cluster[grp]
            for k in _metric_keys:
                row[f"{k}_cluster_{grp}"] = proj_data.get(k)

    # --- Details block (populated when run_perturbation_once was called with
    # include_details=True). Joined with ';' so a single TSV row carries enough
    # to reconstruct the pathway heatmap, riser sets / blocklist, and a
    # boxplot-compatible |delta_rank| distribution.
    pw_padj = summary.get("top_delta_pathways_padj") or []
    if pw_padj:
        row["top_delta_pathways"] = ";".join(p["pathway"] for p in pw_padj)
        row["top_delta_pathways_padj"] = ";".join(
            f"{p['p_adj']:.3e}" for p in pw_padj)
    risers = summary.get("top_risers") or []
    if risers:
        row["top_risers_genes"] = ";".join(r["gene"] for r in risers)
        row["top_risers_delta"] = ";".join(str(r["delta_rank"]) for r in risers)
        row["top_risers_baseline"] = ";".join(
            f"{r['baseline_importance']:.6f}" for r in risers)
    fallers = summary.get("top_fallers") or []
    if fallers:
        row["top_fallers_genes"] = ";".join(f["gene"] for f in fallers)
        row["top_fallers_delta"] = ";".join(str(f["delta_rank"]) for f in fallers)
    q = summary.get("delta_rank_abs_quantiles") or {}
    if q:
        for k in ("q0", "q25", "q50", "q75", "q95", "q99", "q100",
                  "mean", "n_genes"):
            row[f"delta_rank_abs_{k}"] = q.get(k)
    return row


def _load_de_role_sign(de_file: Path, gene_symbols, de_sign: int = 1):
    """Vecteur (n_genes,) du signe de rôle DE, aligné sur gene_symbols.

    Source du rôle DE pour le readout signé (gnn_futur §8.A, alternative à
    `role_latent`). role_DE = de_sign · sgn(avg_log2FC) — un gène up-en-P16
    (pro-sén) doit recevoir +1. `de_sign` fixe la convention du fichier :
      * +1 (DÉFAUT, vérifié sur `DEGs_P4_vs_P16.csv` : logFC>0 = UP en P16 =
        pro-sén — CDKN2A +4.3, CDKN1A +1.2, LMNB1 −1.3) → role_DE = +sgn(logFC).
      * -1 si le fichier est orienté l'inverse (logFC>0 = up côté jeune/prolif).
    ⚠️ Le nom « P4_vs_P16 » est trompeur : le contraste Seurat a ident.1=P16.
    Gènes absents → 0 (rôle indéfini, exclus du score DE). DE-free reste le
    défaut ; ce canal n'est utilisé que si --de-file est fourni.
    ⚠️ role_DE est un signal de MARQUEUR (niveau d'expression), pas causal :
    un gène compensatoire up-en-P16 mais anti-sén sera mal étiqueté → diagnostic
    seulement ; le headline reste role_pert (cosine_senescent causal, report).
    """
    import numpy as np
    df = pd.read_csv(de_file)
    cols = {c.lower(): c for c in df.columns}
    gcol = cols.get("gene") or cols.get("hgnc_symbol")
    lcol = cols.get("avg_log2fc") or cols.get("log2foldchange") or cols.get("logfc")
    if not (gcol and lcol):
        print(f"[warn] --de-file {de_file.name} : colonnes gene/avg_log2FC "
              "introuvables — rôle DE ignoré.")
        return None
    lfc = dict(zip(df[gcol].astype(str),
                   pd.to_numeric(df[lcol], errors="coerce")))
    out = np.zeros(len(gene_symbols), dtype="float32")
    for i, g in enumerate(gene_symbols):
        v = lfc.get(str(g))
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[i] = de_sign * np.sign(v)
    n = int((out != 0).sum())
    print(f"[de-role] {n}/{len(gene_symbols)} gènes avec rôle DE "
          f"(convention de_sign={de_sign:+d}).")
    return out


def _load_de_axis_anchors(de_file: Path, label: str, top_n: int,
                          rank_col: str, padj_thresh: float):
    """V6 (design_log §14bis.8 méthode A) — construit les ancres de l'axe
    DE-ancré depuis un fichier DE bulk : pôle + (pro-sén) = top-N up, pôle −
    (anti-sén) = top-N down. logFC reste un signal de READOUT (jamais feature
    d'encoder). Renvoie (up_genes, down_genes) pour `effector_axis_genes`.
    Validé GSE98440 : cos(axe scRNA, axe DE) ~0.9 à N≥150 (rank=stat)."""
    import sys as _sys
    _data = Path(__file__).resolve().parent.parent / "data"     # src/data (local)
    if not (_data / "loaders").is_dir():
        _data = Path(__file__).resolve().parent / "data"         # flat fallback
    if str(_data) not in _sys.path:
        _sys.path.insert(0, str(_data))
    from loaders.bulk_rna import load_bulk_rna_de
    de = load_bulk_rna_de(de_file, condition_label=label).dropna(subset=["log_fc"])
    pool = de[de["padj"] < padj_thresh] if "padj" in de.columns else de
    if rank_col not in pool.columns or pool[rank_col].isna().all():
        print(f"[de-axis] colonne '{rank_col}' absente/vide → fallback log_fc")
        rank_col = "log_fc"
    if pool.empty:
        pool = de
    pool = pool.dropna(subset=[rank_col])
    up = pool.nlargest(top_n, rank_col)["gene_symbol"].astype(str).tolist()
    down = pool.nsmallest(top_n, rank_col)["gene_symbol"].astype(str).tolist()
    print(f"[de-axis] {Path(de_file).name} : {len(up)}↑ (pro-sén) + {len(down)}↓ "
          f"(anti-sén) ancres ; rank={rank_col}, padj<{padj_thresh}, label={label}")
    return (up, down)


def _load_model_and_baseline(run_dir: Path, hidden: int, latent: int,
                             n_layers: int, n_heads: int,
                             quiescent_groups=None, p16_groups=None,
                             de_file: Path | None = None, de_sign: int = 1,
                             effector_axis_genes=None, device: str = "cpu"):
    """Load the VGAE run + compute the shared baseline once.

    Imports gnn_perturbation lazily so the subprocess TOP-N mode doesn't
    pull torch when not needed.

    Args:
        quiescent_groups : V4 — tuple/list de groupes côté quiescent pour
            l'axe de sénescence. Défaut historique = ("P4",).
        p16_groups : optionnel — clusters P16 côté sénescent. Auto-dérivé
            si None.
        de_file : optionnel — fichier DE (gene, avg_log2FC) pour le rôle DE
            du readout signé (attaché à data._de_role_sign). DE-free si None.
        de_sign : convention de signe du fichier DE (cf. _load_de_role_sign).
    """
    # Local import — gnn_perturbation needs torch + PyG which are heavy.
    from gnn_perturbation import (  # noqa: E402
        load_run, prepare_baseline,
        load_reactome_gmt as _load_gmt,
        load_background as _load_bg,
    )
    data, model, gene_symbols, gene_to_idx, baseline, group_expr = load_run(
        run_dir, hidden, latent, n_layers, n_heads, device=device)
    if de_file is not None:
        _de = _load_de_role_sign(de_file, gene_symbols, de_sign=de_sign)
        if _de is not None:
            data._de_role_sign = _de
    spec, base_imp, base_rank, z_cg_base, mu_base, axis_global, axes_cluster = prepare_baseline(
        model, data, baseline, gene_symbols, group_expr,
        quiescent_groups=quiescent_groups, p16_groups=p16_groups,
        effector_axis_genes=effector_axis_genes)
    reactome = _load_gmt()
    background = _load_bg()
    print(f"Loaded run ({len(gene_symbols)} genes); baseline computed; "
          f"group_expression={'OK' if group_expr is not None else 'absent'}; "
          f"axes={'OK' if axis_global is not None else 'absent'}.")
    return {
        "data": data, "model": model,
        "gene_symbols": gene_symbols, "gene_to_idx": gene_to_idx,
        "spec": spec, "base_imp": base_imp,
        "base_rank": base_rank, "z_cg_base": z_cg_base,
        "mu_base": mu_base, "group_expr": group_expr,
        "axis_global": axis_global, "axes_cluster": axes_cluster,
        "reactome": reactome, "background": background,
    }


def run_all_genes(ctx: dict, modes: tuple, oe_factor: float,
                  top_k: int, fdr: float, out_prefix: Path,
                  reactome_for_pw: dict | None = None,
                  ko_mode: str = "mask", ko_soft_factor: float = 0.1,
                  kd_factor: float = 0.15, ko_edge_factor: float = 1.0,
                  legacy: bool = False) -> None:
    """Perturb every gene in the graph, for each mode, into one TSV per mode.

    Output files are `{out_prefix}_{mode}.tsv` (e.g.
    `perturbation_all_genes_knockout.tsv`). Running the script again with a
    different mode won't overwrite a previous mode's output.
    """
    gene_symbols = ctx["gene_symbols"]
    n_total = len(gene_symbols) * len(modes)
    print(f"\n[ALL GENES] {len(gene_symbols)} genes × {len(modes)} modes "
          f"= {n_total} perturbations.")

    rows_by_mode: dict[str, list[dict]] = {m: [] for m in modes}
    out_by_mode = {m: out_prefix.with_name(f"{out_prefix.name}_{m}.tsv")
                   for m in modes}
    # Readout signé : accumulation de la table long-format par cible
    # (fan-out 1-hop). Activée via flag sur `data` (lu par run_perturbation_once)
    # → ne dépend pas du wrapper de compat. DE-free ; rôles joints côté report.
    has_axis = ctx.get("axis_global") is not None
    if has_axis:
        ctx["data"]._collect_signed_fanout = True
    fanout_cols = ("source", "target", "mode", "sign_known", "sign_pred",
                   "proj_target", "role_latent_sign")
    fanout_rows: dict[str, list] = {c: [] for c in fanout_cols}
    out_fanout = out_prefix.with_name(f"{out_prefix.name}_signed_fanout.tsv")
    done = 0
    for gene in gene_symbols:
        for mode in modes:
            factor = oe_factor if mode == "overexpress" else None
            summary = _run_perturbation_compat(
                ctx["model"], ctx["data"],
                gene_symbols, ctx["gene_to_idx"],
                ctx["spec"], ctx["base_imp"],
                ctx["base_rank"], ctx["z_cg_base"], ctx["mu_base"],
                targets=[str(gene)], mode=mode, factor=factor,
                top_k=top_k, fdr=fdr,
                reactome=ctx["reactome"], background=ctx["background"],
                group_expr=ctx["group_expr"],
                axis_global=ctx["axis_global"], axes_cluster=ctx["axes_cluster"],
                out_dir=None, include_details=True,
                ko_mode=ko_mode, ko_soft_factor=ko_soft_factor,
                kd_factor=kd_factor, ko_edge_factor=ko_edge_factor, legacy=legacy)
            if summary is None:
                continue
            fo = summary.pop("_signed_fanout_rows", None)
            if fo:
                k = len(fo["source"])
                fanout_rows["source"].extend(fo["source"])
                fanout_rows["target"].extend(fo["target"])
                fanout_rows["mode"].extend([mode] * k)
                fanout_rows["sign_known"].extend(fo["sign_known"])
                fanout_rows["sign_pred"].extend(fo["sign_pred"])
                fanout_rows["proj_target"].extend(fo["proj_target"])
                fanout_rows["role_latent_sign"].extend(fo["role_latent_sign"])
            rows_by_mode[mode].append(
                flatten_summary("gene", str(gene), "", mode, factor, summary))
            done += 1
            if done % 50 == 0 or done == n_total:
                print(f"  [{done}/{n_total}] last: {gene} ({mode})")
                # Flush partial TSV so long runs leave intermediate state.
                for m, rs in rows_by_mode.items():
                    if rs:
                        pd.DataFrame(rs).to_csv(out_by_mode[m],
                                                sep="\t", index=False)

    for m, rs in rows_by_mode.items():
        pd.DataFrame(rs).to_csv(out_by_mode[m], sep="\t", index=False)
        print(f"\nWrote {out_by_mode[m]}  ({len(rs)} rows)")
    if has_axis and fanout_rows["source"]:
        pd.DataFrame(fanout_rows).to_csv(out_fanout, sep="\t", index=False)
        print(f"Wrote {out_fanout}  ({len(fanout_rows['source'])} edge rows)")


def run_all_pathways(ctx: dict, modes: tuple, oe_factor: float,
                     top_k: int, fdr: float, out_prefix: Path,
                     pw_min_size: int, pw_max_size: int,
                     ko_mode: str = "mask",
                     ko_soft_factor: float = 0.1,
                     kd_factor: float = 0.15, ko_edge_factor: float = 1.0,
                     legacy: bool = False) -> None:
    """Perturb every REACTOME pathway (within size bounds), one TSV per mode.

    Output files are `{out_prefix}_{mode}.tsv`.
    """
    reactome = ctx["reactome"]
    pathways = [
        (name, sorted(members))
        for name, members in reactome.items()
        if pw_min_size <= len(members) <= pw_max_size
    ]
    pathways.sort(key=lambda x: x[0])
    n_total = len(pathways) * len(modes)
    print(f"\n[ALL PATHWAYS] {len(pathways)} pathways (size in "
          f"[{pw_min_size},{pw_max_size}]) × {len(modes)} modes "
          f"= {n_total} perturbations.")

    rows_by_mode: dict[str, list[dict]] = {m: [] for m in modes}
    out_by_mode = {m: out_prefix.with_name(f"{out_prefix.name}_{m}.tsv")
                   for m in modes}
    done = 0
    for pw_name, members in pathways:
        for mode in modes:
            factor = oe_factor if mode == "overexpress" else None
            summary = _run_perturbation_compat(
                ctx["model"], ctx["data"],
                ctx["gene_symbols"], ctx["gene_to_idx"],
                ctx["spec"], ctx["base_imp"],
                ctx["base_rank"], ctx["z_cg_base"], ctx["mu_base"],
                targets=list(members), mode=mode, factor=factor,
                top_k=top_k, fdr=fdr,
                reactome=ctx["reactome"], background=ctx["background"],
                group_expr=ctx["group_expr"],
                axis_global=ctx["axis_global"], axes_cluster=ctx["axes_cluster"],
                out_dir=None, include_details=True,
                ko_mode=ko_mode, ko_soft_factor=ko_soft_factor,
                kd_factor=kd_factor, ko_edge_factor=ko_edge_factor, legacy=legacy)
            if summary is None:
                continue
            rows_by_mode[mode].append(
                flatten_summary("pathway", pw_name, pw_name,
                                mode, factor, summary))
            done += 1
            if done % 25 == 0 or done == n_total:
                print(f"  [{done}/{n_total}] last: {pw_name} ({mode})")
                for m, rs in rows_by_mode.items():
                    if rs:
                        pd.DataFrame(rs).to_csv(out_by_mode[m],
                                                sep="\t", index=False)

    for m, rs in rows_by_mode.items():
        pd.DataFrame(rs).to_csv(out_by_mode[m], sep="\t", index=False)
        print(f"\nWrote {out_by_mode[m]}  ({len(rs)} rows)")


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
    ap.add_argument("--ko-mode", choices=["cut", "mask", "soft"],
                    default="mask",
                    help="V5.5 : sémantique du knockout. mask (DÉFAUT) = "
                         "feature=0, GARDE le graphe (corrige le bug KO-cut). "
                         "cut (legacy) = feature=0 + coupe arêtes incidentes. "
                         "soft = feature × ko_soft_factor.")
    ap.add_argument("--ko-soft-factor", type=float, default=0.1,
                    help="Multiplicateur appliqué quand --ko-mode soft "
                         "(défaut 0.1).")
    ap.add_argument("--kd-factor", type=float, default=0.15,
                    help="V5.5 : multiplicateur du knockdown soft (défaut 0.15 "
                         "≈ résidu siRNA). Ignoré si --legacy (KD=feature 0).")
    ap.add_argument("--ko-edge-factor", type=float, default=1.0,
                    help="V5.5 : fraction d'arêtes incidentes conservées au KO "
                         "non-cut (défaut 1.0 = mask ; <1.0 = atténuation).")
    ap.add_argument("--legacy", action="store_true",
                    help="Restaure la sémantique V3.3-V5.4 (KD: feature→0 ; "
                         "KO: cut arêtes). Flag TEMPORAIRE (validation exp.).")
    ap.add_argument("--oe-factor", type=float, default=3.0,
                    help="Multiplier used for overexpress (default 3.0).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if summary.json already exists.")
    # --- ALL mode ---
    ap.add_argument("--all-genes", action="store_true",
                    help="Perturb EVERY gene in the graph (in-process). "
                         "Output: one aggregated TSV, no per-target folder.")
    ap.add_argument("--all-pathways", action="store_true",
                    help="Perturb EVERY REACTOME pathway (within size bounds, "
                         "in-process). Output: one aggregated TSV, no per-target folder.")
    ap.add_argument("--pw-min-size", type=int, default=5,
                    help="Min pathway size when --all-pathways (default 5).")
    ap.add_argument("--pw-max-size", type=int, default=500,
                    help="Max pathway size when --all-pathways (default 500).")
    ap.add_argument("--top-k", type=int, default=100,
                    help="Top-K rising genes for delta-ORA per perturbation "
                         "(default 100). Forwarded to gnn_perturbation.")
    ap.add_argument("--fdr", type=float, default=0.05,
                    help="FDR threshold for delta-ORA (default 0.05).")
    # Hyperparams forwarded to gnn_perturbation.load_run (ALL mode only).
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                    help="device perturbation. 'auto' = cuda si dispo sinon cpu "
                         "(recommandé sur cluster GPU). Défaut cpu.")
    # V4 — axe de sénescence : groupes côté quiescent (et P16 résiduels).
    # Permet d'agréger c0 avec P4 (c0 transcriptionnellement quiescent-like
    # cf. CLAUDE.md §V3.3). Backward-compat : défaut "P4" seul.
    ap.add_argument("--quiescent-groups", default="P4",
                    help="liste séparée par virgules des groupes côté "
                         "quiescent pour l'axe de sénescence "
                         "(défaut V3 : 'P4' ; recommandé V4 : "
                         "'P4,P16_cluster_0').")
    ap.add_argument("--p16-groups", default=None,
                    help="liste séparée par virgules des clusters côté "
                         "sénescent. Si non fourni, dérivé automatiquement "
                         "de --quiescent-groups (P16_cluster_0..3 moins "
                         "ceux passés côté quiescent).")
    ap.add_argument("--de-file", type=Path, default=None,
                    help="optionnel — fichier DE (gene, avg_log2FC) pour le "
                         "rôle DE du readout signé (colonnes signed_*_de). "
                         "DE-free si non fourni.")
    ap.add_argument("--de-sign", type=int, default=1, choices=[-1, 1],
                    help="convention du --de-file. +1 (DÉFAUT, vérifié sur "
                         "DEGs_P4_vs_P16.csv : logFC>0 = UP en P16 = pro-sén, "
                         "CDKN2A +4.3) ; -1 si l'inverse. Le nom 'P4_vs_P16' est "
                         "trompeur (ident.1=P16).")
    # Axe effecteur-ancré (gnn_futur §2.2/§4b) — stabilise l'axe (cas ASNS §7.8).
    ap.add_argument("--effector-axis", action="store_true",
                    help="ancre axis_global sur des effecteurs canoniques "
                         "(axis = unit(μ[pro]−μ[anti])) au lieu du contraste "
                         "d'expression. Référence causale, ne pivote pas avec "
                         "les hyperparamètres. Listes via --effector-pro/-anti.")
    ap.add_argument("--effector-pro",
                    default="CDKN2A,CDKN1A,CDKN2B,SERPINE1,GLB1",
                    help="effecteurs PRO-sénescence (pôle +) pour --effector-axis.")
    ap.add_argument("--effector-anti",
                    default="LMNB1,MKI67,CCNB1,CCNA2,PCNA,TOP2A",
                    help="effecteurs ANTI-sénescence/prolifération (pôle −).")
    # Axe DE-ancré (V6 — readout bulk RNA-seq, design_log §14bis.8 méthode A).
    ap.add_argument("--de-axis-file", type=Path, default=None,
                    help="V6 : ancre axis_global sur un fichier DE bulk "
                         "(logFC/padj) — pôle + = top-N up (pro-sén), pôle − = "
                         "top-N down. Validé GSE98440 : cos(axe scRNA)~0.9 à "
                         "N≥150. logFC JAMAIS en feature d'encoder (anti-circularité).")
    ap.add_argument("--de-axis-label", default="sen_vs_pro",
                    help="condition_label du DE (<A>_vs_<B>, log_fc>0 ⇔ up dans A "
                         "= pôle pro-sén). Défaut sen_vs_pro.")
    ap.add_argument("--de-axis-top-n", type=int, default=200,
                    help="nb d'ancres par pôle (défaut 200 ; ≥150 recommandé).")
    ap.add_argument("--de-axis-rank", default="stat", choices=["stat", "log_fc"],
                    help="colonne de ranking des ancres (défaut stat = Wald DESeq2, "
                         "robuste ; fallback log_fc).")
    ap.add_argument("--de-axis-padj", type=float, default=0.1,
                    help="seuil padj des ancres (défaut 0.1 ; NaN gardés).")
    ap.add_argument("--out-suffix", default="",
                    help="suffixe ajouté au préfixe d'output (avant '_<mode>.tsv'). "
                         "Évite d'écraser les TSV V3 quand on relance la même "
                         "perturbation avec un axe V4 différent. "
                         "Exemple : '--out-suffix _axisV4' → "
                         "perturbation_all_genes_axisV4_knockout.tsv.")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    modes = tuple(args.modes)

    # --------------------------------------------------------------- #
    # ALL mode — in-process, single aggregated TSV per target type.
    # --------------------------------------------------------------- #
    if args.all_genes or args.all_pathways:
        # Parse les groupes V4 axe sénescence (CSV → tuple ou None si défaut)
        _q = tuple(s.strip() for s in args.quiescent_groups.split(",")
                   if s.strip()) if args.quiescent_groups else None
        _p = (tuple(s.strip() for s in args.p16_groups.split(",") if s.strip())
              if args.p16_groups else None)
        _eff = None
        if args.de_axis_file:
            _eff = _load_de_axis_anchors(
                args.de_axis_file, args.de_axis_label, args.de_axis_top_n,
                args.de_axis_rank, args.de_axis_padj)
        elif args.effector_axis:
            _eff = (tuple(s.strip() for s in args.effector_pro.split(",") if s.strip()),
                    tuple(s.strip() for s in args.effector_anti.split(",") if s.strip()))
        ctx = _load_model_and_baseline(run_dir, args.hidden, args.latent,
                                       args.n_layers, args.n_heads,
                                       quiescent_groups=_q, p16_groups=_p,
                                       de_file=args.de_file, de_sign=args.de_sign,
                                       effector_axis_genes=_eff, device=args.device)
        _suffix = args.out_suffix  # "" si non fourni → comportement V3 inchangé
        if args.all_genes:
            run_all_genes(ctx, modes, args.oe_factor,
                          args.top_k, args.fdr,
                          out_prefix=run_dir / f"perturbation_all_genes{_suffix}",
                          ko_mode=args.ko_mode,
                          ko_soft_factor=args.ko_soft_factor,
                          kd_factor=args.kd_factor,
                          ko_edge_factor=args.ko_edge_factor,
                          legacy=args.legacy)
        if args.all_pathways:
            run_all_pathways(ctx, modes, args.oe_factor,
                             args.top_k, args.fdr,
                             out_prefix=run_dir / f"perturbation_all_pathways{_suffix}",
                             pw_min_size=args.pw_min_size,
                             pw_max_size=args.pw_max_size,
                             ko_mode=args.ko_mode,
                             ko_soft_factor=args.ko_soft_factor,
                             kd_factor=args.kd_factor,
                             ko_edge_factor=args.ko_edge_factor,
                             legacy=args.legacy)
        return

    # --------------------------------------------------------------- #
    # TOP-N mode — legacy subprocess path, one folder per target.
    # --------------------------------------------------------------- #
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
