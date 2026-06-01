#!/usr/bin/env python3
"""run_signed_auc_gate.py — wrapper cross-seed pour le gate 1c.5.

Lance `test_signed_auc.py` sur chaque seed d'une config V5 entraînée
(typiquement `output/gnn_vgae/V5/full/v5-full.s{1,2,3}`), aggrège les
AUC + AUPR cross-seed, et émet un verdict global :

    GATE 1c.5 PASSÉ  ⇔  AUC moy ≥ 0.85 sur TOUS les edge_types
                        ET σ inter-seed ≤ 0.05

Inputs :
- `--runs-glob` : motif glob pointant vers les dossiers de runs
                   (ex: 'output/gnn_vgae/V5/full/v5-full.s*').
- `--label`    : étiquette pour le rapport (ex: 'V5.1-full').
- `--gate-threshold` : seuil AUC (défaut 0.85, Liu 2024 *NAR* §3).

Outputs (dans `--out-dir`) :
- `signed_auc_cross_seed.tsv`  : 1 ligne par (edge_type, métrique).
- `signed_auc_per_run.tsv`    : 1 ligne par (run, edge_type).
- `signed_auc_gate_summary.md` : synthèse + verdict gate.

Workflow typique (après run cluster V5.1) :
    rsync -av nautilus:'output/gnn_vgae/V5/full/v5-full.s*' \\
        output/gnn_vgae/V5/full/

    python src/validation/explain/run_signed_auc_gate.py \\
        --runs-glob 'output/gnn_vgae/V5/full/v5-full.s*' \\
        --label V5.1-full \\
        --out-dir output/gnn_vgae/V5/gate_1c5_v5.1-full

Comparaison V5.0 → V5.1 :
    # 1. Mesurer gate sur V5.0 (baseline buggée, AUC attendu ≈ 0.47)
    python src/validation/explain/run_signed_auc_gate.py \\
        --runs-glob 'output/gnn_vgae/V5/full/v5-full.s*' \\
        --label V5.0-full --out-dir output/gnn_vgae/V5/gate_v5.0
    # 2. Mesurer gate sur V5.1 (après rsync)
    python src/validation/explain/run_signed_auc_gate.py \\
        --runs-glob 'output/gnn_vgae/V5.1/full/v5.1-full.s*' \\
        --label V5.1-full --out-dir output/gnn_vgae/V5.1/gate_v5.1
    # 3. Comparer les deux fichiers signed_auc_gate_summary.md.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_paths():
    here = Path(__file__).resolve()
    project_root = here.parents[3] if len(here.parents) > 3 else here.parents[1]
    for p in [project_root / "src", project_root]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


_bootstrap_paths()
# Réutilise le pipeline du script unitaire (évite duplication).
from validation.explain.test_signed_auc import (  # noqa: E402
    auc_per_edge_type,
    auc_per_tf,
    auc_tf_stratified,
    collect_signed_edges,
    encode_full,
    load_run,
)


def _read_holdout_set(run_dir: Path) -> tuple[set[str], int | None]:
    """Lit run_config.json:holdout_signed_tf_set (vide si absent)."""
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return set(), None
    try:
        import json as _j
        with open(cfg_path) as fh:
            cfg = _j.load(fh)
        return set(cfg.get("holdout_signed_tf_set", []) or []), \
               cfg.get("holdout_signed_tf_seed_used", None)
    except Exception:
        return set(), None


def run_single(run_dir: Path, n_splits: int, holdout_frac: float, seed: int,
               min_edges_per_tf: int = 5,
               mode: str = "auto",
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, int]:
    """Évalue 1 run et retourne (overall, holdout, per_tf, mode, n_holdout_tfs).

    Si `mode='holdout'` (ou `auto` et set non vide), filtre les arêtes pour
    ne garder QUE celles incidentes au set hold-out (test rigoureux).
    """
    holdout_tf_set, _ = _read_holdout_set(run_dir)
    effective_mode = (
        ("holdout" if holdout_tf_set else "in-sample")
        if mode == "auto" else mode
    )
    if effective_mode == "holdout" and not holdout_tf_set:
        raise RuntimeError(
            f"{run_dir.name} : --mode holdout demandé mais set hold-out vide "
            f"dans run_config.json. Re-train avec --holdout-signed-tf-fraction > 0."
        )

    model, data, symbols = load_run(run_dir)
    edges = collect_signed_edges(data, symbols)

    if effective_mode == "holdout":
        mask = (edges["src_sym"].isin(holdout_tf_set)
                | edges["dst_sym"].isin(holdout_tf_set))
        edges = edges[mask].reset_index(drop=True)
        if edges.empty:
            raise RuntimeError(
                f"{run_dir.name} : aucune arête incidente au set hold-out "
                f"({len(holdout_tf_set)} TFs)."
            )

    z = encode_full(model, data)
    overall = auc_per_edge_type(edges, z, model)
    holdout = auc_tf_stratified(edges, z, model, n_splits, holdout_frac, seed)
    per_tf = auc_per_tf(edges, z, model, min_edges=min_edges_per_tf)
    overall.insert(0, "run", run_dir.name)
    overall.insert(1, "mode", effective_mode)
    holdout.insert(0, "run", run_dir.name)
    holdout.insert(1, "mode", effective_mode)
    per_tf.insert(0, "run", run_dir.name)
    per_tf.insert(1, "mode", effective_mode)
    return overall, holdout, per_tf, effective_mode, len(holdout_tf_set)


def aggregate_cross_seed(per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggrégat cross-seed sur les colonnes AUC/AUPR par edge_type.

    Pour chaque (edge_type, métrique numérique), calcule moy/std/min/max.
    """
    # Colonnes à aggréger (toutes les float, sauf identifiants)
    id_cols = {"run", "edge_type"}
    num_cols = [c for c in per_run.columns
                if c not in id_cols
                and pd.api.types.is_numeric_dtype(per_run[c])]
    rows = []
    for et, sub in per_run.groupby("edge_type"):
        row = {"edge_type": et, "n_seeds": len(sub)}
        for col in num_cols:
            v = sub[col].dropna().values
            if len(v) == 0:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
                continue
            row[f"{col}_mean"] = round(float(np.mean(v)), 4)
            row[f"{col}_std"] = round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 4)
            row[f"{col}_min"] = round(float(np.min(v)), 4)
            row[f"{col}_max"] = round(float(np.max(v)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def emit_gate_summary(
    out_dir: Path, label: str, n_runs: int,
    overall_cs: pd.DataFrame, holdout_cs: pd.DataFrame,
    gate_threshold: float, n_splits: int, holdout_frac: float,
    mode: str = "in-sample", n_holdout_tfs: int = 0,
) -> dict:
    """Émet le fichier markdown + retourne le verdict (dict)."""
    lines = []
    lines.append(f"# Gate 1c.5 cross-seed — {label}\n\n")
    lines.append(f"- n runs : {n_runs}\n")
    lines.append(f"- mode  : **{mode}**"
                 + (f"  ({n_holdout_tfs} TFs réservés au training)"
                    if mode == "holdout" else "")
                 + "\n")
    lines.append(f"- gate seuil : AUC ≥ {gate_threshold:.2f} (Liu 2024 *NAR* §3)\n")
    lines.append(f"- TF-stratified : {n_splits} splits × {holdout_frac:.0%} TFs\n\n")

    if mode == "holdout":
        lines.append("## ✓ Test rigoureux (phase 2)\n\n")
        lines.append(
            "Tous les runs ont été entraînés avec "
            "`--holdout-signed-tf-fraction > 0` : les signs des TFs hold-out "
            "n'ont PAS été vus par la `signed_aux_loss`. L'évaluation est "
            "restreinte aux arêtes incidentes au set hold-out → vrai test de "
            "généralisation à des TFs jamais utilisés pour la loss.\n\n"
        )
    elif mode == "in-sample":
        lines.append("## ⚠ Caveat in-sample\n\n")
        lines.append(
            "Aucun run n'a de `holdout_signed_tf_set` non vide dans son "
            "`run_config.json` → la `signed_aux_loss` a vu TOUTES les arêtes "
            "signées à l'entraînement. Ce gate distingue surtout V5.0 buggée "
            "(AUC ≈ 0.47) de V5.1 corrigée (AUC attendue > 0.85). Pour un "
            "test rigoureux de généralisation, re-train avec "
            "`--holdout-signed-tf-fraction X` (X=0.2 recommandé).\n\n"
        )
    else:
        lines.append("## ⚠ Mode hétérogène entre runs\n\n")
        lines.append(
            f"Les runs n'ont pas tous le même mode : `{mode}`. Le verdict "
            "agrégé est à interpréter avec prudence. Re-train ou re-évaluer "
            "tous les runs dans le même mode pour comparer rigoureusement.\n\n"
        )

    lines.append("## 1. AUC global in-sample (cross-seed)\n\n")
    cols = ["edge_type", "n_seeds",
            "auc_insample_mean", "auc_insample_std",
            "auc_insample_min", "auc_insample_max",
            "aupr_insample_mean", "aupr_insample_std",
            "aupr_baseline_mean"]
    cols = [c for c in cols if c in overall_cs.columns]
    lines.append("```\n" + overall_cs[cols].to_string(index=False) + "\n```\n\n")
    pass_overall = (overall_cs["auc_insample_mean"] >= gate_threshold).all() if \
        "auc_insample_mean" in overall_cs.columns else False
    lines.append(f"Toutes edge_types ≥ {gate_threshold:.2f} (in-sample) : "
                 f"**{'OUI ✓' if pass_overall else 'NON ✗'}**\n\n")

    lines.append(f"## 2. TF-stratified hold-out ({n_splits} splits × "
                 f"{holdout_frac:.0%} TFs, cross-seed)\n\n")
    cols2 = ["edge_type", "n_seeds",
             "auc_holdout_mean_mean", "auc_holdout_mean_std",
             "auc_holdout_mean_min", "auc_holdout_mean_max",
             "aupr_holdout_mean_mean", "aupr_holdout_mean_std"]
    cols2 = [c for c in cols2 if c in holdout_cs.columns]
    lines.append("```\n" + holdout_cs[cols2].to_string(index=False) + "\n```\n\n")
    pass_ho = (holdout_cs["auc_holdout_mean_mean"] >= gate_threshold).all() \
        if "auc_holdout_mean_mean" in holdout_cs.columns else False
    lines.append(f"Toutes edge_types ≥ {gate_threshold:.2f} (TF-stratified) : "
                 f"**{'OUI ✓' if pass_ho else 'NON ✗'}**\n\n")

    lines.append("## Verdict\n\n")
    _rigor = "RIGOUREUX (phase 2)" if mode == "holdout" else "in-sample"
    if pass_overall and pass_ho:
        verdict = "PASS"
        lines.append(f"✅ **GATE PASSÉ [{_rigor}]** — {label} apprend la "
                     f"sémantique du signe. Garder `--signed-decoder` par "
                     f"défaut.\n")
    elif pass_overall:
        verdict = "PASS_PARTIAL"
        lines.append(f"⚠ **GATE PARTIEL [{_rigor}]** — AUC global OK mais "
                     f"TF-stratified < {gate_threshold} → risque de "
                     f"mémorisation. Examiner per-TF distribution.\n")
    else:
        verdict = "FAIL"
        lines.append(f"❌ **GATE NON PASSÉ [{_rigor}]** — {label} n'apprend "
                     f"pas le signe. AUC global < {gate_threshold}.\n")

    (out_dir / "signed_auc_gate_summary.md").write_text("".join(lines))

    return {
        "label": label,
        "n_runs": n_runs,
        "mode": mode,
        "verdict": verdict,
        "pass_overall": pass_overall,
        "pass_holdout": pass_ho,
        "gate_threshold": gate_threshold,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--runs-glob", required=True,
                   help="Motif glob des dossiers de runs V5 "
                        "(ex: 'output/gnn_vgae/V5/full/v5-full.s*').")
    p.add_argument("--label", required=True,
                   help="Étiquette pour le rapport (ex: 'V5.1-full').")
    p.add_argument("--out-dir", required=True,
                   help="Dossier de sortie pour les TSV + markdown.")
    p.add_argument("--n-splits", type=int, default=100,
                   help="Splits TF-stratifiés par run (défaut 100).")
    p.add_argument("--holdout-frac", type=float, default=0.2,
                   help="Fraction TFs hold-out (défaut 0.2).")
    p.add_argument("--holdout-seed", type=int, default=42,
                   help="Seed RNG des splits TF.")
    p.add_argument("--gate-threshold", type=float, default=0.85,
                   help="Seuil AUC gate (défaut 0.85, Liu 2024).")
    p.add_argument("--min-edges-per-tf", type=int, default=5,
                   help="Min arêtes/TF pour calculer AUC individuel.")
    p.add_argument("--mode",
                   choices=["auto", "in-sample", "holdout"],
                   default="auto",
                   help="Phase 2 1c.5 strict : `holdout` filtre l'évaluation "
                        "aux arêtes incidentes au set TF persisté dans "
                        "run_config.json (vrai test généralisation). "
                        "`in-sample` = ignore le set hold-out. `auto` "
                        "(défaut) = `holdout` si set non vide, sinon "
                        "`in-sample`.")
    args = p.parse_args()

    runs = sorted(Path(d) for d in glob.glob(args.runs_glob) if Path(d).is_dir())
    if not runs:
        print(f"[err] Aucun dossier ne correspond à {args.runs_glob!r}")
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gate 1c.5] label  : {args.label}")
    print(f"[gate 1c.5] runs   : {len(runs)} ({[r.name for r in runs]})")
    print(f"[gate 1c.5] out    : {out_dir}")
    print(f"[gate 1c.5] gate   : AUC ≥ {args.gate_threshold}\n")

    overall_all, holdout_all, per_tf_all = [], [], []
    modes_seen: set[str] = set()
    n_holdout_tfs_seen = 0
    for run in runs:
        print(f"[gate 1c.5] === {run.name} ===")
        try:
            ov, ho, pt, mode_used, n_holdout_tfs = run_single(
                run, args.n_splits, args.holdout_frac, args.holdout_seed,
                args.min_edges_per_tf, mode=args.mode,
            )
        except Exception as e:
            print(f"  [err] échec sur {run.name} : {type(e).__name__}: {e}")
            continue
        modes_seen.add(mode_used)
        n_holdout_tfs_seen = max(n_holdout_tfs_seen, n_holdout_tfs)
        overall_all.append(ov)
        holdout_all.append(ho)
        per_tf_all.append(pt)
        _suffix = (f"  [holdout : {n_holdout_tfs} TFs réservés]"
                   if mode_used == "holdout" else "")
        print(f"  mode = {mode_used}{_suffix}")
        print(f"  AUC par edge_type :")
        for _, r in ov.iterrows():
            print(f"    {r['edge_type']:18s} AUC={r['auc_insample']:.4f}  "
                  f"AUPR={r['aupr_insample']:.4f} (baseline {r['aupr_baseline']:.4f})")

    if not overall_all:
        print("[err] Aucun run évalué avec succès.")
        sys.exit(1)

    per_run_overall = pd.concat(overall_all, ignore_index=True)
    per_run_holdout = pd.concat(holdout_all, ignore_index=True)
    per_run_per_tf = pd.concat(per_tf_all, ignore_index=True) if per_tf_all else pd.DataFrame()

    per_run_overall.to_csv(out_dir / "signed_auc_per_run.tsv", sep="\t", index=False)
    per_run_holdout.to_csv(out_dir / "signed_auc_holdout_per_run.tsv", sep="\t", index=False)
    if not per_run_per_tf.empty:
        per_run_per_tf.to_csv(out_dir / "signed_auc_per_tf_per_run.tsv",
                              sep="\t", index=False)

    overall_cs = aggregate_cross_seed(per_run_overall)
    holdout_cs = aggregate_cross_seed(per_run_holdout)
    overall_cs.to_csv(out_dir / "signed_auc_cross_seed.tsv", sep="\t", index=False)
    holdout_cs.to_csv(out_dir / "signed_auc_holdout_cross_seed.tsv",
                       sep="\t", index=False)

    print(f"\n[gate 1c.5] Cross-seed aggregate par edge_type :")
    print(overall_cs[["edge_type", "n_seeds",
                      "auc_insample_mean", "auc_insample_std",
                      "auc_insample_min", "auc_insample_max"]].to_string(index=False))

    # Mode résolu : si tous les runs sont en mode homogène, on le passe au
    # rapport ; sinon on signale l'hétérogénéité.
    if len(modes_seen) == 1:
        global_mode = next(iter(modes_seen))
    else:
        global_mode = f"mixed:{','.join(sorted(modes_seen))}"

    verdict = emit_gate_summary(
        out_dir, args.label, len(runs),
        overall_cs, holdout_cs,
        args.gate_threshold, args.n_splits, args.holdout_frac,
        mode=global_mode, n_holdout_tfs=n_holdout_tfs_seen,
    )
    print(f"\n[gate 1c.5] mode appliqué : {global_mode}"
          + (f"  ({n_holdout_tfs_seen} TFs hold-out)"
             if global_mode == "holdout" else ""))
    print(f"[gate 1c.5] Verdict : **{verdict['verdict']}** "
          f"(pass_overall={verdict['pass_overall']}, "
          f"pass_holdout={verdict['pass_holdout']})")
    print(f"[gate 1c.5] Rapport : {out_dir}/signed_auc_gate_summary.md")


if __name__ == "__main__":
    main()
