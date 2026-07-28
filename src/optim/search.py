"""Automatic hyper-parameter search — Optuna DRIVING Snakemake.

    calibrate   measure the noise floor before searching anything
    search      run an Optuna study
    report      summarise a finished study

WHY OPTUNA SITS *ABOVE* SNAKEMAKE, NOT INSIDE IT
------------------------------------------------
Snakemake is a file-driven DAG: every job must be known when the DAG is built.
A search proposes its next trial only after seeing the previous one, so it
cannot be expressed as rules. The division of labour is therefore:

    Optuna  chooses the hyper-parameters, one trial at a time
      └── writes a config .yaml derived from a base config
          └── invokes `workflow/run.sh` (which keeps caching, restart,
              SLURM submission — none of that is reimplemented here)
              └── an objective reads the trial's outputs and returns a float

Each trial gets its own `run_tag`, so runs never collide and Snakemake's own
caching still skips whatever a previous trial already built (the graph cache in
particular is seed-independent and shared across trials with identical
graph-affecting flags).

BEFORE YOU SEARCH: CALIBRATE
----------------------------
Two runs of an identical config differ by rho 0.556-0.687 on the ranking. A
search that cannot beat that spread is fitting noise. `calibrate` repeats one
config N times and prints the objective's spread; compare any study's gain to
it before believing the result.

Usage
-----
    python src/optim/search.py calibrate --base-config workflow/config/config.yaml \
        --objective cross_seed_stability --repeats 3

    python src/optim/search.py search --base-config workflow/config/config.yaml \
        --objective cross_seed_stability --n-trials 20 --seeds 3

    python src/optim/search.py report --study-name stgp-v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import objectives as obj  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Espace de recherche
# ─────────────────────────────────────────────────────────────────────────────
# Chaque entrée dit comment échantillonner UN levier et comment il se traduit en
# drapeau CLI de gnn_vgae. Ne mettre ici que des leviers dont on a déjà mesuré
# qu'ils bougent quelque chose — un espace trop large dilue le budget d'essais
# dans des dimensions mortes.
#
# Les bornes ci-dessous encadrent les valeurs déjà explorées à la main :
# kl_beta_max (kl1 = +0.0077 AUC, le seul levier positif net), latent_dim
# (lat128 a DÉGRADÉ, sigma x6 -> borne haute prudente), edge_sample_ratio
# (es03 réfuté, gardé pour vérifier), lr.

SEARCH_SPACE = {
    "kl_beta_max": {
        "flag": "--kl-beta-max",
        "type": "loguniform",
        "low": 1e-5,
        "high": 5e-3,
    },
    "latent_dim": {
        "flag": "--latent-dim",
        "type": "categorical",
        "choices": [32, 64, 96],
    },
    "edge_sample_ratio": {
        "flag": "--edge-sample-ratio",
        "type": "uniform",
        "low": 0.05,
        "high": 0.3,
    },
    "lr": {
        "flag": "--lr",
        "type": "loguniform",
        "low": 1e-3,
        "high": 1e-2,
    },
}


def suggest(trial, space=None) -> dict:
    space = SEARCH_SPACE if space is None else space
    out = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "loguniform":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif kind == "uniform":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif kind == "int":
            out[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"type d'échantillonnage inconnu : {kind}")
    return out


def params_to_flags(params: dict, space=None) -> str:
    space = SEARCH_SPACE if space is None else space
    return " ".join(f"{space[k]['flag']} {v}" for k, v in params.items())


# ─────────────────────────────────────────────────────────────────────────────
# Un essai = une config + une invocation Snakemake
# ─────────────────────────────────────────────────────────────────────────────


def write_trial_config(base_config: Path, run_tag: str, flags: str, seeds: int,
                       out_dir: Path) -> Path:
    """Derive a trial config from the base one. Only what the trial changes."""
    cfg = yaml.safe_load(Path(base_config).read_text())
    vgae = cfg.setdefault("models", {}).setdefault("vgae", {})
    vgae["run_tag"] = run_tag
    vgae["seeds"] = list(range(1, seeds + 1))
    vgae["extra_flags"] = (str(vgae.get("extra_flags", "")) + " " + flags).strip()
    # Le QC doit tourner sur chaque essai : sans plancher de bruit, la
    # comparaison entre essais n'est pas interprétable.
    cfg.setdefault("validation", {}).setdefault("qc", {})["enabled"] = True
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (run_tag.replace("/", "_") + ".yaml")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return path


def run_pipeline(config_path: Path, backend: str, cores: int, dry_run: bool,
                 log_path: Path) -> int:
    cmd = ["bash", "workflow/run.sh", "--backend", backend,
           "--configfile", str(config_path), "--cores", str(cores)]
    if dry_run:
        cmd.append("--dry-run")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh,
                              stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def trial_dirs(base_config: Path, run_tag: str) -> tuple[Path, Path]:
    cfg = yaml.safe_load(Path(base_config).read_text())
    out_base = Path(os.environ.get("GNN_OUT_DIR_BASE")
                    or cfg["paths"]["out_base"])
    if not out_base.is_absolute():
        out_base = REPO_ROOT / out_base
    config_dir = out_base / run_tag
    return config_dir, config_dir / "analysis"


def evaluate(args, run_tag: str, flags: str) -> obj.ObjectiveResult:
    cfg_path = write_trial_config(Path(args.base_config), run_tag, flags,
                                  args.seeds, Path(args.trial_config_dir))
    log = REPO_ROOT / "logs" / "optuna" / (run_tag.replace("/", "_") + ".log")
    rc = run_pipeline(cfg_path, args.backend, args.cores, args.dry_run, log)
    if rc != 0:
        raise obj.ObjectiveError(f"pipeline sorti en {rc} — voir {log}")
    if args.dry_run:
        # Un dry-run ne produit aucune sortie : évaluer l'objectif échouerait
        # forcément. On rend donc une valeur factice, pour que --dry-run teste
        # ce qu'il est censé tester — la génération de config et l'appel au
        # workflow — et rien d'autre.
        return obj.ObjectiveResult(value=float("nan"),
                                   diagnostics={"dry_run": True,
                                                "config": str(cfg_path)})
    config_dir, analysis_dir = trial_dirs(Path(args.base_config), run_tag)
    fn = obj.get(args.objective)
    return fn(config_dir=config_dir, analysis_dir=analysis_dir,
              reference_file=args.reference_file, top_n=args.top_n)


# ─────────────────────────────────────────────────────────────────────────────
# Sous-commandes
# ─────────────────────────────────────────────────────────────────────────────


def cmd_calibrate(args):
    """Repeat ONE config N times: how much does the objective move on its own?"""
    values, diags = [], []
    for i in range(args.repeats):
        tag = f"{args.study_name}/calib_{i}"
        print(f"[calibrate] répétition {i + 1}/{args.repeats} → {tag}", flush=True)
        res = evaluate(args, tag, flags="")
        values.append(float(res))
        diags.append(res.diagnostics)
        print(f"[calibrate]   objectif = {float(res):.6f}", flush=True)

    spread = float(np.max(values) - np.min(values)) if len(values) > 1 else 0.0
    out = {
        "objective": args.objective,
        "repeats": args.repeats,
        "values": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "spread": spread,
        "diagnostics": diags,
    }
    dest = Path(args.out_dir) / f"{args.study_name}_calibration.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 64)
    print(f"PLANCHER DE BRUIT — objectif '{args.objective}'")
    print(f"  moyenne  : {out['mean']:.6f}")
    print(f"  écart-type : {out['std']:.6f}")
    print(f"  amplitude  : {spread:.6f}")
    print("\n  => Tout gain de recherche INFÉRIEUR à cette amplitude est du bruit.")
    print(f"  => Écrit dans {dest}")
    print("=" * 64)
    return out


def cmd_search(args):
    try:
        import optuna
    except ImportError:
        sys.exit("optuna absent : pip install -e '.[optim]'")

    if args.seeds < 3 and not args.allow_few_seeds:
        sys.exit(
            "--seeds < 3 refusé : deux runs d'une config IDENTIQUE diffèrent "
            "de rho 0.556-0.687 (écart médian 942 rangs). Sous 3 seeds, la "
            "recherche optimise le bruit. Forcer avec --allow-few-seeds."
        )

    storage = f"sqlite:///{Path(args.out_dir) / (args.study_name + '.db')}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",   # tous les objectifs sont à MAXIMISER
        load_if_exists=True,    # reprise après interruption
    )

    def objective(trial):
        params = suggest(trial)
        flags = params_to_flags(params)
        tag = f"{args.study_name}/trial_{trial.number:03d}"
        print(f"\n[trial {trial.number}] {flags}", flush=True)
        try:
            res = evaluate(args, tag, flags)
        except obj.ObjectiveError as exc:
            print(f"[trial {trial.number}] écarté : {exc}", flush=True)
            raise optuna.TrialPruned() from exc
        for k, v in res.diagnostics.items():
            if isinstance(v, (int, float, str)):
                trial.set_user_attr(k, v)
        trial.set_user_attr("run_tag", tag)
        print(f"[trial {trial.number}] objectif = {float(res):.6f}", flush=True)
        return float(res)

    study.optimize(objective, n_trials=args.n_trials)
    _print_report(study, args)
    return study


def _print_report(study, args):
    print("\n" + "=" * 64)
    print(f"ÉTUDE '{study.study_name}' — objectif '{args.objective}'")
    done = [t for t in study.trials if t.value is not None]
    print(f"  essais aboutis : {len(done)} / {len(study.trials)}")
    if not done:
        print("  aucun essai exploitable.")
        print("=" * 64)
        return
    print(f"  meilleure valeur : {study.best_value:.6f}")
    print(f"  meilleurs params : {study.best_params}")
    print(f"  run_tag          : {study.best_trial.user_attrs.get('run_tag')}")

    values = [t.value for t in done]
    gain = max(values) - min(values)
    calib = Path(args.out_dir) / f"{args.study_name}_calibration.json"
    print(f"\n  amplitude de l'étude : {gain:.6f}")
    if calib.exists():
        floor = json.loads(calib.read_text())["spread"]
        print(f"  plancher de bruit    : {floor:.6f}")
        verdict = ("EXPLOITABLE" if gain > 2 * floor else
                   "NON CONCLUANT — l'étude n'a pas dépassé le bruit")
        print(f"  VERDICT : {verdict}")
    else:
        print("  plancher de bruit    : NON MESURÉ — lancer `calibrate` d'abord ;")
        print("                         sans lui, ce gain n'est pas interprétable.")
    print("=" * 64)


def cmd_report(args):
    try:
        import optuna
    except ImportError:
        sys.exit("optuna absent : pip install -e '.[optim]'")
    storage = f"sqlite:///{Path(args.out_dir) / (args.study_name + '.db')}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    _print_report(study, args)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser():
    p = argparse.ArgumentParser(
        prog="search.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--base-config", default="workflow/config/config.yaml",
                        help="config servant de base à chaque essai")
        sp.add_argument("--objective", default="cross_seed_stability",
                        choices=obj.available(),
                        help="critère à maximiser (voir objectives.py)")
        sp.add_argument("--study-name", default="stgp-search")
        sp.add_argument("--out-dir", default="optuna",
                        help="études SQLite + calibrations (gitignoré)")
        sp.add_argument("--trial-config-dir", default="optuna/configs")
        sp.add_argument("--seeds", type=int, default=3,
                        help="seeds par essai (3 = plancher, cf. bruit)")
        sp.add_argument("--backend", default="local", choices=["local", "cluster"])
        sp.add_argument("--cores", type=int, default=8)
        sp.add_argument("--dry-run", action="store_true",
                        help="n'exécute pas le pipeline (test du plumbing)")
        sp.add_argument("--reference-file", default=None,
                        help="liste de gènes pour known_driver_recall")
        sp.add_argument("--top-n", type=int, default=100)

    sp = sub.add_parser("calibrate", help="mesurer le plancher de bruit")
    common(sp)
    sp.add_argument("--repeats", type=int, default=3)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("search", help="lancer une étude Optuna")
    common(sp)
    sp.add_argument("--n-trials", type=int, default=20)
    sp.add_argument("--allow-few-seeds", action="store_true")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("report", help="résumer une étude terminée")
    common(sp)
    sp.set_defaults(func=cmd_report)
    return p


if __name__ == "__main__":
    _args = build_parser().parse_args()
    _args.func(_args)
