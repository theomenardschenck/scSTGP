"""``stateshift`` — the console entry point.

This does not replace ``workflow/run.sh``; it generalises it. ``run.sh`` locates
everything through ``dirname $0`` and therefore only ever works inside a clone.
The same logic here resolves the Snakefile, the configs and the SLURM profile
through the *installed package*, so the pipeline runs from any directory, with
or without a clone.

Design constraints, in order of priority:

1. Never change what the pipeline computes. Both entry points build the same
   Snakemake invocation; a run launched either way must be indistinguishable.
2. Fail early and in French, at the level of the user's actual problem —
   "snakemake introuvable" beats a traceback from the Snakefile parser.
3. No dependency beyond pyyaml, which the package already requires. This module
   is imported by the console script on every invocation, including ``doctor``
   on a machine where torch is broken; it must not import the scientific stack.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__, _layout

# Directory where generated artefacts (the resolved SLURM profile) are written.
# Kept inside the working directory rather than in a global cache: two projects
# on the same machine legitimately want different partitions and QOS.
_WORK_SUBDIR = ".stateshift"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _die(message: str, code: int = 1) -> None:
    print(f"[stateshift] {message}", file=sys.stderr)
    raise SystemExit(code)


def _require_snakemake() -> str:
    exe = shutil.which("snakemake")
    if exe is None:
        _die(
            "snakemake introuvable dans le PATH.\n"
            "        Le pipeline en a besoin (version 7.x). Deux voies :\n"
            "          conda env create -f environment.yml && conda activate gnn\n"
            "          pip install 'stateshift[workflow]'\n"
            "        Diagnostic complet : stateshift doctor"
        )
    return exe


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_configfile(given: str | None) -> Path:
    """User-provided config, else the one bundled with the package."""
    if given:
        path = Path(given).expanduser()
        if not path.is_file():
            _die(f"configfile introuvable : {path}")
        return path.resolve()

    default = _layout.config_dir() / "config.yaml"
    if not default.is_file():
        _die(
            "aucun --configfile donné et aucune config par défaut embarquée.\n"
            "        Génère la tienne : stateshift init"
        )
    print(
        f"[stateshift] aucun --configfile : utilisation de la config EMBARQUÉE\n"
        f"             {default}\n"
        f"             (config de production HUVEC — génère la tienne avec "
        f"`stateshift init`)",
        file=sys.stderr,
    )
    return default


def _materialize_profile(source: Path, dest: Path) -> Path:
    """Rewrite the SLURM profile so its submit script is addressable.

    Two things break a packaged profile, and both are silent:

    * ``cluster:`` names ``workflow/profiles/slurm/submit.sh`` relative to the
      working directory — correct only from a clone root.
    * a wheel does not preserve the executable bit on package data, so the
      installed ``submit.sh`` cannot be exec'd directly.

    Prefixing with ``bash`` and an absolute path fixes both at once. The source
    profile is left untouched, so editing it in a clone still takes effect.
    """
    import yaml

    src_cfg = source / "config.yaml"
    if not src_cfg.is_file():
        _die(f"profil SLURM incomplet : {src_cfg} manquant")

    cfg = _load_yaml(src_cfg)
    cluster_cmd = cfg.get("cluster")
    if isinstance(cluster_cmd, str) and "submit.sh" in cluster_cmd:
        tokens = cluster_cmd.split()
        cut = next(i for i, t in enumerate(tokens) if t.endswith("submit.sh"))
        submit = (source / Path(tokens[cut]).name).resolve()
        if not submit.is_file():
            _die(f"script de soumission introuvable : {submit}")
        cfg["cluster"] = " ".join(["bash", str(submit), *tokens[cut + 1:]])

    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "config.yaml"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(
            "# Généré par `stateshift run --backend cluster` — NE PAS ÉDITER.\n"
            f"# Source : {src_cfg}\n"
            "# Édite la source, ou `stateshift profile --copy <dir>` pour en\n"
            "# obtenir une copie modifiable, puis --profile-dir <dir>.\n"
        )
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False,
                       allow_unicode=True)
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# stateshift run
# ─────────────────────────────────────────────────────────────────────────────
def cmd_run(args: argparse.Namespace, passthrough: list[str]) -> int:
    snakemake = _require_snakemake()
    configfile = _resolve_configfile(args.configfile)

    backend = args.backend
    if backend is None:
        cfg = _load_yaml(configfile)
        backend = (cfg.get("compute") or {}).get("backend", "local")
    if backend not in ("local", "cluster"):
        _die(f"backend inconnu : '{backend}' (attendu : local | cluster)")

    cmd = [
        snakemake,
        "-s", str(_layout.snakefile()),
        "--configfile", str(configfile),
        # Mirrors run.sh: a failing VALIDATION must not abort the branch that
        # produces the scientific deliverables.
        "--keep-going",
    ]
    if args.dry_run:
        cmd.append("-n")

    if backend == "local":
        print(
            "⚠️  [stateshift] BACKEND=local — exécution sur CETTE machine. "
            "Les vraies charges\n"
            "        (build du graphe ~40 min, entraînement) devraient tourner "
            "sur le cluster :\n"
            "        relance avec --backend cluster.",
            file=sys.stderr,
        )
        cmd += ["--cores", str(args.cores)]
    else:
        source = Path(args.profile_dir).resolve() if args.profile_dir else _layout.profile_dir()
        if not source.is_dir():
            _die(f"profil SLURM introuvable : {source}")
        profile = _materialize_profile(source, Path(_WORK_SUBDIR).resolve() / "profile")
        print(
            f"[stateshift] BACKEND=cluster — soumission SLURM.\n"
            f"             profil source : {source}\n"
            f"             profil résolu : {profile}",
            file=sys.stderr,
        )
        cmd += ["--profile", str(profile), "--jobs", str(args.jobs)]

    # Interpreter used by the rules. `compute.python: "python"` historically
    # meant "whatever the activated conda env provides"; that assumption breaks
    # the moment snakemake and the package live in different environments. The
    # interpreter running THIS command is the one that certainly owns the
    # package, so it is the better default — but an explicit environment
    # variable still wins, and a clone driven by run.sh is unaffected.
    env = dict(os.environ)
    env.setdefault("STATESHIFT_PYTHON", sys.executable)
    env.setdefault("STATESHIFT_PYTHON_TORCH", sys.executable)

    cmd += passthrough
    print(f"[stateshift] interpréteur des règles : {env['STATESHIFT_PYTHON']}",
          file=sys.stderr)
    print(f"[stateshift] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.call(cmd, env=env)


# ─────────────────────────────────────────────────────────────────────────────
# stateshift init
# ─────────────────────────────────────────────────────────────────────────────
def cmd_init(args: argparse.Namespace, passthrough: list[str]) -> int:
    init_py = _layout.workflow_dir() / "init.py"
    if not init_py.is_file():
        _die(f"assistant introuvable : {init_py}")
    cmd = [sys.executable, str(init_py)]
    if args.preset:
        cmd += ["--preset", args.preset]
    if args.out_dir:
        cmd += ["--out-dir", args.out_dir]
    return subprocess.call(cmd + passthrough)


# ─────────────────────────────────────────────────────────────────────────────
# stateshift path / configs / profile
# ─────────────────────────────────────────────────────────────────────────────
_PATH_KINDS = {
    "package": _layout.package_dir,
    "workflow": _layout.workflow_dir,
    "snakefile": _layout.snakefile,
    "config": _layout.config_dir,
    "profile": _layout.profile_dir,
}


def cmd_path(args: argparse.Namespace, _passthrough: list[str]) -> int:
    """Machine-readable paths — this is what shell scripts should call.

    A SLURM script that needs ``gnn_vgae.py`` should ask for it rather than
    assume a clone: ``SRC=$(stateshift path package)``.
    """
    print(_PATH_KINDS[args.kind]())
    return 0


def cmd_configs(_args: argparse.Namespace, _passthrough: list[str]) -> int:
    cfg_dir = _layout.config_dir()
    print(f"Configs embarquées ({cfg_dir}) :\n")
    for path in sorted(cfg_dir.glob("*.yaml")):
        head = ""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            text = stripped.lstrip("#").strip()
            # Skip rule-off banners (===, ───, ***): they are decoration, and a
            # listing full of separators tells the reader nothing.
            if len(text) > 3 and len(set(text)) > 3:
                head = text
                break
        print(f"  {path.name:<28} {head[:60]}")
    print("\n  Utilisation : stateshift run --configfile <chemin>")
    return 0


def cmd_profile(args: argparse.Namespace, _passthrough: list[str]) -> int:
    source = _layout.profile_dir()
    if args.copy:
        dest = Path(args.copy).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)
        print(f"Profil SLURM copié dans {dest}")
        print("Édite partition/QOS puis :")
        print(f"  stateshift run --backend cluster --profile-dir {dest} --configfile <cfg>")
        return 0
    print(f"# {source / 'config.yaml'}\n")
    print((source / "config.yaml").read_text(encoding="utf-8"))
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# stateshift doctor
# ─────────────────────────────────────────────────────────────────────────────
def _check_import(name: str) -> tuple[str, str]:
    try:
        mod = __import__(name)
    except Exception as exc:  # noqa: BLE001 — a broken install must not crash doctor
        return "MANQUE", str(exc).split("\n")[0][:60]
    return "OK", getattr(mod, "__version__", "?")


def cmd_doctor(_args: argparse.Namespace, _passthrough: list[str]) -> int:
    """Environment report. Exists because the two most costly failures observed
    on this project were both environment drift, not code:

    * ``snakemake`` resolving to a different environment than the package, so
      the pipeline ran against a stale interpreter;
    * a cluster wave launched without the SLURM profile, executing the entire
      DAG on the login node.

    Both are invisible until hours are wasted; both are one line here.
    """
    problems = 0
    print("═" * 68)
    print("  stateshift doctor")
    print("═" * 68)

    print(f"\n▸ Paquet\n    version        {__version__}")
    for key, value in _layout.describe().items():
        print(f"    {key:<14} {value}")

    print(f"\n▸ Interpréteur\n    exécutable     {sys.executable}")
    print(f"    version        {sys.version.split()[0]}")
    if sys.version_info < (3, 12):
        print("    ⚠️  Python 3.12 minimum requis (requires-python >=3.12)")
        problems += 1

    print("\n▸ Snakemake")
    snake = shutil.which("snakemake")
    if snake is None:
        print("    ✗ absent du PATH — le pipeline ne peut pas être lancé")
        problems += 1
    else:
        try:
            version = subprocess.run(
                [snake, "--version"], capture_output=True, text=True, timeout=60
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            version = "?"
        print(f"    exécutable     {snake}")
        print(f"    version        {version}")
        if version and not version.startswith("7."):
            print(f"    ⚠️  version 7.x attendue (le profil SLURM utilise la clé "
                  f"`cluster:`, syntaxe 7.x)")
            problems += 1
        # The drift that matters: snakemake living outside the environment that
        # holds the package means the Snakefile's `import stateshift` resolves
        # elsewhere, or not at all.
        if not snake.startswith(sys.prefix):
            print(f"    ⚠️  snakemake est HORS de l'environnement courant")
            print(f"        env courant  : {sys.prefix}")
            print(f"        snakemake    : {snake}")
            print(f"        → le Snakefile peut ne pas voir le paquet stateshift.")
            problems += 1

    print("\n▸ Pile scientifique")
    for name in ("numpy", "pandas", "scipy", "sklearn", "networkx",
                 "matplotlib", "yaml", "torch", "torch_geometric", "scanpy"):
        status, detail = _check_import(name)
        mark = "✓" if status == "OK" else "✗"
        print(f"    {mark} {name:<16} {detail}")
        if status != "OK" and name in ("numpy", "pandas", "scipy", "yaml"):
            problems += 1
    print("    (torch / torch_geometric / scanpy : requis pour un VRAI run,")
    print("     pas pour la CLI ni les loaders — cf. environment.yml)")

    print("\n▸ Cluster")
    sbatch = shutil.which("sbatch")
    print(f"    sbatch         {sbatch or '— (pas de SLURM ici : backend local seulement)'}")

    print("\n▸ Workflow")
    try:
        n_cfg = len(list(_layout.config_dir().glob("*.yaml")))
        print(f"    Snakefile      {_layout.snakefile()}")
        print(f"    configs        {n_cfg} embarquée(s)")
    except FileNotFoundError as exc:
        print(f"    ✗ {exc}")
        problems += 1

    print("\n" + "═" * 68)
    if problems:
        print(f"  {problems} point(s) à corriger avant un run réel.")
    else:
        print("  Rien à signaler.")
    print("═" * 68)
    return 1 if problems else 0


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stateshift",
        description="stateshift — priorisation des gènes qui pilotent une "
                    "transition d'état cellulaire.",
        epilog="Documentation : https://github.com/theomenardschenck/scSTGP"
               "/tree/main/docs/guide",
    )
    parser.add_argument("--version", action="version",
                        version=f"stateshift {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="assistant interactif de configuration")
    p_init.add_argument("--preset", choices=["quick", "full"], default=None)
    p_init.add_argument("--out-dir", default=None,
                        help="où écrire les configs générées")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser(
        "run", help="lance le pipeline (local ou SLURM)",
        description="Tout ce qui suit `--` est transmis tel quel à snakemake.")
    p_run.add_argument("--backend", choices=["local", "cluster"], default=None,
                       help="défaut : compute.backend du config, sinon local")
    p_run.add_argument("--configfile", default=None)
    p_run.add_argument("--cores", type=int, default=8, help="backend local")
    p_run.add_argument("--jobs", type=int, default=20, help="backend cluster")
    p_run.add_argument("--profile-dir", default=None,
                       help="profil SLURM alternatif (cf. stateshift profile --copy)")
    p_run.add_argument("-n", "--dry-run", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_path = sub.add_parser("path", help="chemin résolu d'un composant")
    p_path.add_argument("kind", choices=sorted(_PATH_KINDS), nargs="?",
                        default="package")
    p_path.set_defaults(func=cmd_path)

    p_cfg = sub.add_parser("configs", help="liste les configs embarquées")
    p_cfg.set_defaults(func=cmd_configs)

    p_prof = sub.add_parser("profile", help="affiche ou copie le profil SLURM")
    p_prof.add_argument("--copy", metavar="DIR", default=None,
                        help="copie le profil dans DIR pour l'éditer")
    p_prof.set_defaults(func=cmd_profile)

    p_doc = sub.add_parser("doctor", help="diagnostic de l'environnement")
    p_doc.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in raw:
        cut = raw.index("--")
        raw, passthrough = raw[:cut], raw[cut + 1:]

    args = build_parser().parse_args(raw)
    try:
        return args.func(args, passthrough)
    except FileNotFoundError as exc:
        _die(str(exc))
    except KeyboardInterrupt:
        print("\n[stateshift] interrompu.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
