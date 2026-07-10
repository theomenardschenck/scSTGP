#!/usr/bin/env bash
# ===========================================================================
# run_omnipath_ablation_wave.sh — déploie une grille d'ablations sur le
# Snakefile existant (inchangé), à raison d'1 seed + perturbation + decoy par
# ablation. Pilote un PROFIL (workflow/config/ablations_{omnipath,legacy}.yaml).
#
# Pour chaque ablation du profil, génère un config qui SURCHARGE le config de
# base (workflow/config/config.yaml) — run_tag, extra_flags, seeds, decoy — puis
# lance `snakemake --configfile base --configfile <généré>` jusqu'à `rule all`
# (train → perturb → cross-seed → driver_baselines/interpret → decoy).
#
# Usage :
#   scripts/run_omnipath_ablation_wave.sh <profil.yaml> [options] [-- snakemake args]
#     --only a,b,c      restreint aux ablations nommées
#     --base-config F   config de base (défaut workflow/config/config.yaml)
#     --dry-run | -n    snakemake -n (dry-run, ne lance rien)
#   Tout ce qui suit est passé tel quel à snakemake (ex. -j 8, --use-conda).
#
# Exemples :
#   # dry-run de toute la vague OmniPath
#   scripts/run_omnipath_ablation_wave.sh workflow/config/ablations_omnipath.yaml -n -j1
#   # lancer 2 ablations ciblées
#   scripts/run_omnipath_ablation_wave.sh workflow/config/ablations_omnipath.yaml \
#       --only base,no-ppi -j 8
#   # la référence legacy
#   scripts/run_omnipath_ablation_wave.sh workflow/config/ablations_legacy.yaml --only base -j 8
# ===========================================================================
set -euo pipefail

PROFILE="${1:?usage: run_omnipath_ablation_wave.sh <profil.yaml> [--only a,b] [-n] [snakemake args...]}"
shift

BASE_CONFIG="workflow/config/config.yaml"
ONLY=""
DRY=""
SNAKE_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)        ONLY="$2"; shift 2;;
    --base-config) BASE_CONFIG="$2"; shift 2;;
    --dry-run|-n)  DRY="-n"; shift;;
    --)            shift; while [[ $# -gt 0 ]]; do SNAKE_ARGS+=("$1"); shift; done;;
    *)             SNAKE_ARGS+=("$1"); shift;;
  esac
done

[[ -f "$PROFILE" ]]     || { echo "profil introuvable : $PROFILE" >&2; exit 1; }
[[ -f "$BASE_CONFIG" ]] || { echo "config de base introuvable : $BASE_CONFIG" >&2; exit 1; }

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# Génère un config par ablation (surcharge minimale) et un index _grid.tsv.
python - "$PROFILE" "$ONLY" "$TMPDIR" <<'PY'
import os, sys, yaml
profile, only, tmpd = sys.argv[1], sys.argv[2], sys.argv[3]
p = yaml.safe_load(open(profile))
grid = p["ablation_grid"]
if only:
    want = set(s.strip() for s in only.split(",") if s.strip())
    grid = [g for g in grid if g["name"] in want]
    missing = want - {g["name"] for g in grid}
    if missing:
        sys.exit(f"[wave] ablations inconnues dans le profil : {sorted(missing)}")
lines = []
for g in grid:
    tag   = f"{p['root_tag']}.{g['name']}"
    extra = (p["base_extra_flags"] + " " + g.get("add_flags", "")).split()
    extra = " ".join(extra)  # normalise les espaces
    cfg = {
        "models": {"vgae": {"enabled": True, "run_tag": tag,
                            "seeds": [int(p.get("seed", 1))],
                            "extra_flags": extra}},
        "validation": {"decoy": {"enabled": bool(p.get("decoy", True))}},
    }
    fp = os.path.join(tmpd, tag.replace("/", "_") + ".yaml")
    with open(fp, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    lines.append(f"{tag}\t{fp}\t{extra}")
with open(os.path.join(tmpd, "_grid.tsv"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"[wave] {len(grid)} ablation(s) à lancer :")
for l in lines:
    t, _, e = l.split("\t")
    print(f"   {t:22s} {e}")
PY

echo
while IFS=$'\t' read -r TAG CFG EXTRA; do
  [[ -z "${TAG:-}" ]] && continue
  echo "================================================================"
  echo "=== [ablation] $TAG"
  echo "================================================================"
  # Un SEUL --configfile avec les 2 fichiers → snakemake les fusionne dans
  # l'ordre (base puis surcharge d'ablation). Deux --configfile séparés ne
  # fusionnent PAS (le second écrase) → 'paths' serait perdu.
  snakemake --configfile "$BASE_CONFIG" "$CFG" $DRY "${SNAKE_ARGS[@]}"
done < "$TMPDIR/_grid.tsv"

echo "[wave] terminé."
