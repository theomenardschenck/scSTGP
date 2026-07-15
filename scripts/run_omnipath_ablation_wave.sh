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
PARALLEL=1               # nb d'ablations lancées CONCURREMMENT (chacune orchestre
                         # ses propres jobs SLURM via le profil). 1 = séquentiel.
SNAKE_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)        ONLY="$2"; shift 2;;
    --base-config) BASE_CONFIG="$2"; shift 2;;
    --parallel)    PARALLEL="$2"; shift 2;;
    --dry-run|-n)  DRY="-n"; shift;;
    --)            shift; while [[ $# -gt 0 ]]; do SNAKE_ARGS+=("$1"); shift; done;;
    *)             SNAKE_ARGS+=("$1"); shift;;
  esac
done
[[ "$PARALLEL" =~ ^[0-9]+$ && "$PARALLEL" -ge 1 ]] || { echo "--parallel doit être un entier ≥1" >&2; exit 1; }

[[ -f "$PROFILE" ]]     || { echo "profil introuvable : $PROFILE" >&2; exit 1; }
[[ -f "$BASE_CONFIG" ]] || { echo "config de base introuvable : $BASE_CONFIG" >&2; exit 1; }

# ── PREFLIGHT : si le profil active l'intégration OmniPath (features/arêtes),
#    l'artefact data/omnipath/graph/ + caches DOIVENT être présents, sinon les
#    features sortent à 0 et les arêtes extra sont vides (run INERTE — cf. V6.1.3).
#    L'artefact est gitignoré → absent d'un clone frais (cluster). On bloque
#    AVANT de lancer des heures de calcul. --force pour outrepasser.
NEEDS_OP=$(grep -Eq "use-omnipath-node-features|omnipath-edges" "$PROFILE" && echo 1 || echo 0)
DATA_DIR="${GNN_DATA_DIR:-$(python - "$BASE_CONFIG" <<'PY'
import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["paths"]["data_root"])
PY
)}"
if [[ "$NEEDS_OP" == "1" ]]; then
  MISSING=()
  for f in omnipath/graph/edges.tsv.gz omnipath/graph/nodes.tsv.gz \
           omnipath/hgnc_biotype_map.tsv.gz omnipath/signaling_omnipath.tsv.gz; do
    [[ -f "$DATA_DIR/$f" ]] || MISSING+=("$DATA_DIR/$f")
  done
  if [[ ${#MISSING[@]} -gt 0 && "${FORCE_WAVE:-0}" != "1" ]]; then
    echo "╔═══════════════════════════════════════════════════════════════════════" >&2
    echo "║ PREFLIGHT ÉCHOUÉ — artefacts OmniPath manquants (run serait INERTE) :"   >&2
    printf '║   • %s\n' "${MISSING[@]}"                                             >&2
    echo "║ Le profil active features/arêtes OmniPath mais l'artefact est absent."   >&2
    echo "║ Sur le FRONTAL (Internet) : "                                            >&2
    echo "║   python scripts/build_omnipath_graph.py --layers all --download --cache-dir $DATA_DIR/omnipath" >&2
    echo "║   python -c \"import sys;sys.path.insert(0,'src/gnn');import hgnc_alias as h;h.build_biotype_map('$DATA_DIR/omnipath',download_if_missing=True)\"" >&2
    echo "║   python scripts/cache_omnipath.py --only signaling --cache-dir $DATA_DIR/omnipath" >&2
    echo "║ OU rsync depuis le local. Puis relancer (FORCE_WAVE=1 pour ignorer)."    >&2
    echo "╚═══════════════════════════════════════════════════════════════════════" >&2
    exit 2
  fi
fi

# ── PREFLIGHT SLURM : sans --profile, snakemake exécute TOUT en LOCAL (frontal) —
#    pas de job dans squeue, séquentiel, très lent. C'était le bug V6.1.3.
if [[ -z "$DRY" ]] && ! printf '%s\n' "${SNAKE_ARGS[@]}" | grep -q -- "--profile"; then
  echo "╔═══════════════════════════════════════════════════════════════════════" >&2
  echo "║ ⚠ AUCUN --profile → snakemake tournerait en LOCAL (nœud frontal) :"       >&2
  echo "║   pas de soumission SLURM, séquentiel, lent (bug V6.1.3)."                >&2
  echo "║ Ajoute :  --profile workflow/profiles/slurm"                             >&2
  echo "║ (FORCE_LOCAL=1 pour exécuter en local volontairement, ex. test rapide.)"  >&2
  echo "╚═══════════════════════════════════════════════════════════════════════" >&2
  [[ "${FORCE_LOCAL:-0}" == "1" ]] || exit 3
fi

# ⚠️ FS PARTAGÉ obligatoire (pas /tmp) : en mode --cluster, chaque job re-invoque
# snakemake sur un COMPUTE NODE et relit --configfile. /tmp est LOCAL au nœud → le
# job ne verrait pas le config généré (FileNotFoundError). On génère donc les
# configs sous le repo courant ($PWD, sur le FS partagé /LAB-DATA), pas /tmp.
TMPDIR="$(mktemp -d "${PWD}/.ablwave.XXXXXX")"
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
        # 1 seed/ablation → cluster_annotation (cross-seed, exige ≥2 runs)
        # planterait et stopperait tout le pipeline. On le désactive ici.
        "validation": {"decoy": {"enabled": bool(p.get("decoy", True))},
                       "cluster_annotation": {"enabled": False}},
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
FAILED=()
# Lance une ablation. En parallèle : sortie vers logs/wave/<tag>.log + --nolock
# (chaque ablation écrit dans des chemins DISJOINTS — run_tag distinct — donc pas
# de collision ; le lock .snakemake unique empêcherait juste la concurrence).
# Un SEUL --configfile avec les 2 fichiers → snakemake les fusionne (base puis
# surcharge). RÉSILIENCE : un échec ne tue pas la vague (set -e).
mkdir -p logs/wave
declare -A PIDTAG
run_one() {  # $1=tag $2=cfg  ; en séquentiel: sortie live ; en parallèle: fichier
  local tag="$1" cfg="$2"
  if [[ "$PARALLEL" -gt 1 ]]; then
    snakemake --configfile "$BASE_CONFIG" "$cfg" $DRY --nolock "${SNAKE_ARGS[@]}" \
      &> "logs/wave/${tag}.log"
  else
    snakemake --configfile "$BASE_CONFIG" "$cfg" $DRY "${SNAKE_ARGS[@]}"
  fi
}

while IFS=$'\t' read -r TAG CFG EXTRA; do
  [[ -z "${TAG:-}" ]] && continue
  if [[ "$PARALLEL" -le 1 ]]; then
    echo "=== [ablation] $TAG ============================================="
    run_one "$TAG" "$CFG" || FAILED+=("$TAG")
  else
    # Throttle : attendre qu'un slot se libère avant de lancer la suivante.
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do wait -n; done
    echo "=== [ablation] $TAG → logs/wave/${TAG}.log (parallèle, slot ≤$PARALLEL)"
    run_one "$TAG" "$CFG" &
    PIDTAG[$!]="$TAG"
  fi
done < "$TMPDIR/_grid.tsv"

# En parallèle : attendre tous les background + collecter les échecs.
if [[ "$PARALLEL" -gt 1 ]]; then
  for pid in "${!PIDTAG[@]}"; do
    wait "$pid" || FAILED+=("${PIDTAG[$pid]}")
  done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[wave] terminé AVEC ÉCHECS : ${FAILED[*]}" >&2
  exit 1
fi
echo "[wave] terminé (toutes les ablations OK)."
