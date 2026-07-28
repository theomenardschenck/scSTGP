#!/usr/bin/env bash
# =============================================================================
# run_ablation_grid.sh — Job array SLURM pour grilles d'ablation VGAE
# =============================================================================
# Soumet UN seul job array SLURM. Le cluster (Nautilus, GLiCID) gère le
# parallélisme via `--array=0-(N-1)%MAX_PARALLEL`. Tu peux fermer la
# session après soumission, c'est SLURM qui pilote.
#
# Interface : produit cartésien `--version × --ablations`.
#
# --version {V3, V4, V4.1, V4.2, V4.2-sweep, V4.3-method-compare, V5, V4_1, V4_2}   (défaut : V4.1)
#     Définit les configs de base à tester.
#         V3   : 1 config (full, no OmniPath, no flags). Reproduit V3.6 baseline.
#         V4   : 4 configs avec edge_types `signaling` / `tf_curated` opt-in.
#                tags v4-{baseline,sig,tf,full}.
#         V4.1 : V4 + `--include-omnipath-genes` (étend gene_to_idx avec
#                endpoints OmniPath). tags v4.1-{baseline,sig,tf,full}.
#         V4.2 : 5 configs sur base V4.1-full + 3 leviers toggleable :
#                coexpr différentielle P4∪P16, γ_t par edge_type,
#                Reactome FI. tags v4.2-{full,coexdiff,coexdiff.gw,
#                coexdiff.rfi,coexdiff.gw.rfi}. PRÉREQUIS :
#                bash scripts/run_diff_coexpr.sh + cache_reactome_fi.sh.
#                Cf. §14bis.6octies du rapport.
#         V4.2-sweep : 3 configs comparant l'élagage du canal coexpr
#                (constat 2026-05-19 : AUC 0.91 vs V4.1 0.97 — coexpr
#                trop dense noie le décodeur). tags v4.2-coex.{k5q5,
#                k3q7,k2q8}. PRÉREQUIS : bash scripts/run_coexpr_prune_sweep.sh
#                (génère les 3 coexpr_diff.*.tsv depuis adjacencies existants).
#                Cf. §14bis.6sexdecies du rapport.
#
# --ablations <list>     (défaut : "" = juste les 4 configs base, AKA validation)
#     Liste séparée par virgules. Chaque entrée peut être :
#       (a) une ablation simple : no-coexpr, no-humess, no-reactome, no-ppi,
#           no-scenic-regulons, no-cgrp, ex-degrees
#       (b) composite par "+"      : no-coexpr+no-humess → flags cumulés sur 1 config
#       (c) pseudo "all-standard"  : applique chaque des 7 ablations standards
#           SÉPARÉMENT, restreint au sub-config `*-full` (sinon ça explose).
#       (d) pseudo "all-other-standard" : idem mais sans no-coexpr et no-humess
#           (= no-reactome, no-ppi, no-scenic-regulons, no-cgrp, ex-degrees)
#
# Cardinalité produite :
#     # Configs = (4 si V4/V4.1, 1 si V3) × (n entrées dans --ablations OU 1)
#     # Tâches  = # Configs × n_seeds
#
# Exemples (les 4 cas demandés sur V4.1, 3 seeds chacun) :
#     # 1. 3 seeds no-coexpr seul → 4 base × 1 ablation × 3 seeds = 12 tâches
#     bash scripts/run_ablation_grid.sh --version V4.1 --ablations no-coexpr \
#         --seeds "1 2 3"
#
#     # 2. 3 seeds no-humess seul → 12 tâches
#     bash scripts/run_ablation_grid.sh --version V4.1 --ablations no-humess \
#         --seeds "1 2 3"
#
#     # 3. 3 seeds no-coexpr ET no-humess (composite, flags cumulés) → 12 tâches
#     bash scripts/run_ablation_grid.sh --version V4.1 --ablations "no-coexpr+no-humess" \
#         --seeds "1 2 3"
#
#     # 4. 3 seeds des 5 autres ablations standard (no-reactome / no-ppi /
#     #    no-scenic-regulons / no-cgrp / ex-degrees), appliquées sur le
#     #    sub-config `v4.1-full` uniquement → 5 × 3 = 15 tâches.
#     bash scripts/run_ablation_grid.sh --version V4.1 --ablations all-other-standard \
#         --seeds "1 2 3"
#
# Validation de version (rejoue la phase 1) :
#     bash scripts/run_ablation_grid.sh --version V4.1 --seeds "1 2 3"
#     # = 4 configs × 3 seeds = 12 tâches, pas d'ablation extra
#
# === V5 (TIER 1c, smoke-testé 2026-05-22) ============================
# BASE V5 = V4.1-full (coexpr p16_only LEGACY par défaut, décision 2026-05-27).
# 4 sous-configs : v5-baseline / v5-sm / v5-sd / v5-full.
#
# Pourquoi base p16_only et pas differential ? V4.2-differential n'est pas
# encore validé cross-seed (AUC k2q8=0.953 vs V4.1 0.972, manque le
# driver_score cross-seed). Greffer V5 sur V4.1 (terrain validé) évite de
# confondre l'effet *signed* avec l'effet *coexpr différentielle V4.2*.
#
# Prérequis : aucun spécifique V5 (mêmes données qu'V4.1).
#             `diff-coexpr` requiert coexpr_diff.tsv (V4.2 préalable).
#
# Validation factorielle 2×2 V5 (les 4 sous-configs sans ablation) :
#     bash scripts/run_ablation_grid.sh --version V5 --seeds "1 2 3"
#     # 4 configs × 3 seeds = 12 tâches (smoke A/B {sm,sd} ON/OFF)
#
# Grille V5 recommandée (5 ablations × v5-full × 3 seeds = 15 tâches) :
#     bash scripts/run_ablation_grid.sh --version V5 \
#         --ablations v5-recommended --seeds "1 2 3"
#     # ablations : no-coexpr, diff-coexpr (override p16_only→differential),
#     # no-humess, no-coexpr+no-humess, dedup-ppi-remove
#     # Restriction à v5-full (= signed-message + signed-decoder activés).
#     # Pour la validation factorielle 2×2 (sm/sd × baseline/full), utiliser
#     # `--version V5` sans --ablations (12 tâches).
#
# Ablations standard sur v5-full uniquement (15 tâches) :
#     bash scripts/run_ablation_grid.sh --version V5 \
#         --ablations all-other-standard --seeds "1 2 3"
#
# Ablation single (composable) :
#     bash scripts/run_ablation_grid.sh --version V5 --ablations diff-coexpr \
#         --seeds "1 2 3"
#     # = 4 sub-configs × diff-coexpr × 3 seeds = 12 tâches
#     # NB : `diff-coexpr` override --coexpr-mode (argparse last-wins).
#     # À lancer une fois V4.2-k2q8 validé cross-seed.
#
# Aliases backward-compat :
#     --mode v4_validation     == --version V4    --ablations ""
#     --mode v4_1_validation   == --version V4.1  --ablations ""
#     --mode v4_no_coexpr      == --version V4    --ablations no-coexpr
#     --mode v3_legacy         == --version V3    --ablations all-standard
#     --mode full_grid         == complexe — gardé via le case legacy
#
# Pré-requis : cwd = racine du projet (gnn_huvec/).
#              data/omnipath/ doit contenir tf_collectri.tsv.gz +
#              signed_ppi_signor.tsv.gz pour V4/V4.1.
# Doc GLiCID : https://doc.glicid.fr/GLiCID-PUBLIC/quickstart_advanced_user.html
# =============================================================================

set -euo pipefail

# Shebang bash robuste pour les jobs (GLiCID/Guix : `/usr/bin/env bash` n'est
# PAS résolu côté nœud, `/bin/bash` absent). Le corps des jobs utilise des
# tableaux/`[[ ]]` bash → on bake le chemin absolu résolu sur le frontal (FS
# partagé ⇒ valide sur les nœuds). Override : env BASH_BIN.
BASH_BIN="${BASH_BIN:-$(command -v bash || true)}"
# Guix : déréférencer le symlink de profil (node-local) vers le vrai chemin
# /gnu/store/… partagé sur tous les nœuds (sinon `bad interpreter` côté calcul).
[[ -n "$BASH_BIN" ]] && BASH_BIN="$(readlink -f "$BASH_BIN" 2>/dev/null || echo "$BASH_BIN")"
[[ -z "$BASH_BIN" ]] && BASH_BIN="/bin/bash"

# --- Paramètres par défaut --------------------------------------------------
DEFAULT_SEEDS=(1 2 3)
VERSION="V5.4"
ABLATIONS=""
MODE=""                        # legacy alias
MAX_PARALLEL=10
TIME_LIMIT="0-03:00:00"
DRY_RUN=0
DATA_ROOT_CLI=""               # --data-root (override DATA_ROOT)

# --- Parsing CLI ------------------------------------------------------------
SEEDS=("${DEFAULT_SEEDS[@]}")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --version)        VERSION="$2"; shift 2 ;;
        --ablations)      ABLATIONS="$2"; shift 2 ;;
        --mode)           MODE="$2"; shift 2 ;;   # legacy
        --seeds)          IFS=' ' read -r -a SEEDS <<< "$2"; shift 2 ;;
        --max-parallel)   MAX_PARALLEL="$2"; shift 2 ;;
        --time)           TIME_LIMIT="$2"; shift 2 ;;
        --data-root)      DATA_ROOT_CLI="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,75p' "$0"; exit 0 ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
done

# --- Backward-compat : si --mode passé, remap sur --version + --ablations --
case "$MODE" in
    "")                ;;
    v4_validation)     VERSION="V4";    ABLATIONS="" ;;
    v4_1_validation)   VERSION="V4.1";  ABLATIONS="" ;;
    v4_no_coexpr)      VERSION="V4";    ABLATIONS="no-coexpr" ;;
    v3_legacy)         VERSION="V3";    ABLATIONS="all-standard" ;;
    full_grid)         VERSION="V4.1";  ABLATIONS="all-standard" ;;
    *) echo "Mode legacy inconnu : $MODE"; exit 1 ;;
esac

# Normalisation : V4_1 → V4.1 / V4_2 → V4.2 pour éviter pièges shell
[[ "$VERSION" == "V4_1" ]] && VERSION="V4.1"
[[ "$VERSION" == "V4_2" ]] && VERSION="V4.2"
[[ "$VERSION" == "V4_2-sweep" || "$VERSION" == "V4_2_sweep" ]] && VERSION="V4.2-sweep"
[[ "$VERSION" == "V5_3" ]] && VERSION="V5.3"

# --- Définition des sub-configs par version ---------------------------------
# Format : "<tag>::<flags>" — flags passés tels quels à gnn_vgae.py.
# Le sub-tag est COMBINÉ avec un tag d'ablation pour former le RUN_TAG final.
OP_SIG="--use-omnipath-signaling"
OP_TF="--use-omnipath-tf-curated"
OP_INC="--include-omnipath-genes"

case "$VERSION" in
    V3)
        BASE_CONFIGS=( "v3-full::" )
        ;;
    V4)
        BASE_CONFIGS=(
            "v4-baseline::"
            "v4-sig::$OP_SIG"
            "v4-tf::$OP_TF"
            "v4-full::$OP_SIG $OP_TF"
        )
        ;;
    V4.1)
        BASE_CONFIGS=(
            "v4.1-baseline::"
            "v4.1-sig::$OP_SIG $OP_INC"
            "v4.1-tf::$OP_TF $OP_INC"
            "v4.1-full::$OP_SIG $OP_TF $OP_INC"
        )
        ;;
    V4.2)
        # V4.2 = V4.1-full + 3 leviers toggleable (cf. §14bis.6octies).
        # Prérequis : coexpr_diff.tsv (bash scripts/run_diff_coexpr.sh)
        # + cache Reactome FI (bash scripts/cache_reactome_fi.sh).
        V41_FULL="$OP_SIG $OP_TF $OP_INC"
        COEXDIFF="--coexpr-mode differential"
        GW="--edge-type-weights ppi=0.1,coexpression=0.5"
        RFI="--use-reactome-fi"
        BASE_CONFIGS=(
            "v4.2-full::$V41_FULL"
            "v4.2-coexdiff::$V41_FULL $COEXDIFF"
            "v4.2-coexdiff.gw::$V41_FULL $COEXDIFF $GW"
            "v4.2-coexdiff.rfi::$V41_FULL $COEXDIFF $RFI"
            "v4.2-coexdiff.gw.rfi::$V41_FULL $COEXDIFF $GW $RFI"
        )
        ;;
    V4.2-sweep)
        # Sweep d'élagage coexpr (cf. §14bis.6sexdecies). Run V4.2 du
        # 2026-05-19 : AUC 0.9051 (vs V4.1 0.97). Hypothèse : K=5 / floor
        # q=0.5 / universe=all = trop d'arêtes coexpr (~150-200k, 1.5×PPI)
        # qui noient le signal PPI signé dans le décodeur.
        # 3 configs avec --diff-coexpr-file vers fichiers pré-générés par
        # run_coexpr_prune_sweep.sh (depuis MÊMES adjacencies, juste re-merge).
        V41_FULL="$OP_SIG $OP_TF $OP_INC"
        COEXDIFF="--coexpr-mode differential"
        DD_BASE="data/pyscenic/diff_coexpr/coexpr_diff"
        BASE_CONFIGS=(
            "v4.2-coex.k5q5::$V41_FULL $COEXDIFF --diff-coexpr-file ${DD_BASE}.k5q5.tsv"
            "v4.2-coex.k3q7::$V41_FULL $COEXDIFF --diff-coexpr-file ${DD_BASE}.k3q7.tsv"
            "v4.2-coex.k2q8::$V41_FULL $COEXDIFF --diff-coexpr-file ${DD_BASE}.k2q8.tsv"
        )
        ;;
    V4.3-method-compare)
        # V4.3 (cf. §14bis.6septdecies) — grille méthode×prune sur la couche
        # coexpr. Chaque config combine :
        #   --coexpr-method ∈ {sklearn, arboreto, corr, mi}
        #   --coexpr-prune  ∈ {topk, quantile, mr, zscore}
        # gnn_vgae.py auto-résout le chemin coexpr_diff.<method>.<prune>.tsv
        # (les 16 fichiers doivent avoir été générés par
        # scripts/run_coexpr_method_compare.sh AVANT cette grille).
        # Par défaut on N'INCLUT QUE les configs dont la combinaison fait
        # sens (méthodes corr/mi n'ont pas de delta P4/P16 enrichi, mais
        # le format adjacencies → merge-adjacencies reste compatible).
        # Variables d'env optionnelles V43_METHODS / V43_PRUNES pour
        # restreindre la grille (e.g. V43_METHODS="sklearn" pour Phase A).
        V41_FULL="$OP_SIG $OP_TF $OP_INC"
        COEXDIFF="--coexpr-mode differential"
        # Grille V4.3 figée 2026-05-29 : {arboreto, sklearn} × {topk, quantile, mr}.
        # Exclus : corr/mi (pas encore générés en routine), zscore (113k/149k
        # arêtes — noie le décodeur, §14bis.6sexdecies). mr utilise K=10
        # (cf. coexpr_diff.{m}.mr.tsv régénérés avec K=10 ; à K=5 perte des
        # drivers ASNS/IL6/IL1B sur arboreto).
        V43_METHODS="${V43_METHODS:-arboreto sklearn}"
        V43_PRUNES="${V43_PRUNES:-topk quantile mr}"
        BASE_CONFIGS=()
        for _m in $V43_METHODS; do
            for _p in $V43_PRUNES; do
                BASE_CONFIGS+=( "v4.3-${_m}.${_p}::$V41_FULL $COEXDIFF --coexpr-method ${_m} --coexpr-prune ${_p}" )
            done
        done
        ;;
    V5)
        # V5 = V4.1-full (coexpr p16_only legacy) + message-passing/décodeur
        # signés. 4 sous-configs pour l'ablation factorielle 2×2 :
        # {sm OFF/ON} × {sd OFF/ON}. Cf. §14bis.6septies + TIER 1c
        # du TODO (smoke-test 2026-05-22).
        #
        # BASE V5 = V4.1-full = OP_SIG + OP_TF + OP_INC.
        # Coexpr legacy (p16_only) PAR DÉFAUT (décision 2026-05-27) : V4.2-
        # differential n'est pas encore validé cross-seed (AUC k2q8=0.953
        # vs V4.1 0.972, pas de driver_score cross-seed dispo). Garder la
        # base sur du terrain V4.1 validé évite de confondre l'effet signed
        # de l'effet V4.2. La coexpr différentielle reste accessible via
        # `--ablations diff-coexpr` quand V4.2 sera validé.
        #
        # Ablations V5 typiquement passées via --ablations :
        #   no-coexpr           : retire complètement coexpr (legacy V3.3 disruptif)
        #   diff-coexpr         : override p16_only → differential (V4.2)
        #   no-humess           : retire HuMess (config la plus défendable V4.1.1)
        #   no-coexpr+no-humess : double ablation
        #   dedup-ppi-remove    : supprime les PPI redondants avec arêtes
        #                         signées (bénéfice prévu PLEIN avec V5,
        #                         cf. §14bis.6quaterdecies)
        # + standards : no-reactome, no-ppi, no-scenic-regulons, no-cgrp,
        #               ex-degrees (via all-other-standard).
        #
        # Prérequis : aucun spécifique V5 (coexpr p16_only legacy = même
        # données que V4.1). `diff-coexpr` requiert coexpr_diff.tsv.
        V41_FULL="$OP_SIG $OP_TF $OP_INC"
        SM="--signed-message"
        SD="--signed-decoder"
        BASE_CONFIGS=(
            "v5-baseline::$V41_FULL"
            "v5-sm::$V41_FULL $SM"
            "v5-sd::$V41_FULL $SD"
            "v5-full::$V41_FULL $SM $SD"
        )
        ;;
    V5.3)
        # V5.3 = V5.1-full + V5.2 (inclusion arêtes signées dans pool reco,
        # déjà actif côté gnn_vgae.py:2652) + tête sub-espace `signed_proj`
        # (V5.3 par défaut quand --signed-decoder) + `dedup-ppi-signed remove`
        # (élimine la redondance PPI quand une arête signée existe sur la
        # même paire — bénéfice plein quand le bilinéaire vit dans son
        # sous-espace, cf. §14bis.6unvicesies).
        #
        # 4 sous-configs comme V5 mais avec `--dedup-ppi-signed remove`
        # ajouté partout. La projection `signed_proj` est active dès que
        # --signed-decoder ON (donc dans v5.3-sd et v5.3-full).
        #
        # signed-decoder-dim optionnel : par défaut LATENT_DIM (= 64) → init
        # identité, équivalence V5.2 au départ. Passer 32 ou 16 pour
        # compresser le sous-espace.
        V41_FULL="$OP_SIG $OP_TF $OP_INC"
        SM="--signed-message"
        SD="--signed-decoder"
        DEDUP="--dedup-ppi-signed remove"
        BASE_CONFIGS=(
            "v5.3-baseline::$V41_FULL $DEDUP"
            "v5.3-sm::$V41_FULL $SM $DEDUP"
            "v5.3-sd::$V41_FULL $SD $DEDUP"
            "v5.3-full::$V41_FULL $SM $SD $DEDUP"
        )
        ;;
    V5.3-finalist)
        # V5.3-finalist (2026-06-02) — Phase 3 : 3-seed validation des
        # finalistes V5.3-compose. 4 configs × 3 seeds = 12 tâches,
        # time limit 4h. Cf. §14bis.6duovicies.
        V41_FULL="$OP_SIG $OP_TF $OP_INC"
        LEG_BASE="$V41_FULL --signed-message --signed-decoder"
        SPLIT="--decoder-split"
        KL1="--kl-beta-max 0.0001"
        RFI="--use-reactome-fi"
        DEDUP="--dedup-ppi-signed remove"
        NOREACT="--no-reactome"
        PAT200="--patience 200"
        BASE_CONFIGS=(
            "v5.3-leg.split.rfi.dedup::$LEG_BASE $SPLIT $RFI $DEDUP $PAT200"
            "v5.3-leg.split.rfi.dedup.kl1::$LEG_BASE $SPLIT $RFI $DEDUP $KL1 $PAT200"
            "v5.3-leg.split.rfi.dedup.no-react::$LEG_BASE $SPLIT $RFI $DEDUP $NOREACT $PAT200"
            "v5.3-leg.split.rfi.dedup.kl1.no-react::$LEG_BASE $SPLIT $RFI $DEDUP $KL1 $NOREACT $PAT200"
        )
        ;;
    V5.4-ablation)
        # V5.4-ablation (2026-06-03) — ablations sur la config gagnante V5.4
        # (= split + rfi + dedup + kl1). Test des sources d'information
        # pour identifier celles qui portent le saut métabolique.
        #
        # Hypothèses :
        #   (a) Si no-rfi ramène le ranking vers V5.1 → c'est Reactome FI
        #       qui porte le basculement métabolique
        #   (b) Si no-coexpr stable → coexpr n'est plus discriminante
        #       en V5.4 (déjà testé V5.3-tune mais ici en signed-aware)
        #   (c) Si no-humess change peu → HuMess ne porte plus l'ETC bias
        #       en V5.4 (Reactome FI a pris le relais)
        #   (d) Si ex-degrees promeut Tier-1 V3.3 → MYC/HMGB1 etc. sont
        #       hub-bias en V5.4 (encore que is_hub_inflated=False)
        #   (e) backbone-only (PPI+REACTOME+signed seulement) = quel est
        #       le minimum nécessaire pour le bilinéaire signed ?
        #
        # 12 configs × 1 seed = 12 tâches ≈ 30h cluster (parallélisable).
        V41_FULL="$OP_SIG $OP_TF $OP_INC"
        # V54_NODEDUP = V5.4 sans le dedup-ppi (pour le test causal no-dedup) ;
        # V54_BASE = V54_NODEDUP + dedup (config V5.4 nominale).
        # 2026-06-04 : --n-epochs 1500 + --patience 200 (marge de convergence ;
        # les 1ers runs plafonnaient AUC ~0.94 à epoch 220-380).
        V54_NODEDUP="$V41_FULL --signed-message --signed-decoder --decoder-split \
                  --use-reactome-fi --kl-beta-max 0.0001 --patience 200 --n-epochs 1500"
        V54_BASE="$V54_NODEDUP --dedup-ppi-signed remove"
        DD="data/pyscenic/diff_coexpr"   # sources coexpr alternatives
        BASE_CONFIGS=(
            # === Baseline V5.4 nominale (split.rfi.dedup.kl1, AUCUNE ablation) ===
            # Comparateur de référence, MÊME régime (1500 epochs / patience 200).
            "v5.4.baseline::$V54_BASE"

            # === 5 ablations standard (isolation par source) ===
            "v5.4.no-coexpr::$V54_BASE --no-coexpr"
            "v5.4.no-humess::$V54_BASE --no-humess"
            "v5.4.no-reactome::$V54_BASE --no-reactome"
            "v5.4.no-scenic::$V54_BASE --no-scenic-regulons"
            "v5.4.ex-degrees::$V54_BASE --exclude-features ppi_degree,reg_degree"

            # === 3 ablations progressives (ajout cumulatif) ===
            "v5.4.no-coexpr+no-humess::$V54_BASE --no-coexpr --no-humess"
            "v5.4.no-coexpr+no-humess+no-scenic::$V54_BASE --no-coexpr --no-humess --no-scenic-regulons"
            "v5.4.no-coexpr+no-humess+no-scenic+ex-degrees::$V54_BASE --no-coexpr --no-humess --no-scenic-regulons --exclude-features ppi_degree,reg_degree"

            # === 4 combos ciblés (mes propositions) ===
            # Test 1 : isole l'effet Reactome FI (= V5.4 sans rfi)
            "v5.4.no-rfi::$V54_BASE --no-reactome-fi"
            # Test 2 : V5.4 sans rfi + anti-hub (récupère Tier-1 V3.3 ?)
            "v5.4.no-rfi+ex-degrees::$V54_BASE --no-reactome-fi --exclude-features ppi_degree,reg_degree"
            # Test 3 : backbone-only (PPI+REACTOME+signed minimaliste)
            "v5.4.backbone::$V54_BASE --no-coexpr --no-humess --no-scenic-regulons"
            # Test 4 : signed-pur (= backbone sans REACTOME pathway, ne reste
            # que PPI + arêtes signées). Test extrême : le bilinéaire suffit-il
            # à porter le signal sans pathway non-orienté ?
            "v5.4.signed-only::$V54_BASE --no-coexpr --no-humess --no-scenic-regulons --no-reactome"
            # Test 5 (2026-06-04) : no-dedup — TEST CAUSAL de la chute MYC/RAN.
            # = V5.4 SANS --dedup-ppi-signed. Prédiction falsifiable :
            #   doit REMONTER MYC (#915→~10) et RAN, et NE PAS bouger
            #   FHL2/HMGB2/H2AFZ (overlap PPI∩signé ~0). Si vérifié ⇒ dedup-ppi
            #   est le levier causal de la démotion des hubs (cf. gnn_futur.md §7).
            "v5.4.no-dedup::$V54_NODEDUP"

            # === Sources coexpr alternatives (2026-06-04) ===
            # V5.4 nominal utilise legacy P16-only sklearn. On teste si une
            # source différentielle P4∪P16 change le ranking en signed-aware.
            # (complète V5.3-coexpr-source qui le testait sans les autres ablations).
            # arboreto.mr : différentielle mutual-rank K=10 (~37k paires)
            "v5.4.coexpr-arboreto.mr::$V54_BASE --coexpr-mode differential --coexpr-method arboreto --coexpr-prune mr --diff-coexpr-file $DD/coexpr_diff.arboreto.mr.tsv"
            # arboreto.mr5 : mutual-rank K=5 (~16k, champion AUC V4.3-tune)
            "v5.4.coexpr-arboreto.mr5::$V54_BASE --coexpr-mode differential --coexpr-method arboreto --coexpr-prune mr --diff-coexpr-file $DD/coexpr_diff.arboreto.mr5.tsv"
            # arboreto.quantile.limited : top-7000 matching cardinalité legacy
            "v5.4.coexpr-arboreto.quantile-limited::$V54_BASE --coexpr-mode differential --coexpr-method arboreto --coexpr-prune quantile --diff-coexpr-file $DD/coexpr_diff.arboreto.quantile.limited.tsv"
            # legacy P16-only arboreto (topk) : contrôle même méthode arboreto
            # mais P16-only (≠ différentielle) → isole méthode vs différentiel
            "v5.4.coexpr-legacy-arboreto::$V54_BASE --coexpr-method arboreto --coexpr-prune topk --diff-coexpr-file $DD/coexpr_diff.arboreto.topk.tsv"
        )
        ;;
    V5.3-coexpr-source)
        # V5.3-coexpr-source (2026-06-03) — diagnostic contrôlé :
        # le saut de ranking V5.4 vs V3.3/V4.1/V5.1 (ρ Spearman 0.33-0.38,
        # Tier-1 V3.3 effondré sauf ENO1/ASNS) est-il dû au fichier
        # source coexpr ou à l'architecture V5 signed-aware ?
        #
        # Plan : tester V3.6 (baseline historique, sans signed) ET V5.4
        # (config gagnante split.rfi.dedup) sur 5 sources coexpr distinctes :
        #   1. legacy P16-only sklearn (defaut V3/V4/V5 : adjacencies_P16.csv)
        #   2. legacy P16-only arboreto (canonique)
        #   3. différentielle P4∪P16 arboreto mutual-rank (mr, 37k paires)
        #   4. différentielle P4∪P16 arboreto quantile (49k paires)
        #   5. différentielle P4∪P16 arboreto quantile LIMITED (7000 paires
        #      matching legacy cardinalité, créé 2026-06-03)
        #
        # Lecture : si V3.6 ranking change autant que V5.4 avec ces sources
        # → coexpr est le facteur. Si V3.6 stable et V5.4 instable →
        # l'archi V5 amplifie l'effet coexpr. Si les deux stables sur
        # leurs propres baselines → c'est rfi+dedup+split qui font tout.
        #
        # 1 seed par config × 10 configs (5 sources × 2 versions) =
        # 10 tâches ≈ 30 h cluster (V3.6 ~2h, V5.4 ~3h).
        V41_FULL="$OP_SIG $OP_TF $OP_INC"
        V5_BASE="$V41_FULL --signed-message --signed-decoder --decoder-split \
                  --use-reactome-fi --dedup-ppi-signed remove \
                  --kl-beta-max 0.0001 --patience 200"
        V3_BASE=""  # V3.6 = aucune option OmniPath, aucun signed
        DD="data/pyscenic/diff_coexpr"
        BASE_CONFIGS=(
            # --- V3.6 × 5 sources ---
            "v3.6.legacy-p16-sklearn::"
            "v3.6.legacy-p16-arboreto::--coexpr-method arboreto --coexpr-prune topk --diff-coexpr-file $DD/coexpr_diff.arboreto.topk.tsv"
            "v3.6.diff-arboreto-mr::--coexpr-mode differential --coexpr-method arboreto --coexpr-prune mr --diff-coexpr-file $DD/coexpr_diff.arboreto.mr.tsv"
            "v3.6.diff-arboreto-quantile::--coexpr-mode differential --coexpr-method arboreto --coexpr-prune quantile --diff-coexpr-file $DD/coexpr_diff.arboreto.quantile.tsv"
            "v3.6.diff-arboreto-quantile-limited::--coexpr-mode differential --coexpr-method arboreto --coexpr-prune quantile --diff-coexpr-file $DD/coexpr_diff.arboreto.quantile.limited.tsv"
            # --- V5.4 × 5 sources (= split.rfi.dedup.kl1 = V5.4 par défaut) ---
            "v5.4.legacy-p16-sklearn::$V5_BASE"
            "v5.4.legacy-p16-arboreto::$V5_BASE --coexpr-method arboreto --coexpr-prune topk --diff-coexpr-file $DD/coexpr_diff.arboreto.topk.tsv"
            "v5.4.diff-arboreto-mr::$V5_BASE --coexpr-mode differential --coexpr-method arboreto --coexpr-prune mr --diff-coexpr-file $DD/coexpr_diff.arboreto.mr.tsv"
            "v5.4.diff-arboreto-quantile::$V5_BASE --coexpr-mode differential --coexpr-method arboreto --coexpr-prune quantile --diff-coexpr-file $DD/coexpr_diff.arboreto.quantile.tsv"
            "v5.4.diff-arboreto-quantile-limited::$V5_BASE --coexpr-mode differential --coexpr-method arboreto --coexpr-prune quantile --diff-coexpr-file $DD/coexpr_diff.arboreto.quantile.limited.tsv"
        )
        ;;
    V5.3-finalist-long)
        # V5.3-finalist-long (2026-06-02 nuit) — mêmes configs que finalist
        # mais N_EPOCHS=1500 + patience 250 pour exploiter la marge de
        # convergence (Phase 1 montrait 4/7 runs frôlant le plafond 1000).
        # Time limit recommandé : --time "0-05:00:00" (1500 epochs lat-free).
        # Sert de **base entraînée pour la vague 2** de perturbation (ko_mode mask).
        V41_FULL="$OP_SIG $OP_TF $OP_INC"
        LEG_BASE="$V41_FULL --signed-message --signed-decoder"
        SPLIT="--decoder-split"
        KL1="--kl-beta-max 0.0001"
        RFI="--use-reactome-fi"
        DEDUP="--dedup-ppi-signed remove"
        NOREACT="--no-reactome"
        PAT250="--patience 250"
        EP1500="--n-epochs 1500"
        BASE_CONFIGS=(
            "v5.3-leg.split.rfi.dedup.long::$LEG_BASE $SPLIT $RFI $DEDUP $PAT250 $EP1500"
            "v5.3-leg.split.rfi.dedup.kl1.long::$LEG_BASE $SPLIT $RFI $DEDUP $KL1 $PAT250 $EP1500"
            "v5.3-leg.split.rfi.dedup.no-react.long::$LEG_BASE $SPLIT $RFI $DEDUP $NOREACT $PAT250 $EP1500"
            "v5.3-leg.split.rfi.dedup.kl1.no-react.long::$LEG_BASE $SPLIT $RFI $DEDUP $KL1 $NOREACT $PAT250 $EP1500"
        )
        ;;
    *)
        echo "Version inconnue : $VERSION (V3 | V4 | V4.1 | V4.2 | V4.2-sweep | V4.3-method-compare | V5 | V5.3 | V5.3-finalist | V5.3-finalist-long | V5.3-coexpr-source | V5.4-ablation)"; exit 1 ;;
esac

# --- Parsing de --ablations en variantes -----------------------------------
# Chaque variante = "<tag>::<flags>". Une variante vide ("::") = pas d'ablation.
# Pour les pseudos all-* on restreint plus tard aux sub-configs `*-full`.
declare -a ABL_VARIANTS=()
RESTRICT_TO_FULL=0   # vrai pour all-standard / all-other-standard

# Helper : mappe une ablation simple → ses flags CLI.
# NB : argparse prend la dernière occurrence d'un flag à choix → on peut
# overrider un flag déjà fixé dans BASE_CONFIGS en le repassant ici
# (utilisé pour `legacy-coexpr` qui override --coexpr-mode differential).
_abl_flags() {
    case "$1" in
        no-coexpr)          echo "--no-coexpr" ;;
        no-humess)          echo "--no-humess" ;;
        no-reactome)        echo "--no-reactome" ;;
        no-ppi)             echo "--no-ppi" ;;
        no-scenic-regulons) echo "--no-scenic-regulons" ;;
        no-cgrp)            echo "--no-cell-group-edges" ;;
        ex-degrees)         echo "--exclude-features ppi_degree,reg_degree" ;;
        # V5-spécifiques (override de flags base via argparse last-wins)
        diff-coexpr)        echo "--coexpr-mode differential" ;;
        legacy-coexpr)      echo "--coexpr-mode p16_only" ;;
        dedup-ppi-remove)   echo "--dedup-ppi-signed remove" ;;
        dedup-ppi-annotate) echo "--dedup-ppi-signed annotate" ;;
        no-rfi)             echo "--no-reactome-fi" ;;
        # Toggles V5 isolés (utiles pour ablations partielles vs v5-full)
        no-signed-message)  echo "" ;;   # placeholder : il faut retirer --signed-message du base (cf. §V5)
        no-signed-decoder)  echo "" ;;
        "")                 echo "" ;;
        *) echo "__INVALID__" ;;
    esac
}

case "$ABLATIONS" in
    ""|none)
        ABL_VARIANTS=( "::" )
        ;;
    all-standard)
        RESTRICT_TO_FULL=1
        for abl in no-coexpr no-humess no-reactome no-ppi no-scenic-regulons no-cgrp ex-degrees; do
            ABL_VARIANTS+=( "${abl}::$(_abl_flags "$abl")" )
        done
        ;;
    all-other-standard)
        RESTRICT_TO_FULL=1
        for abl in no-reactome no-ppi no-scenic-regulons no-cgrp ex-degrees; do
            ABL_VARIANTS+=( "${abl}::$(_abl_flags "$abl")" )
        done
        ;;
    v5-recommended)
        # Grille recommandée V5 (à appliquer sur --version V5) :
        # 5 ablations × 1 sous-config (v5-full) = 5 configs.
        # RESTRICT_TO_FULL=1 : ablations appliquées uniquement sur v5-full
        # (= signed-message + signed-decoder + V4.1-full backbone).
        # Pour isoler l'effet sm/sd pur (sans ablation), utiliser la
        # validation factorielle 2×2 : `--version V5` sans --ablations.
        # Base V5 = p16_only legacy ; `diff-coexpr` override → differential
        # (testable une fois V4.2 validé cross-seed).
        RESTRICT_TO_FULL=1
        for abl in no-coexpr diff-coexpr no-humess no-coexpr+no-humess dedup-ppi-remove; do
            tag="$abl"
            flags=""
            IFS='+' read -ra _COMPS <<< "$abl"
            for c in "${_COMPS[@]}"; do
                f="$(_abl_flags "$c")"
                flags="${flags}${flags:+ }${f}"
            done
            ABL_VARIANTS+=( "${tag}::${flags}" )
        done
        ;;
    *)
        # Custom : split par ',' → variantes séparées.
        # Chaque variante peut elle-même être composite via '+'.
        IFS=',' read -ra _ABL_LIST <<< "$ABLATIONS"
        for v in "${_ABL_LIST[@]}"; do
            tag="$v"          # ex : "no-coexpr+no-humess"
            flags=""
            IFS='+' read -ra _COMPS <<< "$v"
            for c in "${_COMPS[@]}"; do
                f="$(_abl_flags "$c")"
                if [[ "$f" == "__INVALID__" ]]; then
                    echo "Ablation inconnue : '$c' (dans '$v')."
                    echo "Standards : no-coexpr no-humess no-reactome no-ppi no-scenic-regulons no-cgrp ex-degrees"
                    echo "V5        : legacy-coexpr dedup-ppi-remove dedup-ppi-annotate"
                    echo "Pseudos   : all-standard all-other-standard v5-recommended"
                    exit 1
                fi
                flags="${flags}${flags:+ }${f}"
            done
            ABL_VARIANTS+=( "${tag}::${flags}" )
        done
        ;;
esac

# --- Produit cartésien BASE × ABL_VARIANTS → CONFIGS finales ---------------
CONFIGS=()
for base in "${BASE_CONFIGS[@]}"; do
    base_tag="${base%%::*}"
    base_flags="${base#*::}"
    for av in "${ABL_VARIANTS[@]}"; do
        abl_tag="${av%%::*}"
        abl_flags="${av#*::}"
        # Restriction all-* : seules les configs *-full passent
        if [[ $RESTRICT_TO_FULL -eq 1 && "$base_tag" != *"-full" ]]; then
            continue
        fi
        # Compose tag final ; on saute le "+" si pas d'ablation
        if [[ -n "$abl_tag" ]]; then
            tag="${base_tag}+${abl_tag}"
        else
            tag="${base_tag}"
        fi
        # Compose flags (squeeze whitespace)
        flags="${base_flags} ${abl_flags}"
        flags="$(echo $flags | xargs -n1 | xargs)"   # trim multiple spaces
        CONFIGS+=( "${tag}::${flags}" )
    done
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "[grid] aucune config produite. Vérifie --version et --ablations."
    exit 1
fi

# --- Préparation logs + configs.tsv ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# DATA_ROOT : racine des données d'ENTRÉE. DOIT correspondre à ce que
# gnn_vgae.py lit réellement (cf. gnn_vgae.py:381-382, BASE_DIR/data sur
# LAB-DATA). PROJECT_DIR est sur /scratch (code), mais les données
# (omnipath, pyscenic, reactome_fi) sont sur /LAB-DATA → le garde-fou
# DOIT vérifier LAB-DATA, pas $PROJECT_DIR/data. Override : --data-root
# ou env GNN_DATA_ROOT.
# Précédence : --data-root > env GNN_DATA_ROOT > défaut (= gnn_vgae.py:381)
DATA_ROOT="${DATA_ROOT_CLI:-${GNN_DATA_ROOT:-/LAB-DATA/GLiCID/users/${GNN_CLUSTER_USER:-$USER}/gnn/data}}"
# OUT_DIR_BASE des jobs : /scratch (écriture rapide + WRITABLE). Le code est
# déployé à plat (src/gnn_vgae.py) → _REPO_ROOT=dirname³ de gnn_vgae remonte un
# cran trop haut (/scratch/nautilus/users, non writable) → PermissionError l.651.
# On force donc GNN_OUT_DIR_BASE + GNN_DATA_DIR dans le sbatch (cf. heredoc).
OUT_BASE_JOB="${GNN_OUT_DIR_BASE:-/scratch/nautilus/users/${GNN_CLUSTER_USER:-$USER}/gnn_vgae}"
TS="$(date +%Y%m%d_%H%M%S)"
# Slug court pour identifier le run dans le LOG_DIR
ABL_SLUG="${ABLATIONS//,/_}"
ABL_SLUG="${ABL_SLUG//+/-}"
[[ -z "$ABL_SLUG" ]] && ABL_SLUG="validation"
LOG_DIR="$SCRIPT_DIR/logs/ablation_${VERSION}_${ABL_SLUG}_${TS}"
mkdir -p "$LOG_DIR"
CONFIGS_FILE="$LOG_DIR/configs.tsv"

> "$CONFIGS_FILE"
for cfg in "${CONFIGS[@]}"; do
    tag="${cfg%%::*}"
    flags="${cfg#*::}"
    for seed in "${SEEDS[@]}"; do
        printf '%s\t%s\t%s\n' "$tag" "$seed" "$flags" >> "$CONFIGS_FILE"
    done
done

N_TASKS=$(wc -l < "$CONFIGS_FILE")
ARRAY_RANGE="0-$((N_TASKS - 1))%${MAX_PARALLEL}"

echo "[grid] version   : $VERSION"
echo "[grid] ablations : ${ABLATIONS:-<none>}"
echo "[grid] project   : $PROJECT_DIR"
echo "[grid] log dir   : $LOG_DIR"
echo "[grid] configs   : $CONFIGS_FILE (${#CONFIGS[@]} configs × ${#SEEDS[@]} seeds = $N_TASKS tâches)"
echo "[grid] array     : $ARRAY_RANGE   (max $MAX_PARALLEL en //)"
echo "[grid] time      : $TIME_LIMIT par tâche"
echo "[grid] seeds     : ${SEEDS[*]}"
echo

# --- Sécurité : prérequis vérifiés sur DATA_ROOT (= ce que gnn_vgae.py
#     lit réellement, LAB-DATA), PAS $PROJECT_DIR/data (scratch, code) ---
echo "[grid] data root : $DATA_ROOT  (doit = gnn_vgae.py BASE_DIR/data)"
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[warn] DATA_ROOT introuvable : $DATA_ROOT"
    echo "       Vérifie qu'il correspond à gnn_vgae.py:381 (LAB_DIR/gnn/data)."
    echo "       Override : --data-root <path> ou export GNN_DATA_ROOT=<path>"
fi
if [[ "$VERSION" == "V4" || "$VERSION" == "V4.1" || "$VERSION" == "V4.2" || "$VERSION" == "V4.2-sweep" || "$VERSION" == "V5" || "$VERSION" == "V5.3" || "$VERSION" == "V5.3-finalist" || "$VERSION" == "V5.3-finalist-long" || "$VERSION" == "V5.3-coexpr-source" || "$VERSION" == "V5.4-ablation" ]]; then
    CACHE_TF="$DATA_ROOT/omnipath/tf_collectri.tsv.gz"
    CACHE_SIG="$DATA_ROOT/omnipath/signed_ppi_signor.tsv.gz"
    if [[ ! -f "$CACHE_TF" ]]; then
        echo "[warn] cache OmniPath TF absent : $CACHE_TF"
        echo "       lance d'abord : python scripts/cache_omnipath.py --cache-dir $DATA_ROOT/omnipath"
    fi
    if [[ ! -f "$CACHE_SIG" ]]; then
        echo "[warn] cache OmniPath SIGNOR absent : $CACHE_SIG"
        echo "       (les arêtes signaling seront vides — runs +sig dégradés)"
    fi
fi

# --- Sécurité V4.2 : prérequis coexpr différentiel + Reactome FI -----------
# V5 base utilise coexpr p16_only legacy (pas de prérequis spécifique).
# Si --ablations diff-coexpr passé à V5, coexpr_diff.tsv est requis — mais
# c'est l'utilisateur qui choisit, garde-fou softé en warning.
if [[ "$VERSION" == "V4.2" ]]; then
    DIFF_FILE="$DATA_ROOT/pyscenic/diff_coexpr/coexpr_diff.tsv"
    RFI_FILE="$DATA_ROOT/reactome_fi/FIsInGene_with_annotations.txt"
    if [[ ! -f "$DIFF_FILE" ]]; then
        echo "[err] coexpr_diff.tsv absent : $DIFF_FILE"
        echo "      Lance d'abord (workflow en 2 étapes découplées) :"
        echo "        1. python src/data/preprocess/build_diff_coexpr.py extract-matrices \\"
        echo "             --gene-universe graph --graph-genes <cross_seed_gene_ranking.tsv>"
        echo "        2. bash scripts/run_diff_coexpr.sh --step grnboost2"
        echo "        3. (attendre) bash scripts/run_diff_coexpr.sh --step merge"
        exit 1
    fi
    if [[ ! -f "$RFI_FILE" ]]; then
        echo "[warn] Reactome FI absent : $RFI_FILE"
        echo "       Les configs *.rfi auront un edge_type reactome_fi vide."
        echo "       Lance : bash scripts/cache_reactome_fi.sh $DATA_ROOT/reactome_fi"
    fi
fi

# --- Sécurité V5 : warning souple si --ablations contient diff-coexpr ------
if [[ ( "$VERSION" == "V5" || "$VERSION" == "V5.3" ) && "$ABLATIONS" == *"diff-coexpr"* ]]; then
    DIFF_FILE="$DATA_ROOT/pyscenic/diff_coexpr/coexpr_diff.tsv"
    if [[ ! -f "$DIFF_FILE" ]]; then
        echo "[warn] V5 + ablation diff-coexpr demandée mais $DIFF_FILE absent."
        echo "       Les configs *+diff-coexpr échoueront à l'exécution."
        echo "       Pour les configs sans diff-coexpr (base p16_only), tout va bien."
    fi
fi

# --- Sécurité V4.2-sweep : 3 fichiers coexpr_diff.{k5q5,k3q7,k2q8}.tsv ----
if [[ "$VERSION" == "V4.2-sweep" ]]; then
    DD="$DATA_ROOT/pyscenic/diff_coexpr"
    missing=0
    for v in k5q5 k3q7 k2q8; do
        F="$DD/coexpr_diff.${v}.tsv"
        if [[ ! -f "$F" ]]; then
            echo "[err] coexpr_diff.${v}.tsv absent : $F"
            missing=1
        fi
    done
    if [[ $missing -eq 1 ]]; then
        echo "      Lance d'abord : bash scripts/run_coexpr_prune_sweep.sh"
        exit 1
    fi
fi

# --- Génération sbatch -----------------------------------------------------
SBATCH_SCRIPT="$LOG_DIR/sbatch_ablation.sh"
cat > "$SBATCH_SCRIPT" <<EOF
#!${BASH_BIN}
#SBATCH --job-name=vgae_${VERSION}_${ABL_SLUG}
#SBATCH --comment="VGAE ablation ${VERSION} ${ABL_SLUG}"
#SBATCH --output=$LOG_DIR/%x_%A_%a.out
#SBATCH --error=$LOG_DIR/%x_%A_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=quick
#SBATCH --array=$ARRAY_RANGE

set -euo pipefail

# GLiCID/Guix : PATH minimal sur le nœud → bake le PATH du frontal (python3 etc.)
export PATH="$PATH"

cd "$PROJECT_DIR"

# --- Chemins portables : le code est déployé À PLAT (src/gnn_vgae.py) sur
# scratch → _REPO_ROOT=dirname³ de gnn_vgae.py est FAUX (remonte à
# /scratch/nautilus/users, non writable). On force donc les chemins : données
# d'entrée sur LAB-DATA (DATA_ROOT), sorties sur scratch (writable).
export GNN_DATA_DIR="$DATA_ROOT"
export GNN_BASE_DIR="$(dirname "$DATA_ROOT")"
export GNN_SCENIC_DIR="$(dirname "$DATA_ROOT")/output/pyscenic"
export GNN_HUMESS_DIR="$(dirname "$(dirname "$DATA_ROOT")")/humess/output_huvec"
export GNN_OUT_DIR_BASE="$OUT_BASE_JOB"

# Lecture ligne + split TSV via builtins bash (pas de sed/cut sur le nœud).
mapfile -t _CFG_LINES < "$CONFIGS_FILE"
LINE="\${_CFG_LINES[\$SLURM_ARRAY_TASK_ID]}"
IFS=\$'\t' read -r TAG SEED FLAGS <<< "\$LINE"
RUN_TAG="\${TAG}.s\${SEED}"

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID : tag=\$TAG seed=\$SEED run_tag=\$RUN_TAG"
echo "[\$(date +%T)] flags: \$FLAGS"
echo "[\$(date +%T)] node : \$(hostname)"

# shellcheck disable=SC2086
# NB cluster Nautilus : tous les .py sont déployés à plat sous src/
# (pas de sous-dossiers gnn/, perturbation/, validation/ comme en local).
python3 src/gnn_vgae.py \\
    --run-tag "\$RUN_TAG" \\
    --seed "\$SEED" \\
    --reuse-graph \\
    --graph-cache "\$GNN_OUT_DIR_BASE/_graph_cache_\${TAG}.pkl" \\
    \$FLAGS

echo "[\$(date +%T)] task \$SLURM_ARRAY_TASK_ID terminée."
EOF

chmod +x "$SBATCH_SCRIPT"
echo "[grid] sbatch script : $SBATCH_SCRIPT"
echo

# --- Soumission ------------------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] sbatch $SBATCH_SCRIPT"
    echo
    echo "[DRY-RUN] Aperçu du configs.tsv :"
    cat "$CONFIGS_FILE"
    echo
    echo "[DRY-RUN] Aperçu du sbatch script (head) :"
    head -25 "$SBATCH_SCRIPT"
    exit 0
fi

JOB_ID=$(sbatch --parsable "$SBATCH_SCRIPT")
echo "[grid] soumis : job array $JOB_ID ($N_TASKS tâches, ${MAX_PARALLEL} max en //)"
echo "[grid] suivi   : squeue -j $JOB_ID    (ou squeue -u \$USER)"
echo "[grid] cancel  : scancel $JOB_ID"
echo "[grid] logs    : $LOG_DIR/vgae_${VERSION}_${ABL_SLUG}_${JOB_ID}_<task>.{out,err}"
