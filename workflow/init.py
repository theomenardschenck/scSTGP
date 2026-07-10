#!/usr/bin/env python3
"""
init.py — Assistant interactif de configuration du pipeline VGAE.

Pose quelques questions (type de données, groupes A/B, chemins, backend,
seeds, perturbation…) et génère un fichier `workflow/config/config.<nom>.yaml`
prêt à l'emploi pour `workflow/run.sh`.

    python workflow/init.py                 # interactif
    python workflow/init.py --preset quick  # démarre sur le préréglage 'quick'

Ce script n'a AUCUNE dépendance (stdlib pure) : il peut tourner avant même
d'avoir activé l'environnement conda.
"""
from __future__ import annotations

import argparse
import os
import sys

# ── Préréglages : valeurs par défaut selon l'intention ──────────────────────
PRESETS = {
    # test fonctionnel rapide (câblage) : 1 seed, peu d'epochs, cibles, graphe minimal
    "quick": dict(seeds=1, epochs=20, patience=20, modes=["knockout"],
                  perturb="targeted", graph="minimal", ora_top_n=50,
                  cluster_annotation=False),
    # run scientifique complet : 3 seeds, entraînement long, tous modes, tous gènes
    "full":  dict(seeds=3, epochs=1200, patience=150,
                  modes=["knockout", "knockdown", "overexpress"],
                  perturb="total", graph="signed", ora_top_n=100,
                  cluster_annotation=True),
}

GRAPH_FLAGS = {
    "minimal": "",  # PPI + Reactome + coexpr + SCENIC + HuMess (sources par défaut)
    "signed":  ("--use-omnipath-signaling --use-omnipath-tf-curated "
                "--include-omnipath-genes --use-reactome-fi "
                "--signed-message --signed-decoder --decoder-split "
                "--kl-beta-max 0.0001"),
}

# Ablations : source à DÉSACTIVER → flag(s) --no-*. Ajoutés APRÈS les flags graphe
# (argparse : le dernier gagne) → désactivent même une source activée par 'signed'.
ABLATIONS = {
    "ppi":         "--no-ppi",
    "reactome":    "--no-reactome",
    "coexpr":      "--no-coexpr",
    "scenic":      "--no-scenic-regulons",
    "humess":      "--no-humess",
    "cell-groups": "--no-cell-group-edges",
    "omnipath":    "--no-omnipath-signaling --no-omnipath-tf-curated",
    "reactome-fi": "--no-reactome-fi",
}


# ── Helpers de saisie ───────────────────────────────────────────────────────
def ask(prompt, default=None, choices=None):
    """Question texte avec défaut et choix optionnels."""
    suffix = ""
    if choices:
        suffix += f" [{'/'.join(choices)}]"
    if default is not None and default != "":
        suffix += f" (défaut: {default})"
    while True:
        ans = input(f"  {prompt}{suffix} : ").strip()
        if not ans:
            ans = "" if default is None else str(default)
        if choices and ans not in choices:
            print(f"    → réponds parmi : {', '.join(choices)}")
            continue
        return ans


def ask_yesno(prompt, default=True):
    d = "o" if default else "n"
    ans = ask(prompt, default=d, choices=["o", "n"])
    return ans == "o"


def ask_int(prompt, default):
    while True:
        ans = ask(prompt, default=default)
        try:
            return int(ans)
        except ValueError:
            print("    → entier attendu")


def ask_path(prompt, default=None, must_exist=False, repo=None):
    """Chemin ; avertit (sans bloquer) si absent. Résolu relativement au repo."""
    ans = ask(prompt, default=default)
    if ans and must_exist:
        check = ans if os.path.isabs(ans) else os.path.join(repo or ".", ans)
        if not os.path.exists(check):
            print(f"    ⚠️  introuvable pour l'instant : {check} "
                  "(ok si tu le stages plus tard / sur le cluster)")
    return ans


def yaml_list(items):
    return "[" + ", ".join(str(i) for i in items) + "]"


# ── Corps du wizard ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Assistant de config du pipeline VGAE.")
    ap.add_argument("--preset", choices=list(PRESETS), default=None,
                    help="démarre sur un préréglage (quick|full)")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du dépôt

    print("\n" + "=" * 64)
    print("  Assistant de configuration — pipeline VGAE (priorisation gènes)")
    print("=" * 64)
    print("  Entrée vide = valeur par défaut entre parenthèses.\n")

    # 0. Préréglage --------------------------------------------------------
    preset_name = args.preset or ask(
        "Préréglage de départ", default="quick", choices=["quick", "full", "custom"])
    P = dict(PRESETS.get(preset_name, PRESETS["full"]))  # 'custom' part de 'full'

    # 1. Identité du run ---------------------------------------------------
    name = ask("Nom du run (sert au run_tag et au dossier de sortie)", default="myrun")

    # 2. Type de données ---------------------------------------------------
    print("\n— Données —")
    print("    'sc'   = scRNA façon HUVEC (matrice + metadata cellules) ;")
    print("    'bulk' = RNA-seq bulk (matrice gènes × échantillons + samplesheet).")
    rna = ask("Type de données", default="bulk", choices=["bulk", "sc"])

    # 3. Groupes A / B (à la discrétion de l'utilisateur) ------------------
    print("\n— Contraste A vs B (référence vs condition) —")
    grpA = ask("Groupe A / référence (ex. pro, sain, control, WT)", default="pro")
    grpB = ask("Groupe B / condition  (ex. sen, malade, patient, mutant)", default="sen")

    # 4. Chemins -----------------------------------------------------------
    print("\n— Chemins (relatifs au dépôt ou absolus) —")
    data_root  = ask_path("Racine des données (data_root)", default="data",
                          must_exist=True, repo=repo)
    scenic_dir = ask_path("Sorties pySCENIC (regulons/adjacencies)",
                          default="output/pyscenic", repo=repo)
    humess_dir = ask_path("Sorties HuMess (importance métabolique)",
                          default=f"{data_root}/humess/{name}", repo=repo)
    # Auto-détection du fichier DE : cherche DEGs_<A>_vs_<B>*.csv et propose le
    # meilleur (préférence _MAST > _Wald > brut), en listant les alternatives.
    import glob as _glob
    _de_dir = os.path.join(repo, data_root, "gnn_data")
    _de_cands = sorted(_glob.glob(os.path.join(_de_dir, f"DEGs_{grpA}_vs_{grpB}*.csv")))
    _de_default = f"{data_root}/gnn_data/DEGs_{grpA}_vs_{grpB}.csv"
    if _de_cands:
        _rank = lambda p: (0 if "_MAST" in p else 1 if "_Wald" in p else 2, p)
        _best = sorted(_de_cands, key=_rank)[0]
        _de_default = os.path.relpath(_best, repo)
        print("    DE trouvés : " + ", ".join(os.path.basename(c) for c in _de_cands))
        print(f"    → proposé : {os.path.basename(_best)} (préférence MAST/Wald)")
    de_csv     = ask_path("CSV de DE (logFC/pvalue A vs B, pour le readout)",
                          default=_de_default, repo=repo)
    coexpr     = ask_path("Fichier de co-expression (coexpr_diff.tsv)",
                          default=f"{data_root}/pyscenic/diff_coexpr/coexpr_diff.tsv", repo=repo)

    build_block_matrix = build_block_meta = ""
    build_enabled = False
    if rna == "bulk":
        print("\n— Bulk : matrice + samplesheet —")
        build_block_matrix = ask_path("Matrice (gènes × échantillons)",
                                      default=f"{data_root}/bulk/{name}/expr_all.csv",
                                      repo=repo)
        build_block_meta = ask_path("Samplesheet (échantillon → groupe A/B)",
                                    default=f"{data_root}/bulk/{name}/samplesheet.tsv",
                                    repo=repo)
        # start-from-assembly : si coexpr/HuMess existent déjà, le build est sauté.
        build_enabled = True

    # 5. Exécution ---------------------------------------------------------
    print("\n— Exécution —")
    backend = ask("Backend", default="local", choices=["local", "cluster"])
    device  = "auto" if backend == "cluster" else "cpu"
    py      = "python3" if backend == "cluster" else "python"

    # 6. Seeds -------------------------------------------------------------
    print("\n— Robustesse —")
    n_seeds = ask_int("Nombre de seeds (runs répétés)", default=P["seeds"])
    seeds = list(range(1, n_seeds + 1))

    # 7. Perturbation ------------------------------------------------------
    print("\n— Perturbation in silico —")
    modes_ans = ask("Modes (séparés par des espaces : knockout knockdown overexpress)",
                    default=" ".join(P["modes"]))
    modes = [m for m in modes_ans.split() if m]
    scope = ask("Portée", default=P["perturb"], choices=["targeted", "total"])
    genes_file = ""
    if scope == "targeted":
        genes_file = ask_path("Fichier de cibles (1 gène/ligne, # = commentaire)",
                              default=f"{data_root}/gene_sets/priority_drivers_targets.txt",
                              repo=repo)

    # 8. Sources du graphe -------------------------------------------------
    print("\n— Graphe & entraînement —")
    print("    'minimal' = PPI+Reactome+coexpr+SCENIC+HuMess ;")
    print("    'signed'  = + OmniPath (signé) + Reactome FI + décodeur signé.")
    graph = ask("Configuration du graphe", default=P["graph"], choices=["minimal", "signed"])
    epochs = ask_int("Epochs d'entraînement", default=P["epochs"])
    patience = P["patience"]

    # 8b. Ablations (optionnel) : désactiver des sources pour ce run -------
    print("\n— Ablations (optionnel) — sources du graphe à DÉSACTIVER —")
    print("    Choix : " + "  ".join(ABLATIONS))
    print("    (ex. 'humess coexpr' pour un run sans HuMess ni co-expression)")
    abl_ans = ask("Ablations (séparées par espaces, vide = aucune)", default="")
    ablations = [a for a in abl_ans.split() if a]
    _bad = [a for a in ablations if a not in ABLATIONS]
    if _bad:
        print(f"    ⚠️  ignorées (inconnues) : {', '.join(_bad)}")
    ablations = [a for a in ablations if a in ABLATIONS]
    if ablations:
        print(f"    → désactivé : {', '.join(ablations)}  "
              "(pense à nommer le run en conséquence, ex. <nom>-no-humess)")

    # 8c. Axe primaire du driver_score -----------------------------------
    print("\n— Axe primaire (ancrage du driver_score) —")
    print("    phenotypic = contraste A→B des cell_groups (défaut, comme V5.4.1) ;")
    print("    de         = axe DE-ancré (pôles = top-N up/down d'un fichier DE) ;")
    print("    effector   = axe MANUEL, ancré sur tes listes de gènes pro/anti.")
    axis = ask("Axe", default="phenotypic", choices=["phenotypic", "de", "effector"])
    de_axis_file = de_axis_rank = effector_pro = effector_anti = ""
    if axis == "de":
        de_axis_file = ask_path("  Fichier DE pour l'axe", default=de_csv, repo=repo)
        de_axis_rank = ask("  Ranking des ancres", default="stat", choices=["stat", "log_fc"])
    elif axis == "effector":
        effector_pro  = ask_path("  Gènes pôle PRO-sénescence (1/ligne)", default="", repo=repo)
        effector_anti = ask_path("  Gènes pôle ANTI-sénescence (1/ligne)", default="", repo=repo)

    # 8c-bis. Axe POLYCENTRIQUE (centroïdes de cluster ancrés) -------------
    #   Redéfinit les centroïdes P16_cluster_k → change l'axe global + active
    #   les transitions inter-cluster. Réservé au sc (les clusters n'existent
    #   que pour les cell_groups P16_cluster_0..3).
    cluster_anchor_mode = "none"
    cluster_anchors_file = ""
    transition_axes = "none"
    if rna == "sc":
        print("\n— Axe polycentrique / readout multi-état (optionnel, sc) —")
        print("    none       = centroïdes par cell_group bruts (défaut) ;")
        print("    de-markers = centroïdes = top-N marqueurs DE par cluster (DEGs_P16_cluster_k.csv) ;")
        print("    manual     = centroïdes ancrés sur un TSV (group, gene, weight) — ancres Ahn 2025.")
        cluster_anchor_mode = ask("Ancrage des centroïdes de cluster", default="none",
                                  choices=["none", "de-markers", "manual"])
        if cluster_anchor_mode == "manual":
            _def_anchor = os.path.join("gnn_data", "ahn_cluster_anchors.tsv")
            if not os.path.exists(os.path.join(repo, "data", _def_anchor)):
                _def_anchor = ""
            cluster_anchors_file = ask(
                "  Fichier d'ancres TSV (relatif à data_root, ou absolu)",
                default=_def_anchor)
        if cluster_anchor_mode != "none":
            transition_axes = ask(
                "  Axes de transition inter-cluster", default="default",
                choices=["none", "default", "all-pairs"])

    # 8d. Intégrer le logFC/DE comme FEATURE de nœud (--de-features) -------
    print("\n— logFC dans le graphe (features de nœud) —")
    print("    ⚠️  CIRCULARITÉ : injecter le DE (logFC/pvalue) comme feature de nœud")
    print("        rend l'encodeur informé de la cible → fuite si l'axe est DE-ancré.")
    print("        Ton historique : « logFC JAMAIS en feature d'encoder (anti-circularité) ».")
    print("        Conseil : si tu l'actives, évalue avec un axe EFFECTEUR-ancré (pas 'de').")
    de_features = ask_yesno("Intégrer le logFC/DE dans le graphe (--de-features) ?", default=False)
    if de_features and axis == "de":
        print("    ⚠️⚠️  de_features + axe 'de' = double circularité — fortement déconseillé.")

    # 9. Validation --------------------------------------------------------
    ora_top_n = ask_int("ORA : top-N drivers", default=P["ora_top_n"])
    cluster_annotation = P["cluster_annotation"] and n_seeds >= 2  # cross-seed → ≥2 seeds

    # 9b. Decoy de base (deux nulls de sanity) -----------------------------
    print("\n— Decoy de base (nulls de contrôle) —")
    print("    (1) decoy d'ARÊTES : rewire des connexions des top-N drivers")
    print("        (même nombre d'arêtes/type, partenaires+signes aléatoires) ;")
    print("    (2) decoy d'AXE : N axes aléatoires vs l'axe réel (spécificité).")
    print("    → montre que le signal vient du graphe/axe réels, pas d'un artefact.")
    decoy_on = ask_yesno("Activer le decoy de base ?", default=True)
    decoy_top_n = 40
    random_axis = 0
    if decoy_on:
        decoy_top_n = ask_int("  decoy d'arêtes : top-N drivers testés", default=40)
        random_axis = ask_int("  decoy d'axe : nombre d'axes aléatoires (0 = off)",
                              default=20)
        if random_axis > 0 and axis == "phenotypic":
            print("    (axe phénotypique par défaut → la nulle compare à cet axe.)")

    deterministic = ask_yesno(
        "Mode déterministe (bit-exact à seed fixe ; ⚠️ threads=1 → plus LENT)", default=False)

    # ── Base des flags (SANS ablations) ───────────────────────────────────
    base_flags = f"--n-epochs {epochs} --patience {patience}"
    if GRAPH_FLAGS[graph]:
        base_flags += " " + GRAPH_FLAGS[graph]
    if de_features:
        base_flags += " --de-features"

    # ── Mode d'ablation : comment décliner les ablations en configs ───────
    abl_mode = "single"
    if ablations:
        print("\n— Étude d'ablation —")
        print("    single      = 1 config, toutes les ablations choisies ENSEMBLE ;")
        print("    independent = 1 config par ablation (chacune ISOLÉE) ;")
        print("    progressive = ablations CUMULÉES (baseline → +1 → +2 …).")
        print("    → une config RÉFÉRENCE sans ablation est toujours générée.")
        abl_mode = ask("Mode d'ablation", default="single",
                       choices=["single", "independent", "progressive"])

    # variants = [(suffixe_de_run_tag, [clés d'ablation]), …]  ('' = pas d'ablation)
    if not ablations:
        variants = [("", [])]
    else:
        variants = [("_baseline", [])]  # référence sans ablation (comparateur)
        if abl_mode == "single":
            variants.append(("_" + "-".join("no-" + a for a in ablations), ablations))
        elif abl_mode == "independent":
            variants += [("_no-" + a, [a]) for a in ablations]
        elif abl_mode == "progressive":
            for i in range(1, len(ablations) + 1):
                variants.append(("_" + "-".join("no-" + a for a in ablations[:i]),
                                 ablations[:i]))

    out_base = ask("\n— Sortie —\n  Dossier de sortie racine (out_base ; env GNN_OUT_DIR_BASE prime)",
                   default=f"output/{name}")

    # ── Écriture du/des config(s) : une par variante d'ablation ───────────
    written = []
    for _suffix, _abl in variants:
        _tag = name + _suffix
        _flags = base_flags + "".join(" " + ABLATIONS[a] for a in _abl)
        _out = (out_base + _suffix) if _suffix else out_base
        _abldesc = "référence (sans ablation)" if not _abl else "sans " + ", ".join(_abl)
        cfg = f"""# ──────────────────────────────────────────────────────────────────
# config.{_tag}.yaml — généré par workflow/init.py (préréglage: {preset_name})
# Ablation : {_abldesc}
# Lancement :  bash workflow/run.sh --configfile workflow/config/config.{_tag}.yaml
# ──────────────────────────────────────────────────────────────────
run:
  name: "{_tag}"
  description: "VGAE {rna} — {grpA} vs {grpB} ({backend}, {n_seeds} seed(s)) — {_abldesc}"

paths:
  data_root: "{data_root}"
  out_base: "{_out}"
  humess_dir: "{humess_dir}"
  scenic_dir: "{scenic_dir}"
  de_magnitude_csv: "{de_csv}"
  coexpr_file: "{coexpr}"

compute:
  device: "{device}"        # cpu | cuda | auto
  python: "{py}"
  python_torch: "{py}"
  backend: "{backend}"      # local | cluster (wrapper run.sh)
  deterministic: {str(deterministic).lower()}   # bit-exact à seed fixe (threads=1 → plus lent)

input:
  rna_type: {rna}
  degs_path: "{de_csv}"

# Stage 0 : build features data-dérivées (coexpr + HuMess) depuis une matrice.
# enabled=true (bulk) → orchestré ; si coexpr/HuMess existent déjà, sauté.
build:
  enabled: {str(build_enabled).lower()}
  matrix: "{build_block_matrix}"
  metadata: "{build_block_meta}"
  gene_col: "Tracking_id"
  group_col: 2
  young_group: "{grpA}"
  sen_group: "{grpB}"
  conditions: [{grpA!r}, {grpB!r}]
  humess_repo: "../humess"
  humess_cs_nsamples: 1000

models:
  vgae:
    enabled: true
    run_tag: "{_tag}"
    seeds: {yaml_list(seeds)}
    extra_flags: "{_flags}"
  gnn_lite:
    enabled: false

perturbation:
  enabled: true
  modes: [{", ".join(modes)}]
  scope: all_genes
  axis_tag: ""
  out_suffix: ""
  genes_file: "{genes_file}"   # "" = tous les gènes ; sinon sous-ensemble (cibles)
  # Axe primaire du driver_score : phenotypic (A→B) | de (DE-ancré) | effector (manuel)
  axis: "{axis}"
  de_axis_file: "{de_axis_file}"
  de_axis_label: "{grpB}_vs_{grpA}"
  de_axis_rank: "{de_axis_rank or 'stat'}"
  effector_pro: "{effector_pro}"
  effector_anti: "{effector_anti}"
  # Readout multi-état / axe polycentrique (sc). none = comportement historique.
  transition_axes: {transition_axes}          # none | default | all-pairs
  cluster_anchor_mode: {cluster_anchor_mode}     # none | de-markers | manual (ancres Ahn)
  cluster_anchors_file: "{cluster_anchors_file}"  # TSV (group,gene,weight) si mode=manual ; "" = défaut
  cache_delta_z: true         # défaut ON : persiste le cache Δz (re-projection d'axes)

scoring:
  de_significance: "pvalue"
  de_padj_max: 0.05
  de_abs_lfc_min: 0.5

validation:
  ora_top_n: {ora_top_n}
  decoy:
    enabled: {str(decoy_on).lower()}        # decoy d'arêtes (rewire top-N drivers)
    top_n: {decoy_top_n}
    random_axis: {random_axis}         # decoy d'axe : N axes aléatoires (0 = off)
    random_axis_seed: 0
  cluster_annotation:
    enabled: {str(cluster_annotation).lower()}   # nécessite ≥2 seeds

comparison:
  enabled: false

output:
  generate_html_report: false
"""
        out_path = os.path.join(repo, "workflow", "config", f"config.{_tag}.yaml")
        if os.path.exists(out_path) and not ask_yesno(
                f"\n{out_path} existe déjà — écraser ?", default=False):
            print(f"  Ignoré : config.{_tag}.yaml")
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(cfg)
        written.append(os.path.relpath(out_path, repo))

    # ── Récap + prochaines étapes ─────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  ✅ {len(written)} config(s) écrite(s) :")
    for _r in written:
        print(f"       {_r}")
    print("=" * 64)
    if written:
        print("  Lancer (par config) :")
        print(f"    bash workflow/run.sh --configfile {written[0]} --dry-run   # vérifie le DAG")
        print(f"    bash workflow/run.sh --configfile {written[0]}")
        if len(written) > 1:
            print("    # étude d'ablation → répéter pour chaque config, ex. :")
            print(f"    for c in workflow/config/config.{name}*.yaml; do bash workflow/run.sh --configfile \"$c\"; done")
    if backend == "cluster":
        print("    (cluster) adapte la partition/QOS dans workflow/profiles/slurm/config.yaml")
        print("    (cluster) export GNN_OUT_DIR_BASE=/scratch/.../output pour écrire sur scratch")
    if rna == "bulk" and build_enabled:
        print("    ⚠️  bulk : si coexpr/HuMess ne sont pas précalculés, le stage build")
        print("        (HuMess) exige cplex en local — cf. commentaires du config.")
    print()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n  Interrompu.")
        sys.exit(1)
