#!/usr/bin/env python3
"""init.py — Wizard du PREPROCESS (data-adaptatif).

Situe les données single-nucleus/single-cell de l'utilisateur (tar GEO + RDS,
ou RDS directs) et décrit les groupes (condition/donneur/type cellulaire) →
écrit `workflow/preprocess/config.<dataset>.yaml`, consommé par le Snakefile
du preprocess (extract→export→coexpr→SCENIC→HuMess).

Usage :
    python workflow/preprocess/init.py
puis suivre les instructions de lancement affichées (local / cluster).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def ask(prompt, default=None, choices=None):
    suffix = f" [{'/'.join(choices)}]" if choices else ""
    if default not in (None, ""):
        suffix += f" (défaut: {default})"
    while True:
        ans = input(f"  {prompt}{suffix} : ").strip() or ("" if default is None else str(default))
        if choices and ans not in choices:
            print(f"    → réponds parmi : {', '.join(choices)}")
            continue
        return ans


def ask_path(prompt, default=None, must_exist=False):
    ans = ask(prompt, default=default)
    if ans and must_exist and not os.path.exists(ans):
        print(f"    ⚠️  introuvable ici : {ans} (ok si stagé plus tard / sur le cluster)")
    return ans


def main():
    print("=" * 68)
    print("  Wizard PREPROCESS — situer les données + décrire les groupes")
    print("=" * 68)

    name = ask("Nom du dataset (→ config + dossiers de sortie)", default="mydataset")

    print("\n  Source des données :")
    print("    'tar' = archive GEO *_RAW.tar contenant des .rds.gz (counts + metadata)")
    print("    'rds' = deux fichiers .rds déjà extraits (counts genes×cells + metadata)")
    src = ask("Type de source", default="tar", choices=["tar", "rds"])

    cfg = {"dataset": name}
    if src == "tar":
        cfg["tar"] = ask_path("Chemin du .tar GEO", default="data/signleNucleus/<GSE>/<GSE>_RAW.tar", must_exist=True)
        cfg["counts_member"] = ask("Nom du membre COUNTS dans le tar (genes×cells)", default="<GSM>_counts.rds.gz")
        cfg["meta_member"] = ask("Nom du membre METADATA dans le tar", default="<GSM>_metadata.rds.gz")
    else:
        # RDS directs : on les place tels quels dans le work_dir attendu par le Snakefile.
        cfg["_rds_counts"] = ask_path("Chemin du .rds COUNTS (genes×cells)", must_exist=True)
        cfg["_rds_meta"] = ask_path("Chemin du .rds METADATA", must_exist=True)
        print("    ℹ️  source 'rds' : copie/lie ces .rds dans <out_dir>/_rds/ avant de lancer")

    print("\n  Colonnes de la metadata (ouvre le .rds dans R pour vérifier si besoin) :")
    cfg["group_col"] = ask("Colonne CONDITION (ex. type, diagnosis)", default="type")
    cfg["donor_col"] = ask("Colonne DONNEUR (pseudobulk HuMess)", default="donor")
    ct = ask("Filtrer un TYPE cellulaire ? (colonne, vide=non)", default="")
    cfg["celltype_col"] = ct
    cfg["celltype"] = ask("  → valeur du type cellulaire à garder", default="") if ct else ""

    print("\n  Axe (baseline → advanced) = valeurs de la colonne CONDITION :")
    cfg["baseline"] = ask("Valeur BASELINE / référence (ex. CT, Control, pro)", default="CT")
    cfg["advanced"] = ask("Valeur ADVANCED / état avancé (ex. AD, sen, patient)", default="AD")

    cfg["subsample"] = ask("Cellules/condition pour coexpr/SCENIC", default="8000")
    cfg["min_frac"] = ask("Gène gardé si détecté ≥ fraction des cellules", default="0.05")

    out_dir = f"data/pyscenic/{name}"
    humess_dir = f"data/humess/{name}"

    # écriture YAML (chaînes simples, pas de dépendance pyyaml)
    q = '""'   # YAML chaîne vide (évite backslash dans f-string)
    ct_col_y = cfg["celltype_col"] or q
    ct_val_y = cfg["celltype"] or q
    lines = [
        f"# Preprocess {name} — généré par workflow/preprocess/init.py",
        f"dataset: {name}",
    ]
    if src == "tar":
        lines += [
            f"tar:           {cfg['tar']}",
            f"counts_member: {cfg['counts_member']}",
            f"meta_member:   {cfg['meta_member']}",
        ]
    lines += [
        f"out_dir:    {out_dir}",
        f"humess_dir: {humess_dir}",
        f"group_col:    {cfg['group_col']}",
        f"donor_col:    {cfg['donor_col']}",
        f"celltype_col: {ct_col_y}",
        f"celltype:     {ct_val_y}",
        f"baseline:     {cfg['baseline']}",
        f"advanced:     {cfg['advanced']}",
        f"subsample:    {cfg['subsample']}",
        f"min_frac:     {cfg['min_frac']}",
        "tf_list:        data/pyscenic/scenic_refs/allTFs_hg38.txt",
        "humess_repo:    ../humess",
        "humess_env:     humess",
        "humess_cs:      1000",
        "scenic_env:     arboreto",
        "scenic_workers: 8",
        "scenic_mem:     8GB",
        "py:             python",
    ]
    out_cfg = os.path.join(HERE, f"config.{name}.yaml")
    with open(out_cfg, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    D = out_dir
    print("\n" + "=" * 68)
    print(f"  ✅ Config écrite : {out_cfg}")
    print("=" * 68)
    print("\n  LANCEMENT (rappel du découpage HuMess=local / coexpr+SCENIC=cluster) :\n")
    print("  ── Tout en local (petites données + cplex) :")
    print(f"     snakemake -s workflow/preprocess/Snakefile --configfile {out_cfg} -j 8\n")
    print("  ── Split gros volume :")
    print("     # 1) LOCAL — HuMess seul (cplex) + entrées légères :")
    print(f"     snakemake -s workflow/preprocess/Snakefile --configfile {out_cfg} -j 4 \\")
    print(f"         {humess_dir}/models/{cfg['baseline']}/cs/cs_gene_to_importance_{cfg['baseline']}.tsv \\")
    print(f"         {humess_dir}/models/{cfg['advanced']}/cs/cs_gene_to_importance_{cfg['advanced']}.tsv")
    print("     # 2) CLUSTER — export+coexpr+SCENIC en jobs SLURM (sans HuMess) :")
    print(f"     snakemake -s workflow/preprocess/Snakefile --configfile {out_cfg} \\")
    print("         --profile workflow/profiles/slurm --jobs 20 --omit-from humess")
    print("\n  Prérequis cluster (une fois) : conda run -n arboreto pip install \"setuptools<81\" pyscenic")
    print("  Feathers cisTarget dans data/pyscenic/scenic_refs/*.feather")


if __name__ == "__main__":
    sys.exit(main())
