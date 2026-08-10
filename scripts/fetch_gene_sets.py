#!/usr/bin/env python3
"""fetch_gene_sets.py — récupération offline-first des BDD de validation.

Sort les téléchargements HORS du runtime de scoring (comme cache_omnipath.py) :
le pipeline lui-même ne télécharge plus rien (offline-first sur cluster). Lance
ce script une fois pour peupler `data/databases/` et `data/gene_sets/`.

    python scripts/fetch_gene_sets.py            # tout ce qui est connu
    python scripts/fetch_gene_sets.py --name cellage genage
    python scripts/fetch_gene_sets.py --list

Les sets sans URL connue (ex. endosen_up : signature d'un supplément d'article)
sont signalés avec la marche à suivre — l'outil tourne quand même sans eux
(health-check → AUTO_OFF).
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "databases"
GS = ROOT / "data" / "gene_sets"

# name → (url, destination, {member_suffix pour extraire d'un zip})
SOURCES: dict[str, dict] = {
    "genage": {
        "url": "https://genomics.senescence.info/genes/human_genes.zip",
        "dest": DB / "genage_human.csv", "zip_member": ".csv"},
    "cellage": {
        "url": "https://genomics.senescence.info/cells/cellAge.zip",
        "dest": DB / "cellage3.tsv", "zip_member": (".tsv", ".csv")},
    "msigdb_aging": {
        "url": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
               "2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt",
        "dest": DB / "h.all.symbols.gmt"},
    "ageanno": {
        "url": "https://raw.githubusercontent.com/vikkihuangkexin/AgeAnno/"
               "main/scRNA/Aging-related%20DEGs.txt",
        "dest": DB / "ageanno" / "aging_DEGs.txt"},
    "aging_local": {
        "url": None, "dest": ROOT / "data" / "human_age_related_gene.csv",
        "note": "proxy SenMayo local — fourni avec le dépôt, pas de source publique."},
    "endosen_up": {
        "url": None, "dest": GS / "endosen_up.txt",
        "note": ("EndoSEN_up (Guduric-Fuchs et al. 2024, Aging Cell 23:e14240, "
                 "PMC11488300) — FOURNI dans le dépôt (70/75 transcrits de la "
                 "Fig. SF5 ; voir l'en-tête du fichier). Les 5 gènes manquants ne "
                 "sont pas étiquetés dans la figure ; pour les 75 exacts, re-dériver "
                 "depuis GSE160166 ou contacter les auteurs.")},
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def fetch(name: str, spec: dict, force: bool) -> None:
    dest: Path = spec["dest"]
    if dest.exists() and not force:
        print(f"  [cache] {name} → {dest.relative_to(ROOT)}")
        return
    if not spec.get("url"):
        status = "présent" if dest.exists() else "ABSENT"
        print(f"  [manuel] {name} ({status}) — {spec.get('note', '')}")
        return
    print(f"  [get] {name} …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = _get(spec["url"])
    member = spec.get("zip_member")
    if member:
        suffixes = member if isinstance(member, tuple) else (member,)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(suffixes)]
            if not names:
                print(f"    ! aucun membre {suffixes} dans le zip", file=sys.stderr)
                return
            raw = z.read(names[0])
    dest.write_bytes(raw)
    print(f"    OK {len(raw)/1e3:.0f} kB → {dest.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", nargs="*", help="sets à récupérer (défaut: tous)")
    ap.add_argument("--force", action="store_true", help="re-télécharge même si présent")
    ap.add_argument("--list", action="store_true", help="liste les sources connues")
    args = ap.parse_args()

    if args.list:
        for n, s in SOURCES.items():
            kind = "url" if s.get("url") else "manuel"
            print(f"  {n:14s} [{kind}] → {Path(s['dest']).relative_to(ROOT)}")
        return

    todo = args.name or list(SOURCES)
    for n in todo:
        if n not in SOURCES:
            print(f"  ! inconnu: {n} (voir --list)", file=sys.stderr)
            continue
        try:
            fetch(n, SOURCES[n], args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! échec {n}: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
