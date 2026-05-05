#!/usr/bin/env python3
"""
Pré-télécharge les ressources OmniPath et les écrit en cache TSV.

À lancer UNE FOIS sur le frontal Nautilus (ou en local) — qui a accès
Internet — avant de soumettre les jobs SLURM. Les compute nodes
n'ont pas de réseau sortant, donc `gnn_vgae.py --use-omnipath-…`
doit pouvoir lire les TSV directement sans appel HTTP.

Cible 3 ressources :
  - signaling_omnipath.tsv.gz  (kinase-substrat + causal dirigé)
  - tf_collectri.tsv.gz        (TF→target curé, fallback DoRothEA si KO)
  - signed_ppi_signor.tsv.gz   (PPI causal SIGNOR 3.0)

Usage :
    python scripts/cache_omnipath.py
    python scripts/cache_omnipath.py --cache-dir data/omnipath
    python scripts/cache_omnipath.py --only signaling,collectri
"""

from __future__ import annotations

import argparse
import os
import sys

# Permettre l'import depuis src/gnn sans installer le package
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "gnn"))

import omnipath_integration as opi  # noqa: E402


SOURCES = {
    "signaling": ("signaling_omnipath.tsv.gz",
                  opi._fetch_signaling_omnipath),
    "collectri": ("tf_collectri.tsv.gz",
                  opi._fetch_collectri),
    "dorothea":  ("tf_dorothea.tsv.gz",
                  opi._fetch_dorothea),
    "signor":    ("signed_ppi_signor.tsv.gz",
                  opi._fetch_signor_signed_ppi),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default="data/omnipath",
                   help="dossier de sortie (défaut : data/omnipath)")
    p.add_argument("--only", default="signaling,collectri,signor",
                   help="liste séparée par virgules : signaling, collectri, "
                        "dorothea, signor (défaut : signaling,collectri,signor)")
    p.add_argument("--organism", type=int, default=9606,
                   help="taxon NCBI (défaut : 9606 = humain)")
    p.add_argument("--force", action="store_true",
                   help="re-télécharger même si le cache existe")
    args = p.parse_args()

    if not opi.OMNIPATH_AVAILABLE:
        sys.exit("ERREUR : module `omnipath` non installé. "
                 "Lance `pip install 'omnipath>=1.0.7'` dans l'env GNN.")

    os.makedirs(args.cache_dir, exist_ok=True)
    requested = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SOURCES]
    if unknown:
        sys.exit(f"Sources inconnues : {unknown}. Valides : {list(SOURCES)}")

    print(f"[cache_omnipath] cible : {args.cache_dir}")
    print(f"[cache_omnipath] sources : {requested}")

    n_ok, n_skip, n_fail = 0, 0, 0
    for name in requested:
        cache_name, fetcher = SOURCES[name]
        path = os.path.join(args.cache_dir, cache_name)
        if os.path.exists(path) and not args.force:
            print(f"  [skip] {name} : cache existe ({path}) — "
                  f"--force pour re-télécharger")
            n_skip += 1
            continue
        print(f"  [fetch] {name} → {path}")
        try:
            df = fetcher(organism=args.organism)
        except TypeError:
            df = fetcher()
        if df is None or df.empty:
            print(f"  [fail] {name} : aucune donnée")
            n_fail += 1
            continue
        opi._write_cache(df, path)
        print(f"  [ok]   {name} : {len(df)} liens écrits")
        n_ok += 1

    print(f"\n[cache_omnipath] récap : {n_ok} ok, {n_skip} skip, {n_fail} fail")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
