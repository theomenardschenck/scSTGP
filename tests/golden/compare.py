#!/usr/bin/env python3
"""
compare.py — comparateur golden pour le split de gnn_vgae.py.

Compare deux sorties de run (gene_ranking_vgae.csv + gene_embeddings_vgae.csv)
entre une RÉFÉRENCE figée et un run NOUVEAU (après refactor). Sert de filet
iso-comportement : chaque stage du split doit laisser ces sorties inchangées.

Politique de tolérance :
  - ordre des gènes + colonnes non-float (rangs, cluster, flags BDD) : EXACT.
  - colonnes float + matrice d'embeddings : np.allclose(atol, rtol).
    atol/rtol pilotés par env GOLDEN_ATOL / GOLDEN_RTOL (défaut 0 = bit-exact,
    utilisé par la sonde de déterminisme ; relâché ensuite selon son verdict).

Usage : python compare.py <ref_dir> <new_dir>
Exit 0 si dans la tolérance, 1 sinon (+ résumé des écarts).
"""
import os
import sys
import numpy as np
import pandas as pd

ATOL = float(os.environ.get("GOLDEN_ATOL", "0"))
RTOL = float(os.environ.get("GOLDEN_RTOL", "0"))


def _fail(msg):
    print(f"  ❌ {msg}")
    return False


def compare_ranking(ref_dir, new_dir):
    ok = True
    fr = os.path.join(ref_dir, "gene_ranking_vgae.csv")
    fn = os.path.join(new_dir, "gene_ranking_vgae.csv")
    if not (os.path.exists(fr) and os.path.exists(fn)):
        return _fail(f"gene_ranking_vgae.csv manquant (ref={os.path.exists(fr)} new={os.path.exists(fn)})")
    ref = pd.read_csv(fr)
    new = pd.read_csv(fn)
    if list(ref.columns) != list(new.columns):
        ok = _fail(f"colonnes différentes\n     ref={list(ref.columns)}\n     new={list(new.columns)}")
    if ref.shape != new.shape:
        return _fail(f"shape ranking : ref={ref.shape} new={new.shape}")
    # Ordre des gènes : exact (le tri par vgae_importance ne doit pas bouger).
    if not (ref["gene"].values == new["gene"].values).all():
        n_diff = int((ref["gene"].values != new["gene"].values).sum())
        ok = _fail(f"ordre des gènes : {n_diff} positions diffèrent (1re = "
                   f"{ref['gene'].values[(ref['gene'].values != new['gene'].values)][:3]})")
    worst = []
    for col in ref.columns:
        if col == "gene":
            continue
        if pd.api.types.is_float_dtype(ref[col]):
            a, b = ref[col].to_numpy(), new[col].to_numpy()
            if not np.allclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=True):
                d = np.nanmax(np.abs(a - b))
                worst.append((col, d))
        else:
            if not (ref[col].values == new[col].values).all():
                n = int((ref[col].values != new[col].values).sum())
                worst.append((col, f"{n} valeurs (exact requis)"))
    if worst:
        ok = _fail("colonnes hors-tolérance : " + ", ".join(f"{c}(Δ={d})" for c, d in worst))
    if ok:
        print("  ✅ gene_ranking_vgae.csv identique")
    return ok


def compare_embeddings(ref_dir, new_dir):
    fr = os.path.join(ref_dir, "gene_embeddings_vgae.csv")
    fn = os.path.join(new_dir, "gene_embeddings_vgae.csv")
    if not (os.path.exists(fr) and os.path.exists(fn)):
        return _fail(f"gene_embeddings_vgae.csv manquant (ref={os.path.exists(fr)} new={os.path.exists(fn)})")
    ref = pd.read_csv(fr, index_col=0)
    new = pd.read_csv(fn, index_col=0)
    if ref.shape != new.shape:
        return _fail(f"shape embeddings : ref={ref.shape} new={new.shape}")
    if not (ref.index.values == new.index.values).all():
        return _fail("ordre des gènes (index embeddings) diffère")
    a, b = ref.to_numpy(dtype=np.float64), new.to_numpy(dtype=np.float64)
    if not np.allclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=True):
        print(f"  ❌ embeddings hors-tolérance : max|Δ|={np.nanmax(np.abs(a - b)):.2e} "
              f"(atol={ATOL}, rtol={RTOL})")
        return False
    print(f"  ✅ gene_embeddings_vgae.csv identique (max|Δ|={np.nanmax(np.abs(a - b)):.2e})")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python compare.py <ref_dir> <new_dir>")
        sys.exit(2)
    ref_dir, new_dir = sys.argv[1], sys.argv[2]
    print(f"[golden] compare  ref={ref_dir}\n                  new={new_dir}  (atol={ATOL}, rtol={RTOL})")
    r1 = compare_ranking(ref_dir, new_dir)
    r2 = compare_embeddings(ref_dir, new_dir)
    if r1 and r2:
        print("[golden] ✅ PASS — iso-comportement")
        sys.exit(0)
    print("[golden] ❌ FAIL — divergence détectée")
    sys.exit(1)
