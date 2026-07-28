#!/usr/bin/env python3
"""
compare_cache.py — compare deux bundles de cache graphe (§1-7) de façon
ROBUSTE À L'ORDRE. L'ordre des arêtes/gènes dépend du hash-seed du process
(non déterministe entre builds, dans le monolithe comme après refactor) ; on
compare donc des invariants structurels : nb de gènes, ensemble des symboles,
comptes + ensembles de paires (symbole,symbole) par edge_type, et features
alignées par symbole.

Usage : python compare_cache.py <ref.pkl> <test.pkl>
Exit 0 si structurellement identiques.
"""
import sys
import pickle
import numpy as np

ref = pickle.load(open(sys.argv[1], "rb"))
tst = pickle.load(open(sys.argv[2], "rb"))
ok = True

def fail(m):
    global ok; ok = False; print(f"  ❌ {m}")

# n_genes + symboles
if ref.get("n_genes") != tst.get("n_genes"):
    fail(f"n_genes : {ref.get('n_genes')} vs {tst.get('n_genes')}")
rs, ts = ref["gene_symbols"], tst["gene_symbols"]
if set(rs) != set(ts):
    fail(f"ensembles de symboles diffèrent (|ref\\tst|={len(set(rs)-set(ts))}, |tst\\ref|={len(set(ts)-set(rs))})")
else:
    print(f"  ✅ {len(rs)} gènes, ensembles de symboles identiques")

# edge_index_* : comptes + ensembles de paires-symboles
ref_sym = list(rs); tst_sym = list(ts)
edge_keys = sorted(k for k in ref if k.startswith("edge_index_"))
for k in edge_keys:
    a, b = ref[k], tst.get(k)
    if b is None:
        fail(f"{k} absent du test"); continue
    na = a.shape[1] if hasattr(a, "shape") and a.numel() else 0
    nb = b.shape[1] if hasattr(b, "shape") and b.numel() else 0
    if na != nb:
        fail(f"{k} : {na} vs {nb} arêtes"); continue
    if na == 0:
        print(f"  ✅ {k} : 0 arête (identique)"); continue
    # mapper indices → symboles → set de paires (non orienté-agnostique : garder l'ordre src,dst)
    ea = {(ref_sym[i], ref_sym[j]) for i, j in zip(a[0].tolist(), a[1].tolist())}
    eb = {(tst_sym[i], tst_sym[j]) for i, j in zip(b[0].tolist(), b[1].tolist())}
    if ea != eb:
        fail(f"{k} : {na} arêtes mais ensembles de paires diffèrent (|ref\\tst|={len(ea-eb)})")
    else:
        print(f"  ✅ {k} : {na} arêtes, paires-symboles identiques")

# gene_features alignées par symbole
if "gene_features" in ref and "gene_features" in tst:
    fa = np.asarray(ref["gene_features"]); fb = np.asarray(tst["gene_features"])
    if fa.shape != fb.shape:
        fail(f"gene_features shape : {fa.shape} vs {fb.shape}")
    else:
        idx_t = {s: i for i, s in enumerate(tst_sym)}
        perm = [idx_t[s] for s in ref_sym]     # réaligner test sur l'ordre ref
        if np.allclose(fa, fb[perm], atol=0, rtol=0, equal_nan=True):
            print(f"  ✅ gene_features {fa.shape} identiques (alignées par symbole)")
        else:
            fail(f"gene_features diffèrent : max|Δ|={np.nanmax(np.abs(fa - fb[perm])):.2e}")

print("[cache] ✅ STRUCTURELLEMENT IDENTIQUE" if ok else "[cache] ❌ DIVERGENCE")
sys.exit(0 if ok else 1)
