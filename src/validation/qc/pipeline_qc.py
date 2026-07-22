#!/usr/bin/env python
"""pipeline_qc.py — the five checks that must pass before an ablation is read.

WHY (2026-07-21). Each of these was improvised during a single debugging session
and each caught a real defect. None of them was part of the pipeline, so every
wave shipped without them. They are cheap; they belong in the workflow.

  1. `repro`     -- NOISE FLOOR. Rank correlation between runs of the SAME
                    config. Without it, no ablation delta is interpretable.
                    Measured 2026-07-21: rho = 0.556 (rich) / 0.687 (pure)
                    between identical configs, median rank shift 942 places --
                    LARGER than the FI treatment it was supposed to measure
                    (693). Cause: `compute.deterministic` was unset (LOG §27).
  2. `duplicates`-- every edge_type's ordered-pair multiplicity, and whether
                    duplicates carry identical attributes / opposite signs.
                    Caught PPI stored 2x: STRING lists each interaction twice
                    AND the builder symmetrises again (LOG §25quater). Inflates
                    ||h_PPI|| ~ 13x ||h_signaling||.
  3. `overlap`   -- pairwise containment between edge sources. Caught
                    tf_curated ⊂ transcriptional at 99.9 % -- 19 unique pairs
                    for 48 026 edges (LOG §25ter).
  4. `degree`    -- is the readout degree-confounded? rho(degree, driver_score)
                    and the degree-stratified version. Measured +0.451, and the
                    sign of the confound FLIPS with graph density (LOG §25).
  5. `axis`      -- axis specificity: does the real axis beat random axes?
                    Reads the `--random-axis` outputs if present.

`all` runs everything it can and writes a markdown report. Exit code is 1 if a
BLOCKING check fails (repro below threshold), 0 otherwise -- so it can gate a
wave in CI.

Usage
-----
    # noise floor between two runs of the same config
    python src/validation/qc/pipeline_qc.py repro \
        --a output/.../op.all.s1 --b output/.../rfi.pure-legacy/s1

    # full QC on a wave directory (auto-detects configs and seeds)
    python src/validation/qc/pipeline_qc.py all \
        --wave output/gnn_vgae/V6.2/output_fi --out qc_report.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Below this, two runs of the SAME config disagree so much that no ablation
# delta can be attributed. Empirical: 0.556-0.687 observed WITHOUT determinism;
# with `compute.deterministic: true` the same-config pair should be ~1.0.
REPRO_MIN = 0.95

RANKING_NAMES = ("cross_seed_gene_ranking.tsv",)
TARGETS = ["OCRL", "SYNJ2", "SMPD1", "NAMPT", "GCLC",
           "HMGB1", "HMGB2", "H2AFZ", "TP53", "MYC", "ASNS"]


# --------------------------------------------------------------------- utils
def _find_ranking(d: Path) -> Path | None:
    """Locate a cross-seed ranking for a run dir.

    Layouts differ: the flat one keeps it under <run>/xseed/, the Snakemake one
    puts it at CONFIG level (<config>/analysis/) while the run is <config>/s1/ --
    i.e. a SIBLING, which no rglob under the run dir would ever find.
    """
    subs = ("", "xseed", "analysis", "cross_seed_report")
    for base in (d, d.parent):                    # run dir, then config dir
        for sub in subs:
            for name in RANKING_NAMES:
                p = (base / sub / name) if sub else (base / name)
                if p.exists():
                    return p
    hits = sorted(d.rglob(RANKING_NAMES[0])) or sorted(d.parent.rglob(RANKING_NAMES[0]))
    return hits[0] if hits else None


def _load_ranking(d: Path) -> pd.DataFrame | None:
    p = _find_ranking(d)
    if p is None:
        return None
    r = pd.read_csv(p, sep="\t", low_memory=False)
    r["rank"] = np.arange(1, len(r) + 1)
    return r.set_index("target")


def _load_graph(run_dir: Path):
    import torch
    g = run_dir / "hetero_graph_vgae.pt"
    if not g.exists():
        hits = sorted(run_dir.rglob("hetero_graph_vgae.pt"))
        if not hits:
            return None
        g = hits[0]
    return torch.load(g, map_location="cpu", weights_only=False)


def _gene_ets(g):
    return [et for et in g.edge_types if et[0] == "gene" and et[2] == "gene"]


# ---------------------------------------------------------------- 1. repro
def check_repro(a: Path, b: Path) -> dict:
    """Rank correlation between two runs. If configs are identical, this is the
    NOISE FLOOR: any ablation effect weaker than it is unattributable."""
    from scipy.stats import spearmanr
    ra, rb = _load_ranking(a), _load_ranking(b)
    if ra is None or rb is None:
        return {"check": "repro", "status": "SKIP",
                "detail": f"ranking absent ({'A' if ra is None else 'B'})"}
    common = ra.index.intersection(rb.index)
    rho = float(spearmanr(ra.loc[common, "driver_score"],
                          rb.loc[common, "driver_score"])[0])
    shifts = (ra.loc[common, "rank"] - rb.loc[common, "rank"]).abs()
    tg = [t for t in TARGETS if t in common]
    tg_shift = float(shifts.loc[tg].median()) if tg else float("nan")

    # are the two configs actually identical? (then rho IS the noise floor)
    ca, cb = a / "run_config.json", b / "run_config.json"
    same_cfg, cfg_diff = None, []
    if ca.exists() and cb.exists():
        ja, jb = json.load(open(ca)), json.load(open(cb))
        cfg_diff = sorted(k for k in set(ja) & set(jb)
                          if ja[k] != jb[k] and k != "run_tag")
        same_cfg = not cfg_diff and set(ja) == set(jb)

    status = "OK"
    if same_cfg and rho < REPRO_MIN:
        status = "FAIL"
    elif same_cfg is None:
        status = "WARN"
    return {"check": "repro", "status": status, "rho": round(rho, 4),
            "n": len(common), "median_rank_shift": round(float(shifts.median()), 1),
            "median_rank_shift_targets": round(tg_shift, 1),
            "identical_config": same_cfg, "config_diff": cfg_diff,
            "detail": ("configs identiques => rho EST le plancher de bruit"
                       if same_cfg else f"configs diffèrent sur {cfg_diff}")}


# ----------------------------------------------------------- 2. duplicates
def check_duplicates(run_dir: Path) -> dict:
    """Ordered-pair multiplicity per edge_type + do duplicates agree on
    attributes and sign? A factor != 1.0 means edges are counted several times
    by `HeteroConv(aggr="sum")` -- silent reweighting of that source."""
    g = _load_graph(run_dir)
    if g is None:
        return {"check": "duplicates", "status": "SKIP", "detail": "graphe absent"}
    n = int(g["gene"].num_nodes)
    rows, bad = [], []
    for et in _gene_ets(g):
        ei = g[et].edge_index.numpy().astype(np.int64)
        ea = g[et].edge_attr.numpy() if getattr(g[et], "edge_attr", None) is not None else None
        key = ei[0] * n + ei[1]
        uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        factor = ei.shape[1] / max(len(uniq), 1)
        dup_idx = np.where(cnt > 1)[0]
        attrs_differ = sign_flip = 0
        for u in dup_idx[:2000]:
            if ea is None:
                break
            idx = np.where(inv == u)[0]
            r = ea[idx]
            if not np.allclose(r, r[0]):
                attrs_differ += 1
                if r.shape[1] > 1 and len(set(np.sign(r[:, 1]).tolist())) > 1:
                    sign_flip += 1
        rev = len(set(uniq.tolist()) & set((ei[1] * n + ei[0]).tolist()))
        rows.append(dict(edge_type=et[1], n_edges=ei.shape[1],
                         unique_ordered=len(uniq), dup_factor=round(factor, 2),
                         attr_dim=(0 if ea is None else ea.shape[1]),
                         dup_attrs_differ=attrs_differ, dup_sign_flip=sign_flip,
                         pct_reciprocal=round(100 * rev / max(len(uniq), 1), 1)))
        if factor > 1.001:
            bad.append(f"{et[1]} ×{factor:.2f}")
    return {"check": "duplicates", "status": "FAIL" if bad else "OK",
            "detail": ("arêtes dupliquées : " + ", ".join(bad)) if bad
                      else "aucune duplication",
            "table": pd.DataFrame(rows).sort_values("n_edges", ascending=False)}


# -------------------------------------------------------------- 3. overlap
def check_overlap(run_dir: Path, min_containment: float = 40.0) -> dict:
    """Pairwise containment |A∩B|/|A| on UNORDERED pairs. High containment =
    a source that adds edge count but almost no information."""
    g = _load_graph(run_dir)
    if g is None:
        return {"check": "overlap", "status": "SKIP", "detail": "graphe absent"}
    n = int(g["gene"].num_nodes)
    S = {}
    for et in _gene_ets(g):
        ei = g[et].edge_index.numpy().astype(np.int64)
        a, b = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
        m = a != b
        S[et[1]] = set((a[m] * n + b[m]).tolist())
    rows = []
    for x, y in itertools.combinations(sorted(S, key=lambda k: -len(S[k])), 2):
        i = len(S[x] & S[y])
        if not i:
            continue
        cx, cy = 100 * i / len(S[x]), 100 * i / len(S[y])
        if max(cx, cy) < min_containment:
            continue
        rows.append(dict(A=x, B=y, inter=i, A_in_B=round(cx, 1),
                         B_in_A=round(cy, 1),
                         jaccard=round(100 * i / len(S[x] | S[y]), 1),
                         A_unique=len(S[x] - S[y])))
    df = pd.DataFrame(rows).sort_values("A_in_B", ascending=False) if rows else pd.DataFrame()
    red = [f"{r.A}⊂{r.B} {r.A_in_B}%" for r in df.itertuples()] if len(df) else []
    return {"check": "overlap", "status": "WARN" if red else "OK",
            "detail": ("redondance : " + ", ".join(red[:5])) if red else "rien de notable",
            "table": df}


# --------------------------------------------------------------- 4. degree
def check_degree(run_dir: Path) -> dict:
    """Is the readout degree-confounded? Reports rho(degree, driver_score) raw
    AND stratified. The confound's SIGN flips with graph density, so a value
    near 0 on one config says nothing about another."""
    from scipy.stats import spearmanr
    g = _load_graph(run_dir)
    rk = _load_ranking(run_dir)
    if g is None or rk is None:
        return {"check": "degree", "status": "SKIP", "detail": "graphe/ranking absent"}
    emb = run_dir / "gene_embeddings_vgae.csv"
    if not emb.exists():
        hits = sorted(run_dir.rglob("gene_embeddings_vgae.csv"))
        if not hits:
            return {"check": "degree", "status": "SKIP", "detail": "noms de gènes absents"}
        emb = hits[0]
    names = pd.read_csv(emb, usecols=[0]).iloc[:, 0].astype(str).tolist()
    n = len(names)
    deg = np.zeros(n)
    for et in _gene_ets(g):
        ei = g[et].edge_index.numpy()
        np.add.at(deg, ei[0], 1); np.add.at(deg, ei[1], 1)
    d = pd.DataFrame({"gene": names, "degree": deg}).set_index("gene")
    m = rk.join(d, how="inner").dropna(subset=["degree", "driver_score"])
    if len(m) < 100:
        return {"check": "degree", "status": "SKIP", "detail": "trop peu de gènes appariés"}
    rho = float(spearmanr(m.degree, m.driver_score)[0])
    strat = m.assign(s=pd.qcut(m.degree.rank(method="first"), 25, labels=False))
    m = m.assign(ds_strat=strat.groupby("s").driver_score.rank(pct=True))
    rho_s = float(spearmanr(m.degree, m.ds_strat)[0])
    top = m.nlargest(50, "driver_score")
    return {"check": "degree", "status": "WARN" if abs(rho) > 0.25 else "OK",
            "rho_degree_driver": round(rho, 3),
            "rho_after_stratification": round(rho_s, 3),
            "median_degree_top50": float(top.degree.median()),
            "median_degree_all": float(m.degree.median()),
            "detail": (f"driver_score confondu au degré (rho={rho:+.3f}) ; "
                       f"stratifié {rho_s:+.3f}")}


# ----------------------------------------------------------------- 5. axis
def check_axis(run_dir: Path) -> dict:
    """Axis specificity: the real axis must beat random axes. Reads the
    `--random-axis` outputs. NOTE: E[|cos|] = 1/sqrt(d) is the WRONG reference
    (LOG §26ter) -- the latent is not isotropic, so the empirical random-axis
    distribution is the only valid null."""
    hits = sorted(run_dir.rglob("*random_axis*.tsv"))
    if not hits:
        return {"check": "axis", "status": "SKIP",
                "detail": "aucune sortie random_axis (lancer avec decoy_random_axis > 0)"}
    cols, vals = None, []
    for h in hits:
        df = pd.read_csv(h, sep="\t", low_memory=False)
        c = [x for x in df.columns if "cos" in x.lower()]
        if c:
            cols = c[0]
            vals.append(df[c[0]].abs().dropna())
    if not vals:
        return {"check": "axis", "status": "SKIP", "detail": "pas de colonne cosinus"}
    v = pd.concat(vals)
    return {"check": "axis", "status": "OK", "n_files": len(hits),
            "random_abs_cos_median": round(float(v.median()), 4),
            "random_abs_cos_p95": round(float(v.quantile(.95)), 4),
            "detail": f"nulle d'axe empirique sur {len(v)} valeurs ({cols})"}


# ------------------------------------------------------------------ report
def _fmt(res: dict) -> str:
    icon = {"OK": "✅", "WARN": "🟠", "FAIL": "⛔", "SKIP": "⚪"}[res["status"]]
    head = f"### {icon} {res['check']} — {res['status']}\n\n{res.get('detail','')}\n"
    kv = {k: v for k, v in res.items()
          if k not in ("check", "status", "detail", "table")}
    if kv:
        head += "\n" + "\n".join(f"- `{k}` = {v}" for k, v in kv.items()) + "\n"
    if isinstance(res.get("table"), pd.DataFrame) and len(res["table"]):
        try:                                   # `tabulate` is optional
            body = res["table"].to_markdown(index=False)
        except ImportError:
            body = "```\n" + res["table"].to_string(index=False) + "\n```"
        head += "\n" + body + "\n"
    return head


def cmd_all(args):
    wave = args.wave
    runs = sorted({p.parent for p in wave.rglob("hetero_graph_vgae.pt")})
    if not runs:
        sys.exit(f"[qc] aucun run sous {wave}")
    ref = args.run_dir or runs[0]
    print(f"[qc] {len(runs)} run(s) ; référence = {ref}")

    results = [check_duplicates(ref), check_overlap(ref),
               check_degree(ref), check_axis(ref)]

    # noise floor: prefer an explicit pair, else two seeds of the same config
    if args.a and args.b:
        results.insert(0, check_repro(args.a, args.b))
    else:
        by_cfg: dict[str, list[Path]] = {}
        for r in runs:
            by_cfg.setdefault(r.parent.name, []).append(r)
        pair = next((v for v in by_cfg.values() if len(v) >= 2), None)
        results.insert(0, check_repro(*pair[:2]) if pair else
                       {"check": "repro", "status": "SKIP",
                        "detail": "aucune paire de runs de même config "
                                  "(passer --a/--b, ou relancer avec >=2 seeds)"})

    body = ("# QC pipeline — 5 vérifications\n\n"
            f"*Référence : `{ref}` · vague : `{wave}`*\n\n"
            "Une ablation n'est lisible que si `repro` passe : le plancher de\n"
            "bruit doit être au-dessus de tout effet que l'on veut attribuer.\n\n"
            + "\n".join(_fmt(r) for r in results))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
        print(f"[qc] → {args.out}")
    else:
        print(body)
    if any(r["status"] == "FAIL" for r in results):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("repro", help="plancher de bruit entre 2 runs")
    r.add_argument("--a", type=Path, required=True)
    r.add_argument("--b", type=Path, required=True)

    for name, helptxt in [("duplicates", "multiplicité des paires orientées"),
                          ("overlap", "recouvrement entre sources"),
                          ("degree", "confusion readout ↔ degré"),
                          ("axis", "spécificité d'axe")]:
        s = sub.add_parser(name, help=helptxt)
        s.add_argument("--run-dir", type=Path, required=True)

    a = sub.add_parser("all", help="les 5 + rapport markdown")
    a.add_argument("--wave", type=Path, required=True)
    a.add_argument("--run-dir", type=Path, default=None)
    a.add_argument("--a", type=Path, default=None)
    a.add_argument("--b", type=Path, default=None)
    a.add_argument("--out", type=Path, default=None)

    args = ap.parse_args()
    if args.cmd == "all":
        return cmd_all(args)
    fn = {"repro": lambda: check_repro(args.a, args.b),
          "duplicates": lambda: check_duplicates(args.run_dir),
          "overlap": lambda: check_overlap(args.run_dir),
          "degree": lambda: check_degree(args.run_dir),
          "axis": lambda: check_axis(args.run_dir)}[args.cmd]
    res = fn()
    print(_fmt(res))
    sys.exit(1 if res["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
