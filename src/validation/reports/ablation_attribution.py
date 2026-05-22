#!/usr/bin/env python3
"""ablation_attribution.py — attribution automatique des sources d'information.

Étant donné un ranking de référence (typiquement `v4.1-baseline` ou
`v4.1-full`) et un ou plusieurs rankings d'ablation (`+no-coexpr`,
`+no-humess`, …), ce script :

1. **Aligne** les `cross_seed_gene_ranking.tsv` sur le set de gènes commun.
2. **Calcule** `Δrank[gene][ablation] = rank_ref - rank_ablation` (>0 →
   gène promu par l'ablation, <0 → dému).
3. **Identifie** les top-K genes promus et démus par chaque ablation.
4. **Enrichit** chaque set via ORA hypergéométrique :
   - REACTOME pathways (modules biologiques).
   - Bases de données aging (SenMayo, CellAge, GenAge, Fridman).
5. **Produit** :
   - `delta_rank_table.tsv` : wide-format gènes × ablations.
   - `top_movers_<ablation>.tsv` : top démus + top promus avec
     `evidence_tier`, `direction`, `n_aging_dbs`, `is_de_significant`.
   - `ora_<ablation>_<set>.tsv` : ORA REACTOME / aging pour
     démus et promus séparément.
   - `attribution_report.md` : synthèse markdown auto-générée pour
     interprétation rapide. Inclut titre, tableaux résumés, top-20
     gènes par direction × ablation, top-10 pathways enrichis.

Usage
-----
    python src/validation/reports/ablation_attribution.py \\
        --reference output/gnn_vgae/V4.1/cross_seed_v4.1-baseline_axisV4 \\
        --ablations \\
            output/gnn_vgae/V4.1/cross_seed_v4.1-baseline+no-coexpr_axisV4 \\
            output/gnn_vgae/V4.1/cross_seed_v4.1-baseline+no-humess_axisV4 \\
        --top-k 50 \\
        --out-dir output/gnn_vgae/V4.1/attribution_baseline

    # Sur la branche full V4.1 :
    python src/validation/reports/ablation_attribution.py \\
        --reference output/gnn_vgae/V4.1/cross_seed_v4.1-full_axisV4 \\
        --ablations \\
            output/gnn_vgae/V4.1/cross_seed_v4.1-full+no-coexpr_axisV4 \\
            output/gnn_vgae/V4.1/cross_seed_v4.1-full+no-humess_axisV4 \\
        --out-dir output/gnn_vgae/V4.1/attribution_full

Références
----------
- Ramaswamy 2021 *Bioinformatics* : edge perturbation ranking.
- Ying 2019 *NeurIPS* GNNExplainer : interprétation formelle (TODO
  Tier 2 Phase 5) ; ici, on fait une attribution statistique, plus
  rapide mais moins formelle.
- decoupler-py (Badia-i-Mompel 2022) : on utilise ora_consensus.run_ora
  pour l'ORA hypergéométrique.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# Bootstrap pour importer ora_consensus.py (src/validation/ora/) et permettre
# le fallback flat-layout du cluster (tout sous src/).
def _bootstrap_paths():
    here = Path(__file__).resolve()
    candidates = [
        here.parent,                       # src/validation/reports/ (cas inattendu)
        here.parent.parent / "ora",        # src/validation/ora/ (layout local)
        here.parent.parent.parent,         # src/ (fallback flat cluster)
    ]
    for cand in candidates:
        if cand.exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))


_bootstrap_paths()
from ora_consensus import (  # noqa: E402
    load_reactome_gmt,
    load_aging_databases,
    load_background,
    run_ora,
)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
RANKING_FILENAME = "cross_seed_gene_ranking.tsv"
REQUIRED_COLS = [
    "target", "driver_score", "direction", "evidence_tier",
    "n_aging_dbs", "is_de_significant", "is_tf",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _short_name(p: Path) -> str:
    """Extrait un nom court depuis le chemin du report.

    Ex: 'cross_seed_v4.1-baseline+no-coexpr_axisV4' → 'baseline+no-coexpr'.
    """
    s = p.name
    for pref in ("cross_seed_", "cross_seed"):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    for suf in ("_axisV4", "_axisV3", ""):
        if suf and s.endswith(suf):
            s = s[: -len(suf)]
            break
    if s.startswith("v4.1-"):
        s = s[len("v4.1-"):]
    if s.startswith("v4-"):
        s = s[len("v4-"):]
    return s


def load_ranking(report_dir: Path) -> pd.DataFrame:
    p = report_dir / RANKING_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"Ranking introuvable : {p}")
    df = pd.read_csv(p, sep="\t")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{p} : colonnes manquantes {missing}")
    return df


def compute_delta_rank(ref: pd.DataFrame,
                       ablations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calcule Δrank pour chaque gène × ablation sur set commun."""
    common = set(ref["target"])
    for df in ablations.values():
        common &= set(df["target"])
    common = sorted(common)

    rows = {}
    ref_idx = ref.set_index("target").loc[common]
    rows["ref_rank"] = ref_idx["driver_score"].rank(ascending=False, method="min").astype(int)
    rows["ref_score"] = ref_idx["driver_score"]
    rows["direction"] = ref_idx["direction"]
    rows["evidence_tier"] = ref_idx["evidence_tier"]
    rows["n_aging_dbs"] = ref_idx["n_aging_dbs"]
    rows["is_de_significant"] = ref_idx["is_de_significant"]
    rows["is_tf"] = ref_idx["is_tf"]

    for name, df in ablations.items():
        a_idx = df.set_index("target").loc[common]
        a_rank = a_idx["driver_score"].rank(ascending=False, method="min").astype(int)
        rows[f"{name}_rank"] = a_rank
        rows[f"{name}_score"] = a_idx["driver_score"]
        # Δrank > 0 → gène GAGNE en rang quand on ablate
        # (ref_rank > ablation_rank ⇔ ablation a rang plus bas ⇔ meilleur rang)
        rows[f"{name}_delta_rank"] = rows["ref_rank"] - a_rank

    out = pd.DataFrame(rows, index=common)
    out.index.name = "target"
    return out


def top_movers(delta: pd.DataFrame, ablation: str, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (top promoted, top demoted) par |Δrank|."""
    col = f"{ablation}_delta_rank"
    if col not in delta.columns:
        raise KeyError(col)
    keep_cols = [
        "ref_rank", f"{ablation}_rank", "ref_score", f"{ablation}_score",
        col, "direction", "evidence_tier", "n_aging_dbs",
        "is_de_significant", "is_tf",
    ]
    df = delta[keep_cols].copy()
    promoted = df.nlargest(k, col)
    demoted = df.nsmallest(k, col)
    return promoted, demoted


def run_ora_block(gene_set: set[str],
                  background: set[str],
                  reactome: dict[str, set[str]],
                  aging: dict[str, set[str]],
                  ) -> dict[str, pd.DataFrame]:
    """Exécute ORA REACTOME + aging et retourne 2 DataFrames."""
    out: dict[str, pd.DataFrame] = {}
    if reactome:
        rows = run_ora(gene_set, background, reactome,
                       min_overlap=3, min_pw_size=5, max_pw_size=500)
        out["reactome"] = pd.DataFrame([r.__dict__ for r in rows])
    if aging:
        rows = run_ora(gene_set, background, aging,
                       min_overlap=2, min_pw_size=5, max_pw_size=5000)
        out["aging"] = pd.DataFrame([r.__dict__ for r in rows])
    return out


# ---------------------------------------------------------------------------
# Rapport markdown
# ---------------------------------------------------------------------------
def render_markdown(delta: pd.DataFrame,
                    ablation_names: list[str],
                    ref_short: str,
                    top_k: int,
                    ora_results: dict[str, dict[str, dict[str, pd.DataFrame]]],
                    ) -> str:
    """Compose le rapport markdown synthétique.

    `ora_results[ablation][direction][source]` = DataFrame (direction ∈
    {promoted, demoted}, source ∈ {reactome, aging}).
    """
    lines: list[str] = []
    lines.append(f"# Attribution d'ablation — référence `{ref_short}`")
    lines.append("")
    lines.append("Δrank > 0 ⇒ gène **promu** par l'ablation (gagne du rang).")
    lines.append("Δrank < 0 ⇒ gène **dému** par l'ablation (perd du rang). "
                 "Donc l'ablation retire une source qui *soutenait* ce gène.")
    lines.append("")
    lines.append(f"Gènes alignés sur set commun : **{len(delta)}**.")
    lines.append("")

    # Résumé global
    lines.append("## Résumé — magnitude du déplacement par ablation")
    lines.append("")
    summary_rows = []
    for ab in ablation_names:
        col = f"{ab}_delta_rank"
        ser = delta[col]
        summary_rows.append({
            "ablation": ab,
            "median |Δrank|": int(ser.abs().median()),
            "P95 |Δrank|": int(ser.abs().quantile(0.95)),
            "max |Δrank|": int(ser.abs().max()),
            "n_promus > 1000": int((ser > 1000).sum()),
            "n_démus < -1000": int((ser < -1000).sum()),
            "stabilité (% |Δ|<500)": round(100 * (ser.abs() < 500).mean(), 1),
        })
    lines.append("```")
    lines.append(pd.DataFrame(summary_rows).to_string(index=False))
    lines.append("```")
    lines.append("")

    # Détail par ablation
    for ab in ablation_names:
        lines.append("")
        lines.append(f"## Ablation `{ab}` vs `{ref_short}`")
        lines.append("")
        prom, dem = top_movers(delta, ab, top_k)

        lines.append(f"### Top-{top_k} démus (l'ablation retire une source qui les portait)")
        lines.append("")
        lines.append("```")
        lines.append(dem[[
            "ref_rank", f"{ab}_rank", f"{ab}_delta_rank",
            "ref_score", f"{ab}_score", "direction", "evidence_tier",
            "n_aging_dbs", "is_de_significant", "is_tf",
        ]].head(20).to_string())
        lines.append("```")
        lines.append("")

        lines.append(f"### Top-{top_k} promus (l'ablation expose ces gènes en retirant un compétiteur)")
        lines.append("")
        lines.append("```")
        lines.append(prom[[
            "ref_rank", f"{ab}_rank", f"{ab}_delta_rank",
            "ref_score", f"{ab}_score", "direction", "evidence_tier",
            "n_aging_dbs", "is_de_significant", "is_tf",
        ]].head(20).to_string())
        lines.append("```")
        lines.append("")

        # ORA
        ora_blocks = ora_results.get(ab, {})
        for direction in ("demoted", "promoted"):
            for src in ("reactome", "aging"):
                df = ora_blocks.get(direction, {}).get(src)
                if df is None or df.empty:
                    continue
                top10 = df.head(10)[["pathway", "k", "pw_size", "p_adj"]]
                title = "REACTOME" if src == "reactome" else "Aging DBs"
                lines.append(f"#### ORA {title} — {direction} (top-10 par p_adj)")
                lines.append("")
                lines.append("```")
                lines.append(top10.to_string(index=False))
                lines.append("```")
                lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Référence : Ramaswamy 2021 *Bioinformatics* (edge perturbation "
                 "ranking) ; ORA hypergéométrique via "
                 "`src/validation/ora/ora_consensus.py` (BH-FDR sur catalogues).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Attribution automatique des sources d'information "
                    "via Δrank cross-ablation + ORA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage", 1)[-1],
    )
    ap.add_argument("--reference", required=True, type=Path,
                    help="Dossier cross_seed_* du ranking de référence.")
    ap.add_argument("--ablations", required=True, nargs="+", type=Path,
                    help="Un ou plusieurs dossiers cross_seed_* d'ablation.")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Dossier de sortie (créé si absent).")
    ap.add_argument("--top-k", type=int, default=50,
                    help="Top-K mouvements à extraire par ablation (défaut 50).")
    ap.add_argument("--ora-min-overlap", type=int, default=3,
                    help="ORA REACTOME : taille minimale de l'intersection.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ref ] {args.reference}")
    ref = load_ranking(args.reference)
    ref_short = _short_name(args.reference)

    ablations: dict[str, pd.DataFrame] = {}
    for p in args.ablations:
        print(f"[abl ] {p}")
        ablations[_short_name(p)] = load_ranking(p)

    print(f"\n[delta] computing Δrank on common gene set...")
    delta = compute_delta_rank(ref, ablations)
    delta_path = args.out_dir / "delta_rank_table.tsv"
    delta.to_csv(delta_path, sep="\t")
    print(f"[delta] wrote {delta_path} ({len(delta)} genes × {len(ablations)} ablations)")

    # ORA — set d'arrière-plan = tous les gènes du ref
    print(f"\n[ora ] loading REACTOME + aging DBs...")
    try:
        reactome = load_reactome_gmt()
        print(f"[ora ] REACTOME : {len(reactome)} pathways")
    except Exception as e:
        print(f"[ora ] REACTOME indisponible : {e}")
        reactome = {}
    try:
        aging = load_aging_databases()
        print(f"[ora ] aging DBs : {sorted(aging.keys())}")
    except Exception as e:
        print(f"[ora ] aging DBs indisponibles : {e}")
        aging = {}
    background = set(ref["target"])

    ora_results: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for ab in ablations:
        prom, dem = top_movers(delta, ab, args.top_k)
        ora_results[ab] = {}
        for direction, df in [("demoted", dem), ("promoted", prom)]:
            gene_set = set(df.index)
            blocks = run_ora_block(gene_set, background, reactome, aging)
            ora_results[ab][direction] = blocks
            # Export TSV
            for src, ora_df in blocks.items():
                if ora_df.empty:
                    continue
                out = args.out_dir / f"ora_{ab}_{direction}_{src}.tsv"
                ora_df.to_csv(out, sep="\t", index=False)
                print(f"[ora ] {ab} {direction} {src} : {len(ora_df)} hits → {out.name}")
        # Top movers TSV
        tsv_path = args.out_dir / f"top_movers_{ab}.tsv"
        combined = pd.concat([dem.assign(category="demoted"),
                              prom.assign(category="promoted")])
        combined.to_csv(tsv_path, sep="\t")
        print(f"[move] {tsv_path.name}")

    # Markdown
    md = render_markdown(delta, list(ablations.keys()), ref_short,
                         args.top_k, ora_results)
    md_path = args.out_dir / "attribution_report.md"
    md_path.write_text(md)
    print(f"\n[md  ] wrote {md_path}")

    print(f"\n[done] all outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
