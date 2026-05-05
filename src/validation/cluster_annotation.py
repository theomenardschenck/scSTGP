"""Annotation biologique automatique des cell_groups (P4 + c0..c3).

Charge `group_expression.tsv` (mean d'expression par cell_group) et calcule
l'enrichissement de chaque cluster vis-à-vis de signatures connues :
SenMayo (Saul 2022), CellAge (Avelar 2020, Induces/Inhibits), GenAge,
HALLMARK MSigDB (50 gene sets), REACTOME ciblés.

Méthode (§13.10.4 du rapport) :
1. Pour chaque gène : z-score de `mean_<cluster>` à travers les 5 cell_groups
   → matrice gene × cluster de spécificité positive/négative.
2. Pour chaque signature S et chaque cluster c :
   - mean_z(c, S) = moyenne des z-scores des gènes de S présents dans nos
     données.
   - p-value = Welch t-test(z[S, c] vs z[non-S, c]).
   - BH-FDR sur l'ensemble (cluster, signature).
3. Outputs : cluster_annotation.tsv (long format), cluster_dominant_labels.tsv
   (top-3 par cluster, q<0.05), cluster_annotation_heatmap.png.

Refs :
- AUCell : Aibar 2017 Nat Methods (SCENIC).
- decoupleR : Badia-i-Mompel 2022 Bioinform Adv.
- SenMayo : Saul 2022 Nat Commun.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DB_DIR = DATA_DIR / "databases"


def parse_gmt(path: Path, prefix_filter: tuple[str, ...] | None = None) -> dict[str, set[str]]:
    """Parse un fichier MSigDB .gmt → {signature_name: {gene1, gene2, ...}}.

    `prefix_filter` : si fourni, ne garde que les signatures dont le nom
    contient un des motifs (case-insensitive sur le nom complet).
    """
    out: dict[str, set[str]] = {}
    with path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, _url, *genes = parts
            if prefix_filter is not None and not any(p.lower() in name.lower() for p in prefix_filter):
                continue
            out[name] = {g.strip() for g in genes if g.strip()}
    return out


def load_signatures() -> dict[str, set[str]]:
    """Construit le dict {signature_name: gene_set}.

    Cherche dans :
      - data/human_age_related_gene.csv (SenMayo proxy local)
      - data/databases/cellage3.tsv (Induces / Inhibits)
      - data/databases/genage_human.csv
      - data/databases/h.all.symbols.gmt (HALLMARK)
      - data/databases/c2.cp.reactome.symbols.gmt (REACTOME, filtré)
    """
    sigs: dict[str, set[str]] = {}

    # SenMayo proxy local
    p = DATA_DIR / "human_age_related_gene.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "Symbol" in df.columns:
            sigs["SenMayo_local"] = set(df["Symbol"].dropna().astype(str))

    # CellAge — split Induces vs Inhibits (information directionnelle)
    p = DB_DIR / "cellage3.tsv"
    if p.exists():
        df = pd.read_csv(p, sep="\t")
        gcol = "Gene symbol" if "Gene symbol" in df.columns else df.columns[1]
        ecol = "Senescence Effect" if "Senescence Effect" in df.columns else None
        sigs["CellAge_all"] = set(df[gcol].dropna().astype(str))
        if ecol is not None:
            sigs["CellAge_induces"] = set(df.loc[df[ecol].astype(str).str.lower().str.contains("induces", na=False), gcol].dropna().astype(str))
            sigs["CellAge_inhibits"] = set(df.loc[df[ecol].astype(str).str.lower().str.contains("inhibits", na=False), gcol].dropna().astype(str))

    # GenAge
    p = DB_DIR / "genage_human.csv"
    if p.exists():
        df = pd.read_csv(p)
        gcol = "symbol" if "symbol" in df.columns else df.columns[1]
        sigs["GenAge"] = set(df[gcol].dropna().astype(str))

    # HALLMARK 50 (toutes — peu de gene sets, cher pas la peine de filtrer)
    p = DB_DIR / "h.all.symbols.gmt"
    if p.exists():
        sigs.update(parse_gmt(p))

    # REACTOME ciblé : sénescence, SASP, ECM, inflammation, oxphos, cycle, ISR/UPR
    p = DB_DIR / "c2.cp.reactome.symbols.gmt"
    if p.exists():
        keywords = (
            "senescen", "sasp", "extracellular_matrix", "ecm", "collagen",
            "inflammat", "interleukin", "tnf", "nfkb", "interferon",
            "oxidative_phosphorylation", "respiratory_electron",
            "cell_cycle", "mitotic", "g1_s", "g2_m", "e2f",
            "unfolded_protein", "integrated_stress", "atf4", "atf6",
            "tgf_beta", "tgf-beta", "p53", "apoptos", "autophag",
            "complement", "dna_damage", "dna_repair", "chromatin",
            "histone", "telomere",
        )
        sigs.update(parse_gmt(p, prefix_filter=keywords))

    return sigs


def compute_zscore_matrix(group_expr: pd.DataFrame, mean_cols: list[str], min_total_expr: float = 0.05) -> pd.DataFrame:
    """Z-score per gene à travers les colonnes de moyenne d'expression.

    Filtre les gènes silencieux (somme totale < min_total_expr).
    Renvoie un DataFrame (gene × cluster) de z-scores.
    """
    expr = group_expr[mean_cols].astype(float).copy()
    total = expr.sum(axis=1)
    keep = total >= min_total_expr
    expr = expr.loc[keep]
    mu = expr.mean(axis=1)
    sigma = expr.std(axis=1, ddof=0).replace(0, np.nan)
    z = expr.sub(mu, axis=0).div(sigma, axis=0)
    z = z.fillna(0.0)
    z.index = group_expr.loc[keep, "gene"].astype(str).values
    return z


def enrich_signatures(z: pd.DataFrame, sigs: dict[str, set[str]], min_signature_size: int = 5) -> pd.DataFrame:
    """Calcule mean_z et Welch t-test (signature vs background) par cluster."""
    universe = set(z.index)
    rows = []
    for name, gset in sigs.items():
        present = gset & universe
        if len(present) < min_signature_size:
            continue
        idx_in = z.index.isin(present)
        idx_out = ~idx_in
        z_in = z.loc[idx_in]
        z_out = z.loc[idx_out]
        for cluster in z.columns:
            vals_in = z_in[cluster].values
            vals_out = z_out[cluster].values
            mean_in = float(vals_in.mean())
            t_stat, pval = stats.ttest_ind(vals_in, vals_out, equal_var=False)
            rows.append({
                "cluster": cluster,
                "signature": name,
                "n_in_signature": int(len(present)),
                "n_used": int(len(vals_in)),
                "mean_z": mean_in,
                "median_z": float(np.median(vals_in)),
                "t_stat": float(t_stat),
                "pval": float(pval),
            })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    # BH-FDR sur l'ensemble (cluster, signature)
    pvals = df["pval"].values
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(ranked)
    bh = ranked * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    qvals = np.empty(n)
    qvals[order] = np.clip(bh, 0, 1)
    df["qval"] = qvals
    df = df.sort_values(["cluster", "qval", "mean_z"], ascending=[True, True, False])
    return df


def dominant_labels(enrich: pd.DataFrame, top_k: int = 5, qval_thresh: float = 0.05) -> pd.DataFrame:
    """Top-K signatures par cluster (positivement enrichies, q < seuil)."""
    sig_pos = enrich.loc[(enrich["mean_z"] > 0) & (enrich["qval"] < qval_thresh)].copy()
    sig_pos = sig_pos.sort_values(["cluster", "mean_z"], ascending=[True, False])
    rows = []
    for cluster, grp in sig_pos.groupby("cluster"):
        for rank_, (_, row) in enumerate(grp.head(top_k).iterrows(), start=1):
            rows.append({
                "cluster": cluster,
                "rank": rank_,
                "signature": row["signature"],
                "mean_z": row["mean_z"],
                "qval": row["qval"],
                "n_used": row["n_used"],
            })
    return pd.DataFrame(rows)


def heatmap(enrich: pd.DataFrame, out_path: Path, max_signatures: int = 40) -> None:
    """Heatmap mean_z (cluster × signature). Limite aux N signatures les plus
    extrêmes en |mean_z| toutes-clusters confondus pour la lisibilité.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib non disponible, skip heatmap", file=sys.stderr)
        return
    pivot = enrich.pivot(index="signature", columns="cluster", values="mean_z").fillna(0.0)
    extreme = pivot.abs().max(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[extreme.head(max_signatures).index]
    qpivot = enrich.pivot(index="signature", columns="cluster", values="qval").reindex(pivot.index)

    fig, ax = plt.subplots(figsize=(1.2 + 0.5 * pivot.shape[1], 0.25 * pivot.shape[0] + 1.5))
    vmax = float(np.nanmax(np.abs(pivot.values))) or 1.0
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([s[:60] for s in pivot.index], fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            q = qpivot.iat[i, j]
            mark = "**" if q < 0.001 else "*" if q < 0.05 else ""
            if mark:
                ax.text(j, i, mark, ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="mean z-score", shrink=0.6)
    ax.set_title("Cluster annotation — mean z-score per signature\n(*: q<0.05, **: q<0.001)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def align_clusters_hungarian(
    ge_runs: list[pd.DataFrame],
    mean_cols_per_run: list[list[str]],
    reference_idx: int = 0,
) -> list[dict[str, str]]:
    """Apparie les clusters entre runs par Hungarian sur cosine des centroïdes.

    Les indices KMeans (P16_cluster_0..3) ne sont pas stables entre seeds.
    On fixe une numérotation canonique sur le run de référence et on remappe
    chaque autre run par appariement optimal des centroïdes (vecteur
    d'expression moyenne sur les gènes communs).

    Renvoie : liste de dict {col_orig → col_canon} pour chaque run.
    """
    from scipy.optimize import linear_sum_assignment

    common_genes = set(ge_runs[reference_idx]["gene"].astype(str))
    for ge in ge_runs:
        common_genes &= set(ge["gene"].astype(str))
    common_genes = sorted(common_genes)
    print(f"[align] gènes communs aux runs : {len(common_genes)}")

    indexed = []
    for ge, cols in zip(ge_runs, mean_cols_per_run):
        sub = ge.set_index(ge["gene"].astype(str)).loc[common_genes, cols]
        sub = sub.astype(float)
        norms = np.linalg.norm(sub.values, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        indexed.append((sub, norms, cols))

    ref_sub, ref_norms, ref_cols = indexed[reference_idx]
    canonical_names = list(ref_cols)
    mappings: list[dict[str, str]] = []
    for run_i, (sub, norms, cols) in enumerate(indexed):
        if run_i == reference_idx:
            mappings.append({c: c for c in cols})
            continue
        cos = (ref_sub.values.T @ sub.values) / (ref_norms.T * norms)
        cost = -cos
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {}
        for r, c in zip(row_ind, col_ind):
            canon = canonical_names[r]
            orig = cols[c]
            mapping[orig] = canon
        mappings.append(mapping)
        print(f"[align] run {run_i} : " + " ; ".join(
            f"{orig}→{canon} (cos={cos[list(canonical_names).index(canon), list(cols).index(orig)]:.3f})"
            for orig, canon in mapping.items()
        ))
    return mappings


def stouffer_combine(pvals: np.ndarray, signs: np.ndarray) -> tuple[float, float]:
    """Combine N p-values bilatérales avec leurs signes via Stouffer.

    Renvoie (z_combined, pval_combined). Suppose poids égaux.
    """
    pvals = np.clip(pvals, 1e-300, 1.0)
    z_one_sided = stats.norm.ppf(1.0 - pvals / 2.0) * signs
    z_comb = z_one_sided.sum() / np.sqrt(len(z_one_sided))
    p_comb = 2.0 * stats.norm.sf(abs(z_comb))
    return float(z_comb), float(p_comb)


def cross_seed_aggregate(per_run_enrich: list[pd.DataFrame]) -> pd.DataFrame:
    """Agrège les enrichissements par (cluster, signature) à travers les runs.

    Chaque run apporte (mean_z, t_stat, pval). On combine via :
      - mean_z_mean / mean_z_std : moyenne et écart-type sur les runs,
      - n_runs : nombre de runs ayant pu calculer la signature,
      - n_runs_sig : nombre de runs avec qval<0.05,
      - z_stouffer / pval_stouffer : combinaison de Stouffer signée.
    """
    cat = pd.concat(
        [df.assign(run=i) for i, df in enumerate(per_run_enrich)],
        ignore_index=True,
    )
    rows = []
    for (cluster, signature), grp in cat.groupby(["cluster", "signature"]):
        means = grp["mean_z"].values
        signs = np.sign(grp["t_stat"].values)
        signs = np.where(signs == 0, 1.0, signs)
        pvals = grp["pval"].values
        z_st, p_st = stouffer_combine(pvals, signs)
        rows.append({
            "cluster": cluster,
            "signature": signature,
            "n_runs": int(len(grp)),
            "n_runs_sig": int((grp["qval"] < 0.05).sum()),
            "n_used_min": int(grp["n_used"].min()),
            "n_used_max": int(grp["n_used"].max()),
            "mean_z_mean": float(means.mean()),
            "mean_z_std": float(means.std(ddof=0)),
            "mean_z_min": float(means.min()),
            "mean_z_max": float(means.max()),
            "z_stouffer": z_st,
            "pval_stouffer": p_st,
        })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    pvals = out["pval_stouffer"].values
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(ranked)
    bh = ranked * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    qvals = np.empty(n)
    qvals[order] = np.clip(bh, 0, 1)
    out["qval_stouffer"] = qvals
    out = out.sort_values(
        ["cluster", "qval_stouffer", "mean_z_mean"],
        ascending=[True, True, False],
    )
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--group-expression",
        type=Path,
        default=REPO_ROOT / "output" / "gnn_vgae" / "V3.3" / "run1" / "group_expression.tsv",
        help="Chemin vers group_expression.tsv (défaut : V3.3/run1)",
    )
    p.add_argument(
        "--cross-seed-runs",
        type=Path,
        nargs="+",
        default=None,
        help="Mode cross-seed : liste de group_expression.tsv (≥2). Si fourni, "
             "ignore --group-expression et calcule l'annotation agrégée.",
    )
    p.add_argument(
        "--reference-run",
        type=int,
        default=0,
        help="Index du run de référence pour l'appariement Hungarian (0-based).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "gnn_vgae" / "V3.3" / "cluster_annotation",
        help="Répertoire de sortie",
    )
    p.add_argument("--min-total-expr", type=float, default=0.05,
                   help="Filtre gènes silencieux (somme expr < seuil)")
    p.add_argument("--min-signature-size", type=int, default=5,
                   help="Taille minimale d'une signature après intersection")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-K signatures par cluster dans dominant_labels.tsv")
    p.add_argument("--qval-thresh", type=float, default=0.05,
                   help="Seuil de q-value pour dominant_labels.tsv")
    return p.parse_args()


def run_single(ge_path: Path, sigs: dict[str, set[str]], args) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Charge un run, calcule z et enrichissement, renvoie (z, enrich, mean_cols)."""
    ge = pd.read_csv(ge_path, sep="\t")
    mean_cols = [c for c in ge.columns if c.startswith("mean_")]
    if not mean_cols:
        sys.exit(f"Aucune colonne mean_* dans {ge_path}")
    z = compute_zscore_matrix(ge, mean_cols, min_total_expr=args.min_total_expr)
    enrich = enrich_signatures(z, sigs, min_signature_size=args.min_signature_size)
    return z, enrich, mean_cols


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] signatures")
    sigs = load_signatures()
    print(f"  {len(sigs)} signatures chargées")
    if not sigs:
        sys.exit("Aucune signature chargée — vérifier data/databases/")

    if args.cross_seed_runs:
        # ── Mode cross-seed ────────────────────────────────────────────────
        if len(args.cross_seed_runs) < 2:
            sys.exit("--cross-seed-runs nécessite ≥2 chemins")

        print(f"[cross-seed] {len(args.cross_seed_runs)} runs")
        ge_runs, mean_cols_per_run, per_run_enrich = [], [], []
        for path in args.cross_seed_runs:
            print(f"  - {path}")
            ge = pd.read_csv(path, sep="\t")
            cols = [c for c in ge.columns if c.startswith("mean_")]
            ge_runs.append(ge)
            mean_cols_per_run.append(cols)

        # Vérification : même schéma de colonnes (modulo ordre)
        ref_set = set(mean_cols_per_run[args.reference_run])
        for i, cols in enumerate(mean_cols_per_run):
            if set(cols) != ref_set:
                print(f"[warn] run {i} schéma différent : {set(cols)^ref_set}")

        # Apparier les clusters
        mappings = align_clusters_hungarian(
            ge_runs, mean_cols_per_run, reference_idx=args.reference_run,
        )

        # Calculer z + enrichissement par run avec colonnes renommées
        for i, (ge, cols, mapping) in enumerate(zip(ge_runs, mean_cols_per_run, mappings)):
            ge_renamed = ge.rename(columns=mapping)
            cols_renamed = [mapping[c] for c in cols]
            z = compute_zscore_matrix(
                ge_renamed[["gene"] + cols_renamed], cols_renamed,
                min_total_expr=args.min_total_expr,
            )
            enrich = enrich_signatures(z, sigs, min_signature_size=args.min_signature_size)
            enrich.to_csv(args.out_dir / f"cluster_annotation_run{i}.tsv", sep="\t", index=False)
            per_run_enrich.append(enrich)
            print(f"  run {i} : {len(enrich)} lignes (cluster × signature)")

        # Agréger
        print("[cross-seed] agrégation Stouffer")
        agg = cross_seed_aggregate(per_run_enrich)
        out_agg = args.out_dir / "cluster_annotation_cross_seed.tsv"
        agg.to_csv(out_agg, sep="\t", index=False)
        print(f"[write] {out_agg}")

        # Mapping documenté
        map_rows = []
        for i, mapping in enumerate(mappings):
            for orig, canon in mapping.items():
                map_rows.append({"run": i, "orig_label": orig, "canonical_label": canon})
        out_map = args.out_dir / "cluster_alignment_map.tsv"
        pd.DataFrame(map_rows).to_csv(out_map, sep="\t", index=False)
        print(f"[write] {out_map}")

        # Top labels cross-seed
        sig_pos = agg.loc[(agg["mean_z_mean"] > 0) & (agg["qval_stouffer"] < args.qval_thresh)]
        sig_pos = sig_pos.sort_values(["cluster", "mean_z_mean"], ascending=[True, False])
        rows = []
        for cluster, grp in sig_pos.groupby("cluster"):
            for rank_, (_, r) in enumerate(grp.head(args.top_k).iterrows(), start=1):
                rows.append({
                    "cluster": cluster,
                    "rank": rank_,
                    "signature": r["signature"],
                    "mean_z_mean": r["mean_z_mean"],
                    "mean_z_std": r["mean_z_std"],
                    "n_runs_sig": r["n_runs_sig"],
                    "qval_stouffer": r["qval_stouffer"],
                })
        labels = pd.DataFrame(rows)
        out_labels = args.out_dir / "cluster_dominant_labels_cross_seed.tsv"
        labels.to_csv(out_labels, sep="\t", index=False)
        print(f"[write] {out_labels}")

        # Heatmap : utiliser mean_z_mean comme valeur, qval_stouffer pour les étoiles
        heat_df = agg.rename(columns={"mean_z_mean": "mean_z", "qval_stouffer": "qval"})
        heatmap(heat_df, args.out_dir / "cluster_annotation_heatmap_cross_seed.png")

        print(f"\n=== Top signatures cross-seed par cluster (q_stouffer < {args.qval_thresh}, "
              f"{len(args.cross_seed_runs)} runs) ===")
        for cluster, grp in labels.groupby("cluster"):
            print(f"\n  {cluster} :")
            for _, row in grp.iterrows():
                print(f"    [{row['rank']}] {row['signature']:55s}  "
                      f"mean_z={row['mean_z_mean']:+.3f}±{row['mean_z_std']:.2f}  "
                      f"q_st={row['qval_stouffer']:.2e}  n_sig={int(row['n_runs_sig'])}/{len(args.cross_seed_runs)}")
        return

    # ── Mode single-run ───────────────────────────────────────────────────
    print(f"[load] group_expression = {args.group_expression}")
    z, enrich, mean_cols = run_single(args.group_expression, sigs, args)
    print(f"  matrice z : {z.shape[0]} gènes × {z.shape[1]} clusters")
    print(f"  enrichissement : {len(enrich)} lignes")

    labels = dominant_labels(enrich, top_k=args.top_k, qval_thresh=args.qval_thresh)

    out_enrich = args.out_dir / "cluster_annotation.tsv"
    out_labels = args.out_dir / "cluster_dominant_labels.tsv"
    out_z = args.out_dir / "gene_cluster_zscore.tsv"
    out_heatmap = args.out_dir / "cluster_annotation_heatmap.png"
    enrich.to_csv(out_enrich, sep="\t", index=False)
    labels.to_csv(out_labels, sep="\t", index=False)
    z.reset_index().rename(columns={"index": "gene"}).to_csv(out_z, sep="\t", index=False)
    print(f"[write] {out_enrich}")
    print(f"[write] {out_labels}")
    print(f"[write] {out_z}")

    heatmap(enrich, out_heatmap)
    if out_heatmap.exists():
        print(f"[write] {out_heatmap}")

    print("\n=== Top signatures par cluster (q < {:.2g}) ===".format(args.qval_thresh))
    for cluster, grp in labels.groupby("cluster"):
        print(f"\n  {cluster} :")
        for _, row in grp.iterrows():
            print(f"    [{row['rank']}] {row['signature']:55s}  mean_z={row['mean_z']:+.3f}  q={row['qval']:.2e}  n={row['n_used']}")


if __name__ == "__main__":
    main()
