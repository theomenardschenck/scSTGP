#!/usr/bin/env python3
"""test_signed_auc.py — TIER 1c.5 gate de succès du décodeur signed V5.

Référence : Liu et al. 2024 *NAR* SGAT-bilinear §3 — protocole standard
d'évaluation du décodeur bilinéaire signé via AUC(activate vs inhibit).

Protocole (selon TODO 1c.5) :
    Split CollecTRI par TF (20 % hold-out) → AUC(activate vs inhibit) sur
    les arêtes du hold-out > 0.85 ⇒ V5 apprend la sémantique du signe ;
    sinon le BilinearSignedDecoder n'est pas rentable.

CAVEAT (in-sample) — l'implémentation V5 actuelle (gnn_vgae.py
:signed_pos_pool) utilise TOUTES les arêtes signées dans la
`signed_aux_loss` à chaque epoch. Donc tout test post-hoc sur un run V5
existant est **in-sample** : les signs des TFs « hold-out » ont en fait
été vus par la loss pendant l'entraînement.

Trois métriques sont calculées pour distinguer mémorisation et
généralisation :

1. **AUC global in-sample** sur l'ensemble des arêtes signées par
   edge_type. Sanity check : si < 0.6, le décodeur n'a rien appris.

2. **TF-stratified split AUC** (`--n-splits`, défaut 100) : tirages
   aléatoires de 80/20 TFs ; on calcule l'AUC sur les arêtes des TFs
   « hold-out ». Toujours in-sample, MAIS donne une estimation
   pessimiste si les TFs ont des distributions de signe variables —
   isole l'effet « le modèle a appris une fonction sign générale » vs
   « le modèle a mémorisé chaque arête ».

3. **Per-TF AUC** (top-K TFs par nombre d'arêtes) : si la moyenne
   globale est haute mais que la majorité des TFs individuels ont
   AUC≈0.5, c'est de la mémorisation. Si la distribution est tassée
   autour de la moyenne, c'est de la généralisation locale.

Pour un VRAI hold-out (gate 1c.5 strict), il faut un re-train avec
`--holdout-signed-tf-fraction X` (à ajouter à gnn_vgae.py — TODO 1c.5
phase 2). Pour l'instant on caveate.

Usage
-----
    python src/validation/explain/test_signed_auc.py \\
        --run-dir output/gnn_vgae/V5/full/v5-full.s1 \\
        --collectri-cache data/omnipath/tf_collectri.tsv.gz \\
        --n-splits 100 --holdout-frac 0.2 --holdout-seed 42 \\
        --out-dir output/gnn_vgae/V5/full/v5-full.s1/test_signed_auc

    # Cross-seed (mean ± std AUC sur 3 seeds v5-full) :
    for s in 1 2 3; do
        python src/validation/explain/test_signed_auc.py \\
            --run-dir output/gnn_vgae/V5/full/v5-full.s$s \\
            --out-dir output/gnn_vgae/V5/full/v5-full.s$s/test_signed_auc
    done
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bootstrap : importer SignedGATConv / BilinearSignedDecoder / HeteroEncoder
# depuis src/gnn/_vgae_model.py (Tier 2.5 import-safe).
# ---------------------------------------------------------------------------
def _bootstrap_paths():
    here = Path(__file__).resolve()
    project_root = here.parents[3] if len(here.parents) > 3 else here.parents[1]
    for p in [project_root / "src", project_root]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


_bootstrap_paths()
from gnn._vgae_model import (  # noqa: E402
    SIGNED_EDGE_TYPES,
    HeteroEncoder,
    VGAE,
    BilinearSignedDecoder,
)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SIGNED_RELS = {"signaling", "tf_curated", "tf_curated_by", "reactome_fi"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_v5(state_dict: dict) -> tuple[bool, bool]:
    """Détecte si le run est V5 (signed_message / signed_decoder) depuis
    les clés du state_dict.

    - signed_decoder : présence de `bilinear_decoder.W_pos`, etc.
    - signed_message : pas détectable depuis le state_dict pur (la classe
      SignedGATConv ajoute pas de paramètre, juste un buffer non
      persistant). On infère via le run_config.json si dispo.
    """
    has_bilin = any(k.startswith("bilinear_decoder.") for k in state_dict.keys())
    return has_bilin


def _infer_hyperparams(state_dict: dict, gene_in: int, cell_in: int) -> dict:
    """Infère hidden / latent / n_layers / n_heads / head_dim depuis le state."""
    hidden = state_dict["encoder.gene_proj.weight"].shape[0]
    latent = state_dict["encoder.mu_head.weight"].shape[0]
    n_layers = sum(1 for k in state_dict
                   if k.startswith("encoder.norms.") and k.endswith(".gene.weight"))
    att_keys = [k for k in state_dict if k.endswith(".att_src")]
    if not att_keys:
        raise RuntimeError("state_dict sans .att_src — modèle non GAT ?")
    att_shape = state_dict[att_keys[0]].shape  # (1, n_heads, head_dim)
    n_heads = int(att_shape[1])
    head_dim = int(att_shape[2])
    assert hidden == n_heads * head_dim, \
        f"hidden ({hidden}) != n_heads × head_dim ({n_heads}×{head_dim})"
    return dict(hidden=hidden, latent=latent, n_layers=n_layers,
                n_heads=n_heads, head_dim=head_dim,
                gene_in=gene_in, cell_in=cell_in)


def load_run(run_dir: Path) -> tuple[VGAE, "HeteroData", list[str]]:
    """Charge graph + modèle V5 (avec BilinearSignedDecoder), gene_symbols."""
    graph_path = run_dir / "hetero_graph_vgae.pt"
    weights_path = run_dir / "vgae_weights.pt"
    if not graph_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing hetero_graph_vgae.pt ou vgae_weights.pt dans {run_dir}")

    data = torch.load(graph_path, weights_only=False, map_location="cpu")
    state = torch.load(weights_path, weights_only=False, map_location="cpu")

    has_bilin = _detect_v5(state)
    if not has_bilin:
        raise RuntimeError(
            f"{run_dir} : pas de bilinear_decoder.* dans state_dict — "
            f"le run n'a pas été entraîné avec --signed-decoder."
        )

    gene_in = data["gene"].x.shape[1]
    cell_in = data["cell_group"].x.shape[1] if "cell_group" in data.node_types else 3
    hp = _infer_hyperparams(state, gene_in, cell_in)

    encoder = HeteroEncoder(
        gene_in=hp["gene_in"], cell_in=hp["cell_in"],
        hidden=hp["hidden"], latent=hp["latent"],
        n_layers=hp["n_layers"], n_heads=hp["n_heads"],
        available_edge_types=list(data.edge_types),
        signed_message=True,  # inoffensif si run sans signed-message :
        # SignedGATConv hérite GATConv et restera comportement identique
        # quand _current_edge_sign est None.
    )
    bilin = BilinearSignedDecoder(latent_dim=hp["latent"])
    model = VGAE(encoder, bilinear_decoder=bilin)
    model.load_state_dict(state, strict=True)
    model.eval()

    # gene_symbols : conservé soit dans data['gene'].gene_symbols, soit
    # via gene_embeddings_vgae.csv (1ère colonne = symbol, ordre = tensor).
    symbols: list[str] | None = None
    if hasattr(data["gene"], "gene_symbols"):
        symbols = list(data["gene"].gene_symbols)
    elif (run_dir / "gene_embeddings_vgae.csv").exists():
        # Format : `Unnamed: 0,0,1,2,…,63` ou `gene,0,1,…` selon version.
        # La 1ère colonne est le symbol, l'ordre suit `data["gene"].x`.
        df = pd.read_csv(run_dir / "gene_embeddings_vgae.csv", usecols=[0])
        symbols = df.iloc[:, 0].astype(str).tolist()
    if symbols is None:
        raise RuntimeError(f"Impossible de récupérer gene_symbols pour {run_dir}")

    n_genes = data["gene"].x.shape[0]
    assert len(symbols) == n_genes, \
        f"len(symbols)={len(symbols)} != n_genes={n_genes}"
    return model, data, symbols


def collect_signed_edges(data, symbols: list[str]) -> pd.DataFrame:
    """Concatène toutes les arêtes signées dans un DataFrame long-format.

    Colonnes : edge_type (rel), src_idx, dst_idx, src_sym, dst_sym, sign.
    Seules les arêtes avec sign != 0 sont retenues.
    """
    rows = []
    for et in data.edge_types:
        rel = et[1]
        if rel not in SIGNED_RELS:
            continue
        ei = data[et].edge_index
        ea = getattr(data[et], "edge_attr", None)
        if ea is None or ea.ndim < 2 or ea.shape[1] < 2:
            continue
        sign = ea[:, 1].numpy()
        src = ei[0].numpy()
        dst = ei[1].numpy()
        mask = sign != 0
        for s, d, sg in zip(src[mask], dst[mask], sign[mask]):
            rows.append((rel, int(s), int(d), symbols[s], symbols[d], float(sg)))
    df = pd.DataFrame(rows, columns=["edge_type", "src_idx", "dst_idx",
                                     "src_sym", "dst_sym", "sign"])
    return df


def encode_full(model: VGAE, data) -> torch.Tensor:
    """Forward de l'encoder en eval mode → z (= μ, pas d'échantillonnage)."""
    x_dict = {"gene": data["gene"].x, "cell_group": data["cell_group"].x}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types
                       if data[et].edge_index.numel() > 0}
    edge_attr_dict = {et: data[et].edge_attr for et in data.edge_types
                      if getattr(data[et], "edge_attr", None) is not None}
    with torch.no_grad():
        z, mu, _ = model.encode(x_dict, edge_index_dict, edge_attr_dict)
    # En eval, model.reparametrize retourne mu (cf. _vgae_model.py). On
    # utilise mu directement pour la stabilité (pas d'échantillonnage).
    return mu


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC robuste — retourne NaN si une seule classe."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Analyses principales
# ---------------------------------------------------------------------------
def auc_per_edge_type(
    edges: pd.DataFrame, z: torch.Tensor, model: VGAE
) -> pd.DataFrame:
    """AUC + AUPR in-sample par edge_type.

    Score sign-agnostique : `predict_sign_score(z, ei) = logit_pos − logit_neg`.
    """
    rows = []
    for rel, sub in edges.groupby("edge_type"):
        src = torch.tensor(sub["src_idx"].values, dtype=torch.long)
        dst = torch.tensor(sub["dst_idx"].values, dtype=torch.long)
        ei = torch.stack([src, dst])
        with torch.no_grad():
            score = model.bilinear_decoder.predict_sign_score(z, ei)
        y_true = (sub["sign"].values > 0).astype(int)
        y_score = score.numpy()
        auc = _safe_auc(y_true, y_score)
        aupr = _safe_aupr(y_true, y_score)
        n_pos = int((sub["sign"] > 0).sum())
        n_neg = int((sub["sign"] < 0).sum())
        baseline_aupr = n_pos / (n_pos + n_neg) if n_pos + n_neg > 0 else float("nan")
        rows.append({
            "edge_type": rel,
            "n_edges": len(sub),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "frac_pos": round(n_pos / (n_pos + n_neg), 3),
            "auc_insample": round(auc, 4),
            "aupr_insample": round(aupr, 4),
            "aupr_baseline": round(baseline_aupr, 4),
        })
    return pd.DataFrame(rows)


def auc_tf_stratified(
    edges: pd.DataFrame, z: torch.Tensor, model: VGAE,
    n_splits: int, holdout_frac: float, seed: int,
) -> pd.DataFrame:
    """TF-stratified split : tire aléatoirement holdout_frac des TFs,
    calcule AUC sur leurs arêtes, répète n_splits fois.

    Retourne : 1 ligne par edge_type avec auc_holdout_mean ± std (+ idem AUPR).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rel, sub in edges.groupby("edge_type"):
        tfs = sub["src_sym"].unique()
        if len(tfs) < 5:
            rows.append({
                "edge_type": rel, "n_tfs": len(tfs),
                "auc_holdout_mean": float("nan"), "auc_holdout_std": float("nan"),
                "aupr_holdout_mean": float("nan"), "aupr_holdout_std": float("nan"),
                "n_splits_used": 0, "n_splits_skipped": n_splits,
            })
            continue

        aucs, auprs = [], []
        skipped = 0
        for _ in range(n_splits):
            ho_tfs = rng.choice(tfs, size=max(1, int(len(tfs) * holdout_frac)),
                                replace=False)
            ho = sub[sub["src_sym"].isin(ho_tfs)]
            y_true = (ho["sign"].values > 0).astype(int)
            if len(np.unique(y_true)) < 2:
                skipped += 1
                continue
            src = torch.tensor(ho["src_idx"].values, dtype=torch.long)
            dst = torch.tensor(ho["dst_idx"].values, dtype=torch.long)
            ei = torch.stack([src, dst])
            with torch.no_grad():
                score = model.bilinear_decoder.predict_sign_score(z, ei)
            y_score = score.numpy()
            aucs.append(_safe_auc(y_true, y_score))
            auprs.append(_safe_aupr(y_true, y_score))
        rows.append({
            "edge_type": rel,
            "n_tfs": len(tfs),
            "auc_holdout_mean": round(float(np.mean(aucs)), 4) if aucs else float("nan"),
            "auc_holdout_std": round(float(np.std(aucs)), 4) if aucs else float("nan"),
            "aupr_holdout_mean": round(float(np.mean(auprs)), 4) if auprs else float("nan"),
            "aupr_holdout_std": round(float(np.std(auprs)), 4) if auprs else float("nan"),
            "n_splits_used": len(aucs),
            "n_splits_skipped": skipped,
        })
    return pd.DataFrame(rows)


def auc_per_tf(
    edges: pd.DataFrame, z: torch.Tensor, model: VGAE,
    min_edges: int = 5,
) -> pd.DataFrame:
    """AUC in-sample par TF (uniquement TFs avec >=min_edges et au moins
    une arête de chaque signe).
    """
    rows = []
    for (rel, tf), sub in edges.groupby(["edge_type", "src_sym"]):
        if len(sub) < min_edges:
            continue
        if sub["sign"].gt(0).sum() == 0 or sub["sign"].lt(0).sum() == 0:
            continue
        src = torch.tensor(sub["src_idx"].values, dtype=torch.long)
        dst = torch.tensor(sub["dst_idx"].values, dtype=torch.long)
        ei = torch.stack([src, dst])
        with torch.no_grad():
            score = model.bilinear_decoder.predict_sign_score(z, ei)
        y_true = (sub["sign"].values > 0).astype(int)
        y_score = score.numpy()
        rows.append({
            "edge_type": rel,
            "tf": tf,
            "n_edges": len(sub),
            "n_pos": int((sub["sign"] > 0).sum()),
            "n_neg": int((sub["sign"] < 0).sum()),
            "auc_insample": round(_safe_auc(y_true, y_score), 4),
        })
    return pd.DataFrame(rows).sort_values(["edge_type", "auc_insample"],
                                          ascending=[True, False])


# ---------------------------------------------------------------------------
# Rapport markdown
# ---------------------------------------------------------------------------
def emit_report(
    out_dir: Path, run_dir: Path,
    overall: pd.DataFrame, holdout: pd.DataFrame, per_tf: pd.DataFrame,
    gate_threshold: float, holdout_frac: float, n_splits: int,
    effective_mode: str = "in-sample",
    n_holdout_tfs: int = 0,
) -> None:
    lines = []
    lines.append("# Test signed decoder V5 — gate 1c.5\n")
    lines.append(f"Run analysé : `{run_dir}`\n")
    lines.append(f"Mode      : **{effective_mode}**"
                 + (f"  (set hold-out : {n_holdout_tfs} TFs réservés à "
                    f"l'entraînement)" if effective_mode == "holdout" else "")
                 + "\n")
    lines.append(f"Gate de succès (Liu 2024 SGAT-bilinear §3) : "
                 f"**AUC > {gate_threshold:.2f}** sur hold-out TF-stratifié.\n")
    if effective_mode == "holdout":
        lines.append("\n## ✓ Test rigoureux (phase 2)\n")
        lines.append(
            "Ce run a été entraîné avec `--holdout-signed-tf-fraction > 0` : "
            "les signs des TFs hold-out **n'ont PAS été vus par la "
            "`signed_aux_loss`**. Les arêtes incidentes au set hold-out (lu "
            "depuis `run_config.json:holdout_signed_tf_set`) sont évaluées "
            "ici comme un VRAI hold-out → test de généralisation à des TFs "
            "jamais utilisés pour l'apprentissage du signe.\n"
        )
    else:
        lines.append("\n## ⚠ Caveat in-sample\n")
        lines.append(
            "Ce run n'a pas de hold-out signed entraîné "
            "(`run_config.json:holdout_signed_tf_set` vide). La "
            "`signed_aux_loss` a vu TOUTES les arêtes signées → ce test est "
            "*in-sample*, y compris le « hold-out » TF-stratifié ci-dessous "
            "qui partitionne seulement les arêtes à l'évaluation, pas leur "
            "sign label.\n\n"
            "Pour un test rigoureux, re-train avec "
            "`--holdout-signed-tf-fraction X` (TIER 1c.5 phase 2).\n"
        )

    # 1. AUC par edge_type
    lines.append("\n## 1. AUC global in-sample par edge_type\n")
    lines.append("```\n" + overall.to_string(index=False) + "\n```")
    lines.append("\n")
    pass_overall = (overall["auc_insample"] >= gate_threshold).all()
    lines.append(f"Toutes les edge_types ≥ {gate_threshold:.2f} : "
                 f"**{'OUI ✓' if pass_overall else 'NON ✗'}**.\n")

    # 2. TF-stratified hold-out
    lines.append(f"\n## 2. TF-stratified hold-out ({n_splits} splits × "
                 f"{holdout_frac:.0%} TFs)\n")
    lines.append("```\n" + holdout.to_string(index=False) + "\n```")
    lines.append("\n")
    pass_ho = (holdout["auc_holdout_mean"] >= gate_threshold).all()
    lines.append(f"Toutes les edge_types ≥ {gate_threshold:.2f} : "
                 f"**{'OUI ✓' if pass_ho else 'NON ✗'}**.\n")

    # 3. Per-TF distribution
    lines.append("\n## 3. Per-TF AUC distribution (in-sample)\n")
    if per_tf.empty:
        lines.append("(aucun TF avec ≥5 arêtes des deux signes — distribution non calculable)\n")
    else:
        for rel, sub in per_tf.groupby("edge_type"):
            aucs = sub["auc_insample"].values
            lines.append(f"### {rel}\n")
            lines.append(f"- n TFs évaluables : {len(sub)}\n")
            lines.append(f"- AUC moyenne : {np.mean(aucs):.3f}\n")
            lines.append(f"- AUC médiane : {np.median(aucs):.3f}\n")
            lines.append(f"- AUC σ      : {np.std(aucs):.3f}\n")
            lines.append(f"- AUC min   : {np.min(aucs):.3f}\n")
            lines.append(f"- AUC max   : {np.max(aucs):.3f}\n")
            frac_pass = (aucs >= gate_threshold).mean()
            lines.append(f"- fraction TF ≥ {gate_threshold:.2f} : "
                         f"**{frac_pass:.1%}**\n")
            # top + bottom 5
            top = sub.nlargest(5, "auc_insample")[["tf", "n_edges", "n_pos", "n_neg", "auc_insample"]]
            bot = sub.nsmallest(5, "auc_insample")[["tf", "n_edges", "n_pos", "n_neg", "auc_insample"]]
            lines.append(f"\nTop-5 TFs (meilleur AUC) :\n")
            lines.append("```\n" + top.to_string(index=False) + "\n```")
            lines.append("\n\nBottom-5 TFs (pire AUC) :\n")
            lines.append("```\n" + bot.to_string(index=False) + "\n```")
            lines.append("\n")

    # Verdict
    lines.append("\n## Verdict gate 1c.5\n")
    if pass_overall and pass_ho:
        lines.append(f"✅ **GATE PASSÉE** : AUC global ≥ {gate_threshold:.2f} ET "
                     f"AUC hold-out TF ≥ {gate_threshold:.2f} sur toutes les "
                     f"edge_types.\n")
        lines.append("Le décodeur signed apprend une sémantique du signe "
                     "exploitable. Garder `--signed-decoder` en config V5.\n")
    elif pass_overall:
        lines.append(f"⚠ **GATE PARTIELLE** : AUC global ≥ {gate_threshold:.2f} "
                     f"mais hold-out TF < {gate_threshold:.2f} sur certaines "
                     f"edge_types — risque de mémorisation.\n")
    else:
        lines.append(f"❌ **GATE NON PASSÉE** : AUC global < {gate_threshold:.2f}. "
                     f"Le décodeur signed ne distingue pas activate vs "
                     f"inhibit. Reconsidérer `--signed-decoder`.\n")
    lines.append("\nRappel : ce test est *in-sample* — pour validation "
                 "rigoureuse, re-train avec hold-out signed.\n")

    (out_dir / "signed_auc_report.md").write_text("".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="Dossier d'un run V5 entraîné (avec bilinear_decoder).")
    p.add_argument("--out-dir", default=None,
                   help="Sortie. Défaut : <run-dir>/test_signed_auc/")
    p.add_argument("--n-splits", type=int, default=100,
                   help="Nombre de splits TF-stratifiés (défaut 100).")
    p.add_argument("--holdout-frac", type=float, default=0.2,
                   help="Fraction TFs hold-out par split (défaut 0.2).")
    p.add_argument("--holdout-seed", type=int, default=42,
                   help="Seed RNG pour les splits TF-stratifiés.")
    p.add_argument("--gate-threshold", type=float, default=0.85,
                   help="Seuil AUC du gate 1c.5 (défaut 0.85, Liu 2024 NAR).")
    p.add_argument("--min-edges-per-tf", type=int, default=5,
                   help="Min arêtes par TF pour calculer son AUC individuel.")
    p.add_argument("--mode",
                   choices=["auto", "in-sample", "holdout"],
                   default="auto",
                   help="Phase 2 1c.5 strict : si `holdout`, restreint "
                        "l'évaluation aux seules arêtes hold-out persistées "
                        "dans run_config.json (vrai test de généralisation). "
                        "`in-sample` = ignore le set hold-out (test V5.1 "
                        "classique). `auto` (défaut) = `holdout` si le set "
                        "est non vide dans run_config.json, sinon "
                        "`in-sample`.")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else \
        run_dir / "test_signed_auc"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lecture du run_config.json : récupère le set hold-out si présent.
    holdout_tf_set: set[str] = set()
    holdout_seed_used: int | None = None
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        try:
            import json as _j
            with open(cfg_path) as _fh:
                _cfg = _j.load(_fh)
            holdout_tf_set = set(_cfg.get("holdout_signed_tf_set", []) or [])
            holdout_seed_used = _cfg.get("holdout_signed_tf_seed_used", None)
        except Exception as e:
            print(f"[warn] échec lecture run_config.json : {e}")

    # Résolution du mode effectif
    if args.mode == "auto":
        effective_mode = "holdout" if holdout_tf_set else "in-sample"
    else:
        effective_mode = args.mode
    if effective_mode == "holdout" and not holdout_tf_set:
        print(f"[err] --mode holdout demandé mais run_config.json sans set "
              f"hold-out. Re-train avec --holdout-signed-tf-fraction X > 0.")
        sys.exit(1)

    print(f"[1c.5] run_dir = {run_dir}")
    print(f"[1c.5] out_dir = {out_dir}")
    print(f"[1c.5] gate    = AUC > {args.gate_threshold}")
    print(f"[1c.5] mode    = {effective_mode}"
          + (f"  (hold-out : {len(holdout_tf_set)} TFs, seed_used="
             f"{holdout_seed_used})" if effective_mode == "holdout" else ""))

    model, data, symbols = load_run(run_dir)
    edges = collect_signed_edges(data, symbols)
    if edges.empty:
        print("[err] Aucune arête signée trouvée dans le graphe.")
        sys.exit(1)
    n_total = len(edges)

    # Filtrage hold-out : ne garder QUE les arêtes incidentes aux TFs hold-out.
    # Convention training (gnn_vgae.py:signed_holdout_pool) : edge hold-out
    # si src OR dst ∈ holdout_tf_set. On reproduit ici.
    if effective_mode == "holdout":
        mask = (edges["src_sym"].isin(holdout_tf_set)
                | edges["dst_sym"].isin(holdout_tf_set))
        edges = edges[mask].reset_index(drop=True)
        if edges.empty:
            print(f"[err] Aucune arête signée incidente au set hold-out "
                  f"({len(holdout_tf_set)} TFs).")
            sys.exit(1)
    print(f"[1c.5] {len(edges)}/{n_total} arêtes signées retenues "
          f"({effective_mode}) sur {edges['edge_type'].nunique()} edge_types : "
          f"{dict(edges['edge_type'].value_counts())}")

    z = encode_full(model, data)
    print(f"[1c.5] embeddings z shape = {tuple(z.shape)}")

    # 1. AUC global par edge_type
    overall = auc_per_edge_type(edges, z, model)
    overall.to_csv(out_dir / "signed_auc_overall.tsv", sep="\t", index=False)
    print(f"\n[1c.5] AUC global in-sample par edge_type :")
    print(overall.to_string(index=False))

    # 2. TF-stratified hold-out
    print(f"\n[1c.5] TF-stratified hold-out — {args.n_splits} splits "
          f"× {args.holdout_frac:.0%} TFs (seed={args.holdout_seed})…")
    holdout = auc_tf_stratified(edges, z, model, args.n_splits,
                                args.holdout_frac, args.holdout_seed)
    holdout.to_csv(out_dir / "signed_auc_holdout.tsv", sep="\t", index=False)
    print(holdout.to_string(index=False))

    # 3. Per-TF AUC distribution
    per_tf = auc_per_tf(edges, z, model, min_edges=args.min_edges_per_tf)
    per_tf.to_csv(out_dir / "signed_auc_per_tf.tsv", sep="\t", index=False)
    print(f"\n[1c.5] {len(per_tf)} TFs évaluables (≥{args.min_edges_per_tf} "
          f"arêtes des deux signes).")

    emit_report(out_dir, run_dir, overall, holdout, per_tf,
                args.gate_threshold, args.holdout_frac, args.n_splits,
                effective_mode=effective_mode,
                n_holdout_tfs=len(holdout_tf_set))
    print(f"\n[1c.5] Rapport : {out_dir}/signed_auc_report.md")


if __name__ == "__main__":
    main()
