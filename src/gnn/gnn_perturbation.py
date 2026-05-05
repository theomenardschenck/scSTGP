#!/usr/bin/env python3
"""
gnn_perturbation.py — In silico perturbation of the trained VGAE.

Loads a trained VGAE run (output/gnn_vgae/V*_Run*/) and simulates
knockdown / knockout / over-expression of one gene, a set of genes, or
a full REACTOME pathway. Runs a forward pass through the FROZEN encoder
(no retraining), recomputes the composite importance score and reports:

  * delta_ranking.csv       — per-gene baseline vs perturbed rank & importance
  * delta_ora_top_up_reactome.tsv — REACTOME ORA on the top rising genes
  * summary.json            — top movers, stats, signature delta-pathways

Perturbation modes
------------------
  knockdown   : gene feature row -> 0
  overexpress : gene feature row *= FACTOR (default 2.0)
  knockout    : knockdown + remove every edge incident to the gene

Usage
-----
    # Single-gene knockdown on V3_Run2
    python src/gnn_perturbation.py \\
        --run-dir output/gnn_vgae/V3_Run2 \\
        --mode knockdown \\
        --genes PEX11B

    # Pathway-level knockout (gene symbols read from a file)
    python src/gnn_perturbation.py \\
        --run-dir output/gnn_vgae/V3_Run2 \\
        --mode knockout \\
        --gene-list docs/peroxisome_genes.txt \\
        --out-dir output/perturbation/peroxisome_KO

    # Over-expression with a custom factor
    python src/gnn_perturbation.py \\
        --run-dir output/gnn_vgae/V3_Run2 \\
        --mode overexpress \\
        --genes ADM2 --factor 3.0

Design notes
------------
- The 5th importance component (`specificity`) is computed from RNAseq
  expression (entropy across cell_groups), so it is invariant under gene
  feature perturbation. We therefore reuse the baseline `vgae_specificity`
  column instead of recomputing it — keeping the perturbed score directly
  comparable to the baseline published by `gnn_vgae.py`.
- The encoder hyperparameters (hidden/latent/n_layers/n_heads) default to
  the training-script defaults. Override via CLI if a run used different
  values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GATConv, HeteroConv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ora_consensus import (  # noqa: E402
    load_background,
    load_reactome_gmt,
    run_ora,
    write_tsv,
)

DEFAULT_HIDDEN = 128
DEFAULT_LATENT = 64
DEFAULT_LAYERS = 3
DEFAULT_HEADS = 4

# Ordre FIXE des 5 noeuds cell_group, tel que construit dans gnn_vgae.py:468.
# L'index dans ce tuple = l'index du noeud cell_group dans le HeteroData.
CELL_GROUPS = ("P4", "P16_cluster_0", "P16_cluster_1",
               "P16_cluster_2", "P16_cluster_3")


# --------------------------------------------------------------------------- #
# Model (duplicated from gnn_vgae.py to keep this tool self-contained).
# --------------------------------------------------------------------------- #
class HeteroEncoder(nn.Module):
    # Catalogue de tous les edge_types possibles (V3.6+) — mirroir de
    # gnn_vgae.HeteroEncoder.EDGE_TYPE_CATALOG. Filtré dynamiquement à
    # l'init selon les types réellement présents dans le graphe entraîné
    # (modularité : un modèle --no-ppi n'a pas de poids gene__ppi__gene,
    # le state_dict serait incompatible avec un encoder qui les instancie).
    EDGE_TYPE_CATALOG = [
        (("gene", "ppi", "gene"), 1),
        (("gene", "same_pathway", "gene"), None),
        (("gene", "regulates", "gene"), 1),
        (("gene", "regulated_by", "gene"), 1),
        (("cell_group", "expresses", "gene"), 7),
        (("gene", "expressed_in", "cell_group"), 7),
        (("gene", "coexpression", "gene"), 1),
        (("gene", "metabolic_cocatalysis", "gene"), 2),
    ]

    def __init__(self, gene_in, cell_in, hidden, latent, n_layers,
                 n_heads=4, dropout=0.2, available_edge_types=None):
        super().__init__()
        self.n_layers = n_layers
        head_dim = hidden // n_heads
        self.gene_proj = nn.Linear(gene_in, hidden)
        self.cell_proj = nn.Linear(cell_in, hidden)

        if available_edge_types is None:
            edge_types_dims = list(self.EDGE_TYPE_CATALOG)
        else:
            available = {tuple(et) for et in available_edge_types}
            edge_types_dims = [(et, dim) for et, dim in self.EDGE_TYPE_CATALOG
                               if et in available]
            unknown = available - {et for et, _ in self.EDGE_TYPE_CATALOG}
            if unknown:
                print(f"[warn] HeteroEncoder : edge_types inconnus dans le "
                      f"catalogue : {unknown}")
        if not edge_types_dims:
            raise ValueError(
                "HeteroEncoder : aucun edge_type actif. Vérifie le graphe "
                "sauvegardé (data.edge_types)."
            )
        self.edge_dims = {et: dim for et, dim in edge_types_dims}
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {}
            for et, ed in edge_types_dims:
                conv_kwargs = dict(heads=n_heads, concat=True,
                                   dropout=dropout, add_self_loops=False)
                if ed is not None:
                    conv_kwargs["edge_dim"] = ed
                conv_dict[et] = GATConv(hidden, head_dim, **conv_kwargs)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({
                "gene": nn.BatchNorm1d(hidden),
                "cell_group": nn.BatchNorm1d(hidden),
            }))
        self.dropout = nn.Dropout(dropout)
        self.mu_head = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)
        # Stash du hidden cell_group après le dernier forward. Utilisé par
        # l'option A (shift per cell_group après perturbation) sans changer
        # la signature de forward() / VGAE.encode() / compute_importance().
        self.last_cell_group_h: torch.Tensor | None = None

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = {
            "gene": F.relu(self.gene_proj(x_dict["gene"])),
            "cell_group": F.relu(self.cell_proj(x_dict["cell_group"])),
        }
        for i in range(self.n_layers):
            x_prev = {k: v.clone() for k, v in x_dict.items()}
            active = {k: v for k, v in edge_index_dict.items() if v.numel() > 0}
            if edge_attr_dict is not None:
                active_attrs = {
                    k: edge_attr_dict[k]
                    for k in active
                    if k in edge_attr_dict and self.edge_dims.get(k) is not None
                }
                x_dict = self.convs[i](x_dict, active,
                                       edge_attr_dict=active_attrs)
            else:
                x_dict = self.convs[i](x_dict, active)
            for key in x_dict:
                x_dict[key] = self.norms[i][key](x_dict[key])
                x_dict[key] = F.relu(x_dict[key])
                x_dict[key] = self.dropout(x_dict[key])
                x_dict[key] = x_dict[key] + x_prev[key]
        self.last_cell_group_h = x_dict["cell_group"].detach().clone()
        return self.mu_head(x_dict["gene"]), self.logvar_head(x_dict["gene"])


class VGAE(nn.Module):
    def __init__(self, encoder, tau_init=2.0, tau_max=3.0):
        super().__init__()
        self.encoder = encoder
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau_init))))
        self.log_tau_max = np.log(tau_max)

    def encode(self, x_dict, edge_index_dict, edge_attr_dict=None):
        mu, logvar = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        return mu, logvar

    def decode(self, z, edge_index):
        src, dst = edge_index
        z_src = F.normalize(z[src], dim=1)
        z_dst = F.normalize(z[dst], dim=1)
        cos_sim = (z_src * z_dst).sum(dim=1)
        log_tau = torch.clamp(self.log_tau, max=self.log_tau_max)
        return torch.exp(log_tau) * cos_sim


# --------------------------------------------------------------------------- #
# Importance score (mirrors gnn_vgae.py §11).
# --------------------------------------------------------------------------- #
def compute_importance(model, data, baseline_specificity):
    model.eval()
    x_dict = {"gene": data["gene"].x, "cell_group": data["cell_group"].x}
    edge_index_dict = {k: data[k].edge_index for k in data.edge_types}
    edge_attr_dict = {
        k: data[k].edge_attr
        for k in data.edge_types
        if "edge_attr" in data[k] and data[k].edge_attr is not None
    }

    with torch.no_grad():
        mu, logvar = model.encode(x_dict, edge_index_dict, edge_attr_dict)

    mu_np = mu.cpu().numpy()
    n_nodes = mu_np.shape[0]

    # (1) latent norm
    emb_norm = np.linalg.norm(mu_np, axis=1)
    emb_norm_score = emb_norm / (emb_norm.max() + 1e-8)

    # (2) reconstruction fidelity averaged over all gene↔gene edge types
    gene_edges = []
    for et in data.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            ei = data[et].edge_index
            if ei.numel() > 0:
                gene_edges.append(ei)
    if gene_edges:
        all_ei = torch.cat(gene_edges, dim=1)
        with torch.no_grad():
            sc = torch.sigmoid(model.decode(mu, all_ei)).cpu().numpy()
        src = all_ei[0].cpu().numpy()
        dst = all_ei[1].cpu().numpy()
        err = np.zeros(n_nodes, dtype=np.float32)
        cnt = np.zeros(n_nodes, dtype=np.float32)
        np.add.at(err, src, 1.0 - sc)
        np.add.at(err, dst, 1.0 - sc)
        np.add.at(cnt, src, 1.0)
        np.add.at(cnt, dst, 1.0)
        safe = np.where(cnt > 0, cnt, 1.0)
        recon_err = np.where(cnt > 0, err / safe, 0.0)
    else:
        recon_err = np.zeros(n_nodes, dtype=np.float32)
    recon_err_score = recon_err / (recon_err.max() + 1e-8)

    # (3) uncertainty from logvar
    sigma = torch.exp(0.5 * logvar).cpu().numpy()
    uncertainty = sigma.mean(axis=1)
    uncertainty_score = uncertainty / (uncertainty.max() + 1e-8)

    # (4) local density via cosine k-NN + variance regularisation
    k = min(20, n_nodes - 1)
    emb_normed = mu_np / (np.linalg.norm(mu_np, axis=1, keepdims=True) + 1e-8)
    knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    knn.fit(emb_normed)
    dists, _ = knn.kneighbors(emb_normed)
    dists = dists[:, 1:]
    raw_density = 1.0 / (dists.mean(axis=1) + 1e-8)
    density = rankdata(raw_density) / len(raw_density)
    variance = rankdata(dists.var(axis=1)) / len(dists)
    density = density * variance
    density_score = (density / (density.max() + 1e-8)).astype(np.float32)

    # (5) specificity — invariant under gene feature perturbation
    importance = (
        emb_norm_score
        + density_score
        + (1.0 - recon_err_score)
        + (1.0 - uncertainty_score)
        + baseline_specificity
    ) / 5.0
    importance = importance / (importance.max() + 1e-8)

    return {
        "importance": importance.astype(np.float32),
        "emb_norm": emb_norm_score.astype(np.float32),
        "density": density_score.astype(np.float32),
        "recon_fidelity": (1.0 - recon_err_score).astype(np.float32),
        "certainty": (1.0 - uncertainty_score).astype(np.float32),
        "mu": mu_np.astype(np.float32),   # (n_genes, latent_dim) — requis par
                                           # cell_group_shift_gene_weighted()
    }


# --------------------------------------------------------------------------- #
# Perturbation.
# --------------------------------------------------------------------------- #
def apply_perturbation(data, target_idx: torch.Tensor, mode: str, factor: float):
    pert = data.clone()
    x = pert["gene"].x.clone()
    if mode == "knockdown":
        x[target_idx] = 0.0
    elif mode == "overexpress":
        x[target_idx] = x[target_idx] * factor
    elif mode == "knockout":
        x[target_idx] = 0.0
    else:
        raise ValueError(f"unknown mode: {mode}")
    pert["gene"].x = x

    if mode == "knockout":
        for et in list(pert.edge_types):
            ei = pert[et].edge_index
            if ei is None or ei.numel() == 0:
                continue
            keep = torch.ones(ei.shape[1], dtype=torch.bool)
            if et[0] == "gene":
                keep &= ~torch.isin(ei[0], target_idx)
            if et[2] == "gene":
                keep &= ~torch.isin(ei[1], target_idx)
            pert[et].edge_index = ei[:, keep]
            if "edge_attr" in pert[et] and pert[et].edge_attr is not None:
                pert[et].edge_attr = pert[et].edge_attr[keep]
    return pert


# --------------------------------------------------------------------------- #
# Shift des cell_group après perturbation.
# --------------------------------------------------------------------------- #
def cell_group_shift(z_base: torch.Tensor,
                     z_pert: torch.Tensor,
                     group_names=CELL_GROUPS) -> pd.DataFrame:
    """Compare les embeddings cell_group avant vs après perturbation.

    Pour chaque noeud cell_group, on rapporte :
      * baseline_norm      : ‖z_base‖₂
      * perturbed_norm     : ‖z_pert‖₂
      * shift_L2           : ‖z_pert − z_base‖₂   (amplitude absolue du déplacement)
      * shift_relative     : shift_L2 / ‖z_base‖  (amplitude relative)
      * cosine_similarity  : cos(z_base, z_pert)  (direction conservée ?)

    Interprétation :
      - shift_L2 grand sur P4 → le gène perturbé est important pour P4.
      - shift_L2 grand sur P16_cluster_X → important pour ce sous-état sénescent.
      - Si le shift est concentré sur 1-2 groupes → gène spécifique de ces états.
      - Si réparti uniformément → gène housekeeping / hub global.
    """
    zb = z_base.cpu().numpy()
    zp = z_pert.cpu().numpy()
    assert zb.shape == zp.shape, (zb.shape, zp.shape)
    base_norm = np.linalg.norm(zb, axis=1)
    pert_norm = np.linalg.norm(zp, axis=1)
    shift_l2 = np.linalg.norm(zp - zb, axis=1)
    shift_rel = shift_l2 / (base_norm + 1e-8)
    cos = (zb * zp).sum(axis=1) / (base_norm * pert_norm + 1e-8)
    n = zb.shape[0]
    # Si le nombre de noeuds ne matche pas CELL_GROUPS, on retombe sur un
    # index numérique — évite un crash silencieux sur un run atypique.
    names = list(group_names) if n == len(group_names) else [f"group_{i}" for i in range(n)]
    return pd.DataFrame({
        "group": names,
        "baseline_norm": base_norm,
        "perturbed_norm": pert_norm,
        "shift_L2": shift_l2,
        "shift_relative": shift_rel,
        "cosine_similarity": cos,
    })


# --------------------------------------------------------------------------- #
# Shift au niveau gène, pondéré par l'expression (nouvelle méthode).
# --------------------------------------------------------------------------- #
def cell_group_shift_gene_weighted(mu_base: np.ndarray,
                                    mu_pert: np.ndarray,
                                    group_expr: pd.DataFrame,
                                    gene_symbols: np.ndarray,
                                    target_idx: torch.Tensor,
                                    group_names=CELL_GROUPS) -> pd.DataFrame:
    """Shift par cell_group mesuré au niveau GÈNE, pondéré par l'expression.

    Pour chaque groupe c, on calcule :

        shift_weighted[c]       = Σᵢ∉T  expr_c[i] · ‖Δzᵢ‖₂
        shift_weighted_norm[c]  = shift_weighted[c] / Σᵢ∉T expr_c[i]
        shift_differential[c]   = Σᵢ∉T  max(0, expr_c[i] − meanₘ expr_m[i]) · ‖Δzᵢ‖₂
        direct_target_contrib[c]= Σᵢ∈T  expr_c[i] · ‖Δzᵢ‖₂   (sanity)

    où T = gènes cibles (exclus pour ne pas noyer le signal, risque 1), Δzᵢ = mu_base[i] − mu_pert[i].

    Pourquoi ces 4 colonnes :
      - shift_weighted       : magnitude brute — dépend du voisinage de la cible.
      - shift_weighted_norm  : comparable entre KO de gènes différents (risque 4).
      - shift_differential   : isole ce qui est spécifique au groupe (risque 2 :
                               gomme l'effet housekeeping en soustrayant la
                               moyenne inter-groupe ; clippé à 0 pour rester positif).
      - direct_target_contrib: trace la contribution de la cible elle-même,
                               utile pour vérifier qu'elle ne domine pas.

    Args:
        mu_base    : (n_genes, latent) embedding baseline.
        mu_pert    : (n_genes, latent) embedding post-perturbation.
        group_expr : DataFrame avec colonnes 'gene' + 'mean_<group>' pour chaque
                     group_name. Chargé depuis group_expression.tsv.
        gene_symbols : (n_genes,) symboles, ordre = nodes du graphe.
        target_idx : tensor des indices des gènes cibles (à exclure).

    Returns:
        DataFrame (1 ligne par groupe) avec les 4 métriques + baseline_expr_sum
        (somme totale d'expression dans le groupe, hors cibles — dénominateur
        utilisé pour shift_weighted_norm).
    """
    assert mu_base.shape == mu_pert.shape, (mu_base.shape, mu_pert.shape)
    n_genes = mu_base.shape[0]
    # Ré-aligner group_expr sur l'ordre des gene_symbols du graphe.
    expr_by_gene = group_expr.set_index("gene")
    missing = [g for g in gene_symbols if g not in expr_by_gene.index]
    if missing:
        print(f"[warn] {len(missing)} gene(s) absent de group_expression.tsv "
              f"— leur expression sera 0 : {missing[:5]}...")
    expr_by_gene = expr_by_gene.reindex(gene_symbols).fillna(0.0)

    # Matrice (n_genes, n_groups) des moyennes d'expression brutes.
    expr = np.stack([
        expr_by_gene[f"mean_{g}"].to_numpy().astype(np.float32)
        for g in group_names
    ], axis=1)

    # Expression différentielle : expr_c[i] − moyenne_sur_groupes(expr[i]),
    # clippée à 0 (risque 2 : un gène moins exprimé qu'ailleurs dans le
    # groupe c ne doit PAS pondérer négativement, on l'ignore simplement).
    expr_mean_over_groups = expr.mean(axis=1, keepdims=True)
    expr_diff = np.clip(expr - expr_mean_over_groups, 0.0, None)

    delta_z = np.linalg.norm(mu_base - mu_pert, axis=1).astype(np.float32)

    # Masque non-cible (risque 1 : exclure la cible pour ne pas noyer le signal).
    mask_non_target = np.ones(n_genes, dtype=bool)
    t_np = target_idx.cpu().numpy() if isinstance(target_idx, torch.Tensor) else np.asarray(target_idx)
    if t_np.size > 0:
        mask_non_target[t_np] = False

    rows = []
    for c_idx, grp in enumerate(group_names):
        w = expr[:, c_idx]
        w_diff = expr_diff[:, c_idx]
        w_nt = w[mask_non_target]
        dz_nt = delta_z[mask_non_target]

        shift_w = float((w_nt * dz_nt).sum())
        total_w = float(w_nt.sum())
        shift_w_norm = shift_w / (total_w + 1e-8)
        shift_diff = float((w_diff[mask_non_target] * dz_nt).sum())
        direct = float((w[~mask_non_target] * delta_z[~mask_non_target]).sum())

        rows.append({
            "group": grp,
            "shift_weighted": shift_w,
            "shift_weighted_norm": shift_w_norm,
            "shift_differential": shift_diff,
            "direct_target_contrib": direct,
            "baseline_expr_sum": total_w,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Axes sénescence (option 2 : projection signée P4 -> P16).
# --------------------------------------------------------------------------- #
def compute_senescence_axes(mu_base: np.ndarray,
                             group_expr: pd.DataFrame,
                             gene_symbols: np.ndarray,
                             p4_group="P4",
                             p16_groups=("P16_cluster_0", "P16_cluster_1",
                                         "P16_cluster_2", "P16_cluster_3"),
                             quiescent_groups=None):
    """Axes quiescent -> sénescent dans l'espace latent des gènes.

    Centroïdes pondérés par l'expression (par groupe) :
        c_g = Σᵢ expr_g[i] · μ_base[i]  /  Σᵢ expr_g[i]

    Côté QUIESCENT (= début de l'axe) : par défaut le seul groupe "P4".
    Mais V4 permet d'agréger plusieurs groupes — par ex. {P4, P16_cluster_0}
    car c0 est transcriptionnellement quiescent-like (cf. CLAUDE.md §V3.3).
    Dans ce cas le centroïde quiescent est la MOYENNE des centroïdes des
    groupes listés (équipondérée — chacun compte autant indépendamment de
    sa taille). Ce choix évite que le gros cluster c0 écrase P4 dans la
    pondération si on faisait une moyenne par expression cumulée.

    Côté SÉNESCENT : moyenne (équipondérée) des centroïdes p16_groups
    restants — donc en V4 c'est mean(c1, c2, c3) si c0 a basculé côté
    quiescent.

    Axes finaux (unitaires) :
        axis_global         = unit( mean_k c_{p16_groups[k]}  −  c_quiescent )
        axes_cluster[p16_k] = unit( c_{p16_k}  −  c_quiescent )  pour chaque k

    Args:
        p4_group : str ou liste — backward-compat. Si fourni, il est utilisé
            comme groupe quiescent unique (V3.x baseline).
        quiescent_groups : itérable de str (V4). Si fourni, il OVERRIDE
            p4_group et le centroïde quiescent = moyenne sur cette liste.
            Exemple V4 : ("P4", "P16_cluster_0").
        p16_groups : itérable de str — clusters côté sénescent. En V4 on
            passe typiquement ("P16_cluster_1", "P16_cluster_2", "P16_cluster_3")
            quand c0 a basculé côté quiescent.

    Returns:
        axis_global   : (latent_dim,) unit vector
        axes_cluster  : dict[cluster_name -> (latent_dim,) unit vector]
            La clé est le NOM DU CLUSTER P16 ; l'axe pointe quiescent → ce
            cluster spécifiquement.
        centers       : dict[group_name -> (latent_dim,)] + clé spéciale
            "_quiescent" = centroïde quiescent agrégé.
    """
    # Normalise la spec quiescent (V3 single str ↔ V4 liste)
    if quiescent_groups is None:
        quiescent_list = ([p4_group] if isinstance(p4_group, str)
                          else list(p4_group))
    else:
        quiescent_list = list(quiescent_groups)
    p16_list = list(p16_groups)

    expr_by_gene = group_expr.set_index("gene").reindex(gene_symbols).fillna(0.0)
    centers: dict[str, np.ndarray] = {}
    for g in set(quiescent_list + p16_list):
        w = expr_by_gene[f"mean_{g}"].to_numpy().astype(np.float32)
        w_sum = float(w.sum()) + 1e-8
        centers[g] = (w[:, None] * mu_base).sum(axis=0) / w_sum

    # Centroïde quiescent agrégé (équipondéré sur les groupes listés)
    quiescent_center = np.mean(np.stack([centers[g] for g in quiescent_list]),
                               axis=0)
    centers["_quiescent"] = quiescent_center

    def unit(v: np.ndarray, name: str) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            print(f"[warn] axis '{name}' almost zero (‖·‖={n:.2e}) — "
                  "quiescent et sénescent confondus dans le latent ? "
                  "Projection peu fiable.")
        return (v / (n + 1e-8)).astype(np.float32)

    p16_mean = np.mean(np.stack([centers[g] for g in p16_list]), axis=0)
    axis_global = unit(p16_mean - quiescent_center, "global")
    axes_cluster = {
        g: unit(centers[g] - quiescent_center, g) for g in p16_list
    }
    print(f"  [axes] quiescent = mean({quiescent_list}) ; "
          f"sénescent = mean({p16_list})")
    return axis_global, axes_cluster, centers


def cell_group_shift_projected(mu_base: np.ndarray,
                                mu_pert: np.ndarray,
                                group_expr: pd.DataFrame,
                                gene_symbols: np.ndarray,
                                target_idx: torch.Tensor,
                                axis_global: np.ndarray,
                                axes_cluster: dict,
                                group_names=CELL_GROUPS,
                                target_degree: int | None = None,
                                extent_threshold: float = 1e-3) -> pd.DataFrame:
    """Shift SIGNÉ : projection de Δzᵢ sur les axes sénescence P4 -> P16.

    Δzᵢ = μ_pert[i] − μ_base[i].

    Pour chaque groupe c et chaque axe u ∈ {axis_global, axis_cluster_k (si k=c)} :

        proj_signed[c, u]          = Σᵢ∉T  expr_c[i] · (Δzᵢ · u)
        proj_signed_diff[c, u]     = Σᵢ∉T  w_diff_c[i] · (Δzᵢ · u)
        proj_signed_norm[c, u]     = proj_signed_diff / (Σ w_diff + eps)
                                     -> moyenne par unité de sur-expression (≈ constant par groupe)
        proj_signed_amplitude[c,u] = proj_signed_diff / (Σ w_diff[i] · ‖Δzᵢ‖₂ + eps)
                                     -> fraction du mouvement latent pondéré qui va dans la
                                        direction axis. ∈ [−1, 1]. Correction de hub directe.
        proj_signed_extent[c, u]   = proj_signed_diff / max(n_affected, 1)
                                     avec n_affected = #{i ∉ T : |Δzᵢ · u| > extent_threshold}
                                     -> effet moyen par gène effectivement déplacé.
        proj_signed_degree[c, u]   = proj_signed_diff / max(target_degree, 1)
                                     -> pénalise les hubs : diff par unité de degré PPI.
        proj_signed_cosine[c, u]   = cos(Δz_weighted_mean_c, u)   (Option B)
                                     avec Δz_weighted_mean_c = (Σ w_diff · Δz) / (Σ w_diff + eps)
                                     -> cosinus entre l'effet moyen pondéré et axis. ∈ [−1, 1].
                                        Totalement comparable entre gènes, voies et runs.
        proj_direct_target[c, u]   = Σᵢ∈T  expr_c[i] · (Δzᵢ · u)    (sanity)
        w_diff_motion_sum          = Σᵢ∉T w_diff_c[i] · ‖Δzᵢ‖₂
        n_affected_genes_c         = cardinal des gènes "déplacés" sur cet axe

    Convention de signe :
      * proj > 0  → la perturbation pousse les gènes VERS P16 (pro-sénescent).
      * proj < 0  → pousse VERS P4 (anti-sénescent).
      * |proj| ≈ 0 → pas de déplacement net dans cette direction.

    Args:
        axis_global  : sortie de compute_senescence_axes().
        axes_cluster : idem (dict par cluster P16).
        target_degree : degré PPI total du/des gène(s) cible(s). Si None, la
            métrique proj_signed_degree vaudra proj_signed_diff (division par 1).
        extent_threshold : seuil absolu sur |Δzᵢ · u| pour qu'un gène soit
            compté comme "affecté" (défaut 1e-3).
    """
    assert mu_base.shape == mu_pert.shape, (mu_base.shape, mu_pert.shape)
    delta_z_vec = (mu_pert - mu_base).astype(np.float32)   # (n_genes, latent)
    delta_z_norm = np.linalg.norm(delta_z_vec, axis=1)     # (n_genes,)

    expr_by_gene = group_expr.set_index("gene").reindex(gene_symbols).fillna(0.0)
    expr = np.stack([
        expr_by_gene[f"mean_{g}"].to_numpy().astype(np.float32)
        for g in group_names
    ], axis=1)
    expr_mean_over_groups = expr.mean(axis=1, keepdims=True)
    expr_diff = np.clip(expr - expr_mean_over_groups, 0.0, None)

    n_genes = mu_base.shape[0]
    mask_non_target = np.ones(n_genes, dtype=bool)
    t_np = target_idx.cpu().numpy() if isinstance(target_idx, torch.Tensor) else np.asarray(target_idx)
    if t_np.size > 0:
        mask_non_target[t_np] = False

    deg = max(int(target_degree) if target_degree is not None else 1, 1)
    rows = []

    def _row(grp: str, axis_type: str, axis_u: np.ndarray, c_idx: int):
        w = expr[:, c_idx]
        w_diff = expr_diff[:, c_idx]
        w_nt = w[mask_non_target]
        w_diff_nt = w_diff[mask_non_target]
        proj_u = delta_z_vec @ axis_u                     # (n_genes,)
        p_nt = proj_u[mask_non_target]
        dz_nt = delta_z_vec[mask_non_target]
        dz_norm_nt = delta_z_norm[mask_non_target]
        total_w = float(w_nt.sum())
        total_w_diff = float(w_diff_nt.sum())
        proj_signed_diff_val = float((w_diff_nt * p_nt).sum())
        # Amplitude : mouvement latent total pondéré par w_diff.
        w_motion = float((w_diff_nt * dz_norm_nt).sum())
        # Extent : #gènes réellement déplacés le long de u.
        n_affected = int((np.abs(p_nt) > extent_threshold).sum())
        # Cosine (Option B) : cos(Δz_moyen_pondéré, u). Indépendant de toute
        # amplitude. Numerateur = (Σ w·Δz)·u = proj_signed_diff (def).
        dz_mean = (w_diff_nt[:, None] * dz_nt).sum(axis=0) / (total_w_diff + 1e-8)
        dz_mean_norm = float(np.linalg.norm(dz_mean))
        proj_cosine = float(np.dot(dz_mean, axis_u) / (dz_mean_norm + 1e-8))
        row = {
            "group": grp,
            "axis_type": axis_type,
            "proj_signed":           float((w_nt * p_nt).sum()),
            "proj_signed_diff":      proj_signed_diff_val,
            "proj_signed_norm":      proj_signed_diff_val / (total_w_diff + 1e-8),
            "proj_signed_amplitude": proj_signed_diff_val / (w_motion + 1e-8),
            "proj_signed_extent":    proj_signed_diff_val / max(n_affected, 1),
            "proj_signed_degree":    proj_signed_diff_val / deg,
            "proj_signed_cosine":    proj_cosine,
            "proj_direct_target":    float((w[~mask_non_target] *
                                            proj_u[~mask_non_target]).sum()),
            "baseline_expr_sum":      total_w,
            "baseline_expr_diff_sum": total_w_diff,
            "w_diff_motion_sum":      w_motion,
            "n_affected_genes":       n_affected,
            "target_degree":          deg,
        }
        return row

    # --- Axe global : même u pour tous les groupes, pondération par expr_c.
    for c_idx, grp in enumerate(group_names):
        rows.append(_row(grp, "global", axis_global, c_idx))

    # --- Axe par cluster : u_k spécifique au cluster P16.
    for cluster_name, axis_k in axes_cluster.items():
        if cluster_name not in group_names:
            continue
        c_idx = group_names.index(cluster_name)
        rows.append(_row(cluster_name, "cluster", axis_k, c_idx))

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Run loader.
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path, hidden, latent, n_layers, n_heads):
    data = torch.load(run_dir / "hetero_graph_vgae.pt", weights_only=False)
    baseline = pd.read_csv(run_dir / "gene_ranking_vgae.csv")
    emb_df = pd.read_csv(run_dir / "gene_embeddings_vgae.csv", index_col=0)
    gene_symbols = np.array(emb_df.index.astype(str))
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}

    # Expression par groupe (optionnel — requis pour le shift gene-weighted).
    # Si absent (runs pré-V3.3), on fallback : le shift gene-weighted sera sauté.
    group_expr_path = run_dir / "group_expression.tsv"
    if group_expr_path.exists():
        group_expr = pd.read_csv(group_expr_path, sep="\t")
    else:
        print(f"[warn] {group_expr_path.name} absent — le shift gene-weighted "
              "sera sauté (relancer gnn_vgae.py pour le régénérer).")
        group_expr = None

    encoder = HeteroEncoder(
        gene_in=data["gene"].x.shape[1],
        cell_in=data["cell_group"].x.shape[1],
        hidden=hidden, latent=latent, n_layers=n_layers, n_heads=n_heads,
        # V3.6 : ne créer que les GATConv pour les edge_types réellement
        # présents dans le graphe entraîné — sinon le state_dict refuse
        # de charger sur les ablations (--no-ppi, --no-coexpr, etc.).
        available_edge_types=list(data.edge_types),
    )
    model = VGAE(encoder)
    state_path = run_dir / "best_vgae.pt"
    if not state_path.exists():
        state_path = run_dir / "vgae_weights.pt"
    model.load_state_dict(torch.load(state_path, weights_only=True))
    model.eval()
    return data, model, gene_symbols, gene_to_idx, baseline, group_expr


# --------------------------------------------------------------------------- #
# Reusable API — called both by main() and by perturb_top_genes.py (--all-*).
# --------------------------------------------------------------------------- #
def prepare_baseline(model, data, baseline_df, gene_symbols, group_expr=None,
                     quiescent_groups=None, p16_groups=None):
    """Compute the baseline once; reused across many perturbations.

    Args:
        quiescent_groups : optionnel. Liste de groupes côté quiescent pour
            l'axe de sénescence (V4 : ("P4", "P16_cluster_0")). Si None,
            défaut historique = ("P4",).
        p16_groups : optionnel. Liste des clusters P16 côté sénescent. Si
            quiescent_groups inclut "P16_cluster_0", on le retire
            automatiquement de p16_groups (sauf si l'utilisateur a fourni
            sa propre liste).

    Returns:
        spec         : (n_genes,) baseline vgae_specificity aligned on node idx.
        base_imp     : (n_genes,) composite baseline importance.
        base_rank    : (n_genes,) ordinal rank of base_imp (1 = best).
        z_cg_base    : (5, hidden) cell_group hidden states from the baseline pass.
        mu_base      : (n_genes, latent) embedding baseline — shift gene-weighted.
        axis_global  : (latent,) axe quiescent -> mean(sénescent). None si group_expr absent.
        axes_cluster : dict[P16_cluster_k -> (latent,)] axes par cluster. None idem.
    """
    spec_map = dict(zip(baseline_df["gene"].astype(str),
                        baseline_df["vgae_specificity"].astype(float)))
    spec = np.array([spec_map.get(g, 0.0) for g in gene_symbols],
                    dtype=np.float32)
    base = compute_importance(model, data, spec)
    base_imp = base["importance"]
    base_rank = rankdata(-base_imp, method="ordinal").astype(int)
    z_cg_base = model.encoder.last_cell_group_h.clone()
    mu_base = base["mu"]

    # Axes sénescence (option 2). Invariants sur toutes les perturbations —
    # calculés une fois à partir de mu_base + expression par groupe.
    if group_expr is not None:
        # Auto-derive p16_groups si non fourni : retire les clusters qui
        # ont basculé côté quiescent_groups (ex. V4 : c0 retiré du côté
        # sénescent quand inclus côté quiescent).
        if p16_groups is None:
            default_p16 = ["P16_cluster_0", "P16_cluster_1",
                           "P16_cluster_2", "P16_cluster_3"]
            if quiescent_groups is not None:
                p16_groups = [g for g in default_p16
                              if g not in set(quiescent_groups)]
            else:
                p16_groups = default_p16
        axis_global, axes_cluster, _centers = compute_senescence_axes(
            mu_base, group_expr, gene_symbols,
            p16_groups=tuple(p16_groups),
            quiescent_groups=(tuple(quiescent_groups)
                              if quiescent_groups is not None else None))
        print(f"  axis_global ‖·‖ pre-unit = {np.linalg.norm(axis_global):.4f} "
              f"(1.0 attendu post-normalisation)")
    else:
        axis_global, axes_cluster = None, None

    return spec, base_imp, base_rank, z_cg_base, mu_base, axis_global, axes_cluster


def run_perturbation_once(model, data, gene_symbols, gene_to_idx,
                          spec, base_imp, base_rank, z_cg_base, mu_base,
                          targets, mode, factor, top_k, fdr,
                          reactome, background, group_expr=None,
                          axis_global=None, axes_cluster=None,
                          out_dir=None, write_full=True,
                          include_details=False):
    """Run a single perturbation and return its summary dict.

    Args:
        targets         : list[str] gene symbols to perturb (pre-dedup).
        mode            : "knockdown" | "knockout" | "overexpress".
        factor          : multiplier for overexpress (ignored otherwise).
        mu_base         : (n_genes, latent) embedding baseline.
        group_expr      : DataFrame loaded from group_expression.tsv, or None.
                          Requis pour le shift gene-weighted ; si None → sauté.
        out_dir         : if None, write nothing (in-memory only).
                          if set, always write summary.json.
        write_full      : if True AND out_dir set, also write delta_ranking.csv,
                          delta_ora_top_up_reactome.tsv, cell_group_shift.tsv,
                          and cell_group_shift_gene.tsv.
        include_details : if True, embed extra fields into the returned
                          summary that would normally only live in the
                          per-run TSVs (top-K risers / fallers with their
                          delta_rank, top-K ORA pathways with p_adj, and
                          |delta_rank| quantiles). Used by the ALL-mode
                          aggregator so a single flat TSV carries enough
                          to reconstruct most figures.

    Returns:
        summary (dict) or None if no target is present in the graph.
    """
    n_genes = len(gene_symbols)
    hit = [g for g in targets if g in gene_to_idx]
    miss = [g for g in targets if g not in gene_to_idx]
    if not hit:
        return None
    target_idx = torch.tensor([gene_to_idx[g] for g in hit], dtype=torch.long)

    pert_data = apply_perturbation(data, target_idx, mode, factor)
    pert = compute_importance(model, pert_data, spec)
    pert_imp = pert["importance"]
    pert_rank = rankdata(-pert_imp, method="ordinal").astype(int)
    z_cg_pert = model.encoder.last_cell_group_h.clone()
    mu_pert = pert["mu"]

    # --- Ancienne méthode : shift sur le hidden cell_group (conservée en
    # comparaison méthodologique). Signal dilué par l'agrégation + BatchNorm.
    shift_df = cell_group_shift(z_cg_base, z_cg_pert)
    shift_map = dict(zip(shift_df["group"], shift_df["shift_relative"]))
    max_shift_row = shift_df.loc[shift_df["shift_relative"].idxmax()]

    # --- Nouvelle méthode : shift au niveau gène pondéré par expression.
    # Attendu : signal ×100-×1000, discrimine mieux les groupes.
    shift_gene_df = None
    if group_expr is not None:
        shift_gene_df = cell_group_shift_gene_weighted(
            mu_base, mu_pert, group_expr, gene_symbols, target_idx)

    # --- Option 2 : shift SIGNÉ par projection sur l'axe P4 -> P16.
    # Magnitude + direction (pro / anti-sénescence). Deux axes : global
    # (moyenne P16) et par cluster (spécifique à chaque sous-état P16).
    shift_proj_df = None
    if (group_expr is not None and axis_global is not None
            and axes_cluster is not None):
        # Degré PPI total de la/des cible(s) : utilisé pour la métrique
        # proj_signed_degree (correction explicite du biais hub).
        ppi_key = ("gene", "ppi", "gene")
        target_degree = 0
        if ppi_key in data.edge_types:
            ppi_ei = data[ppi_key].edge_index
            target_degree = int(torch.isin(ppi_ei[0], target_idx).sum().item()
                                + torch.isin(ppi_ei[1], target_idx).sum().item())
        shift_proj_df = cell_group_shift_projected(
            mu_base, mu_pert, group_expr, gene_symbols, target_idx,
            axis_global, axes_cluster,
            target_degree=target_degree)

    delta_rank = base_rank - pert_rank
    delta_imp = pert_imp - base_imp

    ranking = pd.DataFrame({
        "gene": gene_symbols,
        "baseline_importance": base_imp,
        "perturbed_importance": pert_imp,
        "delta_importance": delta_imp,
        "baseline_rank": base_rank,
        "perturbed_rank": pert_rank,
        "delta_rank": delta_rank,
        "is_target": np.isin(np.arange(n_genes),
                             target_idx.cpu().numpy()).astype(int),
    }).sort_values("delta_rank", ascending=False).reset_index(drop=True)

    non_target = ranking[ranking["is_target"] == 0]
    top_up = non_target.head(top_k)
    top_down = non_target.sort_values("delta_rank").head(top_k)

    gl = set(top_up["gene"].astype(str))
    rows = run_ora(gl, background, reactome)
    sig = sum(1 for r in rows if r.p_adj < fdr)

    summary = {
        "mode": mode,
        "factor": factor if mode == "overexpress" else None,
        "n_targets_in_graph": len(hit),
        "targets_in_graph": hit,
        "targets_missing": miss,
        "top_k": top_k,
        "n_rising": int((ranking["delta_rank"] > 0).sum()),
        "n_falling": int((ranking["delta_rank"] < 0).sum()),
        "median_abs_delta_rank": float(np.median(np.abs(delta_rank))),
        "max_up_gene": top_up.iloc[0]["gene"] if len(top_up) else None,
        "max_up_delta_rank": (int(top_up.iloc[0]["delta_rank"])
                              if len(top_up) else None),
        "max_down_gene": top_down.iloc[0]["gene"] if len(top_down) else None,
        "max_down_delta_rank": (int(top_down.iloc[0]["delta_rank"])
                                if len(top_down) else None),
        "n_sig_delta_pathways": sig,
        "fdr_threshold": fdr,
        "top5_delta_pathways": [r.pathway for r in rows[:5]],
        # --- Ancienne méthode (shift sur hidden cell_group) : conservée en
        # comparaison méthodologique. Typiquement ~0.001 de magnitude, faible
        # discrimination inter-groupe à cause de l'agrégation + BatchNorm.
        "cell_group_shift_relative": {k: float(v) for k, v in shift_map.items()},
        "max_shift_group": str(max_shift_row["group"]),
        "max_shift_relative": float(max_shift_row["shift_relative"]),
    }

    # --- Nouvelle méthode (shift gene-weighted) : 4 métriques par groupe.
    if shift_gene_df is not None:
        _gw = shift_gene_df.set_index("group")
        summary["cell_group_shift_gene_weighted"] = {
            g: float(_gw.loc[g, "shift_weighted"]) for g in _gw.index
        }
        summary["cell_group_shift_gene_weighted_norm"] = {
            g: float(_gw.loc[g, "shift_weighted_norm"]) for g in _gw.index
        }
        summary["cell_group_shift_gene_differential"] = {
            g: float(_gw.loc[g, "shift_differential"]) for g in _gw.index
        }
        summary["cell_group_shift_gene_direct_target"] = {
            g: float(_gw.loc[g, "direct_target_contrib"]) for g in _gw.index
        }
        # Groupe maximal — on utilise 'shift_differential' car c'est la
        # métrique qui isole le signal group-specific (risque 2 atténué).
        max_gw_group = _gw["shift_differential"].idxmax()
        summary["max_shift_gene_differential_group"] = str(max_gw_group)
        summary["max_shift_gene_differential"] = float(
            _gw.loc[max_gw_group, "shift_differential"])

    # --- Option 2 : shift SIGNÉ (projection sur axe sénescence).
    # Deux axes : global (moyenne P16) et par cluster P16.
    if shift_proj_df is not None:
        _proj_global = shift_proj_df[shift_proj_df["axis_type"] == "global"]
        _proj_cluster = shift_proj_df[shift_proj_df["axis_type"] == "cluster"]

        _metric_keys = (
            "proj_signed", "proj_signed_diff", "proj_signed_norm",
            "proj_signed_amplitude", "proj_signed_extent",
            "proj_signed_degree", "proj_signed_cosine",
        )

        def _pack(subdf):
            return {
                row["group"]: {k: float(row[k]) for k in _metric_keys}
                for _, row in subdf.iterrows()
            }

        summary["cell_group_shift_projected_global"] = _pack(_proj_global)
        if len(_proj_cluster) > 0:
            summary["cell_group_shift_projected_cluster"] = _pack(_proj_cluster)

        # Degré PPI et n_affected médian — utiles pour interpréter la métrique
        # proj_signed_degree et la relation hub/étendue cross-gene.
        if "target_degree" in shift_proj_df.columns:
            summary["target_ppi_degree"] = int(shift_proj_df["target_degree"].iloc[0])

        # Indicateurs max/min : un par métrique de projection, avec le groupe /
        # axe associé. On reporte la valeur signée (pas |.|), mais on choisit
        # l'entrée par |valeur| la plus grande.
        all_proj = pd.concat([_proj_global, _proj_cluster], ignore_index=True)
        if len(all_proj) > 0:
            for key in _metric_keys:
                if key == "proj_signed":
                    continue  # pas d'indicateur résumé pour la version brute
                idx = all_proj[key].abs().idxmax()
                row = all_proj.loc[idx]
                suffix = key.replace("proj_signed_", "")
                # Noms de sortie : max_proj_signed_diff, max_proj_signed_norm,
                # max_proj_signed_amplitude, max_proj_signed_extent,
                # max_proj_signed_degree, max_proj_signed_cosine.
                out_name = f"max_proj_signed_{suffix}"
                summary[f"{out_name}_group"] = f"{row['group']}/{row['axis_type']}"
                summary[out_name] = float(row[key])

    if include_details:
        # Keep per-target ORA with p_adj (up to top_k, filter irrelevant ones
        # with p_adj > 0.5 to keep the payload bounded).
        summary["top_delta_pathways_padj"] = [
            {"pathway": r.pathway, "p_adj": float(r.p_adj)}
            for r in rows[:top_k] if r.p_adj <= 0.5
        ]
        # Top-K risers / fallers with their delta_rank + baseline_importance.
        summary["top_risers"] = [
            {"gene": str(g), "delta_rank": int(d), "baseline_importance": float(b)}
            for g, d, b in zip(top_up["gene"].tolist(),
                               top_up["delta_rank"].tolist(),
                               top_up["baseline_importance"].tolist())
        ]
        summary["top_fallers"] = [
            {"gene": str(g), "delta_rank": int(d), "baseline_importance": float(b)}
            for g, d, b in zip(top_down["gene"].tolist(),
                               top_down["delta_rank"].tolist(),
                               top_down["baseline_importance"].tolist())
        ]
        # |delta_rank| distribution summary (boxplot-compatible).
        abs_dr = np.abs(non_target["delta_rank"].to_numpy())
        if abs_dr.size:
            qs = np.percentile(abs_dr, [0, 25, 50, 75, 95, 99, 100])
            summary["delta_rank_abs_quantiles"] = {
                "q0": float(qs[0]), "q25": float(qs[1]), "q50": float(qs[2]),
                "q75": float(qs[3]), "q95": float(qs[4]), "q99": float(qs[5]),
                "q100": float(qs[6]),
                "mean": float(abs_dr.mean()),
                "n_genes": int(abs_dr.size),
            }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        if write_full:
            ranking.to_csv(out_dir / "delta_ranking.csv", index=False)
            shift_df.to_csv(out_dir / "cell_group_shift.tsv",
                            sep="\t", index=False)
            if shift_gene_df is not None:
                shift_gene_df.to_csv(
                    out_dir / "cell_group_shift_gene.tsv",
                    sep="\t", index=False)
            if shift_proj_df is not None:
                shift_proj_df.to_csv(
                    out_dir / "cell_group_shift_projected.tsv",
                    sep="\t", index=False)
            write_tsv(rows, out_dir / "delta_ora_top_up_reactome.tsv")
    return summary


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--mode", choices=["knockdown", "knockout", "overexpress"],
                    default="knockdown")
    ap.add_argument("--genes", nargs="+", default=None,
                    help="Gene symbol(s) to perturb.")
    ap.add_argument("--gene-list", type=Path, default=None,
                    help="File with one gene symbol per line (e.g. full pathway).")
    ap.add_argument("--factor", type=float, default=2.0,
                    help="Multiplier used in --mode overexpress (default 2.0).")
    ap.add_argument("--top-k", type=int, default=100,
                    help="Top-K rising genes used for delta-ORA (default 100).")
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--latent", type=int, default=DEFAULT_LATENT)
    ap.add_argument("--n-layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--n-heads", type=int, default=DEFAULT_HEADS)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: run-dir/perturbation/<tag>).")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--summary-only", action="store_true",
                    help="Skip writing delta_ranking.csv / cell_group_shift.tsv "
                         "/ delta_ora_*.tsv. Only summary.json is written.")
    # V4 — axe sénescence : groupes côté quiescent / sénescent.
    ap.add_argument("--quiescent-groups", default="P4",
                    help="liste séparée par virgules (défaut V3 : 'P4' ; "
                         "recommandé V4 : 'P4,P16_cluster_0').")
    ap.add_argument("--p16-groups", default=None,
                    help="liste séparée par virgules ; auto-dérivé si None.")
    args = ap.parse_args()
    _q = (tuple(s.strip() for s in args.quiescent_groups.split(",") if s.strip())
          if args.quiescent_groups else None)
    _p = (tuple(s.strip() for s in args.p16_groups.split(",") if s.strip())
          if args.p16_groups else None)

    targets: list[str] = []
    if args.genes:
        targets.extend(args.genes)
    if args.gene_list:
        targets.extend(
            g.strip() for g in args.gene_list.read_text().splitlines()
            if g.strip() and not g.strip().startswith("#")
        )
    targets = sorted(set(targets))
    if not targets:
        ap.error("must supply --genes and/or --gene-list")

    data, model, gene_symbols, gene_to_idx, baseline, group_expr = load_run(
        args.run_dir, args.hidden, args.latent, args.n_layers, args.n_heads)
    print(f"Run loaded: {len(gene_symbols)} genes, "
          f"{len(data.edge_types)} edge types, "
          f"group_expression={'OK' if group_expr is not None else 'absent'}.")

    hit = [g for g in targets if g in gene_to_idx]
    miss = [g for g in targets if g not in gene_to_idx]
    if miss:
        print(f"[warn] {len(miss)} target(s) absent from the graph: "
              f"{miss[:8]}{'...' if len(miss) > 8 else ''}")
    if not hit:
        print("No target present in the graph. Aborting.")
        return

    tag = (f"{args.mode}_{'_'.join(hit[:3])}"
           f"{'_etc' if len(hit) > 3 else ''}")
    out_dir = args.out_dir or (args.run_dir / "perturbation" / tag)
    print(f"Mode={args.mode} | targets in graph={len(hit)} | "
          f"tag={tag} | out={out_dir}")

    spec, base_imp, base_rank, z_cg_base, mu_base, axis_global, axes_cluster = prepare_baseline(
        model, data, baseline, gene_symbols, group_expr,
        quiescent_groups=_q, p16_groups=_p)

    print("Loading REACTOME + background ...")
    reactome = load_reactome_gmt()
    background = load_background()

    summary = run_perturbation_once(
        model, data, gene_symbols, gene_to_idx,
        spec, base_imp, base_rank, z_cg_base, mu_base,
        targets=hit, mode=args.mode, factor=args.factor,
        top_k=args.top_k, fdr=args.fdr,
        reactome=reactome, background=background,
        group_expr=group_expr,
        axis_global=axis_global, axes_cluster=axes_cluster,
        out_dir=out_dir, write_full=not args.summary_only)
    summary["run_dir"] = str(args.run_dir)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nWrote:")
    print(f"  {out_dir / 'summary.json'}")
    if not args.summary_only:
        print(f"  {out_dir / 'delta_ranking.csv'}")
        print(f"  {out_dir / 'delta_ora_top_up_reactome.tsv'}")
        print(f"  {out_dir / 'cell_group_shift.tsv'}")
        if group_expr is not None:
            print(f"  {out_dir / 'cell_group_shift_gene.tsv'}")


if __name__ == "__main__":
    main()
