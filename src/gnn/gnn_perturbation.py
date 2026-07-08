#!/usr/bin/env python3
"""
gnn_perturbation.py — In silico perturbation of the trained VGAE.

Loads a trained VGAE run (output/gnn_vgae/V*_Run*/) and simulates
knockdown / knockout / over-expression of one gene, a set of genes, or
a full REACTOME pathway. Runs a forward pass through the FROZEN encoder
(no retraining), recomputes the composite importance score and reports:

  * delta_ranking.csv       — per-gene baseline vs perturbed rank & importance
  * delta_ora_top_up_reactome.tsv — REACTOME ORA on the top rising genes
  * summary.json            — top movers, stats, signature delta-pathways ;
                              projections signées sur l'axe global, les axes
                              P4→cluster ET les axes de transition inter-cluster
                              (2026-07 ; centroïdes ancrables DE/Ahn, cf.
                              compute_senescence_axes)
  * signed_fanout.tsv       — readout signé 1-hop (V5.5, gnn_futur §8.A/§9.2) :
                              par cible T du fan-out signé de S, le signe
                              prédit (bilinéaire) / curé de S→T + la réponse
                              Δz_T·u. Brique du score signed_readout / cohérence
                              (rôles de cible joints côté perturb_report).

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

# Permettre l'import d'ora_consensus depuis src/validation/ora/ (layout local)
# OU src/ (fallback flat-layout cluster).
_HERE = Path(__file__).resolve().parent
for _p in [_HERE, _HERE.parent / "validation" / "ora", _HERE.parent]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
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
# V5 (TIER 1c) — edge_types qui portent un `sign` (colonne 1 de edge_attr).
# DOIT rester synchro avec gnn_vgae.SIGNED_EDGE_TYPES.
SIGNED_EDGE_TYPES: set = {
    ("gene", "signaling", "gene"),
    ("gene", "tf_curated", "gene"),
    ("gene", "tf_curated_by", "gene"),
    ("gene", "reactome_fi", "gene"),
}


class SignedGATConv(GATConv):
    """GATConv qui multiplie chaque message par son `sign` d'arête (V5).

    Duplique `_vgae_model.SignedGATConv` pour import-safety (Tier 2.5).
    Aucun nouveau paramètre vs GATConv — les state_dicts sont
    interchangeables, seul le forward diffère.
    """

    def __init__(self, *args, sign_col: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sign_col = sign_col
        self._current_edge_sign = None

    def forward(self, x, edge_index, edge_attr=None, size=None,
                return_attention_weights=None):
        if edge_attr is not None and edge_attr.ndim >= 2 \
                and edge_attr.shape[1] > self.sign_col:
            self._current_edge_sign = edge_attr[:, self.sign_col]
        else:
            self._current_edge_sign = None
        return super().forward(x, edge_index, edge_attr=edge_attr, size=size,
                               return_attention_weights=return_attention_weights)

    def message(self, x_j, alpha):
        out = super().message(x_j, alpha)
        sign = self._current_edge_sign
        if sign is None:
            return out
        scale = torch.where(sign == 0, torch.ones_like(sign), sign)
        scale = scale.view(-1, *([1] * (out.ndim - 1)))
        return out * scale


class BilinearSignedDecoder(nn.Module):
    """Décodeur bilinéaire signé V5.3 — duplicate import-safe.

    V5.3 (§14bis.6unvicesies) : ajout de `signed_proj : R^latent → R^signed_dim`
    pour filtrer le gradient signed loss et réduire la concurrence avec le
    décodeur cosinus. En perturbation, on n'appelle pas `forward_signed()`
    (l'importance utilise le décodeur cosinus), mais la classe doit exister
    pour que le state_dict V5.x charge.

    Backward-compat V5.1/V5.2 : si signed_proj absent du state_dict, init
    à identité + 0 bias ⇒ équivalent numérique exact.
    """

    def __init__(self, latent_dim: int, signed_dim: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.signed_dim = signed_dim if signed_dim is not None else latent_dim
        self.signed_proj = nn.Linear(latent_dim, self.signed_dim, bias=True)
        with torch.no_grad():
            if self.signed_dim == latent_dim:
                self.signed_proj.weight.copy_(torch.eye(latent_dim))
            else:
                init = torch.zeros(self.signed_dim, latent_dim)
                init[:self.signed_dim, :self.signed_dim] = torch.eye(self.signed_dim)
                self.signed_proj.weight.copy_(init)
            self.signed_proj.bias.zero_()
        eye = torch.eye(self.signed_dim)
        self.W_pos = nn.Parameter(eye + 0.01 * torch.randn(self.signed_dim, self.signed_dim))
        self.W_neg = nn.Parameter(eye + 0.01 * torch.randn(self.signed_dim, self.signed_dim))
        self.W_zero = nn.Parameter(eye + 0.01 * torch.randn(self.signed_dim, self.signed_dim))

    def _project(self, z):
        return self.signed_proj(z)

    def forward_signed(self, z, edge_index, edge_sign):
        z_signed = self._project(z)
        z_src = z_signed[edge_index[0]]
        z_dst = z_signed[edge_index[1]]
        logit_pos = (z_src @ self.W_pos * z_dst).sum(dim=-1)
        logit_neg = (z_src @ self.W_neg * z_dst).sum(dim=-1)
        logit_zero = (z_src @ self.W_zero * z_dst).sum(dim=-1)
        mask_pos = (edge_sign > 0).float()
        mask_neg = (edge_sign < 0).float()
        mask_zero = (edge_sign == 0).float()
        return mask_pos * logit_pos + mask_neg * logit_neg + mask_zero * logit_zero

    def predict_sign_score(self, z, edge_index):
        z_signed = self._project(z)
        z_src = z_signed[edge_index[0]]
        z_dst = z_signed[edge_index[1]]
        logit_pos = (z_src @ self.W_pos * z_dst).sum(dim=-1)
        logit_neg = (z_src @ self.W_neg * z_dst).sum(dim=-1)
        return logit_pos - logit_neg

    def predict_signed_existence(self, z, edge_index):
        """V5.4 (decoder-split) — existence = logsumexp(W+,W-,W0).
        Non utilisé en perturbation (importance = cosinus) mais présent
        pour cohérence avec gnn_vgae.py / _vgae_model.py."""
        z_signed = self._project(z)
        z_src = z_signed[edge_index[0]]
        z_dst = z_signed[edge_index[1]]
        logit_pos = (z_src @ self.W_pos * z_dst).sum(dim=-1)
        logit_neg = (z_src @ self.W_neg * z_dst).sum(dim=-1)
        logit_zero = (z_src @ self.W_zero * z_dst).sum(dim=-1)
        return torch.logsumexp(
            torch.stack([logit_pos, logit_neg, logit_zero], dim=-1), dim=-1)


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
        # OmniPath V4 — DOIT rester synchro avec gnn_vgae.HeteroEncoder.
        # edge_attr=[score, sign∈{−1,0,+1}], dim=2.
        (("gene", "signaling", "gene"), 2),
        (("gene", "tf_curated", "gene"), 2),
        (("gene", "tf_curated_by", "gene"), 2),
        (("gene", "reactome_fi", "gene"), 2),  # V4.2
    ]

    def __init__(self, gene_in, cell_in, hidden, latent, n_layers,
                 n_heads=4, dropout=0.2, available_edge_types=None,
                 edge_dim_overrides=None,
                 signed_message=False, signed_edge_types=None):
        super().__init__()
        self.n_layers = n_layers
        head_dim = hidden // n_heads
        self.gene_proj = nn.Linear(gene_in, hidden)
        self.cell_proj = nn.Linear(cell_in, hidden)
        # V5 (TIER 1c.2) : flag SignedGATConv pour edge_types signés.
        self._signed_message = bool(signed_message)
        self._signed_edge_types = set(
            tuple(et) for et in (signed_edge_types or SIGNED_EDGE_TYPES)
        )

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
        # V4.2 : edge_dim peut différer du catalogue (coexpr différentielle
        # 1→6, ppi 1→2 sous --dedup-ppi-signed annotate). Les overrides sont
        # inférés depuis data[et].edge_attr.shape[1] côté load_run().
        overrides = edge_dim_overrides or {}
        if overrides:
            edge_types_dims = [
                (et, overrides.get(tuple(et), dim))
                for et, dim in edge_types_dims
            ]
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
                # V5 : SignedGATConv si flag actif ET edge_type signé ET
                # edge_dim≥2 (besoin de la colonne `sign`).
                _use_signed = (
                    self._signed_message
                    and tuple(et) in self._signed_edge_types
                    and ed is not None and ed >= 2
                )
                if _use_signed:
                    conv_dict[et] = SignedGATConv(hidden, head_dim,
                                                  sign_col=1, **conv_kwargs)
                else:
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
            # Restaurer les types de noeuds que HeteroConv aurait retirés
            # faute d'arêtes entrantes (ex : --no-cell-group-edges → aucune
            # arête pointe vers "cell_group", donc HeteroConv le drop). On
            # propage l'état précédent en identité, BatchNorm/ReLU non appliqués
            # à ce niveau pour ne pas modifier des stats sans flux entrant.
            for key, prev in x_prev.items():
                if key not in x_dict:
                    x_dict[key] = prev
            for key in list(x_dict.keys()):
                if key not in x_prev:
                    continue
                # Skip BN/ReLU/dropout pour les clés restaurées en identité
                if x_dict[key] is x_prev[key]:
                    continue
                x_dict[key] = self.norms[i][key](x_dict[key])
                x_dict[key] = F.relu(x_dict[key])
                x_dict[key] = self.dropout(x_dict[key])
                x_dict[key] = x_dict[key] + x_prev[key]
        self.last_cell_group_h = (
            x_dict["cell_group"].detach().clone()
            if "cell_group" in x_dict else None
        )
        return self.mu_head(x_dict["gene"]), self.logvar_head(x_dict["gene"])


class VGAE(nn.Module):
    def __init__(self, encoder, tau_init=2.0, tau_max=3.0,
                 bilinear_decoder=None):
        super().__init__()
        self.encoder = encoder
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau_init))))
        self.log_tau_max = np.log(tau_max)
        # V5 (TIER 1c.3) : décodeur bilinéaire signé optionnel. En
        # perturbation on n'appelle PAS forward_signed() — la métrique
        # d'importance utilise toujours le décodeur cosinus historique —
        # mais la sous-module doit exister pour que load_state_dict accepte
        # les clés `bilinear_decoder.*` d'un checkpoint trainé `--signed-decoder`.
        self.bilinear_decoder = bilinear_decoder

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
def apply_perturbation(data, target_idx: torch.Tensor, mode: str, factor: float,
                       ko_mode: str = "mask", ko_soft_factor: float = 0.1,
                       kd_factor: float = 0.15, ko_edge_factor: float = 1.0,
                       legacy: bool = False):
    """Apply a knockdown / knockout / overexpression to the target genes.

    Sémantique V5.5 (refactor 2026-06-04, biologiquement monotone
    OE > WT > KD > KO ; cf. docs/gnn_futur.md §6.1) :
      knockdown   : feature × kd_factor (défaut 0.15 ≈ résidu siRNA,
                    le mode de la validation in vitro), GARDE les arêtes.
      knockout    : feature → 0 (perte totale du produit) + arêtes GARDÉES
                    par défaut (= ``mask``) ; atténuation partielle optionnelle
                    via ``ko_edge_factor`` < 1.0 (fraction d'arêtes incidentes
                    conservée, tirage déterministe).
      overexpress : feature × factor.

    Le KO ``mask`` (feature=0, arêtes gardées) corrige le bug du KO ``cut``
    (V3.3-V5.4) : retirer toutes les arêtes — y compris IN — provoquait un
    choc topologique (l'embedding du nœud KO s'effondrait, ne pouvant plus
    agréger ses voisins) → KO incohérent avec KD/OE sur ~4 160 gènes.

    Args:
        data        : HeteroData baseline.
        target_idx  : torch.Tensor of gene indices to perturb.
        mode        : "knockdown" | "knockout" | "overexpress".
        factor      : multiplier for overexpress (ignored otherwise).
        ko_mode     : "mask" (défaut, feature→0 arêtes gardées) | "cut"
                      (feature→0 + arêtes coupées, legacy) | "soft"
                      (feature × ko_soft_factor, arêtes gardées).
        ko_soft_factor : multiplicateur quand ko_mode="soft" (défaut 0.1).
        kd_factor   : multiplicateur du knockdown (défaut 0.15, ∈ [0.1, 0.2]).
        ko_edge_factor : fraction des arêtes incidentes à CONSERVER lors d'un
                      KO non-cut (défaut 1.0 = mask intégral ; <1.0 = atténuation
                      partielle déterministe).
        legacy      : si True, restaure le comportement V3.3-V5.4
                      (KD: feature→0 ; KO: feature→0 + cut arêtes). Flag
                      temporaire en attendant la validation expérimentale.
    """
    pert = data.clone()
    x = pert["gene"].x.clone()
    if legacy:
        # --- Comportement historique V3.3-V5.4 ---
        if mode == "knockdown":
            x[target_idx] = 0.0
        elif mode == "overexpress":
            x[target_idx] = x[target_idx] * factor
        elif mode == "knockout":
            x[target_idx] = 0.0
        else:
            raise ValueError(f"unknown mode: {mode}")
    else:
        # --- Sémantique V5.5 ---
        if mode == "knockdown":
            x[target_idx] = x[target_idx] * kd_factor          # soft
        elif mode == "overexpress":
            x[target_idx] = x[target_idx] * factor
        elif mode == "knockout":
            if ko_mode == "soft":
                x[target_idx] = x[target_idx] * ko_soft_factor
            else:                                              # mask | cut
                x[target_idx] = 0.0
        else:
            raise ValueError(f"unknown mode: {mode}")
    pert["gene"].x = x

    # Gestion des arêtes pour le KO :
    #   * cut total : legacy OU ko_mode=="cut" (statu quo)
    #   * atténuation partielle : ko_edge_factor < 1.0 (garde une fraction)
    #   * sinon (mask, défaut) : arêtes intactes.
    cut_all = mode == "knockout" and (legacy or ko_mode == "cut")
    partial = (mode == "knockout" and not cut_all and ko_edge_factor < 1.0)
    if cut_all or partial:
        gen = torch.Generator().manual_seed(0)   # déterministe (reproductible)
        for et in list(pert.edge_types):
            ei = pert[et].edge_index
            if ei is None or ei.numel() == 0:
                continue
            incident = torch.zeros(ei.shape[1], dtype=torch.bool)
            if et[0] == "gene":
                incident |= torch.isin(ei[0], target_idx)
            if et[2] == "gene":
                incident |= torch.isin(ei[1], target_idx)
            if cut_all:
                keep = ~incident
            else:   # garde une fraction ko_edge_factor des arêtes incidentes
                rand = torch.rand(ei.shape[1], generator=gen)
                keep = (~incident) | (rand < ko_edge_factor)
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
                             quiescent_groups=None,
                             effector_axis_genes=None,
                             cluster_anchors=None,
                             transition_pairs=None):
    """Axes quiescent -> sénescent dans l'espace latent des gènes.

    effector_axis_genes : optionnel (pro_list, anti_list) de symboles. Si
        fourni, **axis_global est ANCRÉ sur les effecteurs** :
            axis = unit( mean(μ[pro]) − mean(μ[anti]) )
        au lieu du contraste d'expression des cell_groups. Référence causale
        (endpoints canoniques) plutôt que corrélationnelle → ne pivote pas
        avec les hyperparamètres (cf. ASNS §7.8, gnn_futur §2.2/§4b). Les
        axes_cluster restent expression-based (le headline = axis_global).

    cluster_anchors : optionnel dict[group_name -> anchor_spec]. Redéfinit le
        CENTROÏDE d'un ou plusieurs groupes à partir d'une liste de gènes au
        lieu de la pondération par expression sur tous les gènes. anchor_spec =
        liste de symboles (équipondéré) OU dict {symbole: poids}. Le centroïde
        ancré = Σ wᵢ μᵢ / Σ wᵢ sur les gènes trouvés. Sources typiques :
          - marqueurs DE spécifiques au cluster (DEGs_P16_cluster_k.csv) ;
          - ancrage manuel bibliographique (Ahn et al. 2025 — voir
            data/gnn_data/ahn_cluster_anchors.tsv).
        Seuls les groupes présents dans le dict sont ré-ancrés ; les autres
        gardent leur centroïde expression-based. axis_global, axes_cluster ET
        axes_transition héritent tous des centroïdes ré-ancrés.

    transition_pairs : optionnel liste de tuples (src, dst) de noms de groupes.
        Pour chaque paire, un axe de TRANSITION inter-cluster est calculé :
            axes_transition[(src, dst)] = unit( centers[dst] − centers[src] )
        Permet P4->ck (src=P4), le chemin de trajectoire (Ahn/Monocle3
        c1->c2->c3) ou toute paire ordonnée. None => aucun axe de transition.

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
        axes_transition : dict[(src, dst) -> (latent_dim,) unit vector]
            Axes de transition inter-cluster (vide si transition_pairs=None).
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
    s2i = {g: i for i, g in enumerate(gene_symbols)}

    def _anchor_centroid(spec, name):
        """Centroïde ancré sur une liste de gènes. spec = liste de symboles
        (poids 1) ou dict {symbole: poids}. Renvoie None si aucun gène trouvé."""
        items = list(spec.items()) if isinstance(spec, dict) else [(g, 1.0) for g in spec]
        idx, w = [], []
        for g, wg in items:
            if g in s2i:
                idx.append(s2i[g]); w.append(float(wg))
        if not idx:
            print(f"[warn] cluster_anchors['{name}'] : 0/{len(items)} gène(s) "
                  "trouvé(s) — fallback centroïde expression.")
            return None
        w = np.asarray(w, dtype=np.float32)
        if len(idx) < len(items):
            print(f"  [axes] anchor '{name}' : {len(idx)}/{len(items)} gènes trouvés.")
        return (w[:, None] * mu_base[idx]).sum(axis=0) / (w.sum() + 1e-8)

    cluster_anchors = cluster_anchors or {}
    centers: dict[str, np.ndarray] = {}
    for g in set(quiescent_list + p16_list):
        anchored = _anchor_centroid(cluster_anchors[g], g) if g in cluster_anchors else None
        if anchored is not None:
            centers[g] = anchored.astype(np.float32)
        else:
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

    # Override effecteur-ancré (gnn_futur §2.2/§4b) : remplace l'axe global
    # corrélationnel par un axe causal défini sur des effecteurs canoniques.
    if effector_axis_genes is not None:
        pro, anti = effector_axis_genes
        s2i = {g: i for i, g in enumerate(gene_symbols)}
        pro_i = [s2i[g] for g in pro if g in s2i]
        anti_i = [s2i[g] for g in anti if g in s2i]
        if pro_i and anti_i:
            pro_c = mu_base[pro_i].mean(axis=0)
            anti_c = mu_base[anti_i].mean(axis=0)
            axis_global = unit(pro_c - anti_c, "effector")
            print(f"  [axes] EFFECTOR-anchored axis_global : "
                  f"pro={len(pro_i)}/{len(pro)} anti={len(anti_i)}/{len(anti)} gènes "
                  f"(remplace le contraste d'expression).")
        else:
            print(f"[warn] effector axis : pro={len(pro_i)} anti={len(anti_i)} "
                  "gènes trouvés — fallback axe expression.")

    # Axes de transition inter-cluster (src -> dst). On complète `centers` pour
    # tout groupe référencé par une paire mais absent des listes quiescent/p16.
    axes_transition: dict[tuple, np.ndarray] = {}
    if transition_pairs:
        for src, dst in transition_pairs:
            for g in (src, dst):
                if g not in centers:
                    anchored = (_anchor_centroid(cluster_anchors[g], g)
                                if g in cluster_anchors else None)
                    if anchored is not None:
                        centers[g] = anchored.astype(np.float32)
                    elif f"mean_{g}" in expr_by_gene.columns:
                        w = expr_by_gene[f"mean_{g}"].to_numpy().astype(np.float32)
                        centers[g] = (w[:, None] * mu_base).sum(axis=0) / (float(w.sum()) + 1e-8)
                    else:
                        print(f"[warn] transition group '{g}' absent du "
                              "group_expression — paire ignorée.")
            if src in centers and dst in centers:
                axes_transition[(src, dst)] = unit(centers[dst] - centers[src],
                                                   f"{src}->{dst}")
        print(f"  [axes] {len(axes_transition)} axe(s) de transition : "
              f"{[f'{s}->{d}' for s, d in axes_transition]}")
    return axis_global, axes_cluster, axes_transition, centers


def cell_group_shift_projected(mu_base: np.ndarray,
                                mu_pert: np.ndarray,
                                group_expr: pd.DataFrame,
                                gene_symbols: np.ndarray,
                                target_idx: torch.Tensor,
                                axis_global: np.ndarray,
                                axes_cluster: dict,
                                group_names=CELL_GROUPS,
                                target_degree: int | None = None,
                                extent_threshold: float = 1e-3,
                                collect_dz_mean: bool = False,
                                axes_transition: dict | None = None) -> pd.DataFrame:
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
        collect_dz_mean : si True, stocke dans `df.attrs["dz_mean_global"]`
            le vecteur `dz_mean_c` (effet moyen pondéré par w_diff, AXIS-INDÉPENDANT)
            pour chaque cell_group de l'axe global → permet de **re-projeter sur
            n'importe quel axe** (`cos(dz_mean_c, u)`) sans ré-encoder (cache Δz,
            test de spécificité d'axe, cf. ARCH §10.4.7).
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
    dz_mean_store: dict[str, np.ndarray] = {}   # cache Δz (axis-indépendant)
    # Échelle Σ w_diff par groupe (scalaire AXIS-INDÉPENDANT) : permet de
    # reconstruire proj_signed_diff = Σw_diff · (dz_mean·u) pour TOUT axe u à
    # partir du cache, sans re-perturber (cf. reproject_axes.py).
    w_diff_sum_store: dict[str, float] = {}

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
        # Cache Δz : dz_mean est AXIS-INDÉPENDANT (ne dépend que de w_diff et Δz)
        # → on le stocke une fois (axe global) pour re-projeter sur tout axe u.
        if collect_dz_mean and axis_type == "global":
            dz_mean_store[grp] = dz_mean.astype(np.float32)
            w_diff_sum_store[grp] = total_w_diff
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

    # --- Axe de transition src -> dst : pondération par l'expression du groupe
    # DESTINATION (les gènes caractérisant l'état d'arrivée). Le label de groupe
    # encode la paire ; proj_signed_cosine reste la métrique sans-poids clé.
    for (src, dst), axis_u in (axes_transition or {}).items():
        if dst not in group_names:
            continue
        rows.append(_row(f"{src}->{dst}", "transition", axis_u,
                         group_names.index(dst)))

    df = pd.DataFrame(rows)
    if collect_dz_mean:
        df.attrs["dz_mean_global"] = dz_mean_store      # {group: (latent,) f32}
        df.attrs["w_diff_sum_global"] = w_diff_sum_store  # {group: float} échelle
        df.attrs["dz_group_names"] = list(group_names)
    return df


# --------------------------------------------------------------------------- #
# Readout signé — fan-out 1-hop (Solution A signée, gnn_futur §8.A / §9.2).
# --------------------------------------------------------------------------- #
# Types d'arêtes signées DIRIGÉES où le gène source S est le RÉGULATEUR
# (S → cible T). On exclut `tf_curated_by` qui est l'arête miroir
# (cible → TF), de direction inverse : l'inclure compterait S comme aval.
# `reactome_fi` est conservée malgré une directionnalité parfois faible
# (les FI Reactome sont souvent bidirectionnelles) ; à filtrer si besoin.
FANOUT_EDGE_TYPES: tuple = (
    ("gene", "signaling", "gene"),
    ("gene", "tf_curated", "gene"),
    ("gene", "reactome_fi", "gene"),
)


def precompute_signed_fanout_context(model, data, mu_base: np.ndarray,
                                     axis_global: np.ndarray,
                                     fanout_edge_types=FANOUT_EDGE_TYPES,
                                     sign_col: int = 1) -> dict:
    """Pré-calcule (une seule fois) le contexte du readout signé fan-out.

    Invariant sur toutes les perturbations : ne dépend que de l'encoder gelé
    (mu_base), de l'axe et du graphe signé. Mis en cache sur ``data._sf_ctx``.

    Calcule :
      * ``role_latent_sign`` (n_genes,) : signe du rôle AU REPOS de chaque gène
        sur l'axe sénescence = sgn(mu_base·u − médiane). >0 = lean pro-sén,
        <0 = lean anti-sén. Auto-référencé (médiane de la population de gènes)
        → pas besoin du centroïde quiescent, robuste cross-version.
      * ``proj_base`` (n_genes,) : projection brute mu_base·u (diagnostic).
      * ``fanout`` : dict ``src_idx -> (dst_idx[], sign_known[], sign_pred[])``
        sur les arêtes signées sortantes. ``sign_pred`` = signe du score
        bilinéaire ``predict_sign_score`` (le signe APPRIS, §9.2) ; ``sign_known``
        = signe curé (edge_attr[:, sign_col], CollecTRI/SIGNOR/Reactome).
        Si pas de décodeur bilinéaire (checkpoint non-V5), ``sign_pred`` retombe
        sur ``sign_known`` et ``has_bilinear`` est False.

    Returns: le dict de contexte (aussi attaché à ``data._sf_ctx``).
    """
    axis = np.asarray(axis_global, dtype=np.float32)
    proj_base = (np.asarray(mu_base, dtype=np.float32) @ axis)
    role_med = float(np.median(proj_base))
    role_latent_sign = np.sign(proj_base - role_med).astype(np.float32)

    bilinear = getattr(model, "bilinear_decoder", None)
    has_bilinear = bilinear is not None

    # Collecte des arêtes signées sortantes (régulateur = src).
    src_all, dst_all, sk_all = [], [], []
    present_types = [tuple(et) for et in fanout_edge_types
                     if tuple(et) in set(data.edge_types)]
    for et in present_types:
        store = data[et]
        ei = store.edge_index.cpu().numpy()
        ea = getattr(store, "edge_attr", None)
        if ea is None or ea.ndim < 2 or ea.shape[1] <= sign_col:
            continue
        signs = ea[:, sign_col].cpu().numpy()
        src_all.append(ei[0]); dst_all.append(ei[1]); sk_all.append(signs)

    if not src_all:
        ctx = {"role_latent_sign": role_latent_sign, "proj_base": proj_base,
               "fanout": {}, "has_bilinear": has_bilinear,
               "n_signed_edges": 0, "edge_types": []}
        data._sf_ctx = ctx
        return ctx

    src = np.concatenate(src_all)
    dst = np.concatenate(dst_all)
    sign_known = np.sign(np.concatenate(sk_all)).astype(np.float32)

    # Retrait des self-loops (T == S) qui ne sont pas un fan-out.
    keep = src != dst
    src, dst, sign_known = src[keep], dst[keep], sign_known[keep]

    # Signe APPRIS via le décodeur bilinéaire (un seul passage sur z=mu_base).
    if has_bilinear:
        dev = next(bilinear.parameters()).device
        z = torch.as_tensor(np.asarray(mu_base, dtype=np.float32), device=dev)
        ei_t = torch.as_tensor(np.stack([src, dst]), dtype=torch.long, device=dev)
        with torch.no_grad():
            logit = bilinear.predict_sign_score(z, ei_t).cpu().numpy()
        sign_pred = np.sign(logit).astype(np.float32)
        # logit == 0 (rare) → retombe sur le signe connu.
        zero = sign_pred == 0
        sign_pred[zero] = sign_known[zero]
    else:
        sign_pred = sign_known.copy()

    # Regroupement par source pour un slicing O(1) par gène perturbé.
    order = np.argsort(src, kind="stable")
    src_s, dst_s = src[order], dst[order]
    sk_s, sp_s = sign_known[order], sign_pred[order]
    boundaries = np.searchsorted(src_s, np.arange(src_s.max() + 2))
    fanout = {}
    uniq = np.unique(src_s)
    for s in uniq:
        a, b = boundaries[s], boundaries[s + 1]
        fanout[int(s)] = (dst_s[a:b], sk_s[a:b], sp_s[a:b])

    ctx = {
        "role_latent_sign": role_latent_sign,
        "proj_base": proj_base,
        "fanout": fanout,
        "has_bilinear": has_bilinear,
        "n_signed_edges": int(src.size),
        "edge_types": ["/".join(et) for et in present_types],
    }
    data._sf_ctx = ctx
    return ctx


def signed_fanout_scalars(data, mu_base: np.ndarray, mu_pert: np.ndarray,
                          target_idx, axis_global: np.ndarray,
                          de_role_sign: np.ndarray | None = None,
                          return_table: bool = False):
    """Readout signé + cohérence de fan-out pour la/les cible(s) perturbée(s) S.

    Pour chaque arête signée sortante S→T (S ∈ target_idx) :
      proj_T   = Δz_T · u            (réponse de T projetée sur l'axe ; signée)
      |proj_T| = magnitude axe-pertinente du mouvement de T
      align_T  = sgn(role_T) · sign_{S→T}   (∈ {−1, +1})
                 role_T = rôle au repos de T ; sign = signe régulateur S→T.
                 Inhiber-un-pro-sén (−·−) et activer-un-anti-sén (+·… ) reçoivent
                 le même signe → s'additionnent au lieu de s'annuler (§8.A).

    Score (pour chaque source de rôle × source de signe) :
      signed_readout = Σ_T |proj_T| · align_T   (>0 pro-sén, <0 anti-sén)
      signed_coherence = |Σ_T align_T| / N_T  ∈ [0,1]  (≈ le « % cohérent » §1.2)

    Convention de signe identique aux projections : >0 pousse VERS P16.

    Args:
        de_role_sign : (n_genes,) signe du rôle DE par gène (∈{−1,0,+1}), ou None.
        return_table : si True, renvoie aussi un DataFrame long par cible.

    Returns:
        (scalars: dict, table: pd.DataFrame|None). dict vide si pas de contexte
        ou pas de fan-out signé pour S.
    """
    ctx = getattr(data, "_sf_ctx", None)
    if ctx is None or not ctx["fanout"]:
        return {}, None

    axis = np.asarray(axis_global, dtype=np.float32)
    dz = (np.asarray(mu_pert, dtype=np.float32)
          - np.asarray(mu_base, dtype=np.float32))
    role_sign = ctx["role_latent_sign"]
    t_np = (target_idx.cpu().numpy() if isinstance(target_idx, torch.Tensor)
            else np.asarray(target_idx))
    src_set = set(int(s) for s in t_np)

    dsts, sk, sp = [], [], []
    for s in src_set:
        if s in ctx["fanout"]:
            d, k, p = ctx["fanout"][s]
            dsts.append(d); sk.append(k); sp.append(p)
    if not dsts:
        return {}, None
    dst = np.concatenate(dsts)
    sign_known = np.concatenate(sk)
    sign_pred = np.concatenate(sp)

    proj_t = dz[dst] @ axis                    # (n_edges,) Δz_T · u
    abs_proj = np.abs(proj_t)
    role_lat = role_sign[dst]                  # sgn(role) latent

    def _score(role_vec, sign_vec):
        align = role_vec * sign_vec            # ∈ {−1,0,+1}
        readout = float((abs_proj * align).sum())
        n = int((align != 0).sum())
        coherence = float(abs(align.sum()) / n) if n else 0.0
        return readout, coherence, n

    ro_lat_pred, coh_lat_pred, n_eff = _score(role_lat, sign_pred)
    ro_lat_known, coh_lat_known, _ = _score(role_lat, sign_known)

    n_fanout = int(dst.size)
    pred_known_agree = (float((sign_pred == sign_known).mean())
                        if n_fanout else None)

    # Scalaires in-perturbation = APERÇU role_latent uniquement (pas de
    # role_pert, qui est report-side). Suffixe `_latent` explicite : aucun
    # « signed_readout » headline (cf. décision 2026-06-09, design_log
    # §14bis.6quatervicies — role_latent n'est qu'un des 3 rôles, et pas le
    # meilleur). Les scores autoritatifs (3 rôles + flag) sont dans
    # cross_seed_gene_ranking.tsv via perturb_report.
    scalars = {
        "signed_fanout_n": n_fanout,
        "signed_readout_latent": ro_lat_pred,        # rôle latent × signe appris
        "signed_coherence_latent": coh_lat_pred,
        "signed_coherence_latent_knownsign": coh_lat_known,  # × signe curé
        "signed_pred_known_agree": pred_known_agree,
        "signed_has_bilinear": bool(ctx["has_bilinear"]),
    }

    if de_role_sign is not None:
        role_de = np.asarray(de_role_sign, dtype=np.float32)[dst]
        ro_de_pred, coh_de_pred, _ = _score(role_de, sign_pred)
        scalars["signed_readout_de"] = ro_de_pred
        scalars["signed_coherence_de"] = coh_de_pred
        # Concordance des deux rôles là où les deux sont définis.
        both = (role_lat != 0) & (role_de != 0)
        scalars["signed_role_latde_agree"] = (
            float((role_lat[both] == role_de[both]).mean())
            if both.any() else None)

    table = None
    if return_table:
        src_col = np.concatenate([
            np.full(len(ctx["fanout"][s][0]), s, dtype=np.int64)
            for s in src_set if s in ctx["fanout"]])
        gene_symbols = None  # rempli par l'appelant si besoin
        table = pd.DataFrame({
            "source_idx": src_col,
            "target_idx": dst,
            "sign_known": sign_known,
            "sign_pred": sign_pred,
            "proj_target": proj_t,
            "abs_proj_target": abs_proj,
            "role_latent_sign": role_lat,
        })
    return scalars, table


# --------------------------------------------------------------------------- #
# Run loader.
# --------------------------------------------------------------------------- #
def resolve_device(device: str = "auto") -> "torch.device":
    """'auto' → cuda si dispo sinon cpu ; 'cuda'/'cpu' forcés."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[device] cuda demandé mais indisponible → fallback cpu.")
        device = "cpu"
    print(f"[device] perturbation sur {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    return torch.device(device)


def load_run(run_dir: Path, hidden, latent, n_layers, n_heads, device="cpu"):
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

    # V4.2 : edge_dim peut différer du catalogue (coexpr 1→6 en differential,
    # ppi 1→2 si --dedup-ppi-signed annotate). On infère depuis edge_attr
    # du graphe sauvé pour aligner l'encoder sur le state_dict.
    edge_dim_overrides = {}
    for et in data.edge_types:
        store = data[et]
        ea = getattr(store, "edge_attr", None)
        if ea is not None and ea.ndim >= 2 and ea.shape[1] > 0:
            edge_dim_overrides[tuple(et)] = int(ea.shape[1])

    # V5 (TIER 1c) : lire les flags signed_message/signed_decoder depuis
    # vgae_metrics.json (persisté à l'entraînement). Sans ces flags, on
    # (i) instancierait des GATConv plain au lieu de SignedGATConv → le
    # state_dict charge mais les signes seraient ignorés à l'inférence, et
    # (ii) refuserait les clés bilinear_decoder.* présentes dans les
    # checkpoints `--signed-decoder`.
    signed_message = False
    signed_decoder = False
    metrics_path = run_dir / "vgae_metrics.json"
    if metrics_path.exists():
        try:
            import json as _json
            with open(metrics_path) as _f:
                _m = _json.load(_f)
            _hp = _m.get("hyperparams", {}) if isinstance(_m, dict) else {}
            signed_message = bool(_hp.get("signed_message", False))
            signed_decoder = bool(_hp.get("signed_decoder", False))
        except Exception as _e:
            print(f"[warn] lecture {metrics_path.name} échouée : {_e} — "
                  f"V5 flags supposés False (backward-compat V4.x).")

    encoder = HeteroEncoder(
        gene_in=data["gene"].x.shape[1],
        cell_in=data["cell_group"].x.shape[1],
        hidden=hidden, latent=latent, n_layers=n_layers, n_heads=n_heads,
        # V3.6 : ne créer que les GATConv pour les edge_types réellement
        # présents dans le graphe entraîné — sinon le state_dict refuse
        # de charger sur les ablations (--no-ppi, --no-coexpr, etc.).
        available_edge_types=list(data.edge_types),
        edge_dim_overrides=edge_dim_overrides or None,
        signed_message=signed_message,
    )
    state_path = run_dir / "best_vgae.pt"
    if not state_path.exists():
        state_path = run_dir / "vgae_weights.pt"
    state_dict = torch.load(state_path, weights_only=True)

    # V5 safety : si vgae_metrics.json absent OU faux négatif sur
    # signed_decoder mais que le state_dict contient bilinear_decoder.*,
    # on bascule signed_decoder=True pour instancier correctement.
    has_bilinear_keys = any(k.startswith("bilinear_decoder.")
                            for k in state_dict.keys())
    if has_bilinear_keys and not signed_decoder:
        print("  [load_run] bilinear_decoder.* détecté dans le state_dict "
              "mais signed_decoder=False dans vgae_metrics.json — "
              "activation a posteriori.")
        signed_decoder = True

    # V5.3 — détecter signed_dim depuis le state_dict (3 cas) :
    #   (a) V5.3 checkpoint : `bilinear_decoder.signed_proj.weight` présent
    #       → signed_dim = signed_proj.weight.shape[0].
    #   (b) V5.1/V5.2 checkpoint (bilinear_decoder.* mais pas signed_proj)
    #       → signed_dim = latent, init identité ⇒ équivalence numérique.
    #   (c) Pas de bilinéaire du tout → bilinear_decoder=None.
    _proj_w = state_dict.get("bilinear_decoder.signed_proj.weight", None)
    if signed_decoder:
        signed_dim = int(_proj_w.shape[0]) if _proj_w is not None else latent
        bilinear_decoder = BilinearSignedDecoder(latent, signed_dim=signed_dim)
    else:
        bilinear_decoder = None
    model = VGAE(encoder, bilinear_decoder=bilinear_decoder)
    if signed_message or signed_decoder:
        print(f"  [load_run] V5 flags : signed_message={signed_message} "
              f"signed_decoder={signed_decoder}"
              + (f" signed_dim={signed_dim}" if signed_decoder else ""))

    # V5.3 strict=False si signed_proj absent (V5.1/V5.2 chargé en V5.3)
    _strict = (_proj_w is not None) if signed_decoder else True
    _missing, _unexpected = model.load_state_dict(state_dict, strict=_strict)
    if not _strict and _unexpected:
        _other = [k for k in _unexpected
                  if not k.startswith("bilinear_decoder.signed_proj.")]
        if _other:
            raise RuntimeError(f"state_dict clés inattendues : {_other}")
        print(f"  [load_run] V5.3 backward-compat : signed_proj initialisé "
              f"à identité (checkpoint V5.1/V5.2 sans signed_proj).")
    model.eval()
    # GPU si demandé (data + model sur le device ; mu ramené en .cpu().numpy()
    # dans compute_importance/encode → reste compatible avec le readout numpy).
    dev = resolve_device(device) if isinstance(device, str) else device
    if dev.type != "cpu":
        data = data.to(dev)
        model = model.to(dev)
    return data, model, gene_symbols, gene_to_idx, baseline, group_expr


# --------------------------------------------------------------------------- #
# Reusable API — called both by main() and by perturb_top_genes.py (--all-*).
# --------------------------------------------------------------------------- #
def prepare_baseline(model, data, baseline_df, gene_symbols, group_expr=None,
                     quiescent_groups=None, p16_groups=None,
                     effector_axis_genes=None, cluster_anchors=None,
                     transition_pairs=None):
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
        axes_transition : dict[(src, dst) -> (latent,)] axes inter-cluster ({} / None).

    Args (suite):
        cluster_anchors : optionnel dict[group -> gene-spec] (voir
            compute_senescence_axes) : ré-ancre les centroïdes de groupe sur des
            marqueurs DE ou des ancres manuelles au lieu de l'expression.
        transition_pairs : optionnel liste de (src, dst) → axes de transition.
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
        axis_global, axes_cluster, axes_transition, _centers = compute_senescence_axes(
            mu_base, group_expr, gene_symbols,
            p16_groups=tuple(p16_groups),
            quiescent_groups=(tuple(quiescent_groups)
                              if quiescent_groups is not None else None),
            effector_axis_genes=effector_axis_genes,
            cluster_anchors=cluster_anchors,
            transition_pairs=transition_pairs)
        print(f"  axis_global ‖·‖ pre-unit = {np.linalg.norm(axis_global):.4f} "
              f"(1.0 attendu post-normalisation)")
    else:
        axis_global, axes_cluster, axes_transition = None, None, {}

    return (spec, base_imp, base_rank, z_cg_base, mu_base,
            axis_global, axes_cluster, axes_transition)


def run_perturbation_once(model, data, gene_symbols, gene_to_idx,
                          spec, base_imp, base_rank, z_cg_base, mu_base,
                          targets, mode, factor, top_k, fdr,
                          reactome, background, group_expr=None,
                          axis_global=None, axes_cluster=None,
                          out_dir=None, write_full=True,
                          include_details=False,
                          ko_mode="mask", ko_soft_factor=0.1,
                          kd_factor=0.15, ko_edge_factor=1.0, legacy=False,
                          axes_transition=None):
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

    pert_data = apply_perturbation(data, target_idx, mode, factor,
                                   ko_mode=ko_mode,
                                   ko_soft_factor=ko_soft_factor,
                                   kd_factor=kd_factor,
                                   ko_edge_factor=ko_edge_factor,
                                   legacy=legacy)
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
            target_degree=target_degree,
            collect_dz_mean=bool(getattr(data, "_cache_dz", False)),
            axes_transition=axes_transition)

    # --- Readout SIGNÉ (Solution A signée, §8.A / §9.2) : pondère chaque
    # cible du fan-out signé de S par rôle_cible × signe_S→T → ne s'annule
    # plus. Contexte (rôle au repos + signes prédits) calculé une seule fois,
    # mis en cache sur `data`. DE-free par défaut ; rôle DE optionnel si
    # `data._de_role_sign` a été attaché par l'appelant.
    signed_scalars, signed_table = {}, None
    _collect_fanout = bool(getattr(data, "_collect_signed_fanout", False))
    if axis_global is not None:
        if getattr(data, "_sf_ctx", None) is None:
            precompute_signed_fanout_context(model, data, mu_base, axis_global)
        signed_scalars, signed_table = signed_fanout_scalars(
            data, mu_base, mu_pert, target_idx, axis_global,
            de_role_sign=getattr(data, "_de_role_sign", None),
            return_table=(_collect_fanout
                          or (out_dir is not None and write_full)))

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
        _proj_transition = shift_proj_df[shift_proj_df["axis_type"] == "transition"]
        # Le cache Δz vit dans shift_proj_df.attrs (dict d'arrays) ; les slices en
        # héritent et casseraient pd.concat (comparaison `==` des attrs). On vide
        # les attrs des slices — le parent shift_proj_df garde l'info (extraite plus bas).
        _proj_global.attrs = {}
        _proj_cluster.attrs = {}
        _proj_transition.attrs = {}

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
        if len(_proj_transition) > 0:
            # Clés = "src->dst" (label de groupe du _row de transition).
            summary["cell_group_shift_projected_transition"] = _pack(_proj_transition)

        # Cache Δz (axis-indépendant) : remonté pour re-projection multi-axe
        # (test de spécificité d'axe, cf. ARCH §10.4.7). Empilé en (n_groups, latent).
        _dz_store = shift_proj_df.attrs.get("dz_mean_global")
        if _dz_store:
            _gn = shift_proj_df.attrs.get("dz_group_names", list(_dz_store.keys()))
            summary["_dz_mean_global"] = np.stack(
                [_dz_store[g] for g in _gn]).astype(np.float32)
            summary["_dz_group_names"] = list(_gn)
            # Échelle Σw_diff par groupe (même ordre) → proj_signed_diff exact
            # reconstruit hors-ligne pour tout axe (reproject_axes.py).
            _ws = shift_proj_df.attrs.get("w_diff_sum_global", {})
            summary["_w_diff_sum_global"] = np.asarray(
                [float(_ws.get(g, 0.0)) for g in _gn], dtype=np.float32)

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

    # --- Readout signé : scalaires par source (fan-out 1-hop). Plats →
    # consommables tels quels par flatten_summary / perturb_report.
    if signed_scalars:
        summary.update(signed_scalars)

    # Table long-format par cible pour accumulation batch (mode all-genes) :
    # primitives d'arête uniquement (les rôles = propriétés cible jointes
    # côté report). Mappée en symboles ici, `mode` ajouté pour l'agrégation.
    if _collect_fanout and signed_table is not None and len(signed_table) > 0:
        t = signed_table
        summary["_signed_fanout_rows"] = {
            "source": gene_symbols[t["source_idx"].to_numpy()].tolist(),
            "target": gene_symbols[t["target_idx"].to_numpy()].tolist(),
            "sign_known": t["sign_known"].to_numpy().tolist(),
            "sign_pred": t["sign_pred"].to_numpy().tolist(),
            "proj_target": t["proj_target"].to_numpy().tolist(),
            "role_latent_sign": t["role_latent_sign"].to_numpy().tolist(),
        }

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
            if signed_table is not None and len(signed_table) > 0:
                tbl = signed_table.copy()
                tbl.insert(0, "source", gene_symbols[tbl["source_idx"].to_numpy()])
                tbl.insert(2, "target", gene_symbols[tbl["target_idx"].to_numpy()])
                tbl.drop(columns=["source_idx", "target_idx"], inplace=True)
                tbl.to_csv(out_dir / "signed_fanout.tsv", sep="\t", index=False)
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
    ap.add_argument("--ko-mode", choices=["cut", "mask", "soft"],
                    default="mask",
                    help="V5.5 : sémantique du knockout (si --mode knockout). "
                         "mask (DÉFAUT) = feature à 0, GARDE le graphe intact "
                         "(corrige le bug 'KD↔OE cohérents, KO opposite' sur "
                         "4 160 gènes). cut (legacy) = feature à 0 + coupe les "
                         "arêtes incidentes (choc topologique). soft = feature × "
                         "ko_soft_factor, garde le graphe.")
    ap.add_argument("--ko-soft-factor", type=float, default=0.1,
                    help="Multiplicateur appliqué quand --ko-mode soft "
                         "(défaut 0.1 = résidu 10%% d'expression).")
    ap.add_argument("--kd-factor", type=float, default=0.15,
                    help="V5.5 : multiplicateur du knockdown soft (défaut 0.15, "
                         "∈ [0.1, 0.2] ≈ résidu siRNA réel). Ignoré si --legacy "
                         "(KD = feature à 0).")
    ap.add_argument("--ko-edge-factor", type=float, default=1.0,
                    help="V5.5 : fraction des arêtes incidentes CONSERVÉES lors "
                         "d'un KO non-cut (défaut 1.0 = mask intégral ; <1.0 = "
                         "atténuation partielle déterministe).")
    ap.add_argument("--legacy", action="store_true",
                    help="Restaure la sémantique V3.3-V5.4 (KD: feature→0 ; "
                         "KO: feature→0 + coupe arêtes). Flag TEMPORAIRE en "
                         "attendant la validation expérimentale du refactor V5.5.")
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

    (spec, base_imp, base_rank, z_cg_base, mu_base,
     axis_global, axes_cluster, axes_transition) = prepare_baseline(
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
        axes_transition=axes_transition,
        out_dir=out_dir, write_full=not args.summary_only,
        ko_mode=args.ko_mode, ko_soft_factor=args.ko_soft_factor,
        kd_factor=args.kd_factor, ko_edge_factor=args.ko_edge_factor,
        legacy=args.legacy)
    summary["ko_mode"] = args.ko_mode
    summary["legacy"] = args.legacy
    summary["kd_factor"] = args.kd_factor if not args.legacy else 0.0
    if args.ko_mode == "soft":
        summary["ko_soft_factor"] = args.ko_soft_factor
    if args.ko_edge_factor < 1.0:
        summary["ko_edge_factor"] = args.ko_edge_factor
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
