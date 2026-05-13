"""
_vgae_model.py — classes VGAE importables sans effets de bord.

**Code duplicate** de `gnn_vgae.py` lignes ~1500-1844. Ce fichier
contient uniquement les définitions de classe (HeteroEncoder + VGAE) +
les imports nécessaires, sans la moindre logique d'entraînement
ou d'I/O.

POURQUOI ce duplicat ?
  `gnn_vgae.py` est un script monolithique (3 100 lignes) qui exécute
  à l'import tout le pipeline (parsing CLI, création de dossiers,
  chargement des données, …). Il n'est donc pas importable depuis
  un autre script comme `edge_attention.py`. La modularisation
  cible (Tier 2.5 du TODO + §3 du pipeline_design) extraira
  proprement les classes dans `src/gnn_huvec/models/vgae.py`.
  En attendant, ce fichier offre un point d'import safe.

**À SYNCHRONISER MANUELLEMENT** avec `gnn_vgae.py` lors de toute
modification de HeteroEncoder ou VGAE. Sera supprimé après refactor
Tier 2.5.

Cf. Kipf & Welling 2016 (VGAE) + Veličković et al. 2018 (GAT) pour
les fondements théoriques. La signature et le state_dict sont
strictement identiques à ceux de `gnn_vgae.py`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv


class HeteroEncoder(nn.Module):
    """Encoder hétérogène multi-couches pour le VGAE. Cf. gnn_vgae.py §8."""

    EDGE_TYPE_CATALOG = [
        (("gene", "ppi", "gene"), 1),
        (("gene", "same_pathway", "gene"), None),
        (("gene", "regulates", "gene"), 1),
        (("gene", "regulated_by", "gene"), 1),
        (("cell_group", "expresses", "gene"), 7),
        (("gene", "expressed_in", "cell_group"), 7),
        (("gene", "coexpression", "gene"), 1),
        (("gene", "metabolic_cocatalysis", "gene"), 2),
        (("gene", "signaling", "gene"), 2),
        (("gene", "tf_curated", "gene"), 2),
        (("gene", "tf_curated_by", "gene"), 2),
    ]

    def __init__(self, gene_in, cell_in, hidden, latent, n_layers,
                 n_heads=4, dropout=0.2, available_edge_types=None):
        super().__init__()
        self.n_layers = n_layers
        head_dim = hidden // n_heads

        self.gene_proj = nn.Linear(gene_in, hidden)
        self.cell_proj = nn.Linear(cell_in, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if available_edge_types is None:
            edge_types_dims = list(self.EDGE_TYPE_CATALOG)
        else:
            available = set(tuple(et) for et in available_edge_types)
            edge_types_dims = [(et, dim) for et, dim in self.EDGE_TYPE_CATALOG
                               if et in available]
            unknown = available - {et for et, _ in self.EDGE_TYPE_CATALOG}
            if unknown:
                print(f"  [warn] edge_types inconnus dans le catalogue : {unknown}")
        if not edge_types_dims:
            raise ValueError(
                "HeteroEncoder : aucun edge_type actif. Vérifie --no-* / la "
                "présence de data.edge_types."
            )
        self.edge_dims = {et: dim for et, dim in edge_types_dims}

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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = {
            "gene": F.relu(self.gene_proj(x_dict["gene"])),
            "cell_group": F.relu(self.cell_proj(x_dict["cell_group"])),
        }

        for i in range(self.n_layers):
            x_prev = {k: v.clone() for k, v in x_dict.items()}
            active_edges = {k: v for k, v in edge_index_dict.items()
                            if v.numel() > 0}
            if edge_attr_dict is not None:
                active_attrs = {
                    k: edge_attr_dict[k]
                    for k in active_edges
                    if k in edge_attr_dict and self.edge_dims.get(k) is not None
                }
                x_dict = self.convs[i](x_dict, active_edges,
                                       edge_attr_dict=active_attrs)
            else:
                x_dict = self.convs[i](x_dict, active_edges)

            for key, prev in x_prev.items():
                if key not in x_dict:
                    x_dict[key] = prev

            for key in list(x_dict.keys()):
                if key not in x_prev:
                    continue
                if x_dict[key] is x_prev[key]:
                    continue
                x_dict[key] = self.norms[i][key](x_dict[key])
                x_dict[key] = F.relu(x_dict[key])
                x_dict[key] = self.dropout(x_dict[key])
                x_dict[key] = x_dict[key] + x_prev[key]

        gene_h = x_dict["gene"]
        mu = self.mu_head(gene_h)
        logvar = self.logvar_head(gene_h)
        return mu, logvar


class SignedGATConv(GATConv):
    """GATConv qui multiplie chaque message par son `sign` d'arête.

    V5 (cf. §14bis.6septies du rapport) — extension du design A pour
    forcer le signe à influencer le MESSAGE (et pas seulement
    l'attention via `edge_dim=2`). Pour les edge_types signés
    (`signaling`, `tf_curated`, `tf_curated_by`), une arête `sign=-1`
    propage `-W·h_j` au lieu de `+W·h_j` — sémantique « inhibition »
    explicitement codée dans la mise à jour d'embedding.

    Référence : Derr et al. 2018 *ICDM* SGCN §3.2 (balance theory
    pour message-passing signé), adapté au cadre GAT.

    Le `sign` est lu depuis `edge_attr[:, sign_col]` (par défaut
    colonne 1 — convention V4 : `edge_attr=[score, sign]`).
    Backward-compat : si `edge_attr` est None ou `sign_col` invalide,
    retombe sur GATConv standard.
    """

    def __init__(self, *args, sign_col: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sign_col = sign_col

    def message(self, x_j, alpha, index, ptr, size_i, **kwargs):
        # Recup standard du message GATConv : α_ij · W·h_j (broadcast head).
        out = super().message(x_j=x_j, alpha=alpha, index=index,
                              ptr=ptr, size_i=size_i, **kwargs)
        # Application du sign si disponible dans edge_attr (via kwargs).
        # PyG passe edge_attr en kwargs si edge_dim défini.
        edge_attr = kwargs.get("edge_attr", None)
        if edge_attr is None or edge_attr.ndim < 2:
            return out
        if edge_attr.shape[1] <= self.sign_col:
            return out
        sign = edge_attr[:, self.sign_col].view(-1, 1, 1)  # broadcast head, feat
        # Si sign ∈ {-1, 0, +1}, sign=0 (info inconnue) → message inchangé.
        # Convention : message *= sign si sign != 0, sinon *= 1.
        scale = torch.where(sign == 0,
                            torch.ones_like(sign),
                            sign)
        return out * scale.squeeze(-1) if out.ndim == 2 else out * scale


class BilinearSignedDecoder(nn.Module):
    """Décodeur bilinéaire à 2 canaux pour reconstruction signée.

    V5 (cf. §14bis.6septies). Pour une arête (i, j) avec sign ∈ {+1, -1} :

        p(activate, i→j) = σ(z_i^T W_+ z_j)
        p(inhibit,  i→j) = σ(z_i^T W_- z_j)

    Au moment de la loss, on prend le canal correspondant au sign cible :
    `logit = sign_pos · logit_+ + sign_neg · logit_-` où sign_pos =
    1{sign > 0} et sign_neg = 1{sign < 0}. Pour sign=0 (PPI non-signé),
    on retombe sur `σ(z_i^T W_0 z_j)` (3e canal "interaction non signée").

    Préserve la compatibilité avec le décodeur cosinus existant pour
    les edge_types unsigned (PPI, coexpr, REACTOME) en exposant
    `forward_cosine()` séparément.

    Référence : Liu et al. 2024 *NAR* SGAT-bilinear ; généralisation du
    DistMult de Yang 2015 *ICLR* à la classification signée.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        # Trois matrices : activation (+), inhibition (-), unsigned (0)
        # Initialisation identité + bruit pour partir de l'inner product
        eye = torch.eye(latent_dim)
        self.W_pos = nn.Parameter(eye + 0.01 * torch.randn(latent_dim, latent_dim))
        self.W_neg = nn.Parameter(eye + 0.01 * torch.randn(latent_dim, latent_dim))
        self.W_zero = nn.Parameter(eye + 0.01 * torch.randn(latent_dim, latent_dim))

    def forward_signed(self, z: torch.Tensor, edge_index: torch.Tensor,
                       edge_sign: torch.Tensor) -> torch.Tensor:
        """Décode des arêtes signées.

        Args:
            z : embedding latent (n_nodes, latent_dim).
            edge_index : (2, E) — paires (src, dst).
            edge_sign : (E,) — sign ∈ {-1, 0, +1}.

        Returns:
            logits : (E,) — logit non-normalisé. Appliquer σ pour proba.
        """
        z_src = z[edge_index[0]]  # (E, D)
        z_dst = z[edge_index[1]]  # (E, D)
        # 3 canaux en parallèle
        logit_pos = (z_src @ self.W_pos * z_dst).sum(dim=-1)
        logit_neg = (z_src @ self.W_neg * z_dst).sum(dim=-1)
        logit_zero = (z_src @ self.W_zero * z_dst).sum(dim=-1)
        # Sélection par signe (vectorized)
        mask_pos = (edge_sign > 0).float()
        mask_neg = (edge_sign < 0).float()
        mask_zero = (edge_sign == 0).float()
        return mask_pos * logit_pos + mask_neg * logit_neg + mask_zero * logit_zero

    def forward_cosine(self, z: torch.Tensor, edge_index: torch.Tensor,
                       tau: float = 1.0) -> torch.Tensor:
        """Fallback compat V4 : décodeur cosinus pour edge_types unsigned.

        Identique à `VGAE.decode` pour la backward-compat.
        """
        z_src = F.normalize(z[edge_index[0]], dim=1)
        z_dst = F.normalize(z[edge_index[1]], dim=1)
        cos = (z_src * z_dst).sum(dim=1)
        return tau * cos


class VGAE(nn.Module):
    """VGAE complet : encoder + reparametrization + décodeur cosinus."""

    def __init__(self, encoder, tau_init=2.0, tau_max=3.0):
        super().__init__()
        self.encoder = encoder
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau_init))))
        self.log_tau_max = np.log(tau_max)

    def reparametrize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x_dict, edge_index_dict, edge_attr_dict=None):
        mu, logvar = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, edge_index):
        src, dst = edge_index
        z_src = F.normalize(z[src], dim=1)
        z_dst = F.normalize(z[dst], dim=1)
        cos_sim = (z_src * z_dst).sum(dim=1)
        log_tau_clamped = torch.clamp(self.log_tau, max=self.log_tau_max)
        tau = torch.exp(log_tau_clamped)
        return tau * cos_sim

    def kl_loss(self, mu, logvar, free_bits=0.0):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim_mean = kl_per_dim.mean(dim=0)
        kl_clamped = torch.clamp(kl_per_dim_mean, min=free_bits)
        return kl_clamped.sum()
