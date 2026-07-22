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


# V5 (TIER 1c) — edge_types qui portent un `sign` (colonne 1 de edge_attr).
# Synchronisé avec gnn_vgae.py.
# V6 Module 1 (extension) — 6 edge_types OmniPath supplémentaires projetables
# sur les nœuds gène (protein↔protein), signés [score, sign]. Ne s'activent que
# si présents dans le HeteroData (--omnipath-edges …). Source : graphe autonome
# omnipath_graph (edges.tsv.gz), pas omnipath_integration (qui porte déjà
# signaling + tf_curated). signaling/collectri_tf exclus (déjà à nous).
OMNIPATH_EXTRA_EDGE_TYPES = [
    "transcriptional", "tf_target", "kinase_substrate",
    "ligand_receptor", "pathway", "enzyme_substrate",
]

SIGNED_EDGE_TYPES: set = {
    ("gene", "signaling", "gene"),
    ("gene", "tf_curated", "gene"),
    ("gene", "tf_curated_by", "gene"),
    ("gene", "reactome_fi", "gene"),
} | {("gene", _et, "gene") for _et in OMNIPATH_EXTRA_EDGE_TYPES}


class _ScaledConv(nn.Module):
    """Wrapper V4.2 : multiplie la sortie d'un conv par γ_t fixe.
    Cf. gnn_vgae.py — synchronisé manuellement (Tier 2.5)."""

    def __init__(self, conv: nn.Module, gamma: float):
        super().__init__()
        self.conv = conv
        self.gamma = float(gamma)

    def forward(self, *args, **kwargs):
        return self.conv(*args, **kwargs) * self.gamma


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
        (("gene", "reactome_fi", "gene"), 2),  # V4.2
        (("gene", "reactome_fi_undirected", "gene"), 2),  # V6.2 (orientation inconnue)
        # V6.3 hypernœuds : le complexe devient un NŒUD, pas une clique. Une
        # arête par membre (étoile) au lieu de n(n-1)/2 arêtes deux-à-deux.
        (("gene", "member_of", "complex"), 1),
        (("complex", "has_member", "gene"), 1),
    ] + [(("gene", _et, "gene"), 2) for _et in OMNIPATH_EXTRA_EDGE_TYPES]  # V6

    def __init__(self, gene_in, cell_in, hidden, latent, n_layers,
                 n_heads=4, dropout=0.2, available_edge_types=None,
                 complex_in=None,
                 edge_dim_overrides=None, edge_type_weights=None,
                 signed_message=False, signed_edge_types=None):
        super().__init__()
        self._edge_dim_overrides = edge_dim_overrides or {}
        self._edge_type_weights = edge_type_weights or {}
        self._signed_message = bool(signed_message)
        self._signed_edge_types = set(
            tuple(et) for et in (signed_edge_types or SIGNED_EDGE_TYPES)
        )
        self.n_layers = n_layers
        head_dim = hidden // n_heads

        self.gene_proj = nn.Linear(gene_in, hidden)
        self.cell_proj = nn.Linear(cell_in, hidden)
        # V6.3 : projection du type `complex`, créée SEULEMENT si le graphe en
        # contient -> aucun paramètre supplémentaire dans un run legacy, donc
        # les checkpoints existants restent chargeables en strict=True.
        self.complex_proj = (nn.Linear(complex_in, hidden)
                             if complex_in is not None else None)

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
        if self._edge_dim_overrides:
            edge_types_dims = [
                (et, self._edge_dim_overrides.get(et, dim))
                for et, dim in edge_types_dims
            ]
        self.edge_dims = {et: dim for et, dim in edge_types_dims}

        def _resolve_gamma(et):
            if et in self._edge_type_weights:
                return float(self._edge_type_weights[et])
            if et[1] in self._edge_type_weights:
                return float(self._edge_type_weights[et[1]])
            return 1.0
        self.edge_gammas = {et: _resolve_gamma(et) for et, _ in edge_types_dims}

        for _ in range(n_layers):
            conv_dict = {}
            for et, ed in edge_types_dims:
                conv_kwargs = dict(heads=n_heads, concat=True,
                                   dropout=dropout, add_self_loops=False)
                if ed is not None:
                    conv_kwargs["edge_dim"] = ed
                # V5 : SignedGATConv pour les edge_types signés si flag actif.
                _use_signed = (
                    self._signed_message
                    and et in self._signed_edge_types
                    and ed is not None and ed >= 2
                )
                if _use_signed:
                    _gat = SignedGATConv(hidden, head_dim, sign_col=1, **conv_kwargs)
                else:
                    _gat = GATConv(hidden, head_dim, **conv_kwargs)
                _g = self.edge_gammas.get(et, 1.0)
                conv_dict[et] = (_ScaledConv(_gat, _g) if _g != 1.0 else _gat)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            _norms = {"gene": nn.BatchNorm1d(hidden),
                      "cell_group": nn.BatchNorm1d(hidden)}
            if self.complex_proj is not None:
                _norms["complex"] = nn.BatchNorm1d(hidden)
            self.norms.append(nn.ModuleDict(_norms))

        self.dropout = nn.Dropout(dropout)
        self.mu_head = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        _x = {
            "gene": F.relu(self.gene_proj(x_dict["gene"])),
            "cell_group": F.relu(self.cell_proj(x_dict["cell_group"])),
        }
        if self.complex_proj is not None and "complex" in x_dict:
            _x["complex"] = F.relu(self.complex_proj(x_dict["complex"]))
        x_dict = _x

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
        # Buffer transient pour le sign de l'edge courant (capturé dans
        # forward(), consommé dans message()). PyG ne passe pas edge_attr
        # à message() par défaut — d'où ce mécanisme « stash ».
        self._current_edge_sign = None

    def forward(self, x, edge_index, edge_attr=None, size=None,
                return_attention_weights=None):
        # Capture du sign AVANT propagate(), pour que message() puisse
        # l'utiliser. edge_attr peut être (E, 1) (unsigned), (E, 2)
        # ([score, sign]) ou plus.
        if edge_attr is not None and edge_attr.ndim >= 2 \
                and edge_attr.shape[1] > self.sign_col:
            self._current_edge_sign = edge_attr[:, self.sign_col]
        else:
            self._current_edge_sign = None
        return super().forward(x, edge_index, edge_attr=edge_attr,
                               size=size,
                               return_attention_weights=return_attention_weights)

    def message(self, x_j, alpha):
        # Sortie GATConv : (E, n_heads, head_dim) — concat=True le rendra
        # (E, n_heads * head_dim) ensuite (au niveau forward, pas message).
        out = super().message(x_j, alpha)
        sign = self._current_edge_sign
        if sign is None:
            return out
        # sign ∈ {-1, 0, +1} → scale ∈ {-1, 1, 1} (sign=0 = info inconnue,
        # message inchangé). Broadcast sur les dims n_heads × head_dim.
        scale = torch.where(sign == 0, torch.ones_like(sign), sign)
        # out a 2 ou 3 dims selon la version pyg ; broadcast via view.
        scale = scale.view(-1, *([1] * (out.ndim - 1)))
        return out * scale


class BilinearSignedDecoder(nn.Module):
    """Décodeur bilinéaire à 3 canaux pour reconstruction signée + tête sub-espace V5.3.

    V5.1+ (cf. §14bis.6septies/octodecies). Pour une arête (i, j) avec
    sign ∈ {+1, -1, 0} :

        z_signed = signed_proj(z)            # V5.3 : projection sub-espace
        logit_pos  = z_signed_i · W_+ · z_signed_j   (activation)
        logit_neg  = z_signed_i · W_- · z_signed_j   (inhibition)
        logit_zero = z_signed_i · W_0 · z_signed_j   (non-signé)
        score sign-agnostique = logit_pos − logit_neg    (gate 1c.5)

    **V5.3 (§14bis.6unvicesies)** : `signed_proj : R^latent → R^signed_dim`
    insère une projection linéaire entre `z` et les bilinéaires. Le gradient
    du `signed_aux_loss` passe par `signed_proj` ET filtre vers `z` —
    `signed_proj` absorbe une partie de la contrainte, ce qui réduit la
    concurrence avec le décodeur cosinus principal (V5.2 mesure : −0.022
    AUC recon dûe à cette concurrence).

    Backward-compat V5.1/V5.2 : `signed_proj` initialisée à identité + bias
    zéro → comportement initial identique à V5.2. Le checkpoint d'une V5.2
    se charge avec `strict=False` + post-init `signed_proj=identity`.

    Paramètres :
        latent_dim  : dim de `z` (= LATENT_DIM, typiquement 64).
        signed_dim  : dim du sous-espace signed. Défaut latent_dim (pas de
                      compression, `signed_proj` peut converger vers
                      identité). signed_dim < latent_dim force la
                      spécialisation par compression.

    Référence : Liu et al. 2024 *NAR* SGAT-bilinear ; généralisation du
    DistMult de Yang 2015 *ICLR* à la classification signée + tête
    sub-espace (inspiré des "task heads" en multi-task learning).
    """

    def __init__(self, latent_dim: int, signed_dim: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.signed_dim = signed_dim if signed_dim is not None else latent_dim

        # V5.3 — projection latent → sous-espace signed. Init = identité +
        # 0 bias pour préserver V5.2 au load (équivalence numérique exacte
        # quand signed_dim == latent_dim).
        self.signed_proj = nn.Linear(latent_dim, self.signed_dim, bias=True)
        with torch.no_grad():
            if self.signed_dim == latent_dim:
                self.signed_proj.weight.copy_(torch.eye(latent_dim))
            else:
                # signed_dim < latent_dim : init "PCA-like" (premières dims).
                init = torch.zeros(self.signed_dim, latent_dim)
                init[:self.signed_dim, :self.signed_dim] = torch.eye(self.signed_dim)
                self.signed_proj.weight.copy_(init)
            self.signed_proj.bias.zero_()

        # 3 matrices bilinéaires dans le sous-espace signed.
        eye = torch.eye(self.signed_dim)
        self.W_pos = nn.Parameter(eye + 0.01 * torch.randn(self.signed_dim, self.signed_dim))
        self.W_neg = nn.Parameter(eye + 0.01 * torch.randn(self.signed_dim, self.signed_dim))
        self.W_zero = nn.Parameter(eye + 0.01 * torch.randn(self.signed_dim, self.signed_dim))

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        """V5.3 — projette `z` dans le sous-espace signed via signed_proj.

        Si signed_dim == latent_dim et signed_proj n'a pas été entraîné :
        identité (équivalence V5.2). Une fois entraîné, peut diverger de
        l'identité pour spécialiser le sous-espace au gate signed.
        """
        return self.signed_proj(z)

    def forward_signed(self, z: torch.Tensor, edge_index: torch.Tensor,
                       edge_sign: torch.Tensor) -> torch.Tensor:
        """Décode des arêtes signées — usage TRAINING uniquement (V5.0 hérité).

        Sélectionne le canal selon `edge_sign` (info connue à l'entraînement).
        ⚠ Inutilisable pour l'évaluation `AUC(activate vs inhibit)` : la
        sélection donne déjà la réponse. Pour l'inférence sign-agnostique,
        utiliser `predict_sign_score()`.

        V5.1+ : conservé pour compat — la `signed_aux_loss` utilise
        désormais `predict_sign_score()` (BCE contrastive sur score
        différentiel) au lieu de cette méthode.
        """
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

    def predict_sign_score(self, z: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
        """Score sign-agnostique pour évaluation AUC(activate vs inhibit).

        Retourne `logit_pos − logit_neg` SANS utiliser le sign cible.
        AUC entre ce score et `(sign > 0).int()` mesure la séparation
        activate / inhibit apprise par le décodeur.

        C'est le score correct pour le gate 1c.5 (Liu 2024 *NAR* §3) et
        celui utilisé par la `signed_aux_loss` V5.1+.
        """
        z_signed = self._project(z)
        z_src = z_signed[edge_index[0]]
        z_dst = z_signed[edge_index[1]]
        logit_pos = (z_src @ self.W_pos * z_dst).sum(dim=-1)
        logit_neg = (z_src @ self.W_neg * z_dst).sum(dim=-1)
        return logit_pos - logit_neg

    def predict_signed_existence(self, z: torch.Tensor,
                                 edge_index: torch.Tensor) -> torch.Tensor:
        """V5.4 (decoder-split) — score d'EXISTENCE = logsumexp(W+,W-,W0).

        Enveloppe différentiable de max (Bishop 2006 §4.5). Reconstruit
        l'existence des arêtes signées (le cosinus ne les voit plus sous
        --decoder-split). Distinct de predict_sign_score (= W+ − W-, le signe).
        Source de vérité, dupliquée dans gnn_vgae.py + gnn_perturbation.py.
        """
        z_signed = self._project(z)
        z_src = z_signed[edge_index[0]]
        z_dst = z_signed[edge_index[1]]
        logit_pos = (z_src @ self.W_pos * z_dst).sum(dim=-1)
        logit_neg = (z_src @ self.W_neg * z_dst).sum(dim=-1)
        logit_zero = (z_src @ self.W_zero * z_dst).sum(dim=-1)
        return torch.logsumexp(
            torch.stack([logit_pos, logit_neg, logit_zero], dim=-1), dim=-1)

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
    """VGAE complet : encoder + reparametrization + décodeur cosinus.

    V5 (TIER 1c.3) : `bilinear_decoder` optionnel ; quand fourni, expose
    `decode_signed(z, edge_index, edge_sign)` en plus du décodeur cosinus
    historique (backward-compat V4.x).
    """

    def __init__(self, encoder, tau_init=2.0, tau_max=3.0,
                 bilinear_decoder=None):
        super().__init__()
        self.encoder = encoder
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau_init))))
        self.log_tau_max = np.log(tau_max)
        self.bilinear_decoder = bilinear_decoder

    def decode_signed(self, z, edge_index, edge_sign):
        if self.bilinear_decoder is None:
            raise AttributeError(
                "VGAE.decode_signed appelé mais bilinear_decoder=None."
            )
        return self.bilinear_decoder.forward_signed(z, edge_index, edge_sign)

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
