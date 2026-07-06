"""
_config_derive.py — CORPS des dérivations de config (modules actifs, feature flags,
run_tag) à partir de CLI_ARGS. Compilé + exécuté par `_config.derive_config()` dans
un namespace {CLI_ARGS: args}. Sémantique module-level exacte du monolithe. Ne pas
importer directement.
"""
# flake8: noqa
_EXCLUDED_FEATURES = {
    s.strip().lower() for s in CLI_ARGS.exclude_features.split(",") if s.strip()
}

# MODULES : dictionnaire compact qui résume la config — référé partout dans
# le pipeline pour gater les blocs (loading + edges + features).
MODULES = {
    "use_ppi":                   CLI_ARGS.use_ppi,
    "use_reactome":              CLI_ARGS.use_reactome,
    "use_coexpr":                CLI_ARGS.use_coexpr,
    "use_scenic_regulons":       CLI_ARGS.use_scenic_regulons,
    "use_humess_edges":          CLI_ARGS.use_humess_edges,
    "use_humess_features":       CLI_ARGS.use_humess_features,
    "use_cell_group_edges":      CLI_ARGS.use_cell_group_edges,
    "use_omnipath_signaling":    CLI_ARGS.use_omnipath_signaling,
    "use_omnipath_tf_curated":   CLI_ARGS.use_omnipath_tf_curated,
    "include_omnipath_genes":    CLI_ARGS.include_omnipath_genes,
    "use_reactome_fi":           CLI_ARGS.use_reactome_fi,
}

# V4.2 : mode coexpression (p16_only = V4.1 ; differential = option A).
COEXPR_MODE = CLI_ARGS.coexpr_mode
COEXPR_DIFFERENTIAL = (COEXPR_MODE == "differential")

# V4.2 : parsing de la pondération γ_t par edge_type (niveau message).
# Format "ppi=0.1,coexpression=0.5". Vide → {} → tous γ=1.0 (V4.1).
EDGE_TYPE_WEIGHTS: dict[str, float] = {}
if CLI_ARGS.edge_type_weights.strip():
    for _kv in CLI_ARGS.edge_type_weights.split(","):
        _k, _, _v = _kv.partition("=")
        _k, _v = _k.strip(), _v.strip()
        if _k and _v:
            try:
                EDGE_TYPE_WEIGHTS[_k] = float(_v)
            except ValueError:
                print(f"  [warn] --edge-type-weights : '{_kv}' ignoré "
                      f"(valeur non numérique)")
if EDGE_TYPE_WEIGHTS:
    print(f"  γ_t edge-type weights (message-level) : {EDGE_TYPE_WEIGHTS}")

# Matrice « feature → activée ? » — recoupe les modules et l'option
# --exclude-features. Une feature est INCLUSE ssi (a) sa source est active
# ET (b) elle n'est pas listée dans --exclude-features.
def _feature_enabled(name, source_ok=True):
    return source_ok and (name not in _EXCLUDED_FEATURES)


GENE_FEATURE_FLAGS = {
    "is_tf":       _feature_enabled("is_tf",      MODULES["use_scenic_regulons"]),
    "variance":    _feature_enabled("variance",   True),  # toujours dispo (group_stats)
    "ppi_degree":  _feature_enabled("ppi_degree", MODULES["use_ppi"]),
    "reg_degree":  _feature_enabled("reg_degree", MODULES["use_scenic_regulons"]),
    "imp_P4":      _feature_enabled("imp_p4",     MODULES["use_humess_features"]),
    "imp_P16":     _feature_enabled("imp_p16",    MODULES["use_humess_features"]),
    "imp_delta":   _feature_enabled("imp_delta",  MODULES["use_humess_features"]),
    "has_humess":  _feature_enabled("has_humess", MODULES["use_humess_features"]),
}


def _build_run_tag():
    """Construit un tag court à partir des modules désactivés.
    'full' si tout est actif. Ex : 'no-ppi.no-humess.s7' si ces deux sont
    off et seed=7. Si --run-tag est explicite, on respecte le tag libre tel
    quel (l'utilisateur est responsable de l'unicité par seed)."""
    tag = CLI_ARGS.run_tag
    if tag != "auto":
        return tag if tag else "full"
    parts = []
    if not MODULES["use_ppi"]:              parts.append("no-ppi")
    if not MODULES["use_reactome"]:         parts.append("no-reactome")
    if not MODULES["use_coexpr"]:           parts.append("no-coexpr")
    if not MODULES["use_scenic_regulons"]:  parts.append("no-scenic")
    if not MODULES["use_humess_edges"] and not MODULES["use_humess_features"]:
        parts.append("no-humess")
    else:
        if not MODULES["use_humess_edges"]:    parts.append("no-humess-edges")
        if not MODULES["use_humess_features"]: parts.append("no-humess-feats")
    if not MODULES["use_cell_group_edges"]: parts.append("no-cgrp")
    if MODULES["use_omnipath_signaling"]:   parts.append("op-sig")
    if MODULES["use_omnipath_tf_curated"]:  parts.append("op-tf")
    if MODULES["include_omnipath_genes"]:   parts.append("op-genes")
    if MODULES["use_reactome_fi"]:          parts.append("rfi")
    if COEXPR_DIFFERENTIAL:                 parts.append("coexdiff")
    # V4.3 : tag des choix méthode×prune (uniquement si != défaut).
    if CLI_ARGS.coexpr_method != "sklearn":
        parts.append(f"grn-{CLI_ARGS.coexpr_method}")
    if CLI_ARGS.coexpr_prune != "topk":
        parts.append(f"prune-{CLI_ARGS.coexpr_prune}")
    if EDGE_TYPE_WEIGHTS:                   parts.append("gw")
    if getattr(CLI_ARGS, "dedup_ppi_signed", "off") != "off":
        parts.append(f"dedup-{CLI_ARGS.dedup_ppi_signed}")
    if _EXCLUDED_FEATURES:                  parts.append("ex-" + "-".join(sorted(_EXCLUDED_FEATURES)))
    if CLI_ARGS.ppi_score_thresh != 900:    parts.append(f"ppi{CLI_ARGS.ppi_score_thresh}")
    if abs(CLI_ARGS.coexpr_top_quantile - 0.98) > 1e-9:
        parts.append(f"q{int(CLI_ARGS.coexpr_top_quantile*1000):04d}")
    if CLI_ARGS.reactome_max_pathway != 20:
        parts.append(f"rxmax{CLI_ARGS.reactome_max_pathway}")
    base = "full" if not parts else ".".join(parts)
    # Toujours suffixer par le seed pour différencier les runs multi-seed.
    return f"{base}.s{CLI_ARGS.seed}"


RUN_TAG = _build_run_tag()
