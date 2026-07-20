"""
_graph_build_body.py — CORPS des sections 1-7 (construction du graphe hétérogène).

⚠️ N'EST PAS un module importable classique : ce fichier est COMPILÉ puis exécuté
par `_graph_build.build_graph()` dans un dict-namespace pré-rempli (config, chemins,
constantes, helpers, imports). Les noms non définis ici (MODULES, DATA_DIR, pd, np,
torch, ...) sont fournis par ce namespace — sémantique IDENTIQUE au niveau module
du monolithe d'origine (assignations ET lectures tapent le même dict → pas de piège
de portée fonction). Ne pas exécuter/importer directement.

Extrait verbatim du monolithe gnn_vgae.py (split Tier 2.5). À la fin, toutes les
variables de `_CACHE_VARS` existent dans le namespace → récupérées comme bundle.
"""
# flake8: noqa  (noms résolus à l'exécution via le namespace injecté)
# =============================================================================
# 1. CHARGEMENT DES DONNÉES scRNA-seq
# =============================================================================
# Les données proviennent de GSE102090 : des HUVEC (Human Umbilical Vein
# Endothelial Cells) en scRNA-seq, deux passages :
#   - P4 (passage 4) : cellules jeunes / prolifératives
#   - P16 (passage 16) : cellules sénescentes (sénescence réplicative)
# Le preprocessing a été fait dans Seurat (normalisation LogNormalize,
# clustering des P16 en 4 sous-populations). On charge ici la metadata
# (barcode, passage, cluster) et l'en-tête de la matrice normalisée
# pour récupérer la liste de tous les gènes mesurés.
print("=" * 70)
print("1. Chargement des données scRNA-seq (P4 / P16)")
print("=" * 70)

# metadata : scRNA HUVEC = 1 ligne/cellule (barcode, passage, cluster_P16) ;
#            bulk (GROUP_META) = 1 ligne/échantillon (sample, group). Sert au
#            dénominateur n_cells_norm (§5) → DOIT refléter le bon univers
#            (cellules en scRNA, échantillons en bulk), sinon feature faussée.
if GROUP_META:
    # Bulk : pas de fichier HUVEC ; la "metadata" = la samplesheet sample→group.
    metadata = pd.read_csv(GROUP_META, sep=None, engine="python", header=None)
    metadata = metadata.rename(columns={0: "sample", 1: "group"})
    print(f"  Metadata (bulk) : {len(metadata)} échantillons ; "
          f"groupes : {sorted(metadata['group'].astype(str).unique())}")
    p16_meta = pd.DataFrame(); p16_clusters = []   # non utilisés en bulk
else:
    _meta_path = os.path.join(GNN_DATA_DIR, "merged_P4_P16_metadata.csv")
    if not os.path.exists(_meta_path):
        raise FileNotFoundError(
            f"{_meta_path} introuvable → mode scRNA HUVEC (GNN_GROUP_META vide). "
            "Pour un dataset BULK/custom, définissez GNN_GROUP_META (samplesheet "
            "sample→group) ET GNN_EXPR_MATRIX. Sur cluster : vérifiez que les "
            "données sont bien transférées (data/ pas toujours rsync vers /scratch).")
    metadata = pd.read_csv(_meta_path)
    print(f"  Metadata : {len(metadata)} cellules")
    print(f"    P4  : {(metadata['passage'] == 'P4').sum()} cellules")
    print(f"    P16 : {(metadata['passage'] == 'P16').sum()} cellules")
    # Les P16 sont subdivisées en 4 clusters (clustering Seurat) = sous-états sén.
    p16_meta = metadata[metadata["passage"] == "P16"].copy()
    p16_clusters = sorted(p16_meta["cluster_P16"].dropna().unique())
    print(f"    Clusters P16 : {p16_clusters}")

# POINT CLÉ : on prend TOUS les gènes de la matrice normalisée, sans
# filtrer par DEG (Differentially Expressed Genes). C'est essentiel car :
#   1. Le VGAE est non supervisé — il n'a pas besoin de labels DEG.
#   2. Filtrer par DEG introduirait la circularité qu'on veut éviter
#      (les DEG sont définis par log2FC/padj, qui étaient les features
#       et les labels du pipeline supervisé précédent gnn.py).
# On ne lit que l'en-tête (première ligne) pour économiser la mémoire —
# la matrice complète sera chargée plus tard (section 4) avec usecols.
with open(os.path.join(GNN_DATA_DIR, EXPR_MATRIX)) as f:
    header = f.readline().strip().split(",")
# HUVEC scRNA : 4 colonnes meta (barcode, passage, cluster_P16, cell_state) puis
# gènes. Bulk (GROUP_META) : 1 colonne meta (sample) puis gènes. Le reste = noms
# de gènes (potentiellement entre guillemets).
_n_meta_cols = 1 if GROUP_META else 4
all_available_genes = [g.strip('"') for g in header[_n_meta_cols:]]

# NOTE : cette liste sera filtrée en section 3 pour ne garder que les gènes
# ayant au moins une connexion dans le graphe (PPI, SCENIC, pathway, etc.).
# Un gène isolé (sans arête) ne reçoit aucun message pendant le GNN et
# ne contribue pas à la reconstruction des arêtes → inutile pour le VGAE.
print(f"  Gènes dans la matrice : {len(all_available_genes)}")

# =============================================================================
# 2. CHARGEMENT DES DONNÉES pySCENIC
# =============================================================================
# pySCENIC est un pipeline d'inférence de réseaux de régulation génique
# (Gene Regulatory Networks, GRN) à partir de données scRNA-seq. Il produit :
#   1. regulon_edges : liens TF → gène cible (avec un poids de régulation).
#      Un "regulon" = un TF + l'ensemble de ses gènes cibles prédits.
#   2. tf_activity (AUCell) : score d'activité de chaque TF dans chaque
#      cluster. AUCell évalue si les cibles d'un TF sont enrichies parmi
#      les gènes les plus exprimés d'une cellule.
#   3. adjacencies (GRNBoost2) : réseau de co-expression brut inféré par
#      GRNBoost2 (gradient boosting sur les paires de gènes). Les poids
#      "importance" mesurent la force de la co-expression prédite.
# Ces 3 sorties alimentent 3 types d'arêtes différents dans le graphe :
#   - regulon_edges → arêtes "regulates" (section 6d)
#   - tf_activity → feature d'arête sur "expresses" (section 6a)
#   - adjacencies → arêtes "coexpression" (section 6e)
print("\n" + "=" * 70)
print("2. Chargement des données pySCENIC")
print("=" * 70)

# regulon_edges + tf_activity : lectures SCENIC, gatées par --no-scenic-regulons
# (généralisation V6 : un dataset bulk sans pySCENIC les saute proprement).
# Fallbacks VIDES tolérés downstream : regulon usage gaté (l.~1401) ; tf_activity
# testé par `gene in tf_activity.columns` (section 6a) → 0 si vide.
if MODULES["use_scenic_regulons"]:
    # regulon_edges : colonnes = TF, target_gene, weight ; TF suffixe "(+)" → TF_clean
    regulon_edges = pd.read_csv(os.path.join(SCENIC_DIR, "regulon_edges_TF_to_gene.csv"))
    regulon_edges["TF_clean"] = regulon_edges["TF"].str.replace(r"\(\+\)$", "", regex=True)
    print(f"  Regulon edges : {len(regulon_edges)} interactions TF→cible")
    # tf_activity : matrice (clusters × TFs) — score AUCell moyen par cluster,
    # feature d'arête cell_group→gene (section 6a) si le gène est un TF.
    tf_activity = pd.read_csv(
        os.path.join(SCENIC_DIR, "mean_TF_activity_per_cluster.csv"), index_col=0
    )
    tf_activity.columns = [c.replace("(+)", "") for c in tf_activity.columns]
    print(f"  TF activity : {tf_activity.shape[0]} clusters × {tf_activity.shape[1]} TFs")
else:
    regulon_edges = pd.DataFrame(columns=["TF", "target_gene", "weight", "TF_clean"])
    tf_activity = pd.DataFrame()   # .columns / .index vides → 0 partout
    print("  SCENIC regulons/tf_activity : SKIP (--no-scenic-regulons)")

# adjacencies (GRNBoost2) : réseau de co-expression brut.
# Colonnes = TF, target, importance. On ne garde que le top 2% (COEXPR_TOP_QUANTILE)
# pour avoir un réseau épars de haute confiance. Les poids faibles sont
# vraisemblablement du bruit et ajouteraient des arêtes non informatives.
# Modulaire : si --no-coexpr, on ne charge même pas le fichier (~700 Mo).
if MODULES["use_coexpr"] and COEXPR_DIFFERENTIAL:
    # V4.2 option A : coexpr_diff.tsv produit par build_diff_coexpr.py.
    # Colonnes : TF, target, importance_p4_norm, importance_p16_norm,
    #            delta, cat_shared, cat_p4, cat_p16, ...
    _diff_path = CLI_ARGS.diff_coexpr_file
    if not os.path.isabs(_diff_path):
        _diff_path = os.path.join(BASE_DIR, _diff_path)
    if not os.path.exists(_diff_path):
        raise FileNotFoundError(
            f"--coexpr-mode differential mais {_diff_path} absent. "
            f"Lancer build_diff_coexpr.py (extract-matrices → GRNBoost2 "
            f"cluster → merge-adjacencies) d'abord."
        )
    adjacencies_filtered = pd.read_csv(_diff_path, sep="\t")
    # Le filtrage top-quantile est déjà fait PAR CONDITION dans
    # build_diff_coexpr.py (merge-adjacencies). On garde tout ici.
    print(f"  Adjacencies (V4.2 differential) : {len(adjacencies_filtered)} "
          f"arêtes P4∪P16, "
          f"catégories={adjacencies_filtered['category'].value_counts().to_dict()}")
elif MODULES["use_coexpr"]:
    adjacencies = pd.read_csv(os.path.join(SCENIC_DIR, "adjacencies.csv"))
    importance_thresh = adjacencies["importance"].quantile(COEXPR_TOP_QUANTILE)
    adjacencies_filtered = adjacencies[adjacencies["importance"] >= importance_thresh].copy()
    print(f"  Adjacencies : {len(adjacencies)} → {len(adjacencies_filtered)} "
          f"(top {100*(1-COEXPR_TOP_QUANTILE):.0f}%, seuil={importance_thresh:.2f})")
else:
    print("  Adjacencies : SKIP (--no-coexpr)")
    adjacencies_filtered = pd.DataFrame(columns=["TF", "target", "importance"])

# =============================================================================
# 2.5. PRÉ-CHARGEMENT OMNIPATH (V4.1) — pour étendre gene_to_idx
# =============================================================================
# Charge les caches OmniPath UNE FOIS, avant la section 3 (sélection des
# gènes), pour pouvoir étendre `selected_genes` avec les endpoints d'OmniPath.
# Sans ce pré-chargement, section 6g intervient TROP TARD : `gene_to_idx`
# est déjà figé sur PPI/SCENIC/coexpr/REACTOME/HuMess, et `_project_to_graph`
# filtre strictement → ~700 TFs CollecTRI étaient éliminés silencieusement.
#
# Note : le set retourné inclut TOUS les symboles présents dans les caches
# OmniPath actifs. L'intersection avec `available_set` (gènes scRNA-mesurés)
# se fait en section 3 — on n'invente pas de gènes hors mesure.
# =============================================================================
# 2.5. PRÉ-SCAN endpoints de TOUTES les sources d'arêtes actives (V6.2, 2026-07-15)
# =============================================================================
# Le filtre de connectivité (§3) ne comptait QUE ppi/coexpr/scenic/reactome →
# un gène connecté UNIQUEMENT par OmniPath (signaling/tf_curated/transcriptional/
# enzyme_substrate/ligand_receptor) ou reactome_fi était jeté AVANT que ces arêtes
# soient construites (poule-et-œuf : les arêtes ne relient que des gènes déjà dans
# l'univers). Résultat : op.all tombait à 1742 gènes alors qu'OmniPath+rfi en
# connectent ~10 000. On pré-scanne donc les endpoints de TOUTES les sources
# actives, sans condition (le filtre doit garder un gène connecté, peu importe
# l'origine de l'arête). Ajoutés à l'univers en §3 (alias-aware).
omnipath_endpoints: set[str] = set()
_any_op_source = (MODULES["use_omnipath_signaling"]
                  or MODULES["use_omnipath_tf_curated"]
                  or bool(OMNIPATH_EXTRA_EDGES)
                  or MODULES["use_reactome_fi"])
if _any_op_source:
    print("\n" + "=" * 70)
    print("2.5. Pré-scan endpoints toutes-sources (connectivité §3)")
    print("=" * 70)
    if not CLI_ARGS.omnipath_download_if_missing:
        os.environ["GNN_OMNIPATH_OFFLINE"] = "1"
    # (a) signaling / tf_curated (caches omnipath_integration, prod)
    if MODULES["use_omnipath_signaling"] or MODULES["use_omnipath_tf_curated"]:
        try:
            from omnipath_integration import (
                get_omnipath_endpoints as _opi_endpoints,
                silence_omnipath_logging as _silence_opi,
            )
            if not CLI_ARGS.omnipath_download_if_missing:
                _silence_opi()
            _src = []
            if MODULES["use_omnipath_signaling"]:  _src += ["signaling", "signor"]
            if MODULES["use_omnipath_tf_curated"]: _src += ["collectri"]
            _e = _opi_endpoints(cache_dir=OMNIPATH_CACHE_DIR, sources=_src,
                                download_if_missing=CLI_ARGS.omnipath_download_if_missing)
            omnipath_endpoints |= _e
            print(f"  signaling/tf_curated : +{len(_e)} endpoints")
        except ImportError as _e:
            print(f"  [warn] omnipath_integration KO ({_e})")
    # (b) arêtes OmniPath supplémentaires (graphe autonome edges.tsv.gz)
    if OMNIPATH_EXTRA_EDGES:
        try:
            import omnipath_graph as _opg
            _, _opg_edges = _opg.load_omnipath_graph(OMNIPATH_CACHE_DIR)
            _sub = _opg_edges[_opg_edges["edge_type"].isin(OMNIPATH_EXTRA_EDGES)]
            _e = (set(_sub["source_symbol"].astype(str))
                  | set(_sub["target_symbol"].astype(str)))
            omnipath_endpoints |= _e
            print(f"  extra edges {OMNIPATH_EXTRA_EDGES} : +{len(_e)} endpoints")
        except (FileNotFoundError, ImportError) as _e:
            print(f"  [warn] graphe OmniPath absent pour endpoints extra ({_e})")
    # (c) reactome_fi : pré-scan Gene1/Gene2 du fichier FI (même résolution qu'en 6)
    if MODULES["use_reactome_fi"]:
        _fip = CLI_ARGS.reactome_fi_file
        if not os.path.isabs(_fip):
            _fip = os.path.join(DATA_DIR,
                                _fip[5:] if _fip.startswith("data/") else _fip)
        if os.path.exists(_fip):
            try:
                _fi = pd.read_csv(_fip, sep="\t", usecols=["Gene1", "Gene2"])
                _e = (set(_fi["Gene1"].astype(str))
                      | set(_fi["Gene2"].astype(str)))
                omnipath_endpoints |= _e
                print(f"  reactome_fi : +{len(_e)} endpoints")
            except Exception as _e:
                print(f"  [warn] lecture reactome_fi KO ({_e})")
        else:
            print(f"  [warn] reactome_fi absent ({_fip})")
    print(f"  → endpoints toutes-sources : {len(omnipath_endpoints)} "
          f"(avant ∩ gènes mesurés)")

# HGNC alias map (V6) — canonicalise symboles OmniPath ↔ nos gene_to_idx en
# symbole approuvé, à l'expansion du gene universe (section 3) ET à la
# projection des arêtes (section 6g). Sans ça, les gènes dérivés de
# nomenclature (H2AFZ↔H2AZ1) perdent toutes leurs arêtes OmniPath curées.
# Vide ({}) = fallback identité (offline sans cache, ou --no-omnipath-hgnc-alias).
omnipath_alias_map: dict = {}
if MODULES["omnipath_hgnc_alias"] and _any_op_source:
    try:
        from hgnc_alias import build_alias_map as _build_hgnc_alias
        omnipath_alias_map = _build_hgnc_alias(
            cache_dir=OMNIPATH_CACHE_DIR,
            download_if_missing=CLI_ARGS.omnipath_download_if_missing,
        )
        if omnipath_alias_map:
            print(f"  [hgnc] normalisation d'alias ON "
                  f"({len(omnipath_alias_map)} variants)")
        else:
            print("  [hgnc] alias map vide (cache absent + download OFF) "
                  "→ fallback identité")
    except ImportError as _e:
        print(f"  [warn] import hgnc_alias KO ({_e}) — alias normalization OFF.")

# =============================================================================
# 3. SÉLECTION DES GÈNES — BASÉE SUR LA CONNECTIVITÉ (pas les DEGs)
# =============================================================================
# JUSTIFICATION : dans un GNN, le message passing propage l'information le
# long des arêtes du graphe. Un gène ISOLÉ (sans arête) :
#   - Ne reçoit aucun message de ses voisins (pas d'agrégation)
#   - Ne contribue à aucune arête à reconstruire (pas de signal de loss)
#   - Son embedding sera uniquement basé sur ses features → pas d'utilité GNN
# On ne garde donc que les gènes connectés dans au moins un réseau :
# SCENIC (régulation), GRNBoost2 (co-expression), STRING (PPI), REACTOME
# (pathway), ou HuMess (cocatalyse métabolique).
# C'est un filtre TOPOLOGIQUE, pas statistique — aucun biais DEG.
print("\n" + "=" * 70)
print("3. Sélection des gènes par connectivité")
print("=" * 70)

# available_set : ensemble des gènes mesurés dans le scRNA-seq.
# Sert de filtre pour ne garder que les gènes qu'on peut observer
# (certains gènes des bases externes ne sont pas dans notre matrice).
available_set = set(all_available_genes)

# --- 3a. Gènes SCENIC (régulation transcriptionnelle) ---
# Un gène est "SCENIC-connecté" s'il est soit un TF qui régule des cibles,
# soit une cible régulée par un TF. On prend l'union des deux.
# L'intersection avec available_set garantit que le gène est dans notre matrice.
scenic_genes = set(regulon_edges["TF_clean"]) | set(regulon_edges["target_gene"])
scenic_genes &= available_set

# --- 3b. Gènes GRNBoost2 (co-expression filtrée au top 2%) ---
# GRNBoost2 produit un réseau dirigé (TF → target) mais en pratique les liens
# sont interprétés comme de la co-expression. On prend l'union des deux côtés.
coexpr_genes = set(adjacencies_filtered["TF"].astype(str)) | set(adjacencies_filtered["target"].astype(str))
coexpr_genes &= available_set

# --- 3c. Gènes STRING PPI (protein-protein interactions) ---
# STRING v12 : base de données d'interactions protéine-protéine (expérimentales,
# co-expression, text mining, etc.). Le "combined_score" (0-1000) agrège
# plusieurs canaux de preuve. On télécharge 2 fichiers :
#   - links : les interactions (protein1, protein2, combined_score)
#   - aliases : mapping identifiant STRING → symbole HGNC (ex : ENSP00000... → TP53)
# Modulaire : si --no-ppi, on saute le téléchargement et la lecture (~1.6 Go),
# ppi_hc / string2sym restent vides, aucune arête PPI ne sera créée.
ppi_genes = set()
ppi_hc = pd.DataFrame(columns=["protein1", "protein2", "combined_score"])
sym2string = {}
string2sym = {}
if MODULES["use_ppi"]:
    PPI_FILE = os.path.join(PPI_DIR, "9606.protein.links.v12.0.txt.gz")
    PPI_ALIAS_FILE = os.path.join(PPI_DIR, "9606.protein.aliases.v12.0.txt.gz")
    download_if_absent(
        "https://stringdb-static.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        PPI_FILE, "STRING links"
    )
    download_if_absent(
        "https://stringdb-static.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
        PPI_ALIAS_FILE, "STRING aliases"
    )

    # Construction du mapping symbole ↔ identifiant STRING.
    # On filtre par "Ensembl_HGNC" pour avoir des symboles de gènes standards
    # (ex : TP53, CDKN1A) et non des alias ambigus. Cela évite les collisions
    # où un alias pourrait correspondre à plusieurs protéines.
    aliases = pd.read_csv(PPI_ALIAS_FILE, sep="\t", compression="gzip")
    aliases_filt = aliases[
        (aliases["alias"].isin(available_set)) &
        (aliases["source"].str.contains("Ensembl_HGNC", na=False))
    ]
    # sym2string : "TP53" → "9606.ENSP00000269305"
    sym2string = dict(zip(aliases_filt["alias"], aliases_filt["#string_protein_id"]))
    # string2sym : inverse, pour reconvertir après filtrage
    string2sym = {v: k for k, v in sym2string.items()}
    string_ids = set(sym2string.values())

    # Chargement du réseau PPI complet puis filtrage :
    # 1. Les deux protéines doivent être dans notre ensemble de gènes
    # 2. Le combined_score doit dépasser PPI_SCORE_THRESH (900 = highest confidence)
    ppi_raw = pd.read_csv(PPI_FILE, sep=" ", compression="gzip")
    ppi_hc = ppi_raw[
        (ppi_raw["protein1"].isin(string_ids)) &
        (ppi_raw["protein2"].isin(string_ids)) &
        (ppi_raw["combined_score"] >= PPI_SCORE_THRESH)
    ]
    # Extraction des symboles de gènes impliqués dans au moins une PPI fiable
    for _, row in ppi_hc.iterrows():
        s1, s2 = string2sym.get(row["protein1"]), string2sym.get(row["protein2"])
        if s1 and s2:
            ppi_genes.update([s1, s2])
else:
    print("  STRING PPI : SKIP (--no-ppi)")

# --- 3d. Gènes REACTOME (pathways biologiques) ---
# REACTOME via MSigDB : fichier GMT (Gene Matrix Transposed) où chaque ligne
# est un pathway avec ses gènes membres. Format : NOM_PATHWAY <tab> URL <tab> GENE1 <tab> GENE2 ...
# On filtre par taille : un pathway de 2-20 gènes est informatif. Au-delà de
# REACTOME_MAX_PATHWAY gènes, le pathway est trop générique (ex : "Metabolism")
# et connecterait des gènes sans rapport fonctionnel direct.
# Modulaire : si --no-reactome, on saute le téléchargement et le parse.
reactome_pathways = {}   # nom_pathway → set(gènes)
reactome_genes = set()   # union de tous les gènes dans au moins un pathway
if MODULES["use_reactome"]:
    MSIGDB_REACTOME = os.path.join(DB_DIR, "c2.cp.reactome.symbols.gmt")
    download_if_absent(
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c2.cp.reactome.v2024.1.Hs.symbols.gmt",
        MSIGDB_REACTOME, "MSigDB REACTOME"
    )
    # Parse du fichier GMT REACTOME : on ne garde que les pathways de taille
    # raisonnable (2 à REACTOME_MAX_PATHWAY gènes) après intersection avec
    # les gènes disponibles dans notre matrice scRNA-seq.
    with open(MSIGDB_REACTOME) as f:
        for line in f:
            parts = line.strip().split("\t")
            # parts[0] = nom du pathway, parts[1] = URL, parts[2:] = gènes
            genes_in_pw = set(parts[2:]) & available_set
            if 2 <= len(genes_in_pw) <= REACTOME_MAX_PATHWAY:
                reactome_pathways[parts[0]] = genes_in_pw
                reactome_genes |= genes_in_pw
else:
    print("  REACTOME : SKIP (--no-reactome)")

# --- 3e. UNION FINALE : tous les gènes connectés par au moins un réseau ---
# connected_genes = SCENIC ∪ GRNBoost2 ∪ PPI ∪ REACTOME (∪ OmniPath en V4.1)
# (HuMess sera ajouté plus tard en section 6f, mais ses gènes sont typiquement
# déjà dans l'un des réseaux ci-dessus.)
# V4.1 : si --include-omnipath-genes, on ajoute les endpoints de signaling
# (kinase-substrat + SIGNOR) et tf_curated (CollecTRI). L'intersection avec
# `available_set` garantit qu'on n'invente pas de gènes hors scRNA-mesurés.
# gene_symbols : array trié de tous les gènes retenus (l'index dans cet
#   array = l'identifiant du noeud dans le graphe PyG).
# gene_to_idx : dictionnaire inverse, symbole → index dans le graphe.
connected_genes = scenic_genes | coexpr_genes | ppi_genes | reactome_genes
_connected_before_opi = set(connected_genes & available_set)
# OmniPath endpoints = symboles approuvés ; nos gènes mesurés = symboles
# legacy → l'intersection brute manque les gènes alias-driftés. On garde les
# gènes MESURÉS dont la forme approuvée est un endpoint (alias-aware).
_opi_measured: set[str] = set()
if omnipath_endpoints:
    if omnipath_alias_map:
        _ep_canon = {omnipath_alias_map.get(e, e) for e in omnipath_endpoints}
        _opi_measured = {g for g in available_set
                         if omnipath_alias_map.get(g, g) in _ep_canon}
    else:
        _opi_measured = omnipath_endpoints & available_set
    connected_genes |= _opi_measured
gene_symbols = np.array(sorted(connected_genes & available_set))
gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
n_genes = len(gene_symbols)

print(f"  Gènes connectés : {n_genes}")
print(f"    SCENIC       : {len(scenic_genes & set(gene_symbols))}")
print(f"    Co-expression: {len(coexpr_genes & set(gene_symbols))}")
print(f"    PPI          : {len(ppi_genes & set(gene_symbols))}")
print(f"    REACTOME     : {len(reactome_genes & set(gene_symbols))}")
if omnipath_endpoints:
    _opi_in_graph = _opi_measured & set(gene_symbols)
    _opi_new = _opi_in_graph - _connected_before_opi
    print(f"    OmniPath     : {len(_opi_in_graph)} "
          f"(dont {len(_opi_new)} nouveaux ∉ des 4 sources V3)")

# =============================================================================
# 4. CALCUL DE L'EXPRESSION PAR GROUPE — FEATURES D'ARÊTES
# =============================================================================
# ARCHITECTURE CLÉ : l'expression génique est placée sur les ARÊTES
# (cell_group → gene) et NON sur les noeuds "gene". C'est le point central
# de la suppression de la circularité :
#   - Pipeline supervisé (gnn.py) : features de noeud = log2FC, padj → labels = DEG
#     → circularité car les features CONTIENNENT l'information des labels.
#   - Pipeline VGAE (ici) : features de noeud = topologiques (is_tf, variance, degree)
#     → l'expression est sur les arêtes, le score d'importance ÉMERGE de
#     l'espace latent sans jamais voir le log2FC/padj.
#
# On calcule 6 statistiques d'expression pour chaque combinaison (groupe, gène) :
#   1. mean_expression : expression moyenne (LogNormalize) dans le groupe
#   2. pct_expressing : fraction de cellules du groupe qui expriment le gène (> 0)
#   3. std_expression : écart-type intra-groupe (mesure la variabilité cellulaire)
#   4. cv_expression : coefficient de variation = std/mean (variabilité relative)
#   5. q25, q75 : quartiles d'expression (caractérisent la distribution)
# + 1 feature supplémentaire (tf_activity) ajoutée en section 6a.
print("\n" + "=" * 70)
print("4. Expression par groupe cellulaire (features d'arêtes)")
print("=" * 70)

# Les 5 groupes cellulaires du graphe :
# P4 = cellules jeunes (1 seul groupe car pas de sous-clusters)
# P16_cluster_0..3 = 4 sous-populations sénescentes (identifiées par Seurat)
# Configurable (généralisation V6 : un dataset bulk a ses propres groupes, ex.
# "pro,sen") via env GNN_CELL_GROUPS. Défaut HUVEC = 5 groupes (rétro-compat).
CELL_GROUPS = [g.strip() for g in os.environ.get(
    "GNN_CELL_GROUPS",
    "P4,P16_cluster_0,P16_cluster_1,P16_cluster_2,P16_cluster_3").split(",")
    if g.strip()]

# On charge la matrice normalisée (LogNormalize de Seurat) mais UNIQUEMENT
# pour les gènes retenus en section 3 (gene_symbols). usecols évite de charger
# les ~20000 gènes en mémoire quand on n'en utilise que ~5000.
print("  Chargement de la matrice normalisée...")
if GROUP_META:
    # ── Chemin BULK (généralisation V6) : matrice échantillons×gènes (1re col =
    #    sample) + meta sample→group ; le groupe est lu directement (pas de
    #    passage/cluster_P16). logFC JAMAIS chargé ici (idem HUVEC : on n'utilise
    #    que l'expression par groupe).
    _full = pd.read_csv(os.path.join(GNN_DATA_DIR, EXPR_MATRIX))
    _sample_col = _full.columns[0]
    # robuste : sniff séparateur (tsv/csv), sans en-tête (col0=sample, col1=group)
    _meta = pd.read_csv(GROUP_META, sep=None, engine="python", header=None)
    _g = dict(zip(_meta.iloc[:, 0].astype(str), _meta.iloc[:, 1].astype(str)))
    _full["group"] = _full[_sample_col].astype(str).map(_g)
    _keep = [g for g in gene_symbols.tolist() if g in _full.columns]
    normalized = _full.dropna(subset=["group"])[["group"] + _keep].copy()
    print(f"  [bulk] matrice : {normalized.shape} ; groupes : "
          f"{sorted(normalized['group'].unique())}")
else:
    cols_to_read = ["barcode", "passage", "cluster_P16", "cell_state"] + gene_symbols.tolist()
    normalized = pd.read_csv(
        os.path.join(GNN_DATA_DIR, EXPR_MATRIX),
        usecols=cols_to_read,
    ).copy()  # .copy() défragmente le DataFrame (pandas alloue des blocs contigus)

    def assign_group(row):
        """Assigne chaque cellule à son groupe (P4 ou P16_cluster_X)."""
        if row["passage"] == "P4":
            return "P4"
        c = row["cluster_P16"]
        if pd.notna(c):
            return f"P16_cluster_{int(c)}"
        return None  # Cellules sans cluster assigné → seront supprimées

    normalized["group"] = normalized.apply(assign_group, axis=1)
    normalized = normalized.dropna(subset=["group"])  # Supprime les cellules sans groupe
print(f"  Matrice : {normalized.shape}")

# Calcul des statistiques par groupe cellulaire.
# Pour chaque groupe, on calcule les 6 stats sur les colonnes de gènes.
# Ces stats deviendront les features d'arêtes "expresses" en section 6a.
print("  Calcul des statistiques (mean, pct, std, cv, q25, q75)...")
group_stats = {}
for grp in CELL_GROUPS:
    mask = normalized["group"] == grp
    sub = normalized.loc[mask, gene_symbols]  # Sous-matrice (cellules du groupe × gènes)
    n_cells = mask.sum()

    # mean_expression : expression moyenne normalisée du gène dans ce groupe
    mean_expr = sub.mean(axis=0).values.astype(np.float32)
    # pct_expressing : fraction de cellules exprimant ce gène (dropout rate = 1 - pct)
    pct_expr = (sub > 0).mean(axis=0).values.astype(np.float32)
    # std_expression : variabilité intra-groupe (forte = gène hétérogène)
    std_expr = sub.std(axis=0).values.astype(np.float32)
    # cv_expression : coefficient de variation (normalise la std par la mean)
    # Un CV élevé = bimodal ou très variable. +1e-8 évite la division par zéro.
    cv_expr = std_expr / (mean_expr + 1e-8)
    # q25, q75 : quartiles de la distribution d'expression
    q25 = sub.quantile(0.25, axis=0).values.astype(np.float32)
    q75 = sub.quantile(0.75, axis=0).values.astype(np.float32)

    group_stats[grp] = {
        "mean_expression": mean_expr, "pct_expressing": pct_expr,
        "std_expression": std_expr, "cv_expression": cv_expr,
        "q25": q25, "q75": q75, "n_cells": n_cells,
    }
    print(f"    {grp:20s} : {n_cells} cellules, mean={mean_expr.mean():.3f}")

# Libération de la matrice normalisée (plusieurs Go) — les stats sont calculées.
del normalized

# =============================================================================
# 5. FEATURES DES NOEUDS (topologiques uniquement — PAS de log2FC/padj)
# =============================================================================
# POINT ANTI-CIRCULARITÉ : les features de noeuds "gene" ne doivent JAMAIS
# contenir les statistiques différentielles (log2FC, padj, delta_pct) car :
#   - Le pipeline supervisé précédent (gnn.py) utilisait ces features ET ces
#     mêmes stats comme labels DEG → circularité.
#   - Ici, on utilise uniquement des propriétés TOPOLOGIQUES / INTRINSÈQUES :
#     1. is_tf : le gène est-il un facteur de transcription ? (booléen, propriété
#        intrinsèque du gène qui ne dépend pas du contexte expérimental)
#     2. variance_across_groups : variabilité de l'expression ENTRE les 5 groupes
#        cellulaires. C'est une mesure brute de variabilité (pas un test statistique
#        comme le log2FC). Un gène avec haute variance = expression différente
#        entre P4 et les clusters P16 → potentiellement intéressant.
#     3. ppi_degree : nombre de voisins PPI (ajouté après section 6)
#     4. reg_degree : nombre de liens de régulation (ajouté après section 6)
#   + En V3 (avec HuMess) : 4 features métaboliques supplémentaires (section 6f)
# Note : les features de degré (3-4) sont ajoutées APRÈS la construction des
# arêtes en section 6, car on a besoin des arêtes pour les compter.
print("\n" + "=" * 70)
print("5. Features des noeuds (topologiques)")
print("=" * 70)

# Feature 1 : is_tf — binaire, 1.0 si le gène est un TF.
# V3 : pySCENIC uniquement (~62 TFs après filtres motif + expression HUVEC).
# V4.1+ : union pySCENIC ∪ CollecTRI sources lorsque
# --use-omnipath-tf-curated est actif. CollecTRI (Müller-Dott 2023) couvre
# ~1186 TFs curés depuis la littérature ; intersection avec les gènes
# mesurés (gene_symbols) garantit qu'on n'invente aucun TF hors scRNA.
# Permet à la logique TF-aware downstream (suffixe interpretation,
# threshold B_discovery relaxé) de couvrir tous les TFs curés, pas
# uniquement le sous-ensemble pySCENIC.
scenic_tfs = set(regulon_edges["TF_clean"].unique())
collectri_tfs: set[str] = set()
if MODULES["use_omnipath_tf_curated"]:
    _collectri_cache = os.path.join(OMNIPATH_CACHE_DIR, "tf_collectri.tsv.gz")
    if os.path.exists(_collectri_cache):
        try:
            import gzip as _gz
            with _gz.open(_collectri_cache, "rt") as _f:
                next(_f, None)  # header
                for _line in _f:
                    _src = _line.split("\t", 1)[0]
                    # On exclut les hétérodimères type "NFKB1_REL"
                    if _src and "_" not in _src:
                        collectri_tfs.add(_src)
            print(f"  is_tf : CollecTRI source TFs lus = {len(collectri_tfs)} "
                  f"(union avec pySCENIC ci-dessous)")
        except Exception as _e:
            print(f"  is_tf : [warn] lecture CollecTRI échouée ({_e}) — "
                  f"on garde pySCENIC seul")
    else:
        print(f"  is_tf : [warn] cache CollecTRI absent à {_collectri_cache} — "
              f"on garde pySCENIC seul")
all_tfs = scenic_tfs | collectri_tfs
is_tf = np.array([1.0 if g in all_tfs else 0.0 for g in gene_symbols],
                 dtype=np.float32)
print(f"  is_tf : pySCENIC={len(scenic_tfs)} + CollecTRI={len(collectri_tfs)} "
      f"→ union ∩ available = {int(is_tf.sum())} TFs")

# Feature 2 : variance_across_groups — variance de l'expression moyenne entre
# les 5 groupes cellulaires. Calculée sur les mean_expression déjà calculées
# en section 4. Normalisée par le max pour être dans [0, 1].
# Intuition : un gène dont l'expression est stable entre P4 et P16 aura une
# faible variance ; un gène fortement dérégulé aura une forte variance.
mean_expr_per_group = np.array([
    group_stats[grp]["mean_expression"] for grp in CELL_GROUPS
])  # Shape : (5, n_genes) — 5 groupes × n_genes
variance_across = mean_expr_per_group.var(axis=0).astype(np.float32)  # Variance sur l'axe des groupes
variance_norm = variance_across / (variance_across.max() + 1e-8)

# ── Cell group features ─────────────────────────────────────────────────────
# Les noeuds "cell_group" ont aussi des features (3 dimensions) :
#   1. is_senescent : 0 pour P4, 1 pour tous les P16 → encode la condition
#   2. n_cells_norm : fraction du nombre total de cellules dans ce groupe
#      → encode la taille relative du groupe
#   3. cluster_idx : index normalisé du cluster (0 pour P4, 0.25/0.5/0.75/1.0
#      pour les clusters P16) → encode l'identité du sous-cluster
# Ces features permettent au modèle de distinguer les groupes cellulaires
# et de moduler les messages "expresses" en conséquence.
cell_group_features_list = []
for grp_idx, grp in enumerate(CELL_GROUPS):
    # baseline = 1er groupe (HUVEC : P4 ; bulk : ex. pro). Générique au nommage.
    is_senescent = 0.0 if grp == CELL_GROUPS[0] else 1.0
    n_cells_norm = group_stats[grp]["n_cells"] / metadata.shape[0]
    # index normalisé de POSITION (agnostique au nommage). Pour les 5 groupes
    # HUVEC, grp_idx/(5-1) = 0, 0.25, 0.5, 0.75, 1.0 = identique à (c+1)/4.
    cluster_idx = grp_idx / max(1, len(CELL_GROUPS) - 1)
    cell_group_features_list.append([is_senescent, n_cells_norm, cluster_idx])
cell_group_features = torch.tensor(cell_group_features_list, dtype=torch.float)

# =============================================================================
# 6. CONSTRUCTION DES ARÊTES
# =============================================================================
# Le graphe hétérogène contient 7 types d'arêtes (+ les reverses) :
#   a. expresses     : cell_group → gene (expression, 7 features)
#   b. ppi           : gene ↔ gene (STRING, 1 feature = score normalisé)
#   c. same_pathway  : gene ↔ gene (REACTOME, pas de features)
#   d. regulates     : TF → cible (pySCENIC, 1 feature = poids de régulation)
#   e. coexpression  : gene ↔ gene (GRNBoost2 top 2%, 1 feature = importance)
#   f. cocatalysis   : gene ↔ gene (HuMess, 2 features = [in_P4, in_P16])
# Toutes les arêtes gene↔gene sont rendues BIDIRECTIONNELLES (i→j ET j→i)
# car le message passing dans un GNN est directionnel : un noeud agrège les
# messages de ses voisins entrants. Sans bidirectionnel, un gène ne verrait
# que ses voisins dans un sens (ex : un TF verrait ses cibles mais pas l'inverse).
print("\n" + "=" * 70)
print("6. Construction des arêtes")
print("=" * 70)

# ── 6a. cell_group → gene (expression) ──────────────────────────────────────
# Arêtes bipartites reliant chaque groupe cellulaire à chaque gène.
# 7 features par arête = les stats d'expression calculées en section 4 +
# le score d'activité TF (AUCell) pour ce gène dans ce cluster.
# C'est un graphe COMPLET (chaque groupe est connecté à chaque gène),
# donc n_arêtes = 5 groupes × n_genes.
expr_src, expr_dst, expr_attrs = [], [], []
if MODULES["use_cell_group_edges"]:
    for grp_idx, grp in enumerate(CELL_GROUPS):
        stats = group_stats[grp]
        # cluster_id sert à récupérer le score AUCell : None pour le groupe
        # baseline ; sinon l'index numérique du cluster (HUVEC P16_cluster_N) ou
        # le nom du groupe (bulk, ex. "sen" → tf_activity vide → lookup = 0).
        if grp == CELL_GROUPS[0]:
            cluster_id = None
        else:
            _suf = grp.split("_")[-1]
            cluster_id = int(_suf) if _suf.isdigit() else grp

        for gene_idx in range(n_genes):
            gene_name = gene_symbols[gene_idx]
            # tf_activity : score AUCell du TF dans ce cluster. Vaut 0 si le
            # gène n'est pas un TF ou si on est dans P4 (pas de clusters).
            tf_act = 0.0
            if cluster_id is not None and gene_name in tf_activity.columns:
                if cluster_id in tf_activity.index:
                    tf_act = float(tf_activity.loc[cluster_id, gene_name])

            expr_src.append(grp_idx)     # Index du noeud cell_group source
            expr_dst.append(gene_idx)    # Index du noeud gene destination
            expr_attrs.append([
                float(stats["mean_expression"][gene_idx]),   # Feature 1 : expression moyenne
                float(stats["pct_expressing"][gene_idx]),     # Feature 2 : % de cellules exprimant
                tf_act,                                       # Feature 3 : activité TF (AUCell)
                float(stats["std_expression"][gene_idx]),     # Feature 4 : écart-type
                float(stats["cv_expression"][gene_idx]),      # Feature 5 : coefficient de variation
                float(stats["q25"][gene_idx]),                # Feature 6 : 1er quartile
                float(stats["q75"][gene_idx]),                # Feature 7 : 3ème quartile
            ])

# Conversion en tenseurs PyG (format COO : [2, n_edges])
edge_index_expresses = (torch.tensor([expr_src, expr_dst], dtype=torch.long)
                        if expr_src else torch.zeros((2, 0), dtype=torch.long))
edge_attr_expresses = (torch.tensor(expr_attrs, dtype=torch.float)
                       if expr_attrs else torch.zeros((0, 7), dtype=torch.float))
# Z-score normalisation colonne par colonne pour que chaque feature ait
# mean=0 et std=1. Important pour GATConv qui calcule des scores d'attention
# sur les features d'arête — sans normalisation, les features à grande
# échelle domineraient le calcul d'attention.
if edge_attr_expresses.numel() > 0:
    for col in range(edge_attr_expresses.shape[1]):
        col_data = edge_attr_expresses[:, col]
        mu, std = col_data.mean(), col_data.std() + 1e-8
        edge_attr_expresses[:, col] = (col_data - mu) / std
print(f"  expresses : {edge_index_expresses.shape[1]} arêtes, "
      f"7 features [mean, pct, tf_act, std, cv, q25, q75]"
      + ("" if MODULES["use_cell_group_edges"] else " [SKIP --no-cell-group-edges]"))

# ── 6b. PPI STRING ──────────────────────────────────────────────────────────
# Arêtes protéine-protéine de STRING (>= 900, highest confidence).
# Chaque interaction est BIDIRECTIONNELLE (i→j ET j→i).
# Feature = combined_score / 1000 (normalisé dans [0, 1]).
# Rôle biologique : encode les interactions physiques entre protéines.
# Les hubs PPI (gènes avec beaucoup de partenaires) reçoivent plus de
# messages et tendent à avoir des embeddings plus informatifs.
ppi_src, ppi_dst, ppi_w = [], [], []
if MODULES["use_ppi"]:
    for _, row in ppi_hc.iterrows():
        s1, s2 = string2sym.get(row["protein1"]), string2sym.get(row["protein2"])
        if s1 and s2 and s1 in gene_to_idx and s2 in gene_to_idx:
            i, j = gene_to_idx[s1], gene_to_idx[s2]
            # Bidirectionnel : on ajoute les deux directions
            ppi_src.extend([i, j])
            ppi_dst.extend([j, i])
            ppi_w.extend([row["combined_score"] / 1000.0] * 2)  # Même poids dans les deux sens

edge_index_ppi = (torch.tensor([ppi_src, ppi_dst], dtype=torch.long)
                  if ppi_src else torch.zeros((2, 0), dtype=torch.long))
edge_attr_ppi = (torch.tensor(ppi_w, dtype=torch.float).unsqueeze(1)
                 if ppi_w else torch.zeros((0, 1), dtype=torch.float))
print(f"  ppi : {len(ppi_src)//2} interactions ({len(ppi_src)} arêtes)"
      + ("" if MODULES["use_ppi"] else " [SKIP --no-ppi]"))

# ── 6c. REACTOME (same_pathway) ─────────────────────────────────────────────
# Pour chaque pathway REACTOME (2 à 20 gènes), on crée une arête entre
# TOUTES les paires de gènes du pathway (graphe complet intra-pathway).
# Pas de features d'arête — la simple existence de l'arête encode le fait
# que les deux gènes partagent un pathway biologique.
# react_pairs déduplique : si deux gènes partagent plusieurs pathways,
# on ne crée qu'une seule arête (bidirectionnelle).
# Rôle biologique : les gènes d'un même pathway coopèrent fonctionnellement.
# Le message passing propagera l'information au sein des modules fonctionnels.
react_src, react_dst = [], []
react_pairs = set()  # Déduplique les paires (un gène peut être dans plusieurs pathways)
if MODULES["use_reactome"]:
    for pw_genes in reactome_pathways.values():
        gene_list = sorted(pw_genes)
        # Double boucle : toutes les paires (i, j) avec i < j (évite les doublons a↔b / b↔a)
        for i_idx in range(len(gene_list)):
            for j_idx in range(i_idx + 1, len(gene_list)):
                g1, g2 = gene_list[i_idx], gene_list[j_idx]
                if g1 in gene_to_idx and g2 in gene_to_idx:
                    pair = (min(gene_to_idx[g1], gene_to_idx[g2]),
                            max(gene_to_idx[g1], gene_to_idx[g2]))
                    if pair not in react_pairs:
                        react_pairs.add(pair)
                        # Bidirectionnel
                        react_src.extend([pair[0], pair[1]])
                        react_dst.extend([pair[1], pair[0]])

edge_index_pathway = (torch.tensor([react_src, react_dst], dtype=torch.long)
                     if react_src else torch.zeros((2, 0), dtype=torch.long))
print(f"  pathway : {len(react_pairs)} paires ({len(react_src)} arêtes)"
      + ("" if MODULES["use_reactome"] else " [SKIP --no-reactome]"))

# ── 6d. Regulon (pySCENIC) — arêtes "regulates" ────────────────────────────
# Liens DIRIGÉS TF → gène cible (pas bidirectionnel à ce stade).
# Feature = poids de régulation (confiance du lien dans le regulon).
# Ces arêtes encodent la structure régulatrice : un TF qui active ou
# réprime ses cibles. Le sens est important biologiquement.
# MAIS on crée aussi l'arête inverse "regulated_by" (cible → TF) plus bas,
# pour que les TFs reçoivent aussi de l'information de leurs cibles
# pendant le message passing. Biologiquement, ça n'a pas de sens causal,
# mais pour le GNN c'est nécessaire : sans ça, les cibles ne propagent
# pas d'information vers les TFs qui les régulent.
reg_src, reg_dst, reg_w = [], [], []
reg_pairs = set()  # Déduplique si un TF régule la même cible dans plusieurs regulons
if MODULES["use_scenic_regulons"]:
    for _, row in regulon_edges.iterrows():
        tf, target = row["TF_clean"], row["target_gene"]
        if tf in gene_to_idx and target in gene_to_idx:
            pair = (gene_to_idx[tf], gene_to_idx[target])
            if pair not in reg_pairs:
                reg_pairs.add(pair)
                reg_src.append(pair[0])      # TF (source)
                reg_dst.append(pair[1])      # Cible (destination)
                reg_w.append(float(row["weight"]))  # Poids de régulation

edge_index_regulates = (torch.tensor([reg_src, reg_dst], dtype=torch.long)
                       if reg_src else torch.zeros((2, 0), dtype=torch.long))
edge_attr_regulates = (torch.tensor(reg_w, dtype=torch.float).unsqueeze(1)
                      if reg_w else torch.zeros((0, 1), dtype=torch.float))
# Normalisation min-max par le poids max (les poids SCENIC ne sont pas bornés)
if edge_attr_regulates.numel() > 0:
    edge_attr_regulates = edge_attr_regulates / (edge_attr_regulates.max() + 1e-8)
# Arête inverse "regulated_by" : cible → TF (même poids, sens inversé).
# Permet aux TFs de recevoir de l'information de leurs cibles pendant
# le message passing.
edge_index_regulated_by = (torch.tensor([reg_dst, reg_src], dtype=torch.long)
                          if reg_src else torch.zeros((2, 0), dtype=torch.long))
print(f"  regulates : {len(reg_pairs)} liens TF→cible"
      + ("" if MODULES["use_scenic_regulons"] else " [SKIP --no-scenic-regulons]"))

# ── 6e. Co-expression (GRNBoost2) ──────────────────────────────────────────
# Arêtes de co-expression inférées par GRNBoost2 (top 2% des poids).
# GRNBoost2 utilise le gradient boosting pour prédire l'expression d'un
# gène à partir de tous les autres. Les "TF" et "target" dans GRNBoost2
# ne sont pas nécessairement des TF biologiques — c'est juste la nomenclature
# du modèle (prédicteur → prédit). On traite ces arêtes comme non dirigées.
# Feature = importance GRNBoost2 normalisée par le max.
coexpr_src, coexpr_dst, coexpr_w = [], [], []
coexpr_pairs = set()
# V4.2 : en mode differential, edge_attr = 6 colonnes (option A).
_COEXPR_DIM = 6 if COEXPR_DIFFERENTIAL else 1
if MODULES["use_coexpr"]:
    for _, row in adjacencies_filtered.iterrows():
        g1, g2 = str(row["TF"]), str(row["target"])
        if g1 in gene_to_idx and g2 in gene_to_idx:
            i, j = gene_to_idx[g1], gene_to_idx[g2]
            pair = (min(i, j), max(i, j))
            if pair not in coexpr_pairs:
                coexpr_pairs.add(pair)
                coexpr_src.extend([i, j])
                coexpr_dst.extend([j, i])
                if COEXPR_DIFFERENTIAL:
                    # [imp_p4, imp_p16, delta, cat_shared, cat_p4, cat_p16]
                    feat = [
                        float(row["importance_p4_norm"]),
                        float(row["importance_p16_norm"]),
                        float(row["delta"]),
                        float(row["cat_shared"]),
                        float(row["cat_p4"]),
                        float(row["cat_p16"]),
                    ]
                    coexpr_w.append(feat)
                    coexpr_w.append(feat)  # bidirectionnel, même attr
                else:
                    imp = float(row["importance"])
                    coexpr_w.extend([imp, imp])

# Conversion en tenseurs PyG. Si aucune co-expression n'a été trouvée,
# on crée un tenseur vide (0 arêtes) pour éviter les erreurs en aval.
edge_index_coexpr = torch.tensor(
    [coexpr_src, coexpr_dst] if coexpr_src else [[], []], dtype=torch.long
)
if COEXPR_DIFFERENTIAL:
    # Pas de re-normalisation : imp_p4/imp_p16 déjà min-max normalisés
    # dans build_diff_coexpr.py ; delta ∈ [-1, 1] ; cat_* ∈ {0, 1}.
    coexpr_w_tensor = (torch.tensor(coexpr_w, dtype=torch.float)
                       if coexpr_w else torch.zeros((0, _COEXPR_DIM)))
else:
    coexpr_w_tensor = (torch.tensor(coexpr_w, dtype=torch.float).unsqueeze(1)
                       if coexpr_w else torch.zeros((0, 1)))
    # Normalisation min-max par le max
    if coexpr_w_tensor.numel() > 0:
        coexpr_w_tensor = coexpr_w_tensor / (coexpr_w_tensor.max() + 1e-8)
print(f"  coexpression : {len(coexpr_pairs)} paires ({len(coexpr_src)} arêtes)"
      + ("" if MODULES["use_coexpr"] else " [SKIP --no-coexpr]"))

# ── 6f. HuMess : cocatalysis (A) + importance métabolique (B) ───────────────
# HuMess (Métabolisme Humain par Échantillonnage de Solutions) est un pipeline
# de modélisation métabolique qui produit deux informations complémentaires :
#
# PARTIE A — Arêtes "metabolic_cocatalysis" :
#   Deux gènes sont reliés s'ils catalysent la même réaction métabolique
#   (déduit des GPR rules = Gene-Protein-Reaction rules de CarveMe).
#   Les GPR rules associent chaque réaction du modèle métabolique à une
#   combinaison booléenne de gènes (ex : "(ALDOB or ALDOC) and GAPDH").
#   Feature d'arête = [in_P4, in_P16] (binaire) : le lien peut exister
#   dans P4 seul, P16 seul, ou les deux → le modèle apprend la
#   CONDITION-SPÉCIFICITÉ du lien métabolique.
#
# PARTIE B — Features de noeud "gene" (ajoutées au vecteur de features) :
#   Importance métabolique par gène = max(importance) sur les réactions
#   catalysées par ce gène, calculée par Corner Sampling.
#   On ajoute 4 features : imp_P4_z, imp_P16_z, imp_delta, has_humess.
#   has_humess est un masque binaire (1 si le gène a des données HuMess)
#   pour éviter que les 0 imputés pour les gènes absents du modèle
#   métabolique soient interprétés comme "importance nulle" (ce serait
#   un faux signal — l'absence de données ≠ absence d'importance).
print("\n  HuMess (cocatalysis + importance)...")

# Regex pour extraire les symboles de gène d'une règle GPR.
# Exemple : "(ALDOB or ALDOC) and GAPDH" → {"ALDOB", "ALDOC", "GAPDH"}
_gene_token_re = re.compile(r"[A-Za-z0-9_\-\.]+")
_gpr_skip = {"and", "or", "AND", "OR", "And", "Or"}  # Mots-clés booléens à ignorer


def _parse_gpr(gpr_str):
    """
    Extrait les symboles de gène d'une règle GPR (Gene-Protein-Reaction).
    Supprime les parenthèses, puis filtre les mots-clés booléens (and/or).
    Retourne un set de symboles de gènes.
    """
    cleaned = gpr_str.replace("(", " ").replace(")", " ")
    return {t for t in _gene_token_re.findall(cleaned) if t not in _gpr_skip}


# --- PARTIE A : parse des GPR rules par condition (arêtes cocatalysis) ---
# Pour chaque condition (P4, P16), on charge les GPR rules du modèle
# métabolique CarveMe et on identifie quels gènes catalysent quelles réactions.
# Format du fichier : réaction_id <TAB> GPR_rule
# Exemple : "R_PFK → (PFKL or PFKM or PFKP)"
# Modulaire : si --no-humess-edges, on n'ouvre même pas les fichiers GPR.
reaction_to_genes = {}   # condition → {réaction : set(gènes)}
gene_to_reactions = {c: {} for c in HUMESS_CONDITIONS}  # condition → {gène : set(réactions)}
cocat_pair_flags = {}  # (index_gene_i, index_gene_j) → [in_P4, in_P16]
cocat_src, cocat_dst, cocat_attr = [], [], []

if MODULES["use_humess_edges"]:
    for cond in HUMESS_CONDITIONS:
        path = os.path.join(HUMESS_DIR, "models", cond, "stats", "carveme.gr-rules.tsv")
        cond_map = {}
        if not os.path.exists(path):
            print(f"    [warn] introuvable : {path}")
            reaction_to_genes[cond] = cond_map
            continue
        with open(path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                rxn, gpr = parts[0], parts[1]
                genes = _parse_gpr(gpr)  # Extrait les gènes de la règle GPR
                cond_map[rxn] = genes
                # Index inverse : pour chaque gène, stocker les réactions qu'il catalyse
                for g in genes:
                    gene_to_reactions[cond].setdefault(g, set()).add(rxn)
        reaction_to_genes[cond] = cond_map
        print(f"    {cond} : {len(cond_map)} réactions, "
              f"{len(gene_to_reactions[cond])} gènes")

    # Construction des paires de cocatalyse (indices graphe).
    # Deux gènes sont co-catalytiques s'ils apparaissent dans la même GPR rule
    # (= catalysent la même réaction). Le flag [in_P4, in_P16] indique dans
    # quelle(s) condition(s) ce lien existe. Exemple :
    #   [1, 1] = la réaction existe dans les deux conditions
    #   [1, 0] = la réaction n'existe que dans P4 (perdue en sénescence)
    #   [0, 1] = la réaction apparaît en P16 (gain métabolique en sénescence)
    for cond_idx, cond in enumerate(HUMESS_CONDITIONS):
        for rxn, genes in reaction_to_genes[cond].items():
            # Convertir les symboles en indices du graphe (ignorer les gènes absents)
            idx_list = sorted({gene_to_idx[g] for g in genes if g in gene_to_idx})
            # Toutes les paires (graphe complet intra-réaction)
            for a in range(len(idx_list)):
                for b in range(a + 1, len(idx_list)):
                    pair = (idx_list[a], idx_list[b])
                    flags = cocat_pair_flags.setdefault(pair, [0.0, 0.0])
                    flags[cond_idx] = 1.0  # Marquer la condition dans le flag

    # Conversion en arêtes bidirectionnelles avec features [in_P4, in_P16]
    for (i, j), flags in cocat_pair_flags.items():
        cocat_src.extend([i, j])       # Bidirectionnel
        cocat_dst.extend([j, i])
        cocat_attr.append(flags)       # Même flags dans les deux sens
        cocat_attr.append(flags)
else:
    print("    cocatalysis : SKIP (--no-humess-edges)")

edge_index_cocat = torch.tensor(
    [cocat_src, cocat_dst] if cocat_src else [[], []], dtype=torch.long
)
edge_attr_cocat = (torch.tensor(cocat_attr, dtype=torch.float)
                   if cocat_attr else torch.zeros((0, 2)))
if MODULES["use_humess_edges"]:
    print(f"    cocatalysis : {len(cocat_pair_flags)} paires "
          f"({edge_index_cocat.shape[1]} arêtes, attr=[in_P4, in_P16])")

# --- PARTIE B : importance métabolique par gène ---
# Corner Sampling (CS) estime l'importance de chaque gène dans le modèle
# métabolique en mesurant l'impact de sa suppression sur l'espace des flux.
# Un gène peut catalyser plusieurs réactions → on prend le MAX d'importance
# (le gène est "aussi important que sa réaction la plus importante").
gene_importance = {c: np.zeros(n_genes, dtype=np.float32) for c in HUMESS_CONDITIONS}
gene_in_model = {c: np.zeros(n_genes, dtype=np.float32) for c in HUMESS_CONDITIONS}

if MODULES["use_humess_features"]:
    for cond in HUMESS_CONDITIONS:
        path = os.path.join(HUMESS_DIR, "models", cond, "cs",
                            f"cs_gene_to_importance_{cond}.tsv")
        if not os.path.exists(path):
            print(f"    [warn] introuvable : {path}")
            continue
        df_imp = pd.read_csv(path, sep="\t")
        # Format : symbol, bigg (réaction), importance (une ligne par (gène, réaction))
        df_imp = df_imp.dropna(subset=["symbol", "importance"])
        # Max importance par symbole : agrège les réactions → 1 valeur par gène
        agg = df_imp.groupby("symbol")["importance"].max()
        for sym, val in agg.items():
            if sym in gene_to_idx:
                gene_importance[cond][gene_to_idx[sym]] = float(val)
                gene_in_model[cond][gene_to_idx[sym]] = 1.0  # Marquer comme "a des données HuMess"
else:
    print("    HuMess gene importance : SKIP (--no-humess-features)")

# Transformation log1p + z-score par condition.
# log1p : compresse les valeurs extrêmes (l'importance brute peut varier
#   sur plusieurs ordres de grandeur).
# z-score : centre et réduit, calculé UNIQUEMENT sur les gènes présents
#   dans le modèle métabolique (mask). Les gènes absents restent à 0.
# Cela évite que les gènes sans données HuMess (la majorité) polluent
# la moyenne et l'écart-type du z-score.
def _log1p_zscore(arr, mask):
    """log1p + z-score sur les éléments masqués ; les autres restent à 0."""
    out = np.zeros_like(arr)
    if mask.sum() < 2:
        return out
    vals = np.log1p(arr[mask.astype(bool)])
    mu, sd = vals.mean(), vals.std() + 1e-8
    out[mask.astype(bool)] = (vals - mu) / sd
    return out

# imp_P4_z, imp_P16_z : importance métabolique z-scorée par condition. Clés
# génériques = HUMESS_CONDITIONS (HUVEC : P4,P16 ; bulk : pro,sen). Les NOMS de
# features (imp_P4/imp_P16/imp_delta) restent des labels internes (cf. GENE_
# FEATURE_FLAGS) : imp_P4 = baseline (cond[0]), imp_P16 = état avancé (cond[-1]).
_hc0, _hc1 = HUMESS_CONDITIONS[0], HUMESS_CONDITIONS[-1]
imp_P4_z = _log1p_zscore(gene_importance[_hc0], gene_in_model[_hc0])
imp_P16_z = _log1p_zscore(gene_importance[_hc1], gene_in_model[_hc1])
# imp_delta : différence cond[-1] - cond[0], positif = gène plus important
# métaboliquement à l'état avancé. Signal de REMODELAGE MÉTABOLIQUE.
imp_delta = imp_P16_z - imp_P4_z
# has_humess : masque binaire (1 si le gène apparaît dans le modèle métabolique
# d'au moins une condition). Distingue "importance=0 car absent" vs "présent".
has_humess = ((gene_in_model[_hc0] + gene_in_model[_hc1]) > 0).astype(np.float32)

print(f"    importance  : {_hc0}={int(gene_in_model[_hc0].sum())} gènes, "
      f"{_hc1}={int(gene_in_model[_hc1].sum())} gènes, "
      f"has_humess={int(has_humess.sum())}/{n_genes}")

# ── 6g. OmniPath (V4) : signaling dirigé signé + TF curé ────────────────────
# DEUX nouveaux edge_types optionnels :
#
#   ("gene", "signaling", "gene")  — kinase-substrat OmniPath + SIGNOR causal
#       directionnel, edge_attr = [score, sign∈{−1,0,+1}].
#       Active une couche de message passing CAUSALE (asymétrique) où
#       l'attention GAT peut apprendre que activation et inhibition portent
#       des messages de polarité différente. Réf : Türei et al. Nat Commun
#       2021 ; Lo Surdo et al. NAR 2023 (SIGNOR 3.0).
#
#   ("gene", "tf_curated", "gene") + reverse  — CollecTRI (fallback DoRothEA)
#       TF→cible curé sur ~1186 TFs (vs ~50 pySCENIC), edge_attr = [score, sign].
#       Co-existe avec "regulates" (pySCENIC, HUVEC-spécifique inféré du
#       scRNA-seq) — laisse le GNN apprendre les deux sources distinctement
#       (option (c) du design V4). Réf : Müller-Dott et al. Genome Biol 2023.
#
# Les deux sources sont chargées depuis un cache TSV pré-téléchargé
# (`scripts/cache_omnipath.py` à lancer sur le frontal). En cas d'absence
# de cache et de --omnipath-download-if-missing OFF, on saute proprement
# l'arête sans crasher le run (modularité préservée).
#
# Default OFF : à activer explicitement via --use-omnipath-signaling /
# --use-omnipath-tf-curated pour les ablations V4.
op_sig_src = op_sig_dst = np.array([], dtype=np.int64)
op_sig_attr = np.zeros((0, 2), dtype=np.float32)
op_tf_src  = op_tf_dst  = np.array([], dtype=np.int64)
op_tf_attr = np.zeros((0, 2), dtype=np.float32)

if MODULES["use_omnipath_signaling"] or MODULES["use_omnipath_tf_curated"]:
    print("\n  OmniPath (V4)…")
    # Idem section 2.5 : force offline si download non autorisé (compute node).
    # Idempotent (déjà fait en 2.5 si --include-omnipath-genes, no-op sinon).
    if not CLI_ARGS.omnipath_download_if_missing:
        os.environ["GNN_OMNIPATH_OFFLINE"] = "1"
    try:
        from omnipath_integration import (
            load_signaling_directed,
            load_collectri_tf_target,
            load_signed_ppi_signor,
            merge_signed_directed,
            silence_omnipath_logging,
        )
        _OPI = True
        # Coupe le bruit quand on sait qu'on ne télécharge pas. Les compute
        # nodes Nautilus déclenchent des centaines de "WARNING:root:Failed
        # to download" sinon, à cause des metadata pre-fetches d'omnipath-py.
        if not CLI_ARGS.omnipath_download_if_missing:
            silence_omnipath_logging()
    except ImportError as _e:
        print(f"    [warn] import omnipath_integration KO ({_e}) — "
              f"aucune arête OmniPath ne sera ajoutée.")
        _OPI = False

    if _OPI:
        _opi_kwargs = dict(
            cache_dir=OMNIPATH_CACHE_DIR,
            available_genes=available_set,
            gene_to_idx=gene_to_idx,
            download_if_missing=CLI_ARGS.omnipath_download_if_missing,
            alias_map=omnipath_alias_map,  # {} = identité (alias OFF)
        )

        if MODULES["use_omnipath_signaling"]:
            # Fusion kinase-substrat OmniPath + PPI causal SIGNOR — même
            # sémantique sémantique (lien causal signé), même edge_type.
            sig_kin = load_signaling_directed(**_opi_kwargs)
            sig_sgn = load_signed_ppi_signor(**_opi_kwargs)
            op_sig_src, op_sig_dst, op_sig_attr = merge_signed_directed(
                (sig_kin[0], sig_kin[1], sig_kin[2]),
                (sig_sgn[0], sig_sgn[1], sig_sgn[2]),
            )
            n_pos = int((op_sig_attr[:, 1] > 0).sum())
            n_neg = int((op_sig_attr[:, 1] < 0).sum())
            n_neu = int((op_sig_attr[:, 1] == 0).sum())
            print(f"    signaling (kinase+SIGNOR) : {len(op_sig_src)} arêtes "
                  f"[+:{n_pos} −:{n_neg} 0:{n_neu}]")

        if MODULES["use_omnipath_tf_curated"]:
            tf_op = load_collectri_tf_target(**_opi_kwargs)
            op_tf_src, op_tf_dst, op_tf_attr = tf_op[0], tf_op[1], tf_op[2]
            n_pos = int((op_tf_attr[:, 1] > 0).sum())
            n_neg = int((op_tf_attr[:, 1] < 0).sum())
            n_neu = int((op_tf_attr[:, 1] == 0).sum())
            n_tfs = len(set(op_tf_src.tolist())) if op_tf_src.size > 0 else 0
            print(f"    tf_curated (CollecTRI)    : {len(op_tf_src)} arêtes, "
                  f"{n_tfs} TFs uniques [+:{n_pos} −:{n_neg} 0:{n_neu}]")

# Conversion en tenseurs PyG (vides si désactivé ou cache absent)
edge_index_signaling = (torch.tensor([op_sig_src.tolist(), op_sig_dst.tolist()],
                                     dtype=torch.long)
                        if op_sig_src.size > 0
                        else torch.zeros((2, 0), dtype=torch.long))
edge_attr_signaling = (torch.tensor(op_sig_attr, dtype=torch.float)
                       if op_sig_attr.size > 0
                       else torch.zeros((0, 2), dtype=torch.float))
edge_index_tf_curated = (torch.tensor([op_tf_src.tolist(), op_tf_dst.tolist()],
                                      dtype=torch.long)
                         if op_tf_src.size > 0
                         else torch.zeros((2, 0), dtype=torch.long))
edge_attr_tf_curated = (torch.tensor(op_tf_attr, dtype=torch.float)
                        if op_tf_attr.size > 0
                        else torch.zeros((0, 2), dtype=torch.float))
# Reverse pour message passing arrière (target → TF), même attr.
edge_index_tf_curated_by = (torch.tensor([op_tf_dst.tolist(),
                                          op_tf_src.tolist()],
                                         dtype=torch.long)
                            if op_tf_src.size > 0
                            else torch.zeros((2, 0), dtype=torch.long))

# ── V4.2 : Reactome FI signé ─────────────────────────────────────────────────
# Reactome Functional Interactions (Wu 2010 Genome Biol). Fichier
# FIsInGene_*_with_annotations.txt : Gene1, Gene2, Annotation, Direction,
# Score. La colonne Direction encode le signe : '|' = inhibition, '>'/'<'
# = activation/direction, '-'/'<->' = non signé. On exclut les FI
# 'predicted' (computationnels non curés). edge_attr = [score, sign].
# Apport mesuré (audit 2026-05-12) : ~45k arêtes signées NOUVELLES (75%
# absentes de PPI/SIGNOR/CollecTRI). Cf. §14bis.6quater du rapport.
reactome_fi_src, reactome_fi_dst, reactome_fi_attr = [], [], []
# V6.2 (2026-07-16) : arêtes reactome_fi à orientation INCONNUE (Direction '-')
# séparées dans un edge_type distinct, suppressible. Vide en mode legacy.
reactome_fi_und_src, reactome_fi_und_dst, reactome_fi_und_attr = [], [], []
if MODULES["use_reactome_fi"]:
    _fi_path = CLI_ARGS.reactome_fi_file
    if not os.path.isabs(_fi_path):
        # Reactome FI lives in the data tree → resolve under DATA_DIR so a single
        # GNN_DATA_DIR (e.g. LAB-DATA clone) covers it like PPI/databases. Strip a
        # leading "data/" so it works whether DATA_DIR ends in /data or not.
        _rel = _fi_path[5:] if _fi_path.startswith("data/") else _fi_path
        _fi_path = os.path.join(DATA_DIR, _rel)
    if not os.path.exists(_fi_path):
        print(f"  [warn] Reactome FI : fichier absent ({_fi_path}) — "
              f"edge_type 'reactome_fi' vide. Télécharger via "
              f"scripts/cache_reactome_fi.sh")
    else:
        _fi = pd.read_csv(_fi_path, sep="\t")
        _n_raw = len(_fi)
        # ── V6.2 flags (défaut = LEGACY V5/V6 strict) ────────────────────────
        #   --reactome-fi-directed   : exploite la colonne Direction pour
        #       ORIENTER les arêtes (au lieu de tout symétriser). Sépare les
        #       arêtes à orientation inconnue ('-') dans `reactome_fi_undirected`.
        #   --reactome-fi-predicted  : inclut les FI 'predicted' (défaut exclues).
        #   --no-reactome-fi-undirected : en mode directed, JETTE les arêtes non
        #       orientées (ne garde que le causal ->/<-/<->/-|/|-).
        _fi_directed = bool(getattr(CLI_ARGS, "reactome_fi_directed", False))
        _fi_keep_pred = bool(getattr(CLI_ARGS, "reactome_fi_predicted", False))
        _fi_keep_und = not bool(getattr(CLI_ARGS, "no_reactome_fi_undirected", False))
        if not _fi_keep_pred:                # exclure les FI purement prédites
            _fi = _fi[~_fi["Annotation"].astype(str).str.contains(
                "predicted", case=False, na=False)]

        def _fi_sign(d: str) -> float:       # LEGACY : signe seul (V5/V6)
            d = str(d)
            if "|" in d:
                return -1.0
            if ">" in d or "<" in d:
                return 1.0
            return 0.0

        def _fi_parse(d: str):
            """Direction → (orientation, sign). Le symbole marque l'EXTRÉMITÉ
            CIBLE : droite ('>'/'|') = Gene2 (forward), gauche ('<'/'|') =
            Gene1 (backward) ; '|' = inhibition. '-' = orientation inconnue."""
            d = str(d).strip()
            if d in ("", "-", "nan"):
                return "none", 0.0
            left, right = d[0] in "<|", d[-1] in ">|"
            sign = -1.0 if "|" in d else 1.0
            if left and right:
                return "both", sign
            if right:
                return "forward", sign
            if left:
                return "backward", sign
            return "none", sign

        _seen_fi = set()
        for _, r in _fi.iterrows():
            g1, g2 = str(r["Gene1"]), str(r["Gene2"])
            if g1 not in gene_to_idx or g2 not in gene_to_idx:
                continue
            i, j = gene_to_idx[g1], gene_to_idx[g2]
            pair = (min(i, j), max(i, j))
            if pair in _seen_fi:
                continue
            _seen_fi.add(pair)
            score = float(r["Score"]) if not pd.isna(r["Score"]) else 1.0
            if not _fi_directed:
                # LEGACY : signe seul, symétrique (comportement V5/V6).
                sign = _fi_sign(r["Direction"])
                reactome_fi_src.extend([i, j])
                reactome_fi_dst.extend([j, i])
                reactome_fi_attr.extend([[score, sign], [score, sign]])
                continue
            orient, sign = _fi_parse(r["Direction"])
            if orient == "none":             # orientation inconnue → type séparé
                if _fi_keep_und:
                    reactome_fi_und_src.extend([i, j])
                    reactome_fi_und_dst.extend([j, i])
                    reactome_fi_und_attr.extend([[score, sign], [score, sign]])
            elif orient == "forward":        # Gene1 → Gene2
                reactome_fi_src.append(i)
                reactome_fi_dst.append(j)
                reactome_fi_attr.append([score, sign])
            elif orient == "backward":       # Gene2 → Gene1
                reactome_fi_src.append(j)
                reactome_fi_dst.append(i)
                reactome_fi_attr.append([score, sign])
            else:                            # both : ->/<- attesté dans les 2 sens
                reactome_fi_src.extend([i, j])
                reactome_fi_dst.extend([j, i])
                reactome_fi_attr.extend([[score, sign], [score, sign]])
        n_fi_signed = sum(1 for a in reactome_fi_attr if a[1] != 0)
        if _fi_directed:
            print(f"  Reactome FI [directed] : {_n_raw} brut → "
                  f"{len(reactome_fi_src)} orientées ({n_fi_signed} signées) + "
                  f"{len(reactome_fi_und_src)} non-orientées"
                  + ("" if _fi_keep_und else " [JETÉES --no-reactome-fi-undirected]")
                  + (" [+predicted]" if _fi_keep_pred else ""))
        else:
            print(f"  Reactome FI [legacy] : {_n_raw} brut → {len(_seen_fi)} "
                  f"paires curées ({n_fi_signed//2} signées)")

edge_index_reactome_fi = (torch.tensor([reactome_fi_src, reactome_fi_dst],
                                       dtype=torch.long)
                          if reactome_fi_src
                          else torch.zeros((2, 0), dtype=torch.long))
edge_attr_reactome_fi = (torch.tensor(reactome_fi_attr, dtype=torch.float)
                         if reactome_fi_attr
                         else torch.zeros((0, 2), dtype=torch.float))
edge_index_reactome_fi_und = (torch.tensor(
    [reactome_fi_und_src, reactome_fi_und_dst], dtype=torch.long)
    if reactome_fi_und_src else torch.zeros((2, 0), dtype=torch.long))
edge_attr_reactome_fi_und = (torch.tensor(reactome_fi_und_attr, dtype=torch.float)
                             if reactome_fi_und_attr
                             else torch.zeros((0, 2), dtype=torch.float))

# ── V4.2 : déduplication PPI vs arêtes signées+orientées ─────────────────────
# Pour une paire (a,b), si une arête SIGNÉE ORIENTÉE existe (signaling /
# tf_curated / reactome_fi avec sign≠0), l'arête PPI non-signée
# non-orientée est redondante : elle gonfle le compte et son message
# symétrique s'ajoute (agrégation HeteroConv-sum) au message signé →
# dilution partielle (cf. §14bis.6bis : PPI domine déjà ‖h‖).
#
# `--dedup-ppi-signed {off,remove,annotate}` (DÉFAUT off — comportement
# inchangé). Le DIAGNOSTIC (combien d'arêtes/gènes seraient touchés)
# est TOUJOURS calculé et loggé, même en mode off, pour décider sur
# chiffres. Cf. §14bis.6quaterdecies. NB : bénéfice « bien orienter le
# message » seulement PARTIEL en V4.2 (le signe n'entre qu'en
# attention, design A) ; bénéfice PLEIN avec V5 (SignedGATConv message
# + BilinearSignedDecoder). Les deux sont décorrélés mais
# complémentaires.
def _signed_pair_set(*eis_attrs) -> set[tuple[int, int]]:
    """Paires (min,max) couvertes par une arête signée (sign≠0)."""
    s: set[tuple[int, int]] = set()
    for ei, attr in eis_attrs:
        if ei.numel() == 0:
            continue
        if attr is not None and attr.numel() > 0 and attr.dim() == 2 \
                and attr.shape[1] >= 2:
            sign = attr[:, 1]
            for k in range(ei.shape[1]):
                if float(sign[k]) != 0.0:
                    a, b = int(ei[0, k]), int(ei[1, k])
                    s.add((min(a, b), max(a, b)))
        else:
            for a, b in zip(ei[0].tolist(), ei[1].tolist()):
                s.add((min(a, b), max(a, b)))
    return s


_signed_pairs = _signed_pair_set(
    (edge_index_signaling, edge_attr_signaling),
    (edge_index_tf_curated, edge_attr_tf_curated),
    (edge_index_reactome_fi, edge_attr_reactome_fi),
)
# Diagnostic : combien d'arêtes PPI redondantes ? combien de gènes
# perdraient TOUTES leurs arêtes PPI si on dédupliquait ?
_n_ppi_before = int(edge_index_ppi.shape[1]) if edge_index_ppi.numel() else 0
_ppi_redundant_mask = None
if _n_ppi_before > 0 and _signed_pairs:
    import numpy as _np
    _src = edge_index_ppi[0].numpy()
    _dst = edge_index_ppi[1].numpy()
    _ppi_redundant_mask = _np.array([
        (min(int(a), int(b)), max(int(a), int(b))) in _signed_pairs
        for a, b in zip(_src, _dst)
    ])
    _n_redundant = int(_ppi_redundant_mask.sum())
    # Gènes qui n'ont QUE des arêtes PPI redondantes (perdraient tout PPI)
    _deg_all = _np.zeros(n_genes, dtype=_np.int64)
    _deg_keep = _np.zeros(n_genes, dtype=_np.int64)
    for i, (a, b) in enumerate(zip(_src, _dst)):
        _deg_all[int(a)] += 1
        if not _ppi_redundant_mask[i]:
            _deg_keep[int(a)] += 1
    _n_genes_lose_ppi = int(((_deg_all > 0) & (_deg_keep == 0)).sum())
    _n_genes_with_ppi = int((_deg_all > 0).sum())
    print(f"\n  [dedup-ppi diag] PPI redondant (paire signée existante) : "
          f"{_n_redundant}/{_n_ppi_before} arêtes "
          f"({100*_n_redundant/_n_ppi_before:.1f}%)")
    print(f"  [dedup-ppi diag] gènes perdant TOUTES leurs arêtes PPI si "
          f"dédup : {_n_genes_lose_ppi}/{_n_genes_with_ppi} "
          f"({100*_n_genes_lose_ppi/max(1,_n_genes_with_ppi):.1f}%)")
    _dedup_mode = getattr(CLI_ARGS, "dedup_ppi_signed", "off")
    if _dedup_mode == "remove":
        _keep = ~_ppi_redundant_mask
        edge_index_ppi = edge_index_ppi[:, _keep.tolist()]
        if edge_attr_ppi.numel() > 0:
            edge_attr_ppi = edge_attr_ppi[_keep.tolist()]
        print(f"  [dedup-ppi] mode=remove : PPI {_n_ppi_before} → "
              f"{edge_index_ppi.shape[1]} arêtes")
    elif _dedup_mode == "annotate":
        # edge_dim 1→2 : ajoute une colonne flag (1 si paire signée).
        _flag = torch.tensor(_ppi_redundant_mask.astype("float32")).unsqueeze(1)
        if edge_attr_ppi.numel() > 0:
            edge_attr_ppi = torch.cat([edge_attr_ppi, _flag], dim=1)
        else:
            edge_attr_ppi = _flag
        print(f"  [dedup-ppi] mode=annotate : edge_attr_ppi → dim "
              f"{edge_attr_ppi.shape[1]} (col 1 = has_signed_counterpart)")
    else:
        print(f"  [dedup-ppi] mode=off : aucune modif (diagnostic seul)")

# ── Finaliser les features de noeuds gene avec les degrés ────────────────────
# Les features de degré (nombre de voisins) sont calculées APRÈS la
# construction des arêtes. Elles capturent la "centralité" de chaque gène
# dans les différents réseaux. Un gène hub PPI avec beaucoup d'interactions
# protéine-protéine aura un ppi_degree élevé.
# Feature 3 : ppi_degree — nombre de voisins PPI (normalisé par le max)
ppi_degree = np.zeros(n_genes, dtype=np.float32)
for idx in ppi_src:
    ppi_degree[idx] += 1
ppi_degree_norm = ppi_degree / (ppi_degree.max() + 1e-8)

# Feature 4 : reg_degree — nombre de liens de régulation (TF→cible + cible→TF)
# On compte les deux directions : un TF qui régule 100 cibles aura un haut
# degré, mais une cible régulée par 5 TFs aussi.
reg_degree = np.zeros(n_genes, dtype=np.float32)
for idx in reg_src + reg_dst:
    reg_degree[idx] += 1
reg_degree_norm = reg_degree / (reg_degree.max() + 1e-8)

# ASSEMBLAGE FINAL DES FEATURES DE NOEUDS "gene" (modulaire — V3.5+) :
# Chaque colonne n'est incluse que si GENE_FEATURE_FLAGS[name] est True
# (dépend (a) de la source activée, (b) de --exclude-features).
# Ordre canonique conservé : is_tf, variance, ppi_degree, reg_degree,
# imp_P4, imp_P16, imp_delta, has_humess. AUCUNE feature ne contient
# log2FC, padj, ou delta_pct (anti-circularité).
_FEATURE_VECTORS = [
    ("is_tf",      is_tf),
    ("variance",   variance_norm),
    ("ppi_degree", ppi_degree_norm),
    ("reg_degree", reg_degree_norm),
    ("imp_P4",     imp_P4_z),
    ("imp_P16",    imp_P16_z),
    ("imp_delta",  imp_delta),
    ("has_humess", has_humess),
]

# V6 Module 1 (opt-in) : features de nœud OmniPath (localisation intercell +
# druggabilité + classe moléculaire) AJOUTÉES EN FIN → l'ordre canonique des
# features scRNA reste intact quand OFF. Offline-safe (sources absentes → 0).
if MODULES["use_omnipath_node_features"]:
    try:
        from omnipath_node_features import (
            build_node_feature_arrays as _build_op_nodefeat,
            coverage_summary as _op_nodefeat_cov,
        )
        _op_arrays = _build_op_nodefeat(
            gene_symbols=list(gene_symbols),
            cache_dir=OMNIPATH_CACHE_DIR,
            alias_map=omnipath_alias_map,
            download_if_missing=CLI_ARGS.omnipath_download_if_missing,
        )
        _FEATURE_VECTORS.extend(_op_arrays.items())
        print(_op_nodefeat_cov(_op_arrays))
    except ImportError as _e:
        print(f"  [warn] import omnipath_node_features KO ({_e}) — "
              f"features de nœud OmniPath OFF.")

# V-sup (opt-in, --de-features) : CIRCULAR DE node features (global/per-cluster
# log2FC, -log10 padj, Δpct) APPENDED LAST, so the canonical scRNA feature order
# is untouched when OFF. These deliberately BREAK the anti-circularity invariant
# — that is the point of the circular-ceiling pole. Because they are built here,
# --de-features invalidates the graph cache and shows up in RUN_TAG ('de-feat').
# SUP_LABELS is exported so --supervised reuses it without rebuilding.
SUP_LABELS = None
if getattr(CLI_ARGS, "de_features", False):
    import sys as _sys_de
    _HERE_DE = os.path.dirname(os.path.abspath(__file__))
    for _cand in (os.path.join(_HERE_DE, "..", "data", "preprocess"), _HERE_DE):
        _cand = os.path.abspath(_cand)
        if os.path.isdir(_cand) and _cand not in _sys_de.path:
            _sys_de.path.insert(0, _cand)
    from build_supervised_labels import build_supervised_labels as _build_sup_labels

    SUP_LABELS = _build_sup_labels(
        gene_symbols, GNN_DATA_DIR,
        recompute=getattr(CLI_ARGS, "supervised_recompute_labels", False))
    _de_mat = SUP_LABELS.de_feature_matrix()
    _de_names = SUP_LABELS.de_feature_names()
    assert _de_mat.shape[1] == len(_de_names), (
        f"de_feature_matrix {_de_mat.shape} vs {len(_de_names)} names")
    _FEATURE_VECTORS.extend(zip(_de_names, _de_mat.T))
    print(f"  [V-sup] +{_de_mat.shape[1]} features DE CIRCULAIRES {_de_names}")

_active_feature_names = [n for n, _ in _FEATURE_VECTORS if GENE_FEATURE_FLAGS[n]]
_active_feature_arrays = [v for n, v in _FEATURE_VECTORS if GENE_FEATURE_FLAGS[n]]

# Garantie d'au moins une feature : si tout est exclu, on retombe sur un
# vecteur de biais constant pour ne pas casser nn.Linear(0, hidden).
if not _active_feature_arrays:
    print("    [warn] toutes les features exclues — fallback bias constant")
    _active_feature_names = ["bias"]
    _active_feature_arrays = [np.ones(n_genes, dtype=np.float32)]

gene_features = torch.tensor(
    np.column_stack(_active_feature_arrays),
    dtype=torch.float,
)
print(f"\n  Gene features : {gene_features.shape}")
print(f"    actives : {_active_feature_names}")
print("    /!\\ CIRCULAIRE : log2FC/padj/Δpct INCLUS (--de-features)"
      if SUP_LABELS is not None else
      "    PAS de log2FC, padj, delta_pct (circularité supprimée)")

# =============================================================================
# 7. ASSEMBLAGE DU GRAPHE HÉTÉROGÈNE
# =============================================================================
# On assemble tous les noeuds et arêtes dans un objet HeteroData de PyG.
# HeteroData est le format standard de PyTorch Geometric pour les graphes
# hétérogènes (plusieurs types de noeuds et d'arêtes). Chaque type d'arête
# est identifié par un triplet (source_type, relation, dest_type).
#
# Structure finale du graphe :
#   Noeuds :
#     "gene"       : n_genes noeuds, 8 features topologiques
#     "cell_group" : 5 noeuds (P4 + 4 clusters P16), 3 features
#   Arêtes (jusqu'à 8 types) :
#     cell_group → gene (expresses) : 5 × n_genes, 7 features
#     gene → cell_group (expressed_in) : reverse, mêmes features
#     gene ↔ gene (ppi) : STRING high-confidence, 1 feature
#     gene ↔ gene (same_pathway) : REACTOME, pas de features
#     gene → gene (regulates) : pySCENIC TF→cible, 1 feature
#     gene → gene (regulated_by) : reverse de regulates, mêmes features
#     gene ↔ gene (coexpression) : GRNBoost2 top 2%, 1 feature
#     gene ↔ gene (metabolic_cocatalysis) : HuMess GPR, 2 features
print("\n" + "=" * 70)
print("7. Assemblage du graphe hétérogène")
print("=" * 70)

# HeteroData stocke les features et arêtes indexées par type
data = HeteroData()

# --- Noeuds ---
data["gene"].x = gene_features            # (n_genes, 8)
data["gene"].num_nodes = n_genes
data["cell_group"].x = cell_group_features  # (5, 3)
data["cell_group"].num_nodes = len(CELL_GROUPS)

# --- Arêtes bipartites cell_group ↔ gene (conditionnelles V3.5+) ---
# "expresses" : cell_group → gene (le groupe cellulaire exprime le gène)
# "expressed_in" : gene → cell_group (reverse, mêmes features)
# Skip total si --no-cell-group-edges.
if edge_index_expresses.numel() > 0:
    data["cell_group", "expresses", "gene"].edge_index = edge_index_expresses
    data["cell_group", "expresses", "gene"].edge_attr = edge_attr_expresses
    data["gene", "expressed_in", "cell_group"].edge_index = torch.stack([
        edge_index_expresses[1], edge_index_expresses[0]  # Inverse src/dst
    ])
    data["gene", "expressed_in", "cell_group"].edge_attr = edge_attr_expresses

# --- Arêtes gene ↔ gene (toutes conditionnelles V3.5+) ---
if edge_index_ppi.numel() > 0:
    data["gene", "ppi", "gene"].edge_index = edge_index_ppi
    data["gene", "ppi", "gene"].edge_attr = edge_attr_ppi
if edge_index_pathway.numel() > 0:
    data["gene", "same_pathway", "gene"].edge_index = edge_index_pathway
    # same_pathway n'a pas d'edge_attr (existence binaire suffit)
if edge_index_regulates.numel() > 0:
    data["gene", "regulates", "gene"].edge_index = edge_index_regulates
    data["gene", "regulates", "gene"].edge_attr = edge_attr_regulates
    data["gene", "regulated_by", "gene"].edge_index = edge_index_regulated_by
    data["gene", "regulated_by", "gene"].edge_attr = edge_attr_regulates  # Mêmes poids
if edge_index_coexpr.numel() > 0:
    data["gene", "coexpression", "gene"].edge_index = edge_index_coexpr
    data["gene", "coexpression", "gene"].edge_attr = coexpr_w_tensor
if edge_index_cocat.numel() > 0:
    data["gene", "metabolic_cocatalysis", "gene"].edge_index = edge_index_cocat
    data["gene", "metabolic_cocatalysis", "gene"].edge_attr = edge_attr_cocat
# OmniPath V4 — signaling dirigé signé (kinase-substrat + SIGNOR causal)
if edge_index_signaling.numel() > 0:
    data["gene", "signaling", "gene"].edge_index = edge_index_signaling
    data["gene", "signaling", "gene"].edge_attr = edge_attr_signaling
# OmniPath V4 — TF→cible curé (CollecTRI), edge_type SÉPARÉ de "regulates"
# pour que le GNN apprenne distinctement les deux sources (option (c) du
# design : pySCENIC HUVEC-spécifique vs CollecTRI méta-curation).
if edge_index_tf_curated.numel() > 0:
    data["gene", "tf_curated", "gene"].edge_index = edge_index_tf_curated
    data["gene", "tf_curated", "gene"].edge_attr = edge_attr_tf_curated
    data["gene", "tf_curated_by", "gene"].edge_index = edge_index_tf_curated_by
    data["gene", "tf_curated_by", "gene"].edge_attr = edge_attr_tf_curated
# V4.2 — Reactome FI signé (edge_type séparé, ablation-able via --no-reactome-fi)
if edge_index_reactome_fi.numel() > 0:
    data["gene", "reactome_fi", "gene"].edge_index = edge_index_reactome_fi
    data["gene", "reactome_fi", "gene"].edge_attr = edge_attr_reactome_fi
if edge_index_reactome_fi_und.numel() > 0:   # V6.2 : orientation inconnue séparée
    data["gene", "reactome_fi_undirected", "gene"].edge_index = edge_index_reactome_fi_und
    data["gene", "reactome_fi_undirected", "gene"].edge_attr = edge_attr_reactome_fi_und

# V6 Module 1 (extension) — arêtes OmniPath supplémentaires projetées sur les
# nœuds gène depuis le graphe autonome (edges.tsv.gz), signées [score, sign].
# Chaque type activable/désactivable via --omnipath-edges. Offline-safe :
# graphe absent → aucune arête ajoutée (warn). `omnipath_extra_edges` (dict
# edge_type → (src, dst)) est exposé au pool de reconstruction (_train_body).
omnipath_extra_edges: dict = {}
if OMNIPATH_EXTRA_EDGES:
    print("\n  OmniPath extra edges (V6)…")
    try:
        import omnipath_graph as _opg
        _op_nodes, _op_edges_df = _opg.load_omnipath_graph(OMNIPATH_CACHE_DIR)
        _proj = _opg.project_to_gene_indices(
            _op_edges_df, gene_to_idx,
            edge_types=OMNIPATH_EXTRA_EDGES,
            alias_map=omnipath_alias_map,
        )
        for _et in OMNIPATH_EXTRA_EDGES:
            _tup = _proj.get(_et)
            if _tup is None or len(_tup[0]) == 0:
                print(f"    {_et:16s}: 0 arêtes après projection")
                continue
            _s, _d, _attr, _ = _tup
            data["gene", _et, "gene"].edge_index = torch.tensor(
                [_s.tolist(), _d.tolist()], dtype=torch.long)
            data["gene", _et, "gene"].edge_attr = torch.tensor(
                _attr, dtype=torch.float)
            omnipath_extra_edges[_et] = (_s, _d)
            _npos = int((_attr[:, 1] > 0).sum())
            _nneg = int((_attr[:, 1] < 0).sum())
            print(f"    {_et:16s}: {len(_s)} arêtes [+:{_npos} −:{_nneg}]")
    except FileNotFoundError:
        print(f"    [warn] graphe OmniPath absent sous {OMNIPATH_CACHE_DIR}/"
              f"graph/ — lance build_omnipath_graph. Aucune arête extra.")
    except ImportError as _e:
        print(f"    [warn] import omnipath_graph KO ({_e}) — aucune arête extra.")

print(f"  Noeuds gene       : {n_genes} (features={gene_features.shape[1]})")
print(f"  Noeuds cell_group : {len(CELL_GROUPS)}")
print(f"  Arêtes ppi        : {edge_index_ppi.shape[1]}")
print(f"  Arêtes pathway    : {edge_index_pathway.shape[1]}")
print(f"  Arêtes regulates  : {edge_index_regulates.shape[1]}")
print(f"  Arêtes coexpr     : {edge_index_coexpr.shape[1] if edge_index_coexpr.numel() > 0 else 0}")
print(f"  Arêtes cocat      : {edge_index_cocat.shape[1] if edge_index_cocat.numel() > 0 else 0}")
print(f"  Arêtes signaling  : {edge_index_signaling.shape[1]}"
      + ("" if MODULES["use_omnipath_signaling"] else " [SKIP --no-omnipath-signaling]"))
print(f"  Arêtes tf_curated : {edge_index_tf_curated.shape[1]}"
      + ("" if MODULES["use_omnipath_tf_curated"] else " [SKIP --no-omnipath-tf-curated]"))
print(f"  Arêtes reactome_fi: {edge_index_reactome_fi.shape[1]}"
      + ("" if MODULES["use_reactome_fi"] else " [SKIP --no-reactome-fi]"))
print(f"  Coexpr mode       : {COEXPR_MODE}"
      + (f" (edge_dim={_COEXPR_DIM})" if MODULES["use_coexpr"] else ""))
