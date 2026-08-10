#!/usr/bin/env python3
"""gene_sets.py — registre déclaratif des ensembles de gènes de validation.

Motivation (2026-07-29)
-----------------------
L'outil devient public et sera appliqué à d'autres phénotypes que la
sénescence. Les BDD de validation ne peuvent donc plus être codées en dur
(URLs GenAge/CellAge/AgeAnno…). L'utilisateur pointe *ses* ensembles de gènes
dans un manifeste YAML (`data/gene_sets/registry.yaml`) ; ce module :

  1. **s'habitue** — sniff du format (csv/tsv/txt/gmt) et de la colonne symbole
     (`symbol_col: auto`), comme `de_schema.read_de_auto` le fait pour la DE ;
  2. **émet un signal d'alarme** — un *health-check* compare le recouvrement de
     chaque set avec l'univers de gènes du graphe et classe le set
     `OK` / `WARN` / `AUTO_OFF` (mauvais espace d'ID, mauvaise espèce, fichier
     absent) ;
  3. **tourne sans BDD** — en l'absence de manifeste ou de tout set actif, les
     colonnes `in_<name>` ne sont simplement pas produites et les
     fonctionnalités dépendantes se mettent en OFF (aucun crash, aucune
     AUROC calculée sur 3 gènes).

Aucun de ces ensembles n'entre dans le graphe, les features, l'entraînement ni
la perturbation : ils sont **post-hoc uniquement** (validation / annotation /
ancre optionnelle). Le mode « sans BDD » ne change donc pas le ranking — c'est
aussi ce qui rend l'argument anti-circularité vérifiable (`role: validation`
est mécaniquement interdit en amont du score, cf. `assert_not_upstream`).

Contrat du manifeste (`registry.yaml`, liste d'entrées)
-------------------------------------------------------
    - name: cellage              # → colonne de sortie `in_cellage`
      path: data/databases/cellage3.tsv
      symbol_col: "Gene symbol"  # ou `auto` (sniff)
      format: auto               # auto | csv | tsv | txt | gmt
      role: validation           # validation | annotation | anchor
      direction: null            # up | down | null (signe global du set)
      direction_col: "Senescence Effect"     # (option) colonne directionnelle
      direction_values: {up: [Induces], down: [Inhibits]}
      gmt_keywords: [SENESCENCE, P53]         # (option, format gmt) filtre
      enabled: true              # (option) mise en OFF manuelle

Les chemins relatifs sont résolus depuis la racine projet (dossier contenant
`data/`). Les téléchargements sortent du runtime vers `scripts/fetch_gene_sets.py`
(offline-first, comme `cache_omnipath.py`).

Cf. design_log §gene-set-registry ; technical/gene_sets.md.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml est dans environment.yml
    yaml = None


DEFAULT_REGISTRY = "data/gene_sets/registry.yaml"

# Seuils du health-check (recouvrement set ↔ univers du graphe).
HEALTH_MIN_ABS = 30      # nb minimal de gènes présents pour un verdict OK
HEALTH_MIN_FRAC = 0.05   # …ET au moins 5 % du set doit être présent

VALID_ROLES = ("validation", "annotation", "anchor")
VALID_DIRECTIONS = (None, "up", "down")


# --------------------------------------------------------------------------- #
# Modèle
# --------------------------------------------------------------------------- #
@dataclass
class GeneSet:
    """Un ensemble de gènes déclaré + son état après chargement/health-check."""
    name: str
    path: Path
    role: str = "validation"
    symbol_col: str = "auto"
    fmt: str = "auto"
    direction: str | None = None
    direction_col: str | None = None
    direction_values: dict[str, list[str]] = field(default_factory=dict)
    gmt_keywords: list[str] = field(default_factory=list)
    enabled: bool = True

    # Rempli à la lecture / health-check
    symbols: set[str] = field(default_factory=set)
    subsets: dict[str, set[str]] = field(default_factory=dict)  # ex. up/down
    status: str = "OK"           # OK | WARN | AUTO_OFF | DISABLED | MISSING
    reason: str = ""
    n_total: int = 0             # taille du set (avant intersection graphe)
    n_in_universe: int = 0       # recouvrement avec l'univers du graphe

    @property
    def column(self) -> str:
        """Nom de la colonne d'annotation binaire produite (`in_<name>`)."""
        return f"in_{self.name}"

    @property
    def active(self) -> bool:
        """Le set contribue-t-il aux annotations/validations ?"""
        return self.enabled and self.status in ("OK", "WARN")


# --------------------------------------------------------------------------- #
# Résolution racine projet (aligné sur ora_consensus._find_project_root)
# --------------------------------------------------------------------------- #
def find_project_root(start: Path | None = None) -> Path:
    """Remonte jusqu'au dossier contenant `data/databases/`.

    On teste `data/databases` (et non `data/` seul) pour ne pas confondre avec
    le dossier de code `src/data/`. Aligné sur ora_consensus._find_project_root.
    Override: GNN_PROJECT_ROOT.
    """
    import os
    env = os.environ.get("GNN_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    start = (start or Path(__file__)).resolve()
    for p in [start] + list(start.parents):
        if (p / "data" / "databases").is_dir():
            return p
    return start.parents[3]


# --------------------------------------------------------------------------- #
# Sniff format + colonne symbole
# --------------------------------------------------------------------------- #
def _sniff_delimiter(path: Path) -> str:
    """Détecte le séparateur (\\t vs ,) d'un fichier tabulaire."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        sample = fh.read(8192)
    if not sample:
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        return "\t" if sample.count("\t") >= sample.count(",") else ","


def _resolve_fmt(path: Path, fmt: str) -> str:
    if fmt and fmt != "auto":
        return fmt
    suf = path.suffix.lower()
    if suf == ".gmt":
        return "gmt"
    if suf in (".tsv",):
        return "tsv"
    if suf in (".csv",):
        return "csv"
    # .txt / inconnu : décidé au sniff (une colonne = liste nue)
    return "txt"


def _read_table(path: Path, sep: str) -> pd.DataFrame:
    """Lecture tabulaire tolérante : QUOTE_NONE (ne mange aucune ligne) puis
    dé-guillemetage des en-têtes (gère à la fois cellage — guillemets parasites
    dans "Gene name" — et AgeAnno — en-têtes eux-mêmes guillemetés)."""
    df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip",
                     dtype=str, quoting=3, comment="#")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


def _symbols_from(df: pd.DataFrame, col: str) -> set[str]:
    return set(df[col].dropna().astype(str).str.strip().str.strip('"')) - {""}


def _pick_symbol_col(df: pd.DataFrame, symbol_col: str) -> str:
    """Choisit la colonne symbole : explicite, sinon sniff heuristique."""
    if symbol_col and symbol_col != "auto":
        if symbol_col in df.columns:
            return symbol_col
        # tolérance casse/espaces
        low = {c.lower().strip(): c for c in df.columns}
        if symbol_col.lower().strip() in low:
            return low[symbol_col.lower().strip()]
        raise KeyError(
            f"colonne symbole '{symbol_col}' absente ; colonnes: {list(df.columns)}")
    for key in ("symbol", "gene symbol", "gene_symbol", "gene", "hgnc", "name"):
        for c in df.columns:
            if c.lower().strip() == key:
                return c
    for c in df.columns:  # repli : première colonne contenant "gene"/"symbol"
        cl = c.lower()
        if "symbol" in cl or "gene" in cl or "name" in cl:
            return c
    return df.columns[0]


def _parse_gmt(path: Path, keywords: list[str]) -> set[str]:
    """Union des gènes des gene-sets d'un .gmt (filtrés par mots-clés si fournis)."""
    up_kw = [k.upper() for k in keywords]
    out: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if up_kw and not any(k in parts[0].upper() for k in up_kw):
                continue
            out |= {g.strip() for g in parts[2:] if g.strip()}
    return out


def read_gene_set(gs: GeneSet) -> GeneSet:
    """Lit les symboles d'un `GeneSet` (sniff format + colonne). Ne lève jamais.

    En cas d'échec (fichier absent/illisible), positionne
    `status='MISSING'`/`reason` et laisse `symbols` vide — le health-check
    tranchera ensuite (AUTO_OFF).
    """
    if not gs.enabled:
        gs.status, gs.reason = "DISABLED", "enabled: false"
        return gs
    if not gs.path.exists():
        gs.status = "MISSING"
        gs.reason = (f"fichier absent: {gs.path} — lancer "
                     f"`python scripts/fetch_gene_sets.py --name {gs.name}`")
        return gs

    fmt = _resolve_fmt(gs.path, gs.fmt)
    try:
        if fmt == "gmt":
            gs.symbols = _parse_gmt(gs.path, gs.gmt_keywords)
        elif fmt == "txt":
            # une colonne = liste nue, sinon on retombe sur le sniff tabulaire
            df = _read_table(gs.path, _sniff_delimiter(gs.path))
            if df.shape[1] == 1 and gs.symbol_col == "auto":
                gs.symbols = _symbols_from(df, df.columns[0])
            else:
                col = _pick_symbol_col(df, gs.symbol_col)
                gs.symbols = _symbols_from(df, col)
        else:
            df = _read_table(gs.path, "\t" if fmt == "tsv" else ",")
            col = _pick_symbol_col(df, gs.symbol_col)
            gs.symbols = _symbols_from(df, col)
            _extract_directions(gs, df, col)
    except Exception as exc:  # noqa: BLE001 — jamais fatal
        gs.status = "MISSING"
        gs.reason = f"illisible ({fmt}): {type(exc).__name__}: {exc}"
        gs.symbols = set()
        return gs

    gs.symbols.discard("")
    gs.n_total = len(gs.symbols)
    if gs.n_total == 0:
        gs.status, gs.reason = "MISSING", f"0 symbole lu ({fmt})"
    return gs


def _extract_directions(gs: GeneSet, df: pd.DataFrame, symbol_col: str) -> None:
    """Remplit `gs.subsets['up'/'down']` à partir d'une colonne directionnelle."""
    if not gs.direction_col or not gs.direction_values:
        if gs.direction in ("up", "down"):
            gs.subsets[gs.direction] = set(gs.symbols)
        return
    if gs.direction_col not in df.columns:
        low = {c.lower().strip(): c for c in df.columns}
        if gs.direction_col.lower().strip() not in low:
            return
        gs.direction_col = low[gs.direction_col.lower().strip()]
    col = df[gs.direction_col].astype(str).str.lower()
    for sign, values in gs.direction_values.items():
        vals = [str(v).lower() for v in values]
        mask = col.apply(lambda x: any(v in x for v in vals))  # noqa: B023
        gs.subsets[sign] = _symbols_from(df.loc[mask], symbol_col)


# --------------------------------------------------------------------------- #
# Chargement du registre
# --------------------------------------------------------------------------- #
def load_registry(root: Path | None = None,
                  registry_path: str | Path | None = None) -> list[GeneSet]:
    """Charge et lit tous les sets déclarés. Registre absent ⇒ liste vide.

    Aucune lecture réseau. Chemins relatifs résolus depuis la racine projet.
    """
    root = Path(root) if root else find_project_root()
    reg_path = Path(registry_path) if registry_path else (root / DEFAULT_REGISTRY)
    if not reg_path.exists():
        return []
    if yaml is None:
        raise RuntimeError("pyyaml requis pour lire le registre gene-sets "
                           "(présent dans environment.yml).")
    with open(reg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or []
    entries: list[GeneSet] = []
    for item in raw:
        role = item.get("role", "validation")
        if role not in VALID_ROLES:
            raise ValueError(f"role invalide '{role}' pour {item.get('name')} ; "
                             f"attendu {VALID_ROLES}")
        direction = item.get("direction")
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"direction invalide '{direction}' pour "
                             f"{item.get('name')} ; attendu {VALID_DIRECTIONS}")
        p = Path(item["path"])
        gs = GeneSet(
            name=item["name"],
            path=p if p.is_absolute() else (root / p),
            role=role,
            symbol_col=item.get("symbol_col", "auto"),
            fmt=item.get("format", "auto"),
            direction=direction,
            direction_col=item.get("direction_col"),
            direction_values=item.get("direction_values", {}) or {},
            gmt_keywords=item.get("gmt_keywords", []) or [],
            enabled=item.get("enabled", True),
        )
        entries.append(read_gene_set(gs))
    return entries


# --------------------------------------------------------------------------- #
# Health-check
# --------------------------------------------------------------------------- #
def health_check(sets: list[GeneSet], universe: set[str],
                 alias_map: dict[str, str] | None = None) -> list[GeneSet]:
    """Classe chaque set OK / WARN / AUTO_OFF selon le recouvrement à l'univers.

    - `AUTO_OFF` : 0 gène présent (mauvaise espèce/espace d'ID, ou fichier
      absent) → désactivé, alarme bruyante.
    - `WARN`     : recouvrement > 0 mais sous seuil → gardé, drapeau propagé.
    - `OK`       : ≥ HEALTH_MIN_ABS gènes ET ≥ HEALTH_MIN_FRAC du set présents.

    `alias_map` (facultatif) : {ancien_symbole: symbole_officiel} pour rattraper
    les renommages HGNC (MARCH1→MARCHF1…) avant l'intersection.
    """
    uni = set(universe)
    for gs in sets:
        if gs.status in ("DISABLED", "MISSING"):
            if gs.status == "MISSING":
                gs.status = "AUTO_OFF"  # fichier absent/vide ⇒ OFF explicite
            continue
        syms = gs.symbols
        if alias_map:
            syms = {alias_map.get(s, s) for s in syms}
        present = syms & uni
        gs.n_in_universe = len(present)
        frac = gs.n_in_universe / max(gs.n_total, 1)
        if gs.n_in_universe == 0:
            gs.status = "AUTO_OFF"
            gs.reason = (f"0/{gs.n_total} gène dans l'univers du graphe — "
                         f"espace d'identifiants ou espèce probablement incompatible")
        elif gs.n_in_universe >= HEALTH_MIN_ABS and frac >= HEALTH_MIN_FRAC:
            gs.status, gs.reason = "OK", ""
        else:
            gs.status = "WARN"
            gs.reason = (f"recouvrement faible {gs.n_in_universe}/{gs.n_total} "
                         f"({frac:.1%}) — set conservé mais peu fiable")
    return sets


def health_table(sets: list[GeneSet]) -> pd.DataFrame:
    """Résumé tabulaire du health-check (pour `db_health.tsv` + logs)."""
    return pd.DataFrame([{
        "name": gs.name, "role": gs.role, "status": gs.status,
        "n_total": gs.n_total, "n_in_universe": gs.n_in_universe,
        "column": gs.column if gs.active else "",
        "path": str(gs.path), "reason": gs.reason,
    } for gs in sets])


def log_health(sets: list[GeneSet], print_fn=print) -> None:
    """Affiche le verdict health-check ; alarme bruyante sur AUTO_OFF/WARN."""
    if not sets:
        print_fn("[gene-sets] aucun registre — annotations/validation BDD OFF.")
        return
    print_fn(f"[gene-sets] {len(sets)} set(s) déclaré(s) :")
    for gs in sets:
        mark = {"OK": "✓", "WARN": "⚠", "AUTO_OFF": "✗",
                "DISABLED": "·", "MISSING": "✗"}.get(gs.status, "?")
        line = (f"  {mark} {gs.name:16s} [{gs.role:10s}] {gs.status:9s} "
                f"{gs.n_in_universe:5d}/{gs.n_total:<6d}")
        if gs.reason:
            line += f"  — {gs.reason}"
        print_fn(line)


# --------------------------------------------------------------------------- #
# Annotation + garde anti-circularité
# --------------------------------------------------------------------------- #
def annotate(sets: list[GeneSet], gene_symbols) -> pd.DataFrame:
    """Table `in_<name>` (0/1) pour les sets actifs + `n_gene_sets` (somme).

    Ne produit **que** les sets actifs (OK/WARN). Un set AUTO_OFF/DISABLED ne
    crée pas de colonne (dégradation DB-free propre). DataFrame indexé par gène.
    """
    genes = list(gene_symbols)
    out = pd.DataFrame(index=genes)
    active = [gs for gs in sets if gs.active]
    for gs in active:
        out[gs.column] = [1 if g in gs.symbols else 0 for g in genes]
        # Sous-ensembles directionnels : séparateur `__` (double) pour qu'un set
        # dont le NOM finit par _up/_down (ex. `endosen_up`) ne soit jamais
        # confondu avec une colonne de sous-ensemble par les filtres aval.
        for sign, sub in gs.subsets.items():
            out[f"{gs.column}__{sign}"] = [1 if g in sub else 0 for g in genes]
    base_cols = [gs.column for gs in active]
    out["n_gene_sets"] = out[base_cols].sum(axis=1) if base_cols else 0
    return out


def validation_pairs(sets: list[GeneSet]) -> list[tuple[str, set[str]]]:
    """Liste (label, gènes) des sets utilisables comme vérité de validation.

    Exclut role='anchor' (peut toucher l'axe → interdit en validation) et les
    sets inactifs. C'est l'entrée directe des tests d'enrichissement post-hoc.
    """
    return [(gs.name, gs.symbols) for gs in sets
            if gs.active and gs.role in ("validation", "annotation")]


def assert_not_upstream(sets: list[GeneSet], context: str = "") -> None:
    """Garde-fou anti-circularité : aucun set 'validation' ne doit être lu en
    amont du score. Appelée depuis le graph-build / features pour échouer bruyamment
    si un set de validation y est branché par erreur."""
    bad = [gs.name for gs in sets if gs.role == "validation"]
    if bad:
        raise RuntimeError(
            f"[anti-circularité] sets de rôle 'validation' interdits en amont "
            f"({context}): {bad}. Utilisez role='anchor' pour un usage amont explicite.")
