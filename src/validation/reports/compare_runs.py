#!/usr/bin/env python3
"""
Analyse inter-runs / inter-ablations pour les rankings VGAE.

Combine la découverte de runs, l'agrégation par type d'ablation, le calcul
de stabilité inter-seed, l'impact d'ablation vs baseline, et la génération
de figures de synthèse — en un seul pipeline.

L'abstraction `RankingSource` permettra d'ajouter par la suite les rankings
produits par `perturb_report.py` (driver_score / validation_score /
discovery_score, cross_seed_gene_ranking.tsv, cross_seed_pathway_ranking.tsv).
Pour l'instant, seule la source `vgae` (gene_ranking_vgae.csv par run) est
implémentée.

Usage:
    # Pipeline complet sur la grille V3.6 (8 ablations × 3 seeds)
    python compare_runs.py --auto-discover \\
        --base-dir output/gnn_vgae/V3.6 \\
        --output-dir output/gnn_vgae/V3.6/comparison_ablation

    # Comparaison explicite de quelques runs
    python compare_runs.py --runs full.s1 full.s2 no-ppi.s1 \\
        --base-dir output/gnn_vgae/V3.6

    # Désactiver une étape
    python compare_runs.py --auto-discover --base-dir <...> --no-figures
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Ranking source abstraction
# ---------------------------------------------------------------------------

@dataclass
class LoadedRanking:
    """Ranking d'un run : index = gènes, colonnes = score + rank.

    Convention: `score` = importance/score (plus grand = plus important),
    `rank` = rang (1 = meilleur).
    """
    df: pd.DataFrame  # indexed by gene, contains 'score' and 'rank'
    score_col: str    # nom canonique du score utilisé
    grain: str        # 'per_seed' ou 'per_ablation'


class RankingSource:
    """Interface : fournit le chemin et le chargement d'un ranking par run."""

    name: str = ''
    grain: str = 'per_seed'  # ou 'per_ablation'
    score_col: str = ''      # colonne canonique du score

    def locate(self, base_dir: Path, run: str) -> Optional[Path]:
        raise NotImplementedError

    def load(self, base_dir: Path, run: str) -> Optional[LoadedRanking]:
        raise NotImplementedError


class VGAERankingSource(RankingSource):
    """gene_ranking_vgae.csv par run (per-seed).

    Supporte deux layouts :
      - plat   : <base_dir>/<run>/gene_ranking_vgae.csv
      - nested : <base_dir>/<ablation>/<run>/gene_ranking_vgae.csv
                 (utilisé par la grille V3.6 après re-perturbation)
    """

    name = 'vgae'
    grain = 'per_seed'
    score_col = 'vgae_importance'
    rank_col = 'rank_vgae'
    file_name = 'gene_ranking_vgae.csv'

    def candidate_paths(self, base_dir: Path, run: str) -> list[Path]:
        ablation, _ = parse_ablation_seed(run)
        return [
            base_dir / run / self.file_name,
            base_dir / ablation / run / self.file_name,
        ]

    def locate(self, base_dir: Path, run: str) -> Optional[Path]:
        for p in self.candidate_paths(base_dir, run):
            if p.exists():
                return p
        return None

    def load(self, base_dir: Path, run: str) -> Optional[LoadedRanking]:
        path = self.locate(base_dir, run)
        if path is None:
            return None
        raw = pd.read_csv(path, index_col='gene')
        out = pd.DataFrame({
            'score': raw[self.score_col],
            'rank': raw[self.rank_col],
        }, index=raw.index)
        return LoadedRanking(df=out, score_col=self.score_col, grain=self.grain)


class PerturbDriverRankingSource(RankingSource):
    """`cross_seed_gene_ranking.tsv` produit par `perturb_report --cross-seed`.

    Grain: per_ablation (déjà agrégé sur les seeds par perturb_report).
    Score canonique configurable parmi {driver_score, validation_score,
    discovery_score}. Le « run » manipulé par compare_runs au plus haut
    niveau correspond ici à un nom d'ablation (ex. ``full``,
    ``no-coexpr``).

    Layout attendu : ``<base_dir>/<ablation>/<report_subdir>/<file_name>``.
    """

    name = 'perturb_driver'
    grain = 'per_ablation'
    score_col = 'driver_score'
    file_name = 'cross_seed_gene_ranking.tsv'
    report_subdir = 'cross_seed_report'

    def __init__(self, score_col: str = 'driver_score',
                 report_subdir: str = 'cross_seed_report'):
        self.score_col = score_col
        self.report_subdir = report_subdir

    def locate(self, base_dir: Path, ablation: str) -> Optional[Path]:
        p = base_dir / ablation / self.report_subdir / self.file_name
        return p if p.exists() else None

    def load(self, base_dir: Path, ablation: str) -> Optional[LoadedRanking]:
        path = self.locate(base_dir, ablation)
        if path is None:
            return None
        raw = pd.read_csv(path, sep='\t', index_col='target')
        if self.score_col not in raw.columns:
            return None
        # Le rang est implicite (ordre du TSV, déjà trié par driver_score).
        out = pd.DataFrame({
            'score': raw[self.score_col],
            'rank': np.arange(1, len(raw) + 1),
        }, index=raw.index)
        # On préserve les colonnes utiles pour l'agrégation cross-ablation.
        for c in ('evidence_tier', 'is_de_significant', 'n_aging_dbs',
                  'is_hub_inflated', 'is_low_purity_signal',
                  'canon_cosine', 'target_ppi_degree',
                  'de_log2fc_p4_vs_p16', 'de_neglog10_padj',
                  'direction', 'is_tf'):
            if c in raw.columns:
                out[c] = raw[c]
        return LoadedRanking(df=out, score_col=self.score_col, grain=self.grain)


SOURCES: dict[str, Callable[..., RankingSource]] = {
    'vgae': VGAERankingSource,
    'perturb_driver': PerturbDriverRankingSource,
    'perturb_validation': lambda: PerturbDriverRankingSource('validation_score'),
    'perturb_discovery': lambda: PerturbDriverRankingSource('discovery_score'),
}


# ---------------------------------------------------------------------------
# Run discovery & ablation grouping
# ---------------------------------------------------------------------------

def parse_ablation_seed(name: str) -> tuple[str, Optional[int]]:
    """`full.s1` → ('full', 1) ; `run3` → ('run3', None)."""
    if '.s' in name:
        ablation, _, seed_str = name.rpartition('.s')
        try:
            return ablation, int(seed_str)
        except ValueError:
            pass
    return name, None


def discover_runs(base_dir: Path, source: RankingSource,
                  pattern: Optional[str] = None) -> list[str]:
    """Globe les sous-répertoires et garde ceux qui exposent un fichier de
    ranking valide pour `source`.

    - per_seed sources : pattern par défaut ``*.s*`` (cherche à la racine
      et un niveau plus bas — couvre layouts plat et nested).
    - per_ablation sources : pattern par défaut ``*`` (chaque sous-dir de
      base_dir est candidat ablation).
    """
    if pattern is None:
        pattern = '*.s*' if source.grain == 'per_seed' else '*'

    seen: dict[str, str] = {}  # nom de run → premier match (déduplique)
    globs = [pattern, f'*/{pattern}'] if source.grain == 'per_seed' else [pattern]
    for glob in globs:
        for p in sorted(base_dir.glob(glob)):
            if not p.is_dir():
                continue
            run = p.name
            if run in seen:
                continue
            if source.locate(base_dir, run):
                seen[run] = str(p)
    return sorted(seen.keys(), key=lambda n: parse_ablation_seed(n))


def group_by_ablation(runs: list[str]) -> dict[str, list[str]]:
    """Groupe les runs par type d'ablation (préserve l'ordre par seed)."""
    groups: dict[str, list[str]] = {}
    for run in runs:
        ablation, _ = parse_ablation_seed(run)
        groups.setdefault(ablation, []).append(run)
    return groups


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

class RunComparator:
    """Charge un ensemble de runs via une `RankingSource` et expose les
    analyses inter-seeds / inter-ablations."""

    def __init__(self, base_dir: Path, source: RankingSource):
        self.base_dir = Path(base_dir)
        self.source = source
        self.rankings: dict[str, pd.DataFrame] = {}  # run → DataFrame(score, rank)
        self.runs: list[str] = []

    # ---- Loading ----------------------------------------------------------

    def load_runs(self, runs: list[str]) -> None:
        for run in runs:
            loaded = self.source.load(self.base_dir, run)
            if loaded is None:
                print(f"⚠️  Ranking introuvable pour {run} (source={self.source.name})")
                continue
            self.rankings[run] = loaded.df
            print(f"📂 {run}: {len(loaded.df)} gènes")
        self.runs = list(self.rankings.keys())

    @property
    def ablation_groups(self) -> dict[str, list[str]]:
        return group_by_ablation(self.runs)

    # ---- Pairwise correlations & overlap ----------------------------------

    def pairwise_correlations(self) -> pd.DataFrame:
        """Matrice (n_runs × n_runs) de Spearman ρ sur le score."""
        n = len(self.runs)
        mat = np.full((n, n), np.nan)
        for i, r1 in enumerate(self.runs):
            mat[i, i] = 1.0
            s1 = self.rankings[r1]['score']
            for j in range(i + 1, n):
                r2 = self.runs[j]
                s2 = self.rankings[r2]['score']
                common = s1.index.intersection(s2.index)
                if len(common) < 3:
                    continue
                rho, _ = spearmanr(s1.loc[common], s2.loc[common],
                                   nan_policy='omit')
                mat[i, j] = mat[j, i] = rho
        return pd.DataFrame(mat, index=self.runs, columns=self.runs)

    def topn_overlap(self, n: int = 50) -> pd.DataFrame:
        """Matrice (n_runs × n_runs) du chevauchement |top_n ∩| / n."""
        tops = {r: set(df['score'].nlargest(n).index)
                for r, df in self.rankings.items()}
        m = np.zeros((len(self.runs), len(self.runs)))
        for i, r1 in enumerate(self.runs):
            for j, r2 in enumerate(self.runs):
                m[i, j] = len(tops[r1] & tops[r2]) / n
        return pd.DataFrame(m, index=self.runs, columns=self.runs)

    # ---- Per-ablation stability ------------------------------------------

    def stability_per_ablation(self) -> pd.DataFrame:
        """Pour chaque ablation : Spearman ρ moyen entre seeds + CV des scores.
        Ne retourne rien d'utile si grain = per_ablation (1 seed)."""
        rows = []
        for ablation, group in self.ablation_groups.items():
            if len(group) < 2:
                rows.append({'ablation': ablation, 'n_seeds': len(group),
                             'spearman_mean': np.nan, 'spearman_std': np.nan,
                             'cv_mean': np.nan, 'cv_std': np.nan})
                continue
            # Indexer sur l'intersection (rare qu'un gène soit absent au sein
            # d'une même ablation, mais sécurisé)
            common = self.rankings[group[0]].index
            for r in group[1:]:
                common = common.intersection(self.rankings[r].index)
            scores = np.column_stack(
                [self.rankings[r].loc[common, 'score'].values for r in group]
            )
            corrs = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    rho, _ = spearmanr(scores[:, i], scores[:, j],
                                       nan_policy='omit')
                    corrs.append(rho)
            means = scores.mean(axis=1)
            stds = scores.std(axis=1)
            cv = np.divide(stds, means, where=means > 0,
                           out=np.zeros_like(stds))
            rows.append({
                'ablation': ablation,
                'n_seeds': len(group),
                'spearman_mean': float(np.mean(corrs)),
                'spearman_std': float(np.std(corrs)),
                'cv_mean': float(cv.mean()),
                'cv_std': float(cv.std()),
            })
        return pd.DataFrame(rows).sort_values('spearman_mean', ascending=False)

    # ---- Ablation impact vs baseline -------------------------------------

    def ablation_impact(self, baseline: str = 'full',
                        top_changes: int = 5) -> pd.DataFrame:
        """Pour chaque ablation ≠ baseline : Spearman ρ vs baseline (sur la
        moyenne des seeds), + top gainers/losers en différence de score."""
        if baseline not in self.ablation_groups:
            print(f"⚠️  Baseline '{baseline}' absente parmi {list(self.ablation_groups)}")
            return pd.DataFrame()

        # Moyenne des seeds pour la baseline
        base_runs = self.ablation_groups[baseline]
        base_score = pd.concat(
            [self.rankings[r]['score'].rename(r) for r in base_runs], axis=1
        ).mean(axis=1)
        base_rank = pd.concat(
            [self.rankings[r]['rank'].rename(r) for r in base_runs], axis=1
        ).mean(axis=1)

        rows = []
        for ablation, group in self.ablation_groups.items():
            if ablation == baseline:
                continue
            ab_score = pd.concat(
                [self.rankings[r]['score'].rename(r) for r in group], axis=1
            ).mean(axis=1)
            ab_rank = pd.concat(
                [self.rankings[r]['rank'].rename(r) for r in group], axis=1
            ).mean(axis=1)

            # Aligner sur l'union (NaN si gène manquant côté ablation)
            all_genes = base_score.index.union(ab_score.index)
            bs = base_score.reindex(all_genes)
            ab = ab_score.reindex(all_genes)
            br = base_rank.reindex(all_genes)
            ar = ab_rank.reindex(all_genes)

            n_missing = int(ab.isna().sum())
            rho_score, _ = spearmanr(bs, ab, nan_policy='omit')
            rho_rank, _ = spearmanr(br, ar, nan_policy='omit')

            diff = ab - bs
            valid = diff.dropna().sort_values()
            losers = valid.head(top_changes).index.tolist()
            gainers = valid.tail(top_changes).index.tolist()

            rows.append({
                'ablation': ablation,
                'n_seeds': len(group),
                'n_missing_genes': n_missing,
                'spearman_score_vs_baseline': rho_score,
                'spearman_rank_vs_baseline': rho_rank,
                'top_losers': ', '.join(losers),
                'top_gainers': ', '.join(gainers),
            })
        return (pd.DataFrame(rows)
                .sort_values('spearman_score_vs_baseline', ascending=False))

    # ---- Summary table ----------------------------------------------------

    def export_summary_table(self, out_path: Path, top_n: int = 100,
                             reference_run: Optional[str] = None,
                             union_top: bool = True) -> Path:
        """Table large : 1 ligne par gène, colonnes `<run>_score` / `<run>_rank`.

        Sélection des gènes :
          - union_top=True (défaut) : union des top-N de chaque run.
          - union_top=False : top-N depuis `reference_run`.
        """
        if reference_run is None or reference_run not in self.rankings:
            reference_run = ('full.s1' if 'full.s1' in self.rankings
                             else self.runs[0])
        ref_score = self.rankings[reference_run]['score']

        if union_top:
            gene_set: set[str] = set()
            for r in self.runs:
                gene_set.update(self.rankings[r]['score'].nlargest(top_n).index)
            genes = sorted(gene_set, key=lambda g: -ref_score.get(g, 0.0))
        else:
            genes = ref_score.nlargest(top_n).index.tolist()

        rows = []
        for g in genes:
            row: dict[str, object] = {'gene': g}
            for r in self.runs:
                df = self.rankings[r]
                if g in df.index:
                    row[f'{r}_score'] = df.at[g, 'score']
                    row[f'{r}_rank'] = df.at[g, 'rank']
                else:
                    row[f'{r}_score'] = np.nan
                    row[f'{r}_rank'] = np.nan
            rows.append(row)

        out = pd.DataFrame(rows)
        out.to_csv(out_path, index=False)
        print(f"✓ Table résumé : {out_path} ({len(out)} gènes × {len(self.runs)} runs, "
              f"référence={reference_run}, union_top={union_top})")
        return out_path

    # ---- Cross-ablation robustness (per_ablation grain only) -------------

    def cross_ablation_robustness(self,
                                  baseline: str = 'full',
                                  out_path: Optional[Path] = None,
                                  coexpr_degree: Optional[dict] = None,
                                  ppi_degree: Optional[dict] = None,
                                  ) -> pd.DataFrame:
        """Synthèse par gène à travers toutes les ablations chargées.

        Disponible uniquement pour grain ``per_ablation`` (les rankings
        de ``perturb_report``). Génère un TSV avec :
          - `n_ablations_strong` : #ablations où tier ∈ {A_confirmed, B_discovery}
          - `tier_consensus` : tier majoritaire (mode) sur toutes les ablations
          - `tier_robustness` : fraction d'ablations confirmant le consensus
          - `<ablation>_tier`, `<ablation>_driver_score` pour chaque ablation
          - colonnes du baseline : evidence_tier, driver_score, canon_cosine,
            is_de_significant, n_aging_dbs, direction
          - `coexpr_degree`, `target_ppi_degree` (si fournis)

        Tri : par n_ablations_strong DESC, puis driver_score baseline DESC.
        """
        if self.source.grain != 'per_ablation':
            raise RuntimeError(
                "cross_ablation_robustness exige une source per_ablation "
                f"(reçue: {self.source.grain}).")
        if baseline not in self.rankings:
            raise ValueError(f"Baseline '{baseline}' absent parmi {self.runs}")

        # Union des gènes vus dans au moins une ablation
        all_genes: set[str] = set()
        for r in self.runs:
            all_genes.update(self.rankings[r].index)

        STRONG = {'A_confirmed', 'B_discovery'}
        rows = []
        for g in all_genes:
            row: dict[str, object] = {'gene': g}
            tiers_seen = []
            n_strong = 0
            for r in self.runs:
                df = self.rankings[r]
                if g not in df.index:
                    row[f'{r}_tier'] = '_absent'
                    row[f'{r}_driver_score'] = np.nan
                    continue
                t = str(df.at[g, 'evidence_tier']) if 'evidence_tier' in df.columns else ''
                row[f'{r}_tier'] = t
                row[f'{r}_driver_score'] = float(df.at[g, 'score'])
                tiers_seen.append(t)
                if t in STRONG:
                    n_strong += 1

            # Consensus : tier modal (en cas d'égalité, priorité D > A > B > C > E)
            tier_priority = {'D_hub': 0, 'A_confirmed': 1, 'B_discovery': 2,
                             'C_effector': 3, 'E_noise': 4, '_absent': 5}
            counts: dict[str, int] = {}
            for t in tiers_seen:
                counts[t] = counts.get(t, 0) + 1
            if counts:
                # Tri : (count desc, priority asc → tier le plus fort en premier)
                consensus = sorted(counts.items(),
                                   key=lambda kv: (-kv[1], tier_priority.get(kv[0], 99)))[0][0]
                robust = counts[consensus] / max(len(tiers_seen), 1)
            else:
                consensus = '_absent'
                robust = 0.0

            row['n_ablations_strong'] = n_strong
            row['tier_consensus'] = consensus
            row['tier_robustness'] = round(robust, 3)

            # Snapshot baseline (pour DE / aging / fonction)
            base_df = self.rankings[baseline]
            if g in base_df.index:
                for c in ('score', 'evidence_tier', 'is_de_significant',
                          'n_aging_dbs', 'canon_cosine', 'target_ppi_degree',
                          'de_log2fc_p4_vs_p16', 'de_neglog10_padj',
                          'direction', 'is_tf'):
                    if c in base_df.columns:
                        row[f'baseline_{c}'] = base_df.at[g, c]
                # Renommer baseline_score → baseline_driver_score
                if 'baseline_score' in row:
                    row['baseline_driver_score'] = row.pop('baseline_score')
            else:
                row['baseline_driver_score'] = np.nan
                row['baseline_evidence_tier'] = '_absent'

            if coexpr_degree is not None:
                row['coexpr_degree'] = int(coexpr_degree.get(g, 0))
            if ppi_degree is not None and 'baseline_target_ppi_degree' not in row:
                row['target_ppi_degree'] = int(ppi_degree.get(g, 0))

            rows.append(row)

        df = pd.DataFrame(rows)
        # Ordre de colonnes lisible
        front = ['gene', 'tier_consensus', 'tier_robustness',
                 'n_ablations_strong', 'baseline_driver_score',
                 'baseline_evidence_tier', 'baseline_canon_cosine',
                 'baseline_is_de_significant', 'baseline_n_aging_dbs',
                 'baseline_de_log2fc_p4_vs_p16', 'baseline_de_neglog10_padj',
                 'baseline_direction', 'baseline_is_tf',
                 'baseline_target_ppi_degree']
        if coexpr_degree is not None:
            front.append('coexpr_degree')
        front = [c for c in front if c in df.columns]
        per_ab = sorted([c for c in df.columns if c.endswith('_tier')
                         and not c.startswith('baseline_')
                         and not c.startswith('tier_')])
        per_ds = sorted([c for c in df.columns if c.endswith('_driver_score')
                         and c != 'baseline_driver_score'])
        rest = [c for c in df.columns if c not in front + per_ab + per_ds]
        df = df[front + per_ab + per_ds + rest]

        df = df.sort_values(
            ['n_ablations_strong', 'baseline_driver_score'],
            ascending=[False, False])

        if out_path is not None:
            df.to_csv(out_path, sep='\t', index=False)
            print(f"✓ Cross-ablation robustness : {out_path} "
                  f"({len(df)} gènes, {len(self.runs)} ablations)")
        return df


# ---------------------------------------------------------------------------
# Coexpression / PPI degree helpers (optional enrichments)
# ---------------------------------------------------------------------------

def load_coexpr_degree(adj_path: Path,
                       top_quantile: float = 0.98) -> dict[str, int]:
    """Compte le degré de chaque gène dans le réseau coexpression
    GRNBoost2 filtré au quantile top_quantile (V3.6 default = 0.98).

    Utilisé pour enrichir le ranking cross-ablation : un gène à degree
    GRN élevé risque d'être un coexpression-hub (tier A artificiellement
    porté par GRNBoost2, cf. analyse §17 V3.6 du rapport).
    """
    if not adj_path.exists():
        return {}
    adj = pd.read_csv(adj_path)
    thresh = adj['importance'].quantile(top_quantile)
    top = adj[adj['importance'] >= thresh]
    deg = pd.concat([top['TF'], top['target']]).value_counts().to_dict()
    return {str(k): int(v) for k, v in deg.items()}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_correlation_heatmap(corr: pd.DataFrame, out: Path,
                            title: str = 'Spearman ρ inter-run') -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(corr)),
                                    max(7, 0.45 * len(corr))))
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                annot=False, ax=ax, cbar_kws={'label': 'ρ'})
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_ablation_panel(comparator: RunComparator,
                       stability_df: pd.DataFrame,
                       impact_df: pd.DataFrame,
                       baseline: str,
                       out: Path) -> None:
    """Panel 2×2 résumé : stabilité, impact, scatter baseline-vs-ablations,
    chevauchement top-N."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f'Ablation analysis (source={comparator.source.name}, '
                 f'baseline={baseline})', fontsize=14, fontweight='bold')

    # 1. Stability per ablation (Spearman inter-seed)
    ax = axes[0, 0]
    if not stability_df.empty and stability_df['n_seeds'].max() > 1:
        sub = stability_df.dropna(subset=['spearman_mean']).sort_values('spearman_mean')
        ax.barh(sub['ablation'], sub['spearman_mean'],
                xerr=sub['spearman_std'], color='steelblue', alpha=0.8)
        ax.set_xlabel('Spearman ρ inter-seed')
        ax.set_xlim([0, 1.02])
        ax.set_title('Stabilité (cohérence entre seeds par ablation)')
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'N/A (1 seed par ablation)',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()

    # 2. Impact vs baseline (Spearman ρ baseline-vs-ablation)
    ax = axes[0, 1]
    if not impact_df.empty:
        sub = impact_df.sort_values('spearman_score_vs_baseline')
        colors = ['#d62728' if x < 0.3 else '#ff7f0e' if x < 0.6 else '#2ca02c'
                  for x in sub['spearman_score_vs_baseline']]
        ax.barh(sub['ablation'], sub['spearman_score_vs_baseline'], color=colors)
        ax.axvline(1.0, color='k', linestyle=':', alpha=0.4)
        ax.set_xlabel(f'Spearman ρ vs {baseline}')
        ax.set_xlim([0, 1.02])
        ax.set_title("Impact de l'ablation sur le ranking")
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.set_axis_off()

    # 3. Scatter: baseline_mean vs ablation_mean (one trace per ablation)
    ax = axes[1, 0]
    base_runs = comparator.ablation_groups.get(baseline, [])
    if base_runs:
        base_mean = pd.concat(
            [comparator.rankings[r]['score'].rename(r) for r in base_runs],
            axis=1).mean(axis=1)
        for ablation, group in comparator.ablation_groups.items():
            if ablation == baseline:
                continue
            ab_mean = pd.concat(
                [comparator.rankings[r]['score'].rename(r) for r in group],
                axis=1).mean(axis=1)
            common = base_mean.index.intersection(ab_mean.index)
            ax.scatter(base_mean.loc[common], ab_mean.loc[common],
                       s=8, alpha=0.4, label=ablation)
        lim_lo = float(base_mean.min())
        lim_hi = float(base_mean.max())
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', alpha=0.4)
        ax.set_xlabel(f'{baseline} (score moyen)')
        ax.set_ylabel('ablation (score moyen)')
        ax.set_title('Score baseline vs ablations')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()

    # 4. Top-N overlap (multiway, intersection of all runs of a given ablation
    # vs baseline)
    ax = axes[1, 1]
    top_sizes = [10, 20, 50, 100]
    if base_runs:
        base_tops = {n: set.intersection(*[
            set(comparator.rankings[r]['score'].nlargest(n).index)
            for r in base_runs]) for n in top_sizes}
        for ablation, group in comparator.ablation_groups.items():
            if ablation == baseline:
                continue
            ab_tops = {n: set.intersection(*[
                set(comparator.rankings[r]['score'].nlargest(n).index)
                for r in group]) for n in top_sizes}
            overlaps = [len(base_tops[n] & ab_tops[n]) / max(1, n)
                        for n in top_sizes]
            ax.plot(top_sizes, overlaps, marker='o', label=ablation, linewidth=1.5)
        ax.set_xlabel('Top-N')
        ax.set_ylabel(f'|baseline ∩ ablation| / N')
        ax.set_title(f'Chevauchement top-N (intersection seeds par ablation)')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_topn_overlap_heatmap(overlap: pd.DataFrame, out: Path,
                             top_n: int) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(overlap)),
                                    max(7, 0.45 * len(overlap))))
    sns.heatmap(overlap, cmap='viridis', vmin=0, vmax=1,
                annot=False, ax=ax,
                cbar_kws={'label': f'|top-{top_n} ∩| / {top_n}'})
    ax.set_title(f'Chevauchement top-{top_n} (par paire)')
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Analyse inter-runs/inter-ablations des rankings VGAE.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    p.add_argument('--base-dir', default='output/gnn_vgae/V3.6',
                   help='Répertoire contenant les runs.')
    p.add_argument('--version',
                   help='Si fourni, surcharge --base-dir → output/gnn_vgae/<version>.')

    p.add_argument('--source', default='vgae', choices=sorted(SOURCES.keys()),
                   help='Source de ranking à analyser (défaut: vgae).')

    grp = p.add_mutually_exclusive_group()
    grp.add_argument('--auto-discover', action='store_true',
                     help='Découvre tous les runs `<discover-pattern>` valides.')
    grp.add_argument('--runs', nargs='+',
                     help='Liste explicite de runs (ex: full.s1 no-ppi.s1).')

    p.add_argument('--discover-pattern', default='*.s*',
                   help='Glob utilisé par --auto-discover (défaut: *.s*).')

    p.add_argument('--baseline', default='full',
                   help="Ablation servant de baseline (défaut: 'full').")

    p.add_argument('--top-n', type=int, default=100,
                   help='Top-N pour la table résumé et les chevauchements.')
    p.add_argument('--reference-run', default=None,
                   help='Run de référence pour le tri du résumé '
                        '(défaut: full.s1 si présent).')
    p.add_argument('--no-union-top', action='store_true',
                   help='Utiliser top-N de la run de référence au lieu de l\'union.')

    p.add_argument('--output-dir', default=None,
                   help='Dossier de sortie (défaut: <base-dir>/comparison).')
    p.add_argument('--no-figures', action='store_true',
                   help='Ne pas générer les figures.')
    p.add_argument('--no-table', action='store_true',
                   help='Ne pas exporter la table résumé large.')

    # Cross-ablation robustness (per_ablation grain only)
    p.add_argument('--cross-ablation', action='store_true',
                   help='Génère cross_ablation_robustness.tsv (uniquement '
                        'pour les sources per_ablation : perturb_driver, '
                        'perturb_validation, perturb_discovery).')
    p.add_argument('--coexpr-adjacencies',
                   default='output/pyscenic/adjacencies.csv',
                   help='CSV GRNBoost2 (pour la colonne coexpr_degree). '
                        'Mettre une chaîne vide pour désactiver.')
    p.add_argument('--coexpr-top-quantile', type=float, default=0.98,
                   help='Quantile de filtrage GRNBoost2 (cohérent avec '
                        'gnn_vgae.py). Défaut 0.98.')

    return p


def resolve_runs(args, base_dir: Path, source: RankingSource) -> list[str]:
    if args.auto_discover:
        # Si l'utilisateur n'a pas surchargé le pattern et la source est
        # per_ablation, basculer sur le pattern par défaut de discover_runs
        # (sinon `*.s*` rate les dossiers d'ablation).
        pattern = args.discover_pattern
        if (pattern == '*.s*' and source.grain == 'per_ablation'):
            pattern = None  # discover_runs choisira '*'
        runs = discover_runs(base_dir, source, pattern)
        if not runs:
            raise SystemExit(f"❌ Aucun run découvert dans {base_dir} "
                             f"(pattern={pattern!r}, source={source.name})")
        return runs
    if args.runs:
        return list(args.runs)
    raise SystemExit("❌ Spécifier --auto-discover OU --runs")


def print_section(title: str) -> None:
    bar = '=' * 78
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> None:
    args = build_parser().parse_args()

    base_dir = Path(f'output/gnn_vgae/{args.version}'
                    if args.version else args.base_dir)
    if not base_dir.exists():
        raise SystemExit(f"❌ Base dir inexistant : {base_dir}")

    source_cls = SOURCES[args.source]
    source = source_cls()  # type: ignore[call-arg]
    runs = resolve_runs(args, base_dir, source)

    out_dir = Path(args.output_dir) if args.output_dir else base_dir / 'comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # ----- Header
    print_section(f'📂 SOURCE = {source.name} (grain={source.grain})')
    print(f"   base_dir = {base_dir}")
    print(f"   out_dir  = {out_dir}")
    print(f"   {len(runs)} runs:")
    groups = group_by_ablation(runs)
    for ab in sorted(groups):
        print(f"     {ab:24s} {len(groups[ab])} seed(s) "
              f"({', '.join(groups[ab])})")

    # ----- Load
    cmp = RunComparator(base_dir, source)
    cmp.load_runs(runs)

    if not cmp.runs:
        raise SystemExit("❌ Aucun ranking chargé.")

    # ----- Pairwise correlation
    print_section('📊 CORRÉLATIONS INTER-RUNS')
    corr = cmp.pairwise_correlations()
    corr_path = out_dir / 'pairwise_spearman.csv'
    corr.to_csv(corr_path)
    print(f"✓ {corr_path}")

    # ----- Top-N overlap
    overlap = cmp.topn_overlap(n=args.top_n)
    overlap_path = out_dir / f'pairwise_top{args.top_n}_overlap.csv'
    overlap.to_csv(overlap_path)
    print(f"✓ {overlap_path}")

    # ----- Stability per ablation (per_seed seulement)
    if source.grain == 'per_seed':
        print_section('🎯 STABILITÉ PAR ABLATION (Spearman ρ inter-seed)')
        stability = cmp.stability_per_ablation()
        stab_path = out_dir / 'stability_per_ablation.csv'
        stability.to_csv(stab_path, index=False)
        if not stability.empty:
            print(stability.to_string(index=False, float_format='%.4f'))
        print(f"\n✓ {stab_path}")
    else:
        stability = pd.DataFrame()

    # ----- Ablation impact vs baseline
    print_section(f'🧪 IMPACT D\'ABLATION vs baseline = {args.baseline}')
    impact = cmp.ablation_impact(baseline=args.baseline)
    impact_path = out_dir / f'ablation_impact_vs_{args.baseline}.csv'
    impact.to_csv(impact_path, index=False)
    if not impact.empty:
        cols = ['ablation', 'n_seeds', 'n_missing_genes',
                'spearman_score_vs_baseline', 'spearman_rank_vs_baseline']
        print(impact[cols].to_string(index=False, float_format='%.4f'))
        print()
        for _, r in impact.iterrows():
            print(f"  {r['ablation']:24s}  losers: {r['top_losers']}")
            print(f"  {' ':24s}  gainers: {r['top_gainers']}")
    print(f"\n✓ {impact_path}")

    # ----- Summary table (gene-level wide)
    if not args.no_table:
        print_section('🗒️  TABLE RÉSUMÉ (large format)')
        cmp.export_summary_table(
            out_dir / 'summary_table_genes.csv',
            top_n=args.top_n,
            reference_run=args.reference_run,
            union_top=not args.no_union_top,
        )

    # ----- Cross-ablation robustness (per_ablation only)
    if args.cross_ablation:
        if source.grain != 'per_ablation':
            print(f"⚠️  --cross-ablation ignoré (source {source.name} a grain "
                  f"{source.grain}, requis: per_ablation)")
        else:
            print_section('🧬 ROBUSTESSE CROSS-ABLATION')
            coexpr_lookup: Optional[dict] = None
            if args.coexpr_adjacencies:
                adj_p = Path(args.coexpr_adjacencies)
                coexpr_lookup = load_coexpr_degree(adj_p, args.coexpr_top_quantile)
                if coexpr_lookup:
                    print(f"   coexpr_degree chargé : {len(coexpr_lookup)} gènes "
                          f"(top {(1 - args.coexpr_top_quantile)*100:.0f}% GRN)")
                else:
                    print(f"   ⚠️  Pas de coexpr_degree (fichier: {adj_p})")
            cab = cmp.cross_ablation_robustness(
                baseline=args.baseline,
                out_path=out_dir / 'cross_ablation_robustness.tsv',
                coexpr_degree=coexpr_lookup,
            )
            # Aperçu
            n_max = len(cmp.runs)
            dist = cab['n_ablations_strong'].value_counts().sort_index(ascending=False)
            print(f"   Distribution n_ablations_strong (sur {n_max}) :")
            for n_strong, n_genes in dist.items():
                print(f"     {n_strong}/{n_max}: {n_genes} gènes")
            top_robust = cab[cab['n_ablations_strong'] == n_max]
            print(f"\n   Drivers A/B dans {n_max}/{n_max} ablations "
                  f"({len(top_robust)}) :")
            cols_show = ['gene', 'tier_consensus', 'baseline_driver_score',
                         'baseline_is_de_significant', 'baseline_n_aging_dbs']
            if 'coexpr_degree' in cab.columns:
                cols_show.append('coexpr_degree')
            if not top_robust.empty:
                print(top_robust[cols_show].head(15).to_string(index=False))

    # ----- Figures
    if not args.no_figures:
        print_section('📈 FIGURES')
        fig_correlation_heatmap(
            corr, fig_dir / 'pairwise_spearman_heatmap.png',
            title=f'Spearman ρ inter-run ({source.name})')
        fig_topn_overlap_heatmap(
            overlap, fig_dir / f'pairwise_top{args.top_n}_overlap.png',
            top_n=args.top_n)
        fig_ablation_panel(cmp, stability, impact,
                           baseline=args.baseline,
                           out=fig_dir / 'ablation_panel.png')
        print(f"✓ Figures dans {fig_dir}")

    print_section('✅ Analyse terminée')


if __name__ == '__main__':
    main()
