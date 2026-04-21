#!/usr/bin/env python3
"""
Hypergeometric test for consensus genes from GNN V3 runs against aging databases.

This script:
1. Identifies consensus genes: genes in top 95 of at least 2/3 V3 runs
2. Ranks them by average rank across runs
3. Takes top 95 consensus genes
4. Performs hypergeometric test against SenMayo, CellAge, Fridman, GenAge
5. Reports enrichment and novel residue
"""

import pandas as pd
import numpy as np
from scipy.stats import hypergeom
import os

# Paths
BASE_DIR = "/home/USER/M2/S2/Stage/petry_project"
DATA_DIR = os.path.join(BASE_DIR, "gnn_huvec", "data")
DB_DIR = os.path.join(DATA_DIR, "databases")
OUTPUT_DIR = os.path.join(BASE_DIR, "gnn_huvec", "output", "gnn_vgae")

# Runs
RUNS = ["V3_Run1", "V3_Run2", "V3_Run3"]

# Databases mapping
DATABASES = {
    "GenAge": "genage",
    "CellAge": "cellage", 
    "Fridman": "msigdb_aging",  # Assuming Fridman refers to MSigDB aging
    "SenMayo": "aging_local"   # Assuming SenMayo refers to local aging genes
}

def load_gene_sets():
    """Load gene sets for each database."""
    gene_sets = {}
    
    # GenAge
    genage_df = pd.read_csv(os.path.join(DB_DIR, "genage_human.csv"))
    gene_sets["genage"] = set(genage_df["symbol"].dropna())
    
    # CellAge
    cellage_df = pd.read_csv(os.path.join(DB_DIR, "cellage3.tsv"), sep='\t')
    gene_sets["cellage"] = set(cellage_df["Gene symbol"].dropna().str.strip())
    
    # MSigDB aging (Fridman)
    # Load MSigDB hallmarks and filter for aging-related
    msigdb_file = os.path.join(DB_DIR, "h.all.symbols.gmt")
    if os.path.exists(msigdb_file):
        msigdb_sets = {}
        with open(msigdb_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    msigdb_sets[parts[0]] = set(parts[2:])
        
        AGING_KEYWORDS = ["SENESCENCE", "P53", "APOPTOSIS", "INFLAMMATORY", "TNFA",
                          "IL6", "KRAS", "MTORC", "REACTIVE_OXYGEN", "DNA_REPAIR",
                          "OXIDATIVE", "AGING"]
        msigdb_aging = set()
        for name, genes in msigdb_sets.items():
            if any(kw in name.upper() for kw in AGING_KEYWORDS):
                msigdb_aging |= genes
        gene_sets["msigdb_aging"] = msigdb_aging
    else:
        print("Warning: MSigDB file not found, using empty set for Fridman")
        gene_sets["msigdb_aging"] = set()
    
    # AgeAnno (if needed, but not used)
    ageanno_df = pd.read_csv(os.path.join(DB_DIR, "ageanno", "aging_DEGs.txt"))
    gene_sets["ageanno"] = set(ageanno_df["gene"].dropna())
    
    # Aging local (SenMayo)
    aging_local_df = pd.read_csv(os.path.join(DATA_DIR, "human_age_related_gene.csv"))
    gene_sets["aging_local"] = set(aging_local_df["Symbol"].dropna())
    
    return gene_sets

def get_consensus_genes():
    """Get consensus genes: top 95 genes appearing in at least 2/3 runs."""
    # Load rankings
    rankings = {}
    for run in RUNS:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, run, "gene_ranking_vgae.csv"))
        # Assume sorted by vgae_importance, but check rank_vgae
        df = df.sort_values("rank_vgae")  # Lower rank is better
        top95 = df.head(95)["gene"].tolist()
        rankings[run] = set(top95)
    
    # Find genes in at least 2 runs
    all_genes = set()
    for run in RUNS:
        all_genes |= rankings[run]
    
    gene_counts = {}
    for gene in all_genes:
        count = sum(1 for run in RUNS if gene in rankings[run])
        if count >= 2:
            gene_counts[gene] = count
    
    # For each consensus gene, compute average rank
    gene_ranks = {}
    for gene in gene_counts:
        ranks = []
        for run in RUNS:
            df = pd.read_csv(os.path.join(OUTPUT_DIR, run, "gene_ranking_vgae.csv"))
            if gene in df["gene"].values:
                rank = df[df["gene"] == gene]["rank_vgae"].iloc[0]
                ranks.append(rank)
        if ranks:
            gene_ranks[gene] = np.mean(ranks)
    
    # Sort by average rank (ascending)
    sorted_genes = sorted(gene_ranks.items(), key=lambda x: x[1])
    
    # Take top 95
    consensus_genes = [gene for gene, _ in sorted_genes[:95]]
    
    return consensus_genes

def hypergeometric_test(M, n, N, k):
    """Perform hypergeometric test.
    
    M: population size (total genes)
    n: sample size (consensus genes)
    N: successes in population (genes in database)
    k: successes in sample (overlap)
    
    Returns p-value for enrichment (P(X >= k))
    """
    if k == 0:
        return 1.0
    # P(X >= k) = sum_{i=k to min(n,N)} hypergeom.pmf(i, M, N, n)
    p_val = hypergeom.sf(k-1, M, N, n)  # sf is survival function, P(X >= k)
    return p_val

def main():
    print("Loading gene sets...")
    gene_sets = load_gene_sets()
    
    print("Identifying consensus genes...")
    consensus_genes = get_consensus_genes()
    print(f"Found {len(consensus_genes)} consensus genes")
    
    # Get background genes (from one run)
    bg_df = pd.read_csv(os.path.join(OUTPUT_DIR, "V3_Run1", "gene_ranking_vgae.csv"))
    background_genes = set(bg_df["gene"])
    M = len(background_genes)  # Population size
    n = len(consensus_genes)   # Sample size
    
    print(f"Background genes: {M}")
    print(f"Consensus genes: {n}")
    
    # For each database
    results = []
    all_db_genes = set()
    for db_name, db_key in DATABASES.items():
        db_genes = gene_sets[db_key] & background_genes  # Only genes in background
        N = len(db_genes)  # Successes in population
        k = len(set(consensus_genes) & db_genes)  # Overlap
        
        p_val = hypergeometric_test(M, n, N, k)
        expected = n * N / M
        
        results.append({
            "Database": db_name,
            "Genes in DB": N,
            "Overlap": k,
            "Expected": expected,
            "Fold enrichment": k / expected if expected > 0 else float('inf'),
            "P-value": p_val,
            "Significant": p_val < 0.05
        })
        
        all_db_genes |= db_genes
    
    # Novel residue
    novel = len(set(consensus_genes) - all_db_genes)
    novel_pct = novel / n * 100
    
    print("\nHypergeometric test results:")
    print("=" * 60)
    for res in results:
        print(f"{res['Database']}: {res['Overlap']}/{n} genes (expected {res['Expected']:.1f}), "
              f"fold={res['Fold enrichment']:.2f}, p={res['P-value']:.2e} {'***' if res['Significant'] else ''}")
    
    print(f"\nNovel residue: {novel}/{n} genes ({novel_pct:.1f}%) not in any database")
    
    # Save consensus genes
    with open(os.path.join(BASE_DIR, "gnn_huvec", "data", "pathway_gene_list", "consensus_top95_2of3.txt"), "w") as f:
        for gene in consensus_genes:
            f.write(f"{gene}\n")
    
    print(f"\nConsensus genes saved to data/pathway_gene_list/consensus_top95_2of3.txt")

if __name__ == "__main__":
    main()