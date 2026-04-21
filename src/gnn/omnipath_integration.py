"""
OmniPath Integration Module for GNN VGAE
=========================================
Charge et intègre OmniPath pour 3 types de données :
1. Signaling dirigé (kinase-substrat) avec activation/inhibition
2. Augmentation PPI (complément de STRING)
3. TF-target DoRothEA (complément pySCENIC)
4. PTM et features enrichies
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, Optional, Set

try:
    import omnipath as op
    OMNIPATH_AVAILABLE = True
except ImportError:
    OMNIPATH_AVAILABLE = False


def load_omnipath_signaling(available_genes, gene_to_idx, organism=9606):
    """
    Charge interactions kinase-substrat (signaling dirigé).
    
    Feature d'arête : [confiance, is_activation, is_inhibition]
    
    Biologie :
    - Interaction dirigée : kinase A → substrat B (phosphorylation)
    - is_activation : 1 si activation, 0 sinon
    - is_inhibition : 1 si inhibition, 0 sinon
    - Permet au GNN de distinguer activation vs inhibition
    
    Exemple : MAPK1 → MBP (phosphorylation, activation probable)
    """
    if not OMNIPATH_AVAILABLE:
        print("⚠️ OmniPath non installé")
        return np.array([]), np.array([]), np.array([]), pd.DataFrame()
    
    print("\n  Chargement OmniPath : Signaling (kinase-substrat)...")
    
    try:
        signaling = op.interactions.OmnipathInteractions.get(
            organism=organism,
            sources='omnipath',
            interaction_types=['kinase-substrate'],
            confidence_level='high',
        )
        
        if signaling is None or len(signaling) == 0:
            print("    → Aucune donnée kinase-substrat")
            return np.array([]), np.array([]), np.array([]), pd.DataFrame()
        
        # Filtrer sur gènes disponibles
        signaling_filt = signaling[
            signaling['genesymbol_a'].isin(available_genes) &
            signaling['genesymbol_b'].isin(available_genes)
        ].copy()
        
        sig_src, sig_dst, sig_attrs = [], [], []
        metadata = []
        
        for _, row in signaling_filt.iterrows():
            kinase = row['genesymbol_a']
            substrate = row['genesymbol_b']
            
            if kinase not in gene_to_idx or substrate not in gene_to_idx:
                continue
            
            src_idx = gene_to_idx[kinase]
            dst_idx = gene_to_idx[substrate]
            confidence = float(row.get('score', 0.5))
            
            # is_stimulation : 1=activation, 0=inhibition, NaN=unknown
            is_stim = row.get('is_stimulation', np.nan)
            is_activation = 1.0 if is_stim == 1 else 0.0
            is_inhibition = 1.0 if is_stim == 0 else 0.0
            
            sig_src.append(src_idx)
            sig_dst.append(dst_idx)
            sig_attrs.append([confidence, is_activation, is_inhibition])
            
            effect = 'activation' if is_stim == 1 else ('inhibition' if is_stim == 0 else 'unknown')
            metadata.append({
                'kinase': kinase,
                'substrate': substrate,
                'confidence': confidence,
                'effect': effect,
            })
        
        if sig_src:
            edge_src = np.array(sig_src, dtype=np.int64)
            edge_dst = np.array(sig_dst, dtype=np.int64)
            edge_attr = np.array(sig_attrs, dtype=np.float32)
            
            # Normaliser chaque colonne
            for col in range(edge_attr.shape[1]):
                col_max = edge_attr[:, col].max()
                if col_max > 0:
                    edge_attr[:, col] /= col_max
            
            print(f"    ✓ {len(sig_src)} arêtes signaling")
            print(f"      - activation: {int(edge_attr[:, 1].sum())}")
            print(f"      - inhibition: {int(edge_attr[:, 2].sum())}")
            
            return edge_src, edge_dst, edge_attr, pd.DataFrame(metadata)
        else:
            return np.array([]), np.array([]), np.array([]), pd.DataFrame()
    
    except Exception as e:
        print(f"    ⚠️ Erreur: {e}")
        return np.array([]), np.array([]), np.array([]), pd.DataFrame()


def load_omnipath_ppi(available_genes, gene_to_idx, organism=9606, min_confidence=0.7):
    """
    Charge PPI d'OmniPath (complément ou remplacement de STRING).
    
    Non dirigé (A ↔ B).
    Feature : score de confiance.
    """
    if not OMNIPATH_AVAILABLE:
        return np.array([]), np.array([]), np.array([])
    
    print("\n  Chargement OmniPath : PPI...")
    
    try:
        ppi = op.interactions.OmnipathInteractions.get(
            organism=organism,
            sources='omnipath',
            interaction_types='ppi',
        )
        
        if ppi is None or len(ppi) == 0:
            print("    → Aucune PPI trouvée")
            return np.array([]), np.array([]), np.array([])
        
        ppi_filt = ppi[
            (ppi['score'] >= min_confidence) &
            ppi['genesymbol_a'].isin(available_genes) &
            ppi['genesymbol_b'].isin(available_genes)
        ].copy()
        
        ppi_src, ppi_dst, ppi_w = [], [], []
        
        for _, row in ppi_filt.iterrows():
            g1, g2 = row['genesymbol_a'], row['genesymbol_b']
            if g1 in gene_to_idx and g2 in gene_to_idx:
                i, j = gene_to_idx[g1], gene_to_idx[g2]
                score = float(row.get('score', 0.5))
                
                ppi_src.extend([i, j])
                ppi_dst.extend([j, i])
                ppi_w.extend([score, score])
        
        if ppi_src:
            edge_src = np.array(ppi_src, dtype=np.int64)
            edge_dst = np.array(ppi_dst, dtype=np.int64)
            edge_attr = np.array(ppi_w, dtype=np.float32)
            edge_attr /= (edge_attr.max() + 1e-8)
            
            print(f"    ✓ {len(ppi_filt)} interactions ({len(ppi_src)} arêtes bidirectionnelles)")
            
            return edge_src, edge_dst, edge_attr
        else:
            return np.array([]), np.array([]), np.array([])
    
    except Exception as e:
        print(f"    ⚠️ Erreur: {e}")
        return np.array([]), np.array([]), np.array([])


def load_omnipath_tf_target(available_genes, gene_to_idx, organism=9606):
    """
    Charge interactions TF-cible d'OmniPath (DoRothEA).
    
    Dirigé : TF → cible.
    Complément pySCENIC (basé sur données scRNA-seq).
    DoRothEA = annotations curatées multi-source.
    """
    if not OMNIPATH_AVAILABLE:
        return np.array([]), np.array([]), np.array([])
    
    print("\n  Chargement OmniPath : TF-target (DoRothEA)...")
    
    try:
        tf_target = op.interactions.OmnipathInteractions.get(
            organism=organism,
            sources='dorothea',
            confidence_level='high',
        )
        
        if tf_target is None or len(tf_target) == 0:
            print("    → Aucune relation TF-target trouvée")
            return np.array([]), np.array([]), np.array([])
        
        tf_target_filt = tf_target[
            tf_target['genesymbol_a'].isin(available_genes) &
            tf_target['genesymbol_b'].isin(available_genes)
        ].copy()
        
        tf_src, tf_dst, tf_w = [], [], []
        
        for _, row in tf_target_filt.iterrows():
            tf, cible = row['genesymbol_a'], row['genesymbol_b']
            if tf in gene_to_idx and cible in gene_to_idx:
                i, j = gene_to_idx[tf], gene_to_idx[cible]
                score = float(row.get('score', 0.5))
                
                tf_src.append(i)
                tf_dst.append(j)
                tf_w.append(score)
        
        if tf_src:
            edge_src = np.array(tf_src, dtype=np.int64)
            edge_dst = np.array(tf_dst, dtype=np.int64)
            edge_attr = np.array(tf_w, dtype=np.float32)
            edge_attr /= (edge_attr.max() + 1e-8)
            
            print(f"    ✓ {len(tf_target_filt)} relations TF→cible (DoRothEA)")
            
            return edge_src, edge_dst, edge_attr
        else:
            return np.array([]), np.array([]), np.array([])
    
    except Exception as e:
        print(f"    ⚠️ Erreur: {e}")
        return np.array([]), np.array([]), np.array([])


def enrich_gene_features(available_genes, gene_symbols, gene_to_idx, organism=9606):
    """
    Enrichit les features de gènes avec données OmniPath.
    
    Features ajoutables :
    - is_kinase : gène code une kinase
    - is_tf_omnipath : TF selon OmniPath (plus large que pySCENIC)
    """
    if not OMNIPATH_AVAILABLE:
        return {}
    
    print("\n  Enrichissement features OmniPath...")
    
    features = {}
    
    # is_kinase
    try:
        kinases_op = op.interactions.OmnipathInteractions.get(
            organism=organism,
            interaction_types='kinase-substrate',
        )
        if kinases_op is not None and len(kinases_op) > 0:
            kinase_set = set(kinases_op['genesymbol_a'].unique()) & set(gene_symbols)
            features['is_kinase'] = np.array(
                [1.0 if g in kinase_set else 0.0 for g in gene_symbols],
                dtype=np.float32
            )
            print(f"    ✓ is_kinase: {features['is_kinase'].sum():.0f}")
    except Exception as e:
        print(f"    ⚠️ is_kinase: {e}")
    
    # is_tf_omnipath
    try:
        tf_op = op.interactions.OmnipathInteractions.get(
            organism=organism,
            sources='dorothea',
        )
        if tf_op is not None and len(tf_op) > 0:
            tf_set = set(tf_op['genesymbol_a'].unique()) & set(gene_symbols)
            features['is_tf_omnipath'] = np.array(
                [1.0 if g in tf_set else 0.0 for g in gene_symbols],
                dtype=np.float32
            )
            print(f"    ✓ is_tf_omnipath: {features['is_tf_omnipath'].sum():.0f}")
    except Exception as e:
        print(f"    ⚠️ is_tf_omnipath: {e}")
    
    return features

