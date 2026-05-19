#!/usr/bin/env bash
# =============================================================================
# cache_reactome_fi.sh — Pré-télécharge Reactome FI (V4.2)
# =============================================================================
# Reactome Functional Interactions, source signée additionnelle (~45k
# arêtes signées NOUVELLES vs PPI/SIGNOR/CollecTRI — cf. §14bis.6quater).
#
# À lancer sur une machine avec Internet (frontal Nautilus ou local).
# Le fichier décompressé est ensuite lu offline par gnn_vgae.py
# (--use-reactome-fi --reactome-fi-file data/reactome_fi/...).
#
# Source : https://reactome.org/download-data (section "Reactome FI").
# Version au 2026-05 : FIsInGene_04142025_with_annotations.txt
# (mettre à jour l'URL si Reactome publie une version plus récente).
# =============================================================================
set -e

DEST_DIR="${1:-data/reactome_fi}"
FI_VERSION="${FI_VERSION:-04142025}"
URL="https://reactome.org/download/tools/ReactomeFIs/FIsInGene_${FI_VERSION}_with_annotations.txt.zip"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "[cache_reactome_fi] download $URL"
curl -fsSL -o reactome_fi.zip "$URL"

echo "[cache_reactome_fi] unzip"
unzip -o reactome_fi.zip
# Nom canonique attendu par gnn_vgae.py (--reactome-fi-file défaut)
SRC=$(ls FIsInGene_*_with_annotations.txt | head -1)
cp "$SRC" FIsInGene_with_annotations.txt
rm -f reactome_fi.zip

echo "[cache_reactome_fi] OK :"
ls -la "$PWD"/FIsInGene_with_annotations.txt
echo "[cache_reactome_fi] $(wc -l < FIsInGene_with_annotations.txt) lignes"
echo "[cache_reactome_fi] head :"
head -3 FIsInGene_with_annotations.txt
echo ""
echo "Utilisation :"
echo "  python src/gnn/gnn_vgae.py --use-reactome-fi \\"
echo "      --reactome-fi-file ${DEST_DIR}/FIsInGene_with_annotations.txt ..."
