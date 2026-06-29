#!/usr/bin/env bash
# build_umap_site.sh — bâtit un mini-site statique des UMAP interactives de
# TOUTES les configs d'une version, avec un index, puis l'expose en ligne
# (temporaire + privé) via un tunnel SSH (localhost.run, rien à installer).
#
# Usage :
#   bash scripts/build_umap_site.sh [VERSION] [--serve] [--auth user:pass]
#     VERSION    : défaut V5.4.1
#     --serve    : démarre http.server + tunnel SSH après génération
#     --auth U:P : protège le site par mot de passe (Basic Auth)
#     --force    : régénère les HTML même s'ils existent
#
# Sorties : output/interpretation/<VERSION>/_umap_site/{index.html,<config>.html}
set -euo pipefail
cd "$(dirname "$0")/.."                       # racine gnn_huvec

VERSION="${1:-V5.4.1}"; [[ "${1:-}" == --* ]] && VERSION="V5.4.1"
SERVE=0; AUTH=""; FORCE=0; PORT=8765
for a in "$@"; do case "$a" in
  --serve) SERVE=1;; --force) FORCE=1;;
  --auth) AUTH="__next__";; *) [[ "$AUTH" == "__next__" ]] && AUTH="$a";;
esac; done

CROSS="output/interpretation/${VERSION}/cross_seed"
SITE="output/interpretation/${VERSION}/_umap_site"
RUNS="output/gnn_vgae/${VERSION}"
mkdir -p "$SITE"
[[ -d "$CROSS" ]] || { echo "ABSENT : $CROSS"; exit 1; }

echo "### Génération des UMAP interactives (${VERSION}) → $SITE"
ROWS=""
for d in "$CROSS"/*/; do
  cfg="$(basename "$d")"
  rank="$d/cross_seed_gene_ranking.tsv"
  [[ -f "$rank" ]] || continue
  # résout le run-dir d'embeddings (s1 puis s2)
  rundir=""
  for s in s1 s2; do
    cand="$RUNS/v5.4.${cfg}.${s}"
    [[ -f "$cand/gene_embeddings_vgae.csv" ]] && rundir="$cand" && break
  done
  [[ -z "$rundir" ]] && { echo "  [skip] $cfg : embeddings introuvables"; continue; }

  outhtml="$SITE/${cfg}.html"
  if [[ -f "$outhtml" && $FORCE -eq 0 ]]; then
    echo "  [ok]   $cfg (existant)"
  else
    echo "  [gen]  $cfg ..."
    python src/validation/viz/interpret_embedding.py \
      --run-dir "$rundir" --ranking "$rank" \
      --out-dir "$d/interpret" \
      --umap-only --plotly-cdn --reuse-umap \
      --ablation-configs no-coexpr,no-humess \
      >/dev/null 2>&1 || { echo "  [ERREUR] $cfg"; continue; }
    cp "$d/interpret/umap_interactive.html" "$outhtml"
  fi
  ROWS="${ROWS}<li><a href=\"${cfg}.html\">${cfg}</a></li>"
done

# index.html
cat > "$SITE/index.html" <<HTML
<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>UMAP interactives — ${VERSION}</title>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px}
h1{font-size:20px}li{margin:4px 0;font-size:15px}a{color:#2166ac}
.note{color:#666;font-size:13px;border-left:3px solid #ddd;padding-left:10px}</style></head><body>
<h1>UMAP interactives du latent VGAE — ${VERSION}</h1>
<p class="note">Une carte par configuration d'ablation. Chaque page : sélecteur de
coloration (communautés, pathways, cibles, driver_score, anti/pro, Δrang vs ablation),
recherche de gène, lien GeneCards, zoom. Comparer les configs = lire la circularité
(cf. target.md §9).</p>
<ul>${ROWS}</ul>
<p class="note">Généré le $(date '+%Y-%m-%d %H:%M') — site temporaire/privé.</p>
</body></html>
HTML
echo "### Index : $SITE/index.html"

if [[ $SERVE -eq 1 ]]; then
  if [[ -n "$AUTH" ]]; then
    echo "### Serveur authentifié (Basic Auth) sur :$PORT"
    AUTH="$AUTH" SITE="$SITE" PORT="$PORT" python - <<'PY' &
import base64,functools,http.server,os,socketserver
USER,_,PW=os.environ["AUTH"].partition(":")
TOK="Basic "+base64.b64encode(f"{USER}:{PW}".encode()).decode()
H=functools.partial(http.server.SimpleHTTPRequestHandler,directory=os.environ["SITE"])
class A(H):
    def do_GET(self):
        if self.headers.get("Authorization")!=TOK:
            self.send_response(401);self.send_header("WWW-Authenticate",'Basic realm="umap"')
            self.end_headers();self.wfile.write(b"auth requise");return
        super().do_GET()
socketserver.TCPServer(("127.0.0.1",int(os.environ["PORT"])),A).serve_forever()
PY
  else
    echo "### Serveur (sans auth) sur :$PORT"
    python -m http.server "$PORT" --bind 127.0.0.1 --directory "$SITE" &
  fi
  SRV=$!; sleep 1
  echo "### Tunnel SSH public (localhost.run) — URL ci-dessous, Ctrl-C pour arrêter :"
  ssh -o StrictHostKeyChecking=no -R 80:localhost:$PORT nokey@localhost.run || true
  kill $SRV 2>/dev/null || true
fi
