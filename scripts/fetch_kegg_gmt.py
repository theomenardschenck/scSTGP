#!/usr/bin/env python3
"""Build a KEGG human pathway GMT (gene symbols) from the free KEGG REST API.

Why REST and not MSigDB: the MSigDB KEGG collections are licence-restricted and
version-locked; rest.kegg.jp is open, canonical and reproducible from a URL.

Outputs: data/databases/c2.cp.kegg.symbols.gmt  (name \t url \t symbols...)
"""
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/databases/c2.cp.kegg.symbols.gmt"


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


# 1. pathway id -> human-readable name
names = {}
for line in get("https://rest.kegg.jp/list/pathway/hsa").splitlines():
    pid, desc = line.split("\t", 1)
    desc = desc.split(" - Homo sapiens")[0].strip()
    slug = re.sub(r"_+", "_", re.sub(r"[^A-Z0-9]+", "_", desc.upper())).strip("_")
    names[pid] = "KEGG_" + slug

# 2. gene id -> HGNC symbol (first alias of the KEGG gene entry)
sym = {}
for line in get("https://rest.kegg.jp/list/hsa").splitlines():
    parts = line.split("\t")
    if len(parts) < 4:
        continue
    gid, aliases = parts[0], parts[3]
    sym[gid] = aliases.split(";")[0].split(",")[0].strip()

# 3. pathway <-> gene links
sets: dict[str, set[str]] = {}
for line in get("https://rest.kegg.jp/link/hsa/pathway").splitlines():
    pid, gid = line.split("\t")
    if pid.startswith("path:"):
        pid = pid[5:]
    if pid not in names or gid not in sym:
        continue
    sets.setdefault(pid, set()).add(sym[gid])

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    for pid, genes in sorted(sets.items()):
        f.write(f"{names[pid]}\thttps://www.kegg.jp/entry/{pid}\t" + "\t".join(sorted(genes)) + "\n")

print(f"{len(sets)} KEGG pathways, "
      f"{len(set().union(*sets.values()))} unique symbols -> {OUT}")
