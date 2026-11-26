#!/usr/bin/env bash
# M11.1 — porte d'authentification puis exécution de l'inspecteur CPU-only.
# Usage: run-checkpoint-audit.sh <repertoire-checkpoint>
# Attendu: 26 shards + 2 sidecars GDN de la revision 4b0402b65e523d0183a14c431955438714cb16ff.
set -euo pipefail

CKPT=${1:?usage: run-checkpoint-audit.sh <repertoire-checkpoint>}
REPO=${REPO:-$(git rev-parse --show-toplevel)}
OUT=${OUT:-$(mktemp -d)}
SRC="$REPO/crates/claymore-model-clover/src/checkpoint.rs"
mkdir -p "$OUT"
TABLE="$OUT/pinned-asset-sha256.txt"

# 1. Table epipelee extraite de la source de la PR (aucune transcription manuelle).
python3 - "$SRC" "$TABLE" <<'PY'
import re, sys
src, out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
block = text.split("pub const PINNED_ASSET_SHA256", 1)[1].split("];", 1)[0]
pairs = re.findall(r'\(\s*"([^"]+)"\s*,\s*"([0-9a-f]{64})"\s*,?\s*\)', block)
if len(pairs) != 28:
    raise SystemExit(f"attendu 28 entrees, obtenu {len(pairs)}")
with open(out, "w", encoding="utf-8") as fh:
    for name, digest in pairs:
        fh.write(f"{digest}  {name}\n")
print(f"table_epinglee={out} entrees={len(pairs)}")
PY

cd "$CKPT"

# 2. Tailles annoncees par l'index local, si present.
if [ -f model.safetensors.index.json ]; then
  cp model.safetensors.index.json "$OUT/index-from-snapshot.json"
  echo "index_present=o"
else
  echo "index_present=n — l'inspecteur echouera"
fi

# 3. Inventaire des 28 fichiers.
echo "--- inventaire ---"
while read -r _ name; do
  if [ -f "$name" ]; then printf '%s %s\n' "$(stat -c %s "$name")" "$name"; else printf 'ABSENT %s\n' "$name"; fi
done < "$TABLE" | tee "$OUT/payload-inventory.txt"

# 4. SHA-256 effectifs des 28 fichiers (streaming disque, aucune lecture de tenseur).
echo "--- sha256 ---"
find . -maxdepth 1 -name '*.safetensors' -printf '%P\n' | sort | tee "$OUT/payload-files.txt" | xargs -r sha256sum > "$OUT/payload-sha256-actual.txt"
grep -c . "$OUT/payload-sha256-actual.txt"

# 5. Comparaison contre la table epinglee.
echo "--- comparaison ---"
if sort -k2 "$TABLE" > "$OUT/.exp" && sort -k2 "$OUT/payload-sha256-actual.txt" > "$OUT/.act"; then
  join -j 2 -o 0,1.1,2.1 -e ABSENT -a1 -a2 "$OUT/.exp" "$OUT/.act" \
    | awk '{ if ($2==$3) print "EGAL      " $1; else print "DIFFERENT " $1 " attendu=" $2 " obtenu=" $3 }' \
    | tee "$OUT/payload-digest-comparison.txt"
  NEGAL=$(grep -c '^EGAL' "$OUT/payload-digest-comparison.txt" || true)
  echo "verifies=$NEGAL / 28"
  [ "$NEGAL" -eq 28 ] || { echo "ECART_DE_DIGEST — arrêt, ne pas affaiblir la table"; exit 2; }
fi

# 6. Inspecteur (en-tetes seulement), puis verification streaming complete.
BIN="$REPO/target/debug/inspect_checkpoint"
[ -x "$BIN" ] || { echo "binaire absent: relance cargo build --locked -p claymore-model-clover --bin inspect_checkpoint"; exit 3; }
echo "--- inspect_checkpoint (en-tetes) ---"
CUDA_VISIBLE_DEVICES="" "$BIN" "$CKPT" 2>&1 | tee "$OUT/inspect-headers.log"
echo "--- inspect_checkpoint --verify-sha256 ---"
CUDA_VISIBLE_DEVICES="" "$BIN" "$CKPT" --verify-sha256 2>&1 | tee "$OUT/inspect-verify-sha256.log"
echo "terminé"
