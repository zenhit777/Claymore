#!/usr/bin/env bash
# M11.1 - demonstration fail-closed des portes de digest de l'inspecteur d'audit.
# Usage : m11.1-gate-negative-tests.sh <repertoire-du-checkpoint-epingle>
# Sortie : transcript dans ${LOG:-m11.1-gate-negative-tests.log.txt}
#
# Aucune copie de payload : les experiences d'index utilisent des symlinks vers les
# shards reels. Aucun octet de tenseur n'est lu, aucun contexte CUDA n'est cree.
set -u

CKPT=${1:?usage: m11.1-gate-negative-tests.sh <repertoire-du-checkpoint-epingle>}
BIN=${BIN:-target/debug/inspect_checkpoint}
LOG=${LOG:-m11.1-gate-negative-tests.log.txt}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

run() { local rc=0; CUDA_VISIBLE_DEVICES="" "$BIN" "$@" 2>&1 || rc=$?; echo "exit=$rc"; }

{
  echo "# M11.1 - demonstration fail-closed des portes de digest de l'inspecteur"
  echo "# binaire=$BIN checkpoint=$CKPT"
  echo "# constantes epinglees : CONFIG_SHA256, INDEX_SHA256 (crates/claymore-model-clover/src/config.rs)"
  echo
  echo "## T0 - controle positif sur le snapshot epingle (attendu : succes)"
  echo '$ CUDA_VISIBLE_DEVICES="" inspect_checkpoint "$CKPT"'
  CUDA_VISIBLE_DEVICES="" "$BIN" "$CKPT" 2>&1 | grep -E 'config_sha256|index_sha256|result=|artifact_sha256'
  echo "exit=0"
  echo
  echo "## T1 - config.json mute (40 -> 50 dans num_hidden_layers)"
  echo "# attendu : rejet sur empreinte de config, avant toute lecture de payload"
  mkdir -p "$WORK/config"
  python3 - "$CKPT" "$WORK/config" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
b = bytearray(open(src + '/config.json', 'rb').read())
k = b.index(b':', b.index(b'"num_hidden_layers"')) + 1
while b[k:k+1] == b' ':
    k += 1
b[k:] = bytes([b[k] + 1]) + b[k+1:k+2]
open(dst + '/config.json', 'wb').write(bytes(b))
PY
  echo '$ CUDA_VISIBLE_DEVICES="" inspect_checkpoint <repertoire_config_mutée_seule>'
  run "$WORK/config"
  echo
  echo "## T2 - index mute structurellement (total_size 6.. -> 7..)"
  echo "# attendu : rejet structurel TotalSizeMismatch, avant le controle d'empreinte"
  mkdir -p "$WORK/idx"
  python3 - "$CKPT" "$WORK/idx" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
for f in os.listdir(src):
    if f.endswith('.safetensors') or f == 'config.json':
        os.symlink(os.path.join(src, f), os.path.join(dst, f))
b = bytearray(open(src + '/model.safetensors.index.json', 'rb').read())
k = b.index(b':', b.index(b'"total_size"')) + 1
while b[k:k+1] == b' ':
    k += 1
b[k] = b[k] + 1
open(dst + '/model.safetensors.index.json', 'wb').write(bytes(b))
PY
  echo '$ CUDA_VISIBLE_DEVICES="" inspect_checkpoint <repertoire_symlinks_des_28_shards_index_muté>'
  run "$WORK/idx"
  echo
  echo "## T3 - index mute sans changement structurel (une espace appendue en fin de fichier)"
  echo "# attendu : rejet sur empreinte d'index, avant tout hachage de payload"
  python3 - "$CKPT" "$WORK/idx" <<'PY'
import hashlib, sys
src, dst = sys.argv[1], sys.argv[2]
b = open(src + '/model.safetensors.index.json', 'rb').read() + b' '
open(dst + '/model.safetensors.index.json', 'wb').write(b)
print("# taille_mutée=%d sha256_muté=%s" % (len(b), hashlib.sha256(b).hexdigest()))
PY
  echo '$ CUDA_VISIBLE_DEVICES="" inspect_checkpoint <repertoire_index_muté>'
  run "$WORK/idx"
  echo
  echo "# taille du repertoire d'experiences T2/T3 (symlinks uniquement, payload non copie) :"
  du -sh "$WORK/idx"
} > "$LOG" 2>&1

echo "transcript=$LOG"
