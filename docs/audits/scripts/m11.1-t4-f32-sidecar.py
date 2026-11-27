"""T4 — porte dtype sur le vrai checkpoint : sidecar GDN converti en F32.

Construit un repertoire d'experience sans copie de payload :
  - config.json, model.safetensors.index.json et les 26 shards du modele en symlinks ;
  - bd_gdn_softmax_scalars.safetensors en symlink (BF16 reel) ;
  - bd_gdn_conv_right_weight.safetensors reconstruit avec les memes noms et formes de
    tenseurs mais le dtype F32 (donnees zeros) ; seule la porte dtype est exercee.

Aucune VRAM, aucun contexte CUDA, aucun octet de payload du modele lu.
Usage : python3 t4-f32-sidecar.py <snapshot> <repertoire_de_travail>
"""
import json
import os
import struct
import subprocess
import sys

SNAPSHOT = sys.argv[1]
WORK = sys.argv[2]
SIDE = "bd_gdn_conv_right_weight.safetensors"
BIN = os.environ.get("BIN", "target/debug/inspect_checkpoint")

os.makedirs(WORK, exist_ok=True)

for name in sorted(os.listdir(SNAPSHOT)):
    if name.endswith(".safetensors") and not name.endswith(SIDE):
        os.symlink(os.path.join(SNAPSHOT, name), os.path.join(WORK, name))
for name in ("config.json", "model.safetensors.index.json"):
    os.symlink(os.path.join(SNAPSHOT, name), os.path.join(WORK, name))

# Lecture de l'en-tete Safetensors reelle du sidecar (8 octets de longueur + JSON).
with open(os.path.join(SNAPSHOT, SIDE), "rb") as fh:
    head_len = struct.unpack("<Q", fh.read(8))[0]
    header = json.loads(fh.read(head_len))

meta = header.pop("__metadata__", {})
new_header = {}
if meta:
    new_header["__metadata__"] = meta
max_end = 0
for name, entry in header.items():
    if name == "__metadata__":
        continue
    numel = 1
    for dim in entry["shape"]:
        numel *= dim
    start = max_end
    end = start + numel * 4          # F32 : 4 octets par element
    new_header[name] = {"dtype": "F32", "shape": entry["shape"],
                        "data_offsets": [start, end]}
    max_end = end

payload = bytearray(max_end)          # zeros : contenu sans effet, la porte dtype est en amont
blob = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
pad = (-len(blob)) % 8
blob += b" " * pad
with open(os.path.join(WORK, SIDE), "wb") as fh:
    fh.write(struct.pack("<Q", len(blob)))
    fh.write(blob)
    fh.write(payload)

tensors = len(new_header) - (1 if "__metadata__" in new_header else 0)
print(f"# T4 — repertoire={WORK}")
print(f"# sidecar_muté={SIDE} dtype=F32 tensors={tensors} taille_octets={os.path.getsize(os.path.join(WORK, SIDE))}")
print(f"# shards_et_autres_fichiers=en symlinks ; copie de payload = aucune")
print("$ CUDA_VISIBLE_DEVICES=\"\" inspect_checkpoint <repertoire_sidecar_F32>")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
proc = subprocess.run([BIN, WORK], env=env, capture_output=True, text=True)
sys.stdout.write(proc.stdout)
sys.stderr.write(proc.stderr)
print(f"exit={proc.returncode}")
