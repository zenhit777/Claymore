"""Controles lateraux publies pour le rafraichissement M11.1 du 2026-11-26.

1. Empreintes non-Xet locales contre le tableau du manifeste.
2. Controle arithmetique des 28 artefacts d'apres les en-tetes reels :
   fichier = prefixe u64 (8 o) + en-tete JSON + payload.
3. (optionnel) Table epinglee contre empreintes sha256sum recalculees.

Aucun poids n'est charge. Usage : python3 m11.1-lateral-controls.py <repertoire_preuves> [sha256sum.txt] [snapshot]
"""
import json
import re
import sys

import os

O = sys.argv[1]
MANIFEST = "docs/audits/clover-upstream-manifest.md"
SNAPSHOT = sys.argv[3] if len(sys.argv) > 3 else (
    "/mnt/c/Users/sjteb/.cache/huggingface/hub/hub/models--Diffuspace--Clover-35B-A3B-Preview/"
    "snapshots/4b0402b65e523d0183a14c431955438714cb16ff"
)
HEX64 = re.compile(r"[0-9a-f]{64}")


def read_pairs(path):
    pairs = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2 and HEX64.fullmatch(parts[0]):
                pairs[parts[1]] = parts[0]
    return pairs


def pinned_manifest_table():
    pinned = {}
    with open(MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            name = cells[0].strip("`")
            rev = cells[1].strip("`")
            found = HEX64.search(cells[2])
            if re.fullmatch(r"[0-9a-f]{40}", rev) and found:
                pinned[name] = found.group(0)
    return pinned


def main():
    pinned = pinned_manifest_table()
    local = read_pairs(O + "/nonxet-sha256.txt")
    nonxet = sorted(n for n in local if not n.endswith(".safetensors"))
    lines = []
    bad = 0
    for name in nonxet:
        expected = pinned.get(name)
        ok = expected == local[name]
        bad += 0 if ok else 1
        mark = "EGAL      " if ok else "DIFFERENT "
        lines.append(f"{mark}{name} manifeste={expected} local={local[name]}")
    with open(O + "/nonxet-vs-manifest-comparison.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"tableau_manifeste={len(pinned)} non_xet={len(nonxet)} "
          f"egaux={len(nonxet) - bad} differents={bad}")

    rows = []
    tensors = 0
    with open(O + "/inspect-headers.log", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("file="):
                fields = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)}
                rows.append(fields)
                tensors += fields["tensors"]
    total = lambda key: sum(r[key] for r in rows)  # noqa: E731
    with open(SNAPSHOT + "/model.safetensors.index.json", encoding="utf-8") as fh:
        index_total = json.load(fh)["metadata"]["total_size"]
    off = [r for r in rows if r["bytes"] != 8 + r["header_bytes"] + r["payload_bytes"]]
    control = [
        f"artefacts={len(rows)}",
        f"octets_fichiers={total('bytes')}",
        f"prefices_longueur_u64={8 * len(rows)} (8 o x {len(rows)} artefacts)",
        f"octets_en_tete_json={total('header_bytes')}",
        f"octets_payload={total('payload_bytes')}",
        f"index.metadata.total_size={index_total}",
        f"payload==index_total={'OUI' if total('payload_bytes') == index_total else 'NON'}",
        f"fichier==8+en_tete+payload={'OUI' if not off else 'NON (' + str(len(off)) + ' fichiers)'}",
        f"en_tetes_plus_prefices={total('header_bytes') + 8 * len(rows)}",
        f"tenseurs_total_inventaire={tensors}",
    ]
    with open(O + "/arithmetic-control.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(control) + "\n")
    print("\n".join(control))

    if len(sys.argv) > 2:
        actual = read_pairs(sys.argv[2])
        # Source d'autorite : la table epinglee extraite du code (PINNED_ASSET_SHA256),
        # produite par docs/audits/scripts/m11.1-run-checkpoint-audit.sh. La table du
        # manifeste ne sert que de repli. La passe sha256sum ne porte que sur les 28
        # Safetensors : seuls ces 28-la sont compares.
        code_table = O + "/pinned-asset-sha256.txt"
        if os.path.exists(code_table):
            pinned256 = read_pairs(code_table)
            source = "PINNED_ASSET_SHA256 (checkpoint.rs, via pinned-asset-sha256.txt)"
        else:
            pinned256 = {n: d for n, d in pinned.items() if n.endswith(".safetensors")}
            source = "clover-upstream-manifest.md"
        expected_names = sorted(pinned256)
        cmp_lines = [f"# table epinglee = {source} ({len(expected_names)} entrées)",
                     "# sha256sum = empreintes recalculees sur les octets bruts du snapshot"]
        ecart = 0
        for name in sorted(set(expected_names) | set(actual)):
            p = pinned256.get(name)
            a = actual.get(name)
            ok = p is not None and p == a
            ecart += 0 if ok else 1
            mark = "EGAL      " if ok else "DIFFERENT "
            cmp_lines.append(f"{mark}{name} epingle={p} sha256sum={a}")
        with open(O + "/digest-comparison.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(cmp_lines) + "\n")
        print(f"sha256sum_vs_epingles: source={source} safetensors={len(expected_names)} "
              f"egaux={len(expected_names) - ecart} ecart={ecart}")


if __name__ == "__main__":
    main()
