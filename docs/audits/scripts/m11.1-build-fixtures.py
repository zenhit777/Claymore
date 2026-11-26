"""M11.1 - assemble le fichier de fixtures tokenizer/gabarit publie.

Entrees :
  argv[1] : repertoire du checkpoint epingle (4b0402b65e523d0183a14c431955438714cb16ff)
  argv[2] : sortie brute de m11.1-tokenizer-audit.py
  argv[3] : sortie brute de m11.1-chat-template-audit.py
  argv[4] : fichier de fixtures a ecrire

Aucun poids n'est charge, aucun contexte CUDA n'est cree : ce script ne lit que des
fichiers de configuration/texte et les deux sorties JSON precedentes.
"""
import hashlib
import json
import sys

REVISION = "4b0402b65e523d0183a14c431955438714cb16ff"
MODEL_ID = "Diffuspace/Clover-35B-A3B-Preview"
AUDIT_DATE = "2026-11-26"

ckpt, raw_path, tpl_path, out_path = sys.argv[1:5]
raw = json.load(open(raw_path, encoding="utf-8"))
tpl_fixtures = json.load(open(tpl_path, encoding="utf-8"))


def sha(name):
    with open(f"{ckpt}/{name}", "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def nbytes(name):
    with open(f"{ckpt}/{name}", "rb") as fh:
        return len(fh.read())


tok = raw["tokenizer"]
cfg = raw["config_eos"]
mask = raw["mask_token_registration"]
remap = raw["remapping_151669"]
template = raw["template"]

out = {
    "source": {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "tokenizer_json_sha256": sha("tokenizer.json"),
        "tokenizer_config_json_sha256": sha("tokenizer_config.json"),
        "chat_template_jinja_sha256": sha("chat_template.jinja"),
        "chat_template_jinja_bytes": nbytes("chat_template.jinja"),
        "audit_date": AUDIT_DATE,
        "environment": raw["env"],
    },
    "vocabulary": {
        "tokenizer_class": cfg["tokenizer_config.tokenizer_class"],
        "bpe_vocab_entries": len(json.load(open(f"{ckpt}/tokenizer.json", encoding="utf-8"))["model"]["vocab"]),
        "bpe_merges": remap["merges_total"],
        "added_tokens": len(remap["special_tokens_defined"]),
        "tokenizer_len": tok["vocab_size_len"],
        "bpe_vocab_size": tok["vocab_size_size"],
        "model_config_vocab_size": cfg["config.vocab_size"],
        "unreachable_weight_rows": cfg["config.vocab_size"] - tok["vocab_size_len"],
        "merges_referencing_mask_or_151669": remap["merges_referencing_count"],
    },
    "eos_pad_layers": {
        "tokenizer_eos": {"token": tok["eos_token"], "id": tok["eos_token_id"]},
        "tokenizer_pad": {"token": tok["pad_token"], "id": tok["pad_token_id"]},
        "tokenizer_all_special_ids": tok["all_special_ids"],
        "model_config": {
            "eos_token_id": cfg["config.eos_token_id"],
            "pad_token_id": cfg["config.pad_token_id"],
            "mask_token_id": cfg["config.mask_token_id"],
            "bos_token_id": raw["config_eos"]["generation_config"]["bos_token_id"],
        },
        "generation_config": cfg["generation_config"],
    },
    "mask_token_registration": mask,
    "token_id_map": raw["token_id_map"],
    "chat_template": template,
    "chat_template_fixtures": tpl_fixtures,
    "claims": {
        "payload_tensors_loaded": False,
        "forward_pass_run": False,
        "cuda_initialized": False,
    },
}

json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"fixtures={out_path}")
