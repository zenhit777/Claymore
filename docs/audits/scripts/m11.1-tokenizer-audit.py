"""M11.1 CPU-only tokenizer/chat-template audit (pinned Clover revision 4b0402b).

No model weights are loaded; no CUDA is initialized.
"""
import json, os, sys, hashlib, platform
import transformers, tokenizers, safetensors
from transformers import AutoTokenizer
from tokenizers import Tokenizer


def ids(x):
    if isinstance(x, dict):
        return list(x["input_ids"])
    if hasattr(x, "input_ids"):
        return list(x["input_ids"])
    return list(x)


def _version(mod):
    try:
        return __import__(mod).__version__
    except ImportError:
        return "absent"


ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CKPT", ".")
FIX = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("FIXTURES_OUT", "clover-tokenizer-fixtures.json")
res = {"env": {
    "python": platform.python_version(),
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "safetensors": safetensors.__version__,
    "jinja2": _version("jinja2"),
    "cuda_available": None,
    "checkpoint_dir": ROOT,
}}

cfg = json.load(open(f"{ROOT}/config.json"))
gencfg = json.load(open(f"{ROOT}/generation_config.json"))
tokcfg = json.load(open(f"{ROOT}/tokenizer_config.json"))
res["config_eos"] = {"config.eos_token_id": cfg.get("eos_token_id"),
                     "config.pad_token_id": cfg.get("pad_token_id"),
                     "config.mask_token": cfg.get("mask_token"),
                     "config.mask_token_id": cfg.get("mask_token_id"),
                     "config.vocab_size": cfg.get("vocab_size"),
                     "config.model_type": cfg.get("model_type"),
                     "config.architectures": cfg.get("architectures"),
                     "generation_config": gencfg,
                     "tokenizer_config.eos_token": tokcfg.get("eos_token"),
                     "tokenizer_config.pad_token": tokcfg.get("pad_token"),
                     "tokenizer_config.mask_token": tokcfg.get("mask_token"),
                     "tokenizer_config.model_max_length": tokcfg.get("model_max_length"),
                     "tokenizer_config.tokenizer_class": tokcfg.get("tokenizer_class")}

# ---------------------------------------------------------------- fast tokenizer
tok = AutoTokenizer.from_pretrained(ROOT, trust_remote_code=False)
res["tokenizer"] = {"class": type(tok).__name__,
                    "vocab_size_len": len(tok),
                    "vocab_size_size": tok.vocab_size,
                    "model_max_length": tok.model_max_length,
                    "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id,
                    "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id,
                    "all_special_tokens": sorted(tok.all_special_tokens),
                    "all_special_ids": sorted(tok.all_special_ids)}

PAIRS = [(248044, "<|endoftext|>"), (248045, "<|im_start|>"), (248046, "<|im_end|>"),
         (151669, "<|mask|>"), (248068, "<think>"), (248069, "</think>")]
idmap = []
for tid, s in PAIRS:
    enc = tok.convert_ids_to_tokens(tid)
    toid = tok.convert_tokens_to_ids(s)
    single = tok.encode(s, add_special_tokens=False)
    idmap.append({"id": tid, "expected_string": s, "id->string": enc, "string->id": toid,
                  "tokenize_to_single_id": ids(single) == [tid],
                  "tokenize_ids": ids(single),
                  "ok": enc == s and toid == tid and ids(single) == [tid]})
res["token_id_map"] = idmap

# ------------------------------------------------- special-token registration
raw = json.load(open(f"{ROOT}/tokenizer.json"))
added = {a["content"]: a for a in raw.get("added_tokens", [])}
mask = added.get("<|mask|>")
res["mask_token_registration"] = {
    "in_added_tokens": mask is not None,
    "added_entry": mask,
    "registered_as_special_in_fast": "<|mask|>" in tok.all_special_tokens,
    "in_additional_special_tokens": "<|mask|>" in (getattr(tok, "additional_special_tokens", None) or []),
    "vocab_index": raw["model"]["vocab"].get("<|mask|>"),
    "byte_decoder_present": raw["decoder"]["type"],
    "not_split": idmap[3]["tokenize_to_single_id"],
    "unk_id": raw.get("model", {}).get("unk_token_id"),
}

# --------------------------------------------- remapping / removed BPE merges
vocab = raw["model"]["vocab"]
merges = raw["model"]["merges"]
inv_vocab = {v: k for k, v in vocab.items()}
merge_list = merges if isinstance(merges[0], str) else [" ".join(m) for m in merges]
# un merge ne doit pas produire la chaine <mask> ni reintroduire l'ancien token 151669
old_151669_candidates = ["151669", inv_vocab.get(151669)]
affected = [m for m in merge_list if "<|mask|>" in m or m.endswith(" 151669")]
decoding_of_mask = tok.decode(ids(tok.encode("<|mask|>", add_special_tokens=False)))
res["remapping_151669"] = {
    "merges_total": len(merge_list),
    "token_at_151669": inv_vocab.get(151669),
    "merges_referencing_mask_or_151669": affected[:10],
    "merges_referencing_count": len(affected),
    "encode_decode_roundtrip": tok.decode(ids(tok.encode("<|mask|>", add_special_tokens=False))),
    "special_tokens_defined": raw.get("added_tokens") and [a["content"] for a in raw["added_tokens"]],
}

# ------------------------------------------------------------ chat template
tpl_text = open(f"{ROOT}/chat_template.jinja", encoding="utf-8").read()
res["template"] = {
    "bytes": len(tpl_text.encode()),
    "sha256": hashlib.sha256(tpl_text.encode()).hexdigest(),
    "declared_in_tokenizer_config": "chat_template" in tokcfg,
    "mentions_image": "image" in tpl_text,
    "mentions_video": "video" in tpl_text,
    "mentions_audio": "audio" in tpl_text,
    "mentions_pixels_or_patch": any(k in tpl_text for k in ("pixel_values", "patch", "image_placeholder")),
    "vision_special_tokens_present": [t for t in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")
                                      if t in vocab],
    "thinking_switch_tokens": {t: (t in vocab) for t in ("<think>", "</think>", "think", "nothink")},
}

FIXTURES = [
    ("user_simple", [{"role": "user", "content": "Bonjour, presente-toi en une phrase."}], {}),
    ("system_user_assistant",
     [{"role": "system", "content": "Tu reponds en francais, factuellement."},
      {"role": "user", "content": "Que vaut 2+2 ?"},
      {"role": "assistant", "content": "4."}], {}),
    ("user_thinking_enabled", [{"role": "user", "content": "Compte les voyelles de 'diffusion'."}],
     {"enable_thinking": True}),
    ("user_thinking_disabled", [{"role": "user", "content": "Compte les voyelles de 'diffusion'."}],
     {"enable_thinking": False}),
]
render = []
for name, msgs, kw in FIXTURES:
    for gen_prompt in (True, False):
        entry = {"fixture": name, "messages": msgs, "kw": kw, "generation_prompt": gen_prompt}
        try:
            s = tok.apply_chat_template(msgs, return_tensors=None, tokenize=False,
                                        add_generation_prompt=gen_prompt, **kw)
            tok_ids = list(tok(s, add_special_tokens=False)["input_ids"])
            entry["rendered"] = s
            entry["ids"] = tok_ids
            entry["n_tokens"] = len(tok_ids)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            try:
                s = tok.apply_chat_template(msgs, return_tensors=None, tokenize=False,
                                            add_generation_prompt=gen_prompt)
                entry["rendered_without_kw"] = s
            except Exception as e2:
                entry["error_without_kw"] = f"{type(e2).__name__}: {e2}"
        render.append(entry)
res["chat_template_fixtures"] = render

# verification CPU only
try:
    import torch
    res["env"]["torch"] = torch.__version__
    res["env"]["cuda_available"] = bool(torch.cuda.is_available())
except ImportError:
    res["env"]["torch"] = "absent"
    res["env"]["cuda_available"] = False

json.dump(res, open(FIX, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
