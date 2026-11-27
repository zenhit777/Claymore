"""M11.1 CPU-only chat-template fixture audit (pinned Clover revision 4b0402b).

transformers 5.19.0, no weights loaded, no CUDA.
"""
import json, os, sys
from transformers import AutoTokenizer

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CKPT", ".")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("FIXTURES_OUT", "clover-chat-template-fixtures.json")
tok = AutoTokenizer.from_pretrained(ROOT, trust_remote_code=False)

USER = [{"role": "user", "content": "Bonjour, presente-toi en une phrase."}]
SVA = [{"role": "system", "content": "Tu reponds en francais, factuellement."},
       {"role": "user", "content": "Que vaut 2+2 ?"},
       {"role": "assistant", "content": "<think>\n\n</think>\n\n4."}]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Cherche un fait",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}]

CASES = [
    ("F1_user_genprompt", USER, {"add_generation_prompt": True}),
    ("F2_user_nogenprompt", USER, {"add_generation_prompt": False}),
    ("F3_sys_user_assistant_nogenprompt", SVA, {"add_generation_prompt": False}),
    ("F4_sys_user_assistant_genprompt", SVA, {"add_generation_prompt": True}),
    ("F5_enable_thinking_true", USER, {"add_generation_prompt": True, "enable_thinking": True}),
    ("F6_enable_thinking_false", USER, {"add_generation_prompt": True, "enable_thinking": False}),
    ("F7_preserve_thinking_false", SVA, {"add_generation_prompt": False, "preserve_thinking": False}),
    ("F8_add_non_thinking_prefix_false", SVA,
     {"add_generation_prompt": False, "add_non_thinking_prefix": False}),
    ("F9_tools_genprompt", USER, {"add_generation_prompt": True, "tools": TOOLS}),
    ("F10_multimodal_content_rejet",
     [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "decrit"}]}],
     {"add_generation_prompt": True}),
    ("F11_role_inconnu_rejet", [{"role": "robot", "content": "x"}], {"add_generation_prompt": True}),
    ("F12_liste_vide_rejet", [], {"add_generation_prompt": True}),
]

EXPECTED_REJECTIONS = {
    "F10_multimodal_content_rejet": ("TemplateError", "text content only"),
    "F11_role_inconnu_rejet": ("TemplateError", "No user query found"),
    "F12_liste_vide_rejet": ("ValueError", "empty conversation"),
}

out = []
for name, msgs, kw in CASES:
    e = {"fixture": name, "kwargs": kw, "messages": msgs}
    try:
        s = tok.apply_chat_template(msgs, return_tensors=None, tokenize=False, **kw)
        # apply_chat_template(tokenize=True) renvoie un BatchEncoding (un Mapping,
        # pas un dict) : list() sur ce objet produirait les cles du dictionnaire.
        # Les identifiants sont portes par `input_ids`, parfois batches.
        direct = tok.apply_chat_template(msgs, return_tensors=None, tokenize=True, **kw)
        direct_ids = list(direct["input_ids"] if "input_ids" in direct else direct)
        if direct_ids and isinstance(direct_ids[0], list):
            direct_ids = direct_ids[0]
        rendered_ids = list(tok(s, add_special_tokens=False)["input_ids"])
        e["rendered"] = s
        e["token_ids"] = direct_ids
        e["rendered_token_ids"] = rendered_ids
        e["n_tokens"] = len(direct_ids)
        e["contains_vision_token"] = any(
            t in s for t in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>",
                             "<|video_pad|>", "<|audio_start|>"))
        e["im_start_count"] = s.count("<|im_start|>")
        e["im_end_count"] = s.count("<|im_end|>")
        e["think_open_count"] = s.count("<think>")
        e["think_close_count"] = s.count("</think>")
        e["roundtrip_ids_stable"] = direct_ids == rendered_ids
        if e["roundtrip_ids_stable"]:
            e["status"] = "OK"
        else:
            e["status"] = "ECHEC_INATTENDU"
            e["error"] = "Direct and rendered tokenization produced different token IDs"
    except Exception as ex:
        error = f"{type(ex).__name__}: {ex}"
        expected = EXPECTED_REJECTIONS.get(name)
        if expected and all(part in error for part in expected):
            e["status"] = "REJET_ATTENDU"
        else:
            e["status"] = "ECHEC_INATTENDU"
        e["error"] = error
    out.append(e)

json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
for e in out:
    print("=" * 12, e["fixture"], "->", e["status"])
    if "error" in e:
        print("  ", e["error"][:200])
    else:
        print("   n_tokens=%d im_start=%d im_end=%d think_open=%d think_close=%d vision=%s"
              % (e["n_tokens"], e["im_start_count"], e["im_end_count"], e["think_open_count"],
                 e["think_close_count"], e["contains_vision_token"]))
        print("   direct/rendered token IDs match:", e["roundtrip_ids_stable"])
        print("   ids:", e["token_ids"][:12], "..." if len(e["token_ids"]) > 12 else "")
        print("   rendu:", repr(e["rendered"]))

if any(e["status"] == "ECHEC_INATTENDU" for e in out):
    raise SystemExit(1)
