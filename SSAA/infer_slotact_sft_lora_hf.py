# -*- coding: utf-8 -*-
"""
infer_slotact_stage1_lora_hf.py  (Aligned to new paradigm, NO candidates in inference)
+ Filter dialogs/turns by oracle target-domain activation turns (state.slot_values delta)

Inference paradigm:
- For selected dialogues: ONLY turns that activate ANY TARGET_DOMAIN slot (by state delta) will be inferred.
- For each selected turn, judge ALL slots in TARGET_DOMAIN (from slot_descriptions keys).
- Prompt aligns with training: Turn + t-1 Fine_summary + Activated slots so far + Target slot + Value examples
  + Slot memory + base/contrast desc + (optional) GateBlock.
- Memory/activated_sofar are maintained by rolling predictions (not gold).
- GateBlock is appended ONLY when triggered (CONFIRM / DEIXIS / SHORT_VALUE).
- Value examples read ONLY from slot_descriptions.json: field "value" (also accept "values"/"value_examples").
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ===================== PATHS (EDIT ME) =====================
BASE_MODEL_DIR = "/home/fzus/zzp/model2/Llama-3.1-8B-Instruct"
ADAPTER_DIR = "/home/fzus/zzp2/LLaMA-Factory-main/saves/Llama-3-8B-Instruct/lora/train_2025-12-23-12-48-15/checkpoint-159"

TEST_PATH = "/home/fzus/zzp2/ds_loo/restaurant/test_restaurant_with_candidates_only_restaurant.json"
SLOT_DESC_PATH = "/home/fzus/zzp2/ds_loo/ontology/slot_descriptions.json"
OUT_PATH = "/home/fzus/zzp2/ds_loo/restaurant/restaurant_outcome"

TARGET_DOMAIN = "restaurant"

# ===================== INFER SETTINGS =====================
BATCH_SIZE = 16
MAX_NEW_TOKENS = 3
MAX_ACT_SOFAR = 30
MAX_VALUE_EX = 10

# 可选：如果目标域槽位特别多，想限制每轮最多判别多少个槽位（None=全部）
SLOT_TOPK: Optional[int] = None

SYSTEM_TEXT = (
    "You are a slot activation verifier for dialogue state tracking. "
    "You will be given the current turn, minimal history, a candidate slot, slot memory, and slot descriptions. "
    "Answer strictly with one token: yes or no."
)

# ===================== PROMPT (same as training build) =====================
PROMPT_TMPL_NORMAL = """[Turn]
System: {sys_t}
User: {usr_t}

[History summary]
t-1: {sum_t1}

[Activated slots so far]
{act_sofar}

[Target slot]
Slot: {slot_show}

[Value examples]
{value_examples}

[Slot memory for target slot]
{slot_mem}

[Slot description]
base: {base_desc}
contrast: {contrast_desc}

[Definition]
A slot is ACTIVATED in this turn if the user provides new information that updates the user's goal for this slot
(e.g., provides a value, changes/corrects it, sets dontcare/none, or confirms it).

[Question]
Is the target slot activated in the current turn?

[Answer format]
Output exactly one token: yes or no.
"""

GATE_BLOCK_CONFIRM = """[Gate: CONFIRM]
The user reply is a confirmation/rejection to the system's proposal/question.
Use the system utterance ONLY to interpret what the user is confirming/rejecting.
Do NOT activate the slot if it is only mentioned by the system and not adopted by the user this turn.
"""

GATE_BLOCK_DEIXIS = """[Gate: DEIXIS]
The user reply contains reference/selection (e.g., "that one", "the first").
Use the system utterance ONLY to resolve what the user refers to.
Do NOT activate the slot if it is only mentioned by the system and not adopted by the user this turn.
"""

GATE_BLOCK_SHORT_VALUE = """[Gate: SHORT_VALUE]
The user reply is a short value-only answer.
Use the system question ONLY to infer which slot the value answers (slot focus).
Do NOT activate the slot if it is only mentioned by the system and not adopted by the user this turn.
"""

YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

# ---------- Gate trigger (narrow, same as training build) ----------
CONFIRM_SET = {
    "yes", "yeah", "yep", "no", "nope", "ok", "okay", "correct", "right", "sure", "fine"
}
DEIXIS_PAT = re.compile(r"\b(that one|this one|the first|the second|the other|it|there)\b", re.I)
Q_PAT = re.compile(r"(\?)|(^\s*(what|which|when|where|how)\b)|(^\s*(do you|would you|could you|can you)\b)", re.I)
OFFER_PAT = re.compile(r"\b(i found|the closest|the only|available|option|recommend|is that ok|is that okay|would you like)\b", re.I)

def gate_type(sys_t: str, usr_t: str) -> str:
    s = (sys_t or "").strip().lower()
    u = (usr_t or "").strip().lower()
    toks = [x for x in u.split() if x]
    ulen = len(toks)

    sys_is_q = bool(Q_PAT.search(s))
    sys_is_offer = bool(OFFER_PAT.search(s))
    is_deixis = bool(DEIXIS_PAT.search(u))
    is_confirm = (u in CONFIRM_SET)

    if is_confirm and (sys_is_offer or sys_is_q):
        return "CONFIRM"
    if is_deixis and sys_is_offer:
        return "DEIXIS"
    if (ulen <= 3) and (not is_confirm) and (not is_deixis) and sys_is_q:
        bad_verbs = {"want", "need", "looking", "find", "book", "prefer", "instead", "actually", "change", "not"}
        if not any(w in bad_verbs for w in toks):
            return "SHORT_VALUE"
    return "OFF"

def gate_block(g: str) -> str:
    if g == "CONFIRM":
        return GATE_BLOCK_CONFIRM
    if g == "DEIXIS":
        return GATE_BLOCK_DEIXIS
    if g == "SHORT_VALUE":
        return GATE_BLOCK_SHORT_VALUE
    return ""


# ===================== IO =====================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ===================== Helpers =====================
ABSENT = "__ABSENT__"

def get_turn_slotvalues_from_state(turn: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expect format:
      turn["state"] = {"active_intent": ..., "slot_values": {...}}
    Return a flat dict: {"domain-slot": value, ...}
    """
    st = turn.get("state", None)
    if isinstance(st, dict):
        sv = st.get("slot_values", None)
        if isinstance(sv, dict):
            return {str(k).strip(): sv[k] for k in sv.keys()}
    return {}

def _norm_val(v: Any) -> str:
    if v is None:
        return ABSENT
    if isinstance(v, (list, tuple)):
        return "|".join([str(x).strip() for x in v])
    s = str(v).strip()
    return s if s else ABSENT

def diff_activated_slots(prev_state: Dict[str, Any], cur_state: Dict[str, Any]) -> Set[str]:
    """
    Activated slots = slots whose value differs from prev -> cur (turn-level delta).
    Includes new/update/remove (remove = present in prev, absent in cur).
    """
    keys = set(prev_state.keys()) | set(cur_state.keys())
    delta = set()
    for slot in keys:
        pv = _norm_val(prev_state.get(slot, ABSENT))
        cv = _norm_val(cur_state.get(slot, ABSENT))
        if pv != cv:
            delta.add(slot)
    return delta

def _norm_text(x: Any, default: str = "none") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default

def _norm_summary(x: Any) -> str:
    if x is None:
        return "None"
    s = str(x).strip()
    return s if s else "None"

def split_slot(slot_id: str) -> Tuple[str, str]:
    if "-" not in slot_id:
        return "", slot_id.strip()
    dom, typ = slot_id.split("-", 1)
    return dom.strip(), typ.strip()

def slot_to_domain(slot_id: str) -> str:
    dom, _ = split_slot(slot_id)
    return dom

def get_tminus1_fine_summary(turns: List[Dict[str, Any]], t_idx: int) -> str:
    if t_idx - 1 >= 0:
        return _norm_summary(turns[t_idx - 1].get("Fine_summary", None))
    return "None"

def fmt_act_sofar(slots: Set[str]) -> str:
    if not slots:
        return "None"
    lst = sorted(slots)
    if len(lst) > MAX_ACT_SOFAR:
        lst = lst[:MAX_ACT_SOFAR] + ["..."]
    return ", ".join(lst)

def parse_yesno(gen_text: str) -> str:
    ms = YESNO_RE.findall((gen_text or "").strip().lower())
    if not ms:
        return "no"
    return "yes" if ms[-1].lower() == "yes" else "no"

def compute_target_activation_mask(turns: List[Dict[str, Any]], target_domain: str):
    """
    Oracle mask based on state.slot_values delta:
    - mask[t] = True iff this turn activates ANY target-domain slot (delta non-empty)
    - gold_list[t] = sorted activated target-domain slots (for debugging & eval)
    """
    prev_state = {}
    mask: List[bool] = []
    gold_list: List[List[str]] = []
    for turn in turns:
        cur_state = get_turn_slotvalues_from_state(turn)
        delta = diff_activated_slots(prev_state, cur_state)
        prev_state = cur_state
        gold_target = sorted([s for s in delta if slot_to_domain(s) == target_domain])
        gold_list.append(gold_target)
        mask.append(len(gold_target) > 0)
    return mask, gold_list


# ===================== Value examples (ONLY from slot_descriptions.json) =====================
def _extract_values_obj(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        s = obj.strip()
        return [s] if s else []
    if isinstance(obj, list):
        vals = []
        for x in obj:
            if isinstance(x, str):
                xs = x.strip()
                if xs:
                    vals.append(xs)
            elif isinstance(x, dict):
                for k in ("value", "text", "name"):
                    if k in x and str(x[k]).strip():
                        vals.append(str(x[k]).strip())
                        break
        return vals
    if isinstance(obj, dict):
        for k in ("value", "values", "examples", "value_examples"):
            if k in obj:
                return _extract_values_obj(obj[k])
    return []

def format_value_examples(slot_id: str, slot_desc_map: Dict[str, Dict[str, Any]], max_n: int = MAX_VALUE_EX) -> str:
    desc_obj = slot_desc_map.get(slot_id, {})
    vals = (
        _extract_values_obj(desc_obj.get("value"))
        or _extract_values_obj(desc_obj.get("values"))
        or _extract_values_obj(desc_obj.get("value_examples"))
    )
    if not vals:
        return "None"
    filt = []
    for v in vals:
        lv = v.strip().lower()
        if lv in {"none", "dontcare"}:
            continue
        filt.append(v)
    if not filt:
        return "None"
    seen = set()
    uniq = []
    for v in filt:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    uniq = uniq[:max_n]
    return ", ".join(uniq)


# ===================== Prompt builder =====================
def build_user_prompt(
    sys_t: str,
    usr_t: str,
    sum_t1: str,
    act_sofar: str,
    slot_id: str,
    slot_mem: str,
    slot_desc_map: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    desc_obj = slot_desc_map.get(slot_id, {})

    base_desc = str(desc_obj.get("concise_desc", "")).strip()
    if not base_desc:
        base_desc = str(desc_obj.get("type_desc", "None")).strip() or "None"

    contrast_desc = str(desc_obj.get("contras_desc", "None")).strip() or "None"
    value_examples = format_value_examples(slot_id, slot_desc_map, max_n=MAX_VALUE_EX)

    g = gate_type(sys_t, usr_t)

    prompt = PROMPT_TMPL_NORMAL.format(
        sys_t=sys_t,
        usr_t=usr_t,
        sum_t1=sum_t1,
        act_sofar=act_sofar,
        slot_show=slot_id,
        value_examples=value_examples,
        slot_mem=slot_mem if slot_mem else "NONE",
        base_desc=base_desc,
        contrast_desc=contrast_desc,
    )

    gb = gate_block(g)
    if gb:
        prompt = prompt + "\n" + gb

    return prompt, g


# ===================== Model helpers =====================
def render_chat(tokenizer, system_text: str, user_text: str) -> str:
    msgs = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate_yesno_batch(model, tokenizer, rendered_prompts: List[str], max_new_tokens: int = MAX_NEW_TOKENS) -> List[str]:
    enc = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    device0 = next(model.parameters()).device
    enc = {k: v.to(device0) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,                 # 贪心
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen = out[:, enc["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return texts


# ===================== MAIN =====================
def main():
    print("[LOAD] slot descriptions...")
    slot_desc_map: Dict[str, Dict[str, Any]] = load_json(SLOT_DESC_PATH)

    # collect ALL target-domain slots from slot_desc keys
    target_slots = sorted([s for s in slot_desc_map.keys() if slot_to_domain(s) == TARGET_DOMAIN])
    if SLOT_TOPK is not None:
        target_slots = target_slots[:SLOT_TOPK]

    if not target_slots:
        raise RuntimeError(f"No target slots found in slot_descriptions for domain={TARGET_DOMAIN}")

    print(f"[INFO] target_domain={TARGET_DOMAIN}  #target_slots={len(target_slots)}")
    print("[LOAD] test data...")
    data = load_json(TEST_PATH)
    if isinstance(data, dict):
        data = list(data.values())
    assert isinstance(data, list), "TEST_PATH should be a list (or dict of dialogs)."

    print("[LOAD] tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[LOAD] base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print("[LOAD] lora adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    # eval stats (ONLY on activated target-domain turns, because we skip others)
    tp = fp = fn = 0
    n_eval_turns = 0
    n_correct_turns = 0

    total_calls = 0
    skipped_dialogs = 0
    skipped_turns = 0
    processed_turns = 0

    for dialog in tqdm(data, desc="Infer dialogs"):
        turns = dialog.get("turns", [])
        if not isinstance(turns, list):
            continue

        # ===== Oracle dialog/turn filtering (NOT cumulative, turn-level activation) =====
        mask, gold_by_delta = compute_target_activation_mask(turns, TARGET_DOMAIN)

        # dialog-level skip: no target activation turns at all => no prompt building, no inference
        if not any(mask):
            skipped_dialogs += 1
            dialog["skipped"] = True
            dialog["skip_reason"] = "no_target_domain_activation_turn"
            continue

        # rolling prediction-based history
        activated_sofar: Set[str] = set()
        mem_map: Dict[str, str] = {}  # slot -> last activation Fine_summary

        for t_idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue

            # always write oracle gold (debug)
            gold_target_list = gold_by_delta[t_idx]
            turn["gold_activated_slots_by_state_delta"] = gold_target_list

            # turn-level skip: this turn does NOT activate any target-domain slot => no prompt, no model
            if not mask[t_idx]:
                skipped_turns += 1
                turn["skipped"] = True
                turn["skip_reason"] = "no_target_domain_activation_this_turn"
                turn["pred"] = []
                turn["pred_detail"] = []
                continue

            processed_turns += 1
            turn["skipped"] = False

            sys_t = _norm_text(turn.get("system", "none"), default="none")
            usr_t = _norm_text(turn.get("user", "none"), default="none")
            sum_t1 = get_tminus1_fine_summary(turns, t_idx)
            act_sofar_str = fmt_act_sofar(activated_sofar)

            rendered_inputs: List[str] = []
            metas: List[Dict[str, Any]] = []

            # build prompts for ALL target-domain slots (for this selected turn)
            for slot_id in target_slots:
                slot_mem = mem_map.get(slot_id, "NONE")
                user_prompt, g = build_user_prompt(
                    sys_t=sys_t,
                    usr_t=usr_t,
                    sum_t1=sum_t1,
                    act_sofar=act_sofar_str,
                    slot_id=slot_id,
                    slot_mem=slot_mem,
                    slot_desc_map=slot_desc_map,
                )
                rendered = render_chat(tokenizer, SYSTEM_TEXT, user_prompt)
                rendered_inputs.append(rendered)
                metas.append({"slot": slot_id, "gate": g})

            preds_yes: Set[str] = set()
            details: List[Dict[str, Any]] = []

            # batch inference
            for i in range(0, len(rendered_inputs), BATCH_SIZE):
                sub = rendered_inputs[i:i + BATCH_SIZE]
                sub_metas = metas[i:i + BATCH_SIZE]
                gen_texts = generate_yesno_batch(model, tokenizer, sub, max_new_tokens=MAX_NEW_TOKENS)
                total_calls += len(sub)

                for gen, meta in zip(gen_texts, sub_metas):
                    yn = parse_yesno(gen)
                    details.append({
                        "slot": meta["slot"],
                        "gate": meta["gate"],
                        "raw_gen": (gen or "").strip(),
                        "pred": yn,
                    })
                    if yn == "yes":
                        preds_yes.add(meta["slot"])

            turn["pred"] = sorted(list(preds_yes))
            turn["pred_detail"] = details

            # update rolling memory/state by predictions
            if preds_yes:
                fine_sum_t = _norm_summary(turn.get("Fine_summary", None))
                for s in preds_yes:
                    mem_map[s] = fine_sum_t
                activated_sofar |= preds_yes

            # ====== EVAL (ONLY on selected turns; gold from precomputed delta list) ======
            gold_target = set(gold_target_list)  # non-empty by construction
            pred_target = {s for s in turn.get("pred", []) if slot_to_domain(s) == TARGET_DOMAIN}

            n_eval_turns += 1
            if gold_target == pred_target:
                n_correct_turns += 1

            tp += len(gold_target & pred_target)
            fp += len(pred_target - gold_target)
            fn += len(gold_target - pred_target)

    if n_eval_turns > 0:
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        class_jga = (n_correct_turns / n_eval_turns) if n_eval_turns > 0 else 0.0

        print(f"[EVAL] target_domain={TARGET_DOMAIN}")
        print(f"[EVAL] evaluated_turns={n_eval_turns} correct_turns={n_correct_turns} classJGA={class_jga:.4f}")
        print(f"[EVAL] P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

    save_json(OUT_PATH, data)
    print(f"[WRITEBACK] saved -> {OUT_PATH}")
    print(f"[INFO] total_model_calls={total_calls}")
    print(f"[INFO] processed_turns={processed_turns} skipped_turns={skipped_turns} skipped_dialogs={skipped_dialogs}")


if __name__ == "__main__":
    main()
