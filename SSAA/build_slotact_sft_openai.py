# -*- coding: utf-8 -*-
"""
Build LLaMA-Factory SFT dataset (OpenAI messages JSONL) for Slot Activation (stage-1).

Features:
1) Per-turn skip if gold_slots empty (avoid chit-chat triggering gating).
2) Triggered source-gating (narrow): only on CONFIRM / DEIXIS / SHORT_VALUE turns; otherwise normal prompt.
3) Slot-level memory: store Fine_summary of last activation turn for each slot; inject per target slot (or NONE).
4) Inject activated slots so far (training uses gold history).
5) Neg sampling ratio per pos slot: 1 POS + 2 NEG (NEG-A + NEG-B), unchanged.
   - NEG-B prefers MNA hard negative: top-ranked candidate (Top-K) same domain diff slot-type, NOT in gold this turn.
6) Domain masking per turn (deterministic): mask slot domain prefix in prompt & contrast desc.
7) Add value examples field in prompt per target slot (ONLY from slot_descriptions.json "value"/"values"/"value_examples").

Output JSONL, each line:
{
  "dial_id": ...,
  "turn_id": ...,
  "slot": ...,
  "sample_type": ...,
  "masked": true|false,
  "gate": "OFF|CONFIRM|DEIXIS|SHORT_VALUE",
  "messages": [
    {"role": "system", "content": SYSTEM_TEXT},
    {"role": "user", "content": PROMPT_TEXT},
    {"role": "assistant", "content": "yes|no"}
  ]
}
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

# ===================== USER PATHS =====================
# NOTE: Update these paths for your environment.
TRAIN_PATH = "/home/fzus/zzp2/ds_loo/restaurant/train_merged_with_candidates_restaurant.json"
SLOT_DESC_PATH = "/home/fzus/zzp2/ds_loo/ontology/slot_descriptions.json"

OUT_DIR = "/home/fzus/zzp2/LLaMA-Factory-main/data"
OUT_NAME = "slotact_stage1_train_openai.jsonl"

# ===================== SETTINGS =====================
SEED = 20251218
MASK_TURN_PROB = 0.3     # per-turn domain masking probability
TOPK_MNA = 10            # scan top-K candidates for MNA hard negative (NEG-B)
MAX_VALUE_EX = 10         # max number of value examples shown in prompt
MAX_ACT_SOFAR = 30       # max number of activated slots shown in prompt (truncate for length control)
HELDOUT_DOMAIN = "restaurant"    # if LOO heldout=attraction, set "attraction" to avoid using it as NO negatives



SYSTEM_TEXT = (
    "You are a slot activation verifier for dialogue state tracking. "
    "You will be given the current turn, minimal history, a candidate slot, slot memory, and slot descriptions. "
    "Answer strictly with one token: yes or no."
)

DOMAINS = ["hotel", "restaurant", "attraction", "train", "taxi"]

# ===================== PROMPTS =====================

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

# ===================== utils =====================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

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

def get_history_summary_t1(turns: List[Dict[str, Any]], t_idx: int) -> str:
    # Only (t-1) from Fine_summary
    if t_idx - 1 >= 0:
        return _norm_summary(turns[t_idx - 1].get("Fine_summary", None))
    return "None"

def split_slot(slot_id: str) -> Tuple[str, str]:
    # "hotel-book stay" -> ("hotel", "book stay")
    if "-" not in slot_id:
        return "", slot_id.strip()
    dom, typ = slot_id.split("-", 1)
    return dom.strip(), typ.strip()

def mask_slot_id(slot_id: str) -> str:
    # remove domain prefix: "hotel-pricerange" -> "pricerange"
    _, typ = split_slot(slot_id)
    return typ

def mask_contrast_desc(desc: str) -> str:
    # remove domain prefix in domain-slot references like "hotel-name" -> "name"
    if not desc:
        return "None"
    pat = r"\b(" + "|".join(DOMAINS) + r")-([a-zA-Z][a-zA-Z ]*[a-zA-Z])\b"
    return re.sub(pat, lambda m: m.group(2), desc)

def turn_masked(dial_id: str, turn_id: int) -> bool:
    # deterministic masking by (seed|dial_id|turn_id)
    h = _hash_int(f"{SEED}|mask|{dial_id}|{turn_id}") % 1000
    return h < int(MASK_TURN_PROB * 1000)

def pick_one_by_hash(cands: List[str], key_prefix: str) -> Optional[str]:
    if not cands:
        return None
    best = None
    best_score = None
    for s in cands:
        score = _hash_int(f"{SEED}|{key_prefix}|{s}")
        if best_score is None or score < best_score:
            best_score = score
            best = s
    return best

def gold_slots_from_taklabels(turn: Dict[str, Any]) -> List[str]:
    """
    turn["taklabels"] = [{"slot": "...", "label": ...}, ...]
    Return unique slot list (preserve first appearance).
    """
    out = []
    tl = turn.get("taklabels", [])
    if isinstance(tl, list):
        for x in tl:
            if isinstance(x, dict) and "slot" in x:
                out.append(str(x["slot"]))
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

# ---------- Triggered source-gating (narrow) ----------
CONFIRM_SET = {
    "yes", "yeah", "yep", "no", "nope", "ok", "okay", "correct", "right", "sure", "fine"
}
DEIXIS_PAT = re.compile(r"\b(that one|this one|the first|the second|the other|it|there)\b", re.I)
Q_PAT = re.compile(r"(\?)|(^\s*(what|which|when|where|how)\b)|(^\s*(do you|would you|could you|can you)\b)", re.I)
OFFER_PAT = re.compile(r"\b(i found|the closest|the only|available|option|recommend|is that ok|is that okay|would you like)\b", re.I)

def gate_type(sys_t: str, usr_t: str) -> str:
    """
    Return one of: OFF, CONFIRM, DEIXIS, SHORT_VALUE
    Narrow triggers:
      - CONFIRM: strict confirm token AND system is question/offer
      - DEIXIS: deixis phrase AND system is offer
      - SHORT_VALUE: <=3 tokens, not confirm/deixis, no strong verbs, AND system is question
    """
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

# ---------- Slot value examples (ONLY from slot_descriptions.json) ----------
def _extract_values_obj(obj: Any) -> List[str]:
    """
    Normalize possible shapes:
      - list[str]
      - list[dict] with keys like 'value'/'text'/'name'
      - dict with keys: value/values/examples/value_examples -> list[str]
      - str -> [str]
    """
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

def format_value_examples(slot_id: str,
                          slot_desc: Dict[str, Dict[str, Any]],
                          max_n: int = MAX_VALUE_EX) -> str:
    """
    Read values ONLY from slot_descriptions.json.
    Your file uses key "value": [ ... ] (also accept "values"/"value_examples" if present).
    """
    desc_obj = slot_desc.get(slot_id, {})
    vals = (
        _extract_values_obj(desc_obj.get("value"))
        or _extract_values_obj(desc_obj.get("values"))
        or _extract_values_obj(desc_obj.get("value_examples"))
    )
    if not vals:
        return "None"
    # de-dup, keep order
    seen = set()
    uniq = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    uniq = uniq[:max_n]
    return ", ".join(uniq)

# ---------- Prompt builder ----------
def fmt_act_sofar(slots: Set[str], masked: bool) -> str:
    if not slots:
        return "None"
    lst = sorted(slots)
    if len(lst) > MAX_ACT_SOFAR:
        lst = lst[:MAX_ACT_SOFAR] + ["..."]
    if masked:
        lst = [mask_slot_id(s) if s != "..." else s for s in lst]
    return ", ".join(lst)

def build_prompt(sys_t: str, usr_t: str, sum_t1: str,
                 act_sofar: str, slot_show: str, value_examples: str,
                 slot_mem: str, base_desc: str, contrast_desc: str) -> Tuple[str, str]:
    g = gate_type(sys_t, usr_t)
    prompt = PROMPT_TMPL_NORMAL.format(
        sys_t=sys_t, usr_t=usr_t,
        sum_t1=sum_t1,
        act_sofar=act_sofar,
        slot_show=slot_show,
        value_examples=value_examples,
        slot_mem=slot_mem,
        base_desc=base_desc,
        contrast_desc=contrast_desc
    )
    gb = gate_block(g)
    if gb:
        prompt = prompt + "\n" + gb
    return prompt, g

def build_openai_sample(dial_id: str, turn_id: int, slot_id: str, sample_type: str,
                        masked: bool, gate: str, prompt: str, label: str) -> Dict[str, Any]:
    return {
        "dial_id": dial_id,
        "turn_id": turn_id,
        "slot": slot_id,
        "sample_type": sample_type,
        "masked": masked,
        "gate": gate,
        "messages": [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label},
        ],
    }


# ---------- Negative selection (priority by ranked candidates) ----------

def _slot_ok_for_neg(slot_id: str, gold_set: Set[str]) -> bool:
    """Return True if slot_id can be used as a negative under global constraints."""
    if slot_id in gold_set:
        return False
    dom, _ = split_slot(slot_id)
    if HELDOUT_DOMAIN is not None and dom == HELDOUT_DOMAIN:
        return False
    return True


def select_neg_a(pos_dom: str, pos_typ: str,
                 gold_set: Set[str],
                 cand_list: List[str], cand_set: Set[str],
                 all_slots: List[str],
                 key_prefix: str) -> Optional[str]:
    """NEG-A: same type, different domain.

    Priority:
      1) highest-ranked candidate satisfying condition
      2) a non-candidate slot satisfying condition
    """
    # 1) candidates (ranked)
    for s in cand_list:
        if not _slot_ok_for_neg(s, gold_set):
            continue
        dom, typ = split_slot(s)
        if typ == pos_typ and dom != pos_dom:
            return s

    # 2) fallback: non-candidate slots
    pool = []
    for s in all_slots:
        if s in cand_set:
            continue
        if not _slot_ok_for_neg(s, gold_set):
            continue
        dom, typ = split_slot(s)
        if typ == pos_typ and dom != pos_dom:
            pool.append(s)

    return pick_one_by_hash(pool, key_prefix)


def select_neg_b(pos_dom: str, pos_typ: str,
                 gold_set: Set[str],
                 cand_list: List[str], cand_set: Set[str],
                 all_slots: List[str],
                 used_negs: Set[str],
                 key_prefix: str) -> Optional[str]:
    """NEG-B: same domain, different type.

    Priority:
      1) top-K candidates
      2) remaining candidates
      3) non-candidate slots
    """
    # 1) top-K
    for s in cand_list[:TOPK_MNA]:
        if s in used_negs:
            continue
        if not _slot_ok_for_neg(s, gold_set):
            continue
        dom, typ = split_slot(s)
        if dom == pos_dom and typ != pos_typ:
            return s

    # 2) remaining candidates
    for s in cand_list[TOPK_MNA:]:
        if s in used_negs:
            continue
        if not _slot_ok_for_neg(s, gold_set):
            continue
        dom, typ = split_slot(s)
        if dom == pos_dom and typ != pos_typ:
            return s

    # 3) fallback: non-candidate slots
    pool = []
    for s in all_slots:
        if s in used_negs:
            continue
        if s in cand_set:
            continue
        if not _slot_ok_for_neg(s, gold_set):
            continue
        dom, typ = split_slot(s)
        if dom == pos_dom and typ != pos_typ:
            pool.append(s)

    return pick_one_by_hash(pool, key_prefix)


def select_any_neg(gold_set: Set[str], used_negs: Set[str], all_slots: List[str], key_prefix: str) -> Optional[str]:
    """Last-resort fallback to guarantee 2 negatives per pos, when strict NEG-A/NEG-B are impossible."""
    pool = []
    for s in all_slots:
        if s in used_negs:
            continue
        if not _slot_ok_for_neg(s, gold_set):
            continue
        pool.append(s)
    return pick_one_by_hash(pool, key_prefix)


# ===================== main =====================

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_NAME

    slot_desc: Dict[str, Dict[str, Any]] = load_json(SLOT_DESC_PATH)

    # Global slot universe (for non-candidate fallback sampling)
    all_slots: List[str] = [str(k) for k in slot_desc.keys()]
    data = load_json(TRAIN_PATH)

    n_total = 0
    n_yes = 0
    n_turn_has_pos = 0

    # diagnostics for negative construction
    n_negA_missing = 0
    n_negB1_missing = 0
    n_negB2_missing = 0
    n_last_resort_used = 0

    with open(out_path, "w", encoding="utf-8") as wf:
        # data is list[dialog] or dict-like
        if isinstance(data, dict):
            items = list(data.items())
        elif isinstance(data, list):
            items = [(str(i), d) for i, d in enumerate(data)]
        else:
            raise ValueError("Unsupported data format: expected list or dict.")

        for dial_key, dial in items:
            if not isinstance(dial, dict):
                continue

            dial_id = str(dial.get("dial_id", dial.get("dialogue_id", dial_key)))
            turns = dial.get("turns", None)
            if turns is None:
                turns = dial.get("dialog", None)
            if not isinstance(turns, list):
                continue

            # -------- rolling memory / history activated slots (gold) --------
            mem_map: Dict[str, str] = {}        # slot -> last activation Fine_summary
            activated_sofar: Set[str] = set()   # gold history activated slots up to t-1

            for t_idx, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue
                turn_id = int(turn.get("turn_id", t_idx))

                gold_slots = gold_slots_from_taklabels(turn)

                # (保持原逻辑) skip pure no-activation turns completely
                if len(gold_slots) == 0:
                    continue

                sys_t_raw = _norm_text(turn.get("system", "none"), default="none")
                usr_t_raw = _norm_text(turn.get("user", "none"), default="none")
                sum_t1 = get_history_summary_t1(turns, t_idx)

                candidates = turn.get("candidates", [])
                cand_list = [str(x) for x in candidates] if isinstance(candidates, list) else []
                cand_set = set(cand_list)

                gold_set = set(gold_slots)

                # POS: any gold slot is a positive sample (do NOT require it to appear in candidates)
                pos_slots = gold_slots
                if len(pos_slots) == 0:
                    continue

                n_turn_has_pos += 1

                masked = turn_masked(dial_id, turn_id)
                act_sofar_str = fmt_act_sofar(activated_sofar, masked=masked)

                def _write_slot_sample(slot_id: str, label: str, sample_type: str):
                    nonlocal n_total, n_yes
                    desc_obj = slot_desc.get(slot_id, {})
                    base_desc = (desc_obj.get("type_desc" if masked else "concise_desc", "None") or "None").strip()
                    contrast_desc = (desc_obj.get("contras_desc", "None") or "None").strip()
                    if masked:
                        contrast_desc = mask_contrast_desc(contrast_desc)

                    slot_show = mask_slot_id(slot_id) if masked else slot_id
                    slot_mem = mem_map.get(slot_id, "NONE")  # Fine_summary memory
                    value_examples = format_value_examples(slot_id, slot_desc, max_n=MAX_VALUE_EX)

                    prompt, g = build_prompt(
                        sys_t=sys_t_raw, usr_t=usr_t_raw, sum_t1=sum_t1,
                        act_sofar=act_sofar_str, slot_show=slot_show, value_examples=value_examples,
                        slot_mem=slot_mem, base_desc=base_desc, contrast_desc=contrast_desc
                    )
                    wf.write(json.dumps(build_openai_sample(
                        dial_id=dial_id, turn_id=turn_id, slot_id=slot_id,
                        sample_type=sample_type, masked=masked, gate=g, prompt=prompt, label=label
                    ), ensure_ascii=False) + "\n")
                    n_total += 1
                    if label == "yes":
                        n_yes += 1

                # -------- build samples for each pos slot --------
                used_pos = set()
                for pos in pos_slots:
                    if pos in used_pos:
                        continue
                    used_pos.add(pos)

                    pos_dom, pos_typ = split_slot(pos)

                    # POS
                    _write_slot_sample(pos, "yes", "pos")

                    # NEG: always create exactly 2 negatives per POS.
                    used_negs: Set[str] = set()

                    # (a) Try NEG-A; if unavailable, fallback to 2x NEG-B.
                    neg_a = select_neg_a(
                        pos_dom=pos_dom, pos_typ=pos_typ,
                        gold_set=gold_set,
                        cand_list=cand_list, cand_set=cand_set,
                        all_slots=all_slots,
                        key_prefix=f"{dial_id}|{turn_id}|negA|{pos}",
                    )

                    if neg_a is not None:
                        used_negs.add(neg_a)
                        _write_slot_sample(neg_a, "no", "neg_same_type_diff_domain")

                        # need 1x NEG-B
                        neg_b1 = select_neg_b(
                            pos_dom=pos_dom, pos_typ=pos_typ,
                            gold_set=gold_set,
                            cand_list=cand_list, cand_set=cand_set,
                            all_slots=all_slots,
                            used_negs=used_negs,
                            key_prefix=f"{dial_id}|{turn_id}|negB1|{pos}",
                        )
                        if neg_b1 is None:
                            n_negB1_missing += 1
                            neg_b1 = select_any_neg(gold_set, used_negs, all_slots,
                                                    key_prefix=f"{dial_id}|{turn_id}|anyB1|{pos}")
                            if neg_b1 is not None:
                                n_last_resort_used += 1
                        if neg_b1 is not None:
                            used_negs.add(neg_b1)
                            _write_slot_sample(neg_b1, "no", "neg_same_domain_diff_type")
                        else:
                            # extreme edge case: duplicate NEG-A to keep 2 negatives
                            n_last_resort_used += 1
                            _write_slot_sample(neg_a, "no", "neg_fallback_duplicate")

                    else:
                        n_negA_missing += 1

                        # need 2x NEG-B
                        neg_b1 = select_neg_b(
                            pos_dom=pos_dom, pos_typ=pos_typ,
                            gold_set=gold_set,
                            cand_list=cand_list, cand_set=cand_set,
                            all_slots=all_slots,
                            used_negs=used_negs,
                            key_prefix=f"{dial_id}|{turn_id}|negB1|{pos}",
                        )
                        if neg_b1 is None:
                            n_negB1_missing += 1
                            neg_b1 = select_any_neg(gold_set, used_negs, all_slots,
                                                    key_prefix=f"{dial_id}|{turn_id}|anyB1|{pos}")
                            if neg_b1 is not None:
                                n_last_resort_used += 1
                        if neg_b1 is not None:
                            used_negs.add(neg_b1)
                            _write_slot_sample(neg_b1, "no", "neg_same_domain_diff_type")
                        else:
                            # should be extremely rare; keep going but still try b2
                            n_last_resort_used += 1

                        neg_b2 = select_neg_b(
                            pos_dom=pos_dom, pos_typ=pos_typ,
                            gold_set=gold_set,
                            cand_list=cand_list, cand_set=cand_set,
                            all_slots=all_slots,
                            used_negs=used_negs,
                            key_prefix=f"{dial_id}|{turn_id}|negB2|{pos}",
                        )
                        if neg_b2 is None:
                            n_negB2_missing += 1
                            neg_b2 = select_any_neg(gold_set, used_negs, all_slots,
                                                    key_prefix=f"{dial_id}|{turn_id}|anyB2|{pos}")
                            if neg_b2 is not None:
                                n_last_resort_used += 1
                        if neg_b2 is not None:
                            used_negs.add(neg_b2)
                            _write_slot_sample(neg_b2, "no", "neg_same_domain_diff_type")
                        else:
                            # last resort: duplicate b1 (if exists), else duplicate any neg
                            n_last_resort_used += 1
                            if neg_b1 is not None:
                                _write_slot_sample(neg_b1, "no", "neg_fallback_duplicate")
                            elif neg_a is not None:
                                _write_slot_sample(neg_a, "no", "neg_fallback_duplicate")

                # -------- after building samples for this turn: update gold history & memory --------
                fine_sum_t = _norm_summary(turn.get("Fine_summary", None))
                for s in gold_slots:
                    mem_map[s] = fine_sum_t
                activated_sofar |= set(gold_slots)

    yes_ratio = (n_yes / n_total) if n_total > 0 else 0.0
    print(f"[train] saved -> {out_path}")
    print(f"[train] total_samples={n_total} yes={n_yes} yes_ratio={yes_ratio:.4f}")
    print(f"[train] turns_with_pos={n_turn_has_pos}")
    print(f"[train] negA_missing={n_negA_missing} negB1_missing={n_negB1_missing} negB2_missing={n_negB2_missing} last_resort_used={n_last_resort_used}")
    print(f"[config] TOPK_MNA={TOPK_MNA} MASK_TURN_PROB={MASK_TURN_PROB} MAX_VALUE_EX={MAX_VALUE_EX}")

if __name__ == "__main__":
    main()
