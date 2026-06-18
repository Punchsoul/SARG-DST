#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build LLaMAFactory SFT data for retrieval-augmented Graph-CoT prompting (training only, no inference).
Output format (jsonl): {"instruction": ..., "input": ..., "output": ...}

User-specified hard constraints:
- Tasklabel is HARD constraint.
- Demos (examples) include FULL structured Graph-CoT in input.
- Query includes a CoT scaffold (no final value revealed), so the model follows the same reasoning program.
- Model output includes a deterministic CoT (Chain/Path) plus a final line `Result: <value>`.
- switch = domain switch that introduces new constraints.
- label=none (tasklabel none): Query scaffold only has S1 "NOT activated", no S2/S3, output Result: none.

Paths are written directly in code per user request.
"""

import os
import json
import random
from typing import Any, Dict, List, Tuple, Optional

# =========================
# User-provided fixed paths
# =========================
SLOT_DESC_PATH = "/home/fzus/zzp2/ds_loo/ontology/slot_descriptions.json"
QUERY_JSONL_PATH = "/home/fzus/zzp2/ds_loo/hotel/hoteldata/finetune_hotel_with_retrieval.json"
GRAPH_STORE_JSONL_PATH = "/home/fzus/zzp2/ds_loo/hotel/hoteldata/turn_hotel_partial.jsonl"
DIALOG_STORE_JSONL_PATH = "/home/fzus/zzp2/ds_loo/hotel/hoteldata/dialog_hotel.jsonl"
OUTPUT_DIR = "/home/fzus/zzp2/LLaMA-Factory-main/data"
OUTPUT_BASENAME = "hotel_retrieval"

# =========================
# Hyperparameters
# =========================
POS_DEMO_K = 2          # number of positive demos per sample
MAX_HIST_TURNS = 4      # max history turns in context (Sys/User pairs) before current
NEG_PROB = 0.5          # probability to attach a negative demo if available
RNG_SEED = 42
random.seed(RNG_SEED)

# Allowed relation names (for sanity)
ALLOWED_RELS = {
    "activate_slot",
    "apply_label",
    "triggers_dontcare",
    "refer_to",
    "value_prev",
    "value_curr",
    "neg_slot",
    "value_carry",
}

instruction_text = (
    "You are an expert in cross-domain zero-shot Dialogue State Tracking (DST). "
    "Learn the structured graph-based Chain-of-Thought (Graph-CoT) reasoning process from the positive examples, "
    "and use the provided nodes and edges to build a graph reasoning chain that infers the value of the QUERY target slot from the dialogue.\n\n"
    "If the value candidates form a closed set, choose the answer from the candidates. "
    "If the value candidates are (Open Slot) examples, infer the value from the dialogue context.\n\n"
    "The input slot may be not activated and the label may be wrong, as shown in the negative examples. "
    "If there is not enough evidence to support an update, output: none.\n\n"
    "You only need to output (C) Chain / Path and (D) Result, and end with a single final line: Result: <value>."
)


# =========================
# Utility helpers
# =========================
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def normalize_label2_tokens(label2: str) -> List[str]:
    """
    label-2 can be something like: "refer-clear+ghost", "refer-implicit", etc.
    We split on ',', ';', '+', '|' and strip whitespace.
    """
    s = safe_str(label2).strip()
    if not s:
        return []
    for sep in [",", ";", "|", "+"]:
        s = s.replace(sep, " ")
    toks = [t for t in s.split() if t]
    return toks


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# =========================
# Slot descriptions
# =========================
def load_slot_descriptions(path: str) -> Dict[str, Dict[str, Any]]:
    """
    适配你现在的 slot_descriptions.json 格式，例如：
    "restaurant-area": {
        "type_desc": "...",
        "question_desc": "...",
        "contras_desc": "...",
        "nli_desc": "...",
        "value": ["centre","east","north","south","west"]  或  ["(Open Slot) e.g. 'Pizza Hut'", ...]
    }
    """
    data = load_json(path)
    slot2desc: Dict[str, Dict[str, Any]] = {}
    for slot, info in data.items():
        type_text = safe_str(info.get("type_desc", ""))
        q_text = safe_str(info.get("question_desc", ""))
        c_text = safe_str(info.get("contras_desc", ""))

        # 取出 ontology 中的离散候选/开放示例
        raw_vals = info.get("value", []) or []
        values: List[str] = []
        for v in raw_vals:
            sv = safe_str(v)
            if sv:
                values.append(sv)

        slot2desc[slot] = {
            "type": type_text,
            "question": q_text,
            "contras": c_text,
            "value": values,  # 把 value 列表也存进来
        }
    return slot2desc



def build_slot_desc_text(desc: Dict[str, str]) -> str:
    """
    Human-readable slot description for prompt.
    """
    pieces = []
    if desc.get("type"):
        pieces.append(f"type: {desc['type']}")
    if desc.get("question"):
        pieces.append(f"question: {desc['question']}")
    if desc.get("contras"):
        pieces.append(f"contras: {desc['contras']}")
    return " | ".join(pieces)


# =========================
# Tasklabel parsing (label-1, label-2 -> final label)
# =========================
def parse_tasklabel(label1: str, label2: str) -> str:
    """
    Parse final tasklabel from (label-1, label-2).

    Updated user rule:
    - If label-2 is non-empty, first parse its tokens:
        - If any token contains "refer-clear"      => tasklabel = "refer-clear"
        - Else if any token contains "refer-implicit" => tasklabel = "refer-implicit"
        - Else                                      => tasklabel = "confirm"
    - If label-2 is empty:
        - Use label-1 if non-empty, otherwise default to "confirm".
    """
    tokens = normalize_label2_tokens(label2)
    if tokens:
        for t in tokens:
            tl = safe_str(t).strip().lower()
            if "refer-clear" in tl:
                return "refer-clear"
            if "refer-implicit" in tl:
                return "refer-implicit"
        return "confirm"

    l1 = safe_str(label1).strip().lower()
    return l1 if l1 else "confirm"

def infer_ref_turn_from_state(
    turns: List[Dict[str, Any]],
    cur_idx: int,
    ref_slot: str,
) -> Optional[int]:
    """
    在当前对话的 state.slot_values 中寻找 ref_slot 的首次非空出现回合。
    只在 [0..cur_idx] 范围内找，避免引用“未来”的信息。
    """
    ref_slot = safe_str(ref_slot)
    if not ref_slot:
        return None

    # 从对话开头到当前轮依次扫描
    for t_idx in range(0, cur_idx + 1):
        turn = turns[t_idx]
        st = turn.get("state") or {}
        sv = st.get("slot_values") or {}
        if not isinstance(sv, dict):
            continue
        val = safe_str(sv.get(ref_slot))
        # 非空 且 不是 "none"/"dontcare" 才认为是有效值
        if val and val.lower() not in {"none", "dontcare"}:
            return t_idx

    return None
def build_demo_context_two_turns(
    dial_id: str,
    turn_id: int,
    dialog_idx: Dict[Tuple[str, int], Dict[str, Any]],
    fallback: str,
) -> Tuple[str, List[int]]:
    """
    对 demo 构造两轮上下文：
    - turn_id <= 0: 用 t0 和 t1（若存在）
    - turn_id > 0: 用 t-1 和 t（若存在）
    返回：(拼好的 context 字符串, 实际使用到的 turn 列表)
    """
    if turn_id <= 0:
        cand_turns = [0, 1]
    else:
        cand_turns = [turn_id - 1, turn_id]

    parts: List[str] = []
    used: List[int] = []
    for ti in cand_turns:
        key = (dial_id, ti)
        rec = dialog_idx.get(key)
        if not rec:
            continue
        sys_utt = safe_str(rec.get("system"))
        usr_utt = safe_str(rec.get("user"))
        parts.append(f"[t{ti}][SYS] {sys_utt}")
        parts.append(f"[t{ti}][USR] {usr_utt}")
        used.append(ti)

    if not parts:
        return fallback, []

    return " ".join(parts), used

# =========================
# Context builder
# =========================
def build_full_context(turns: List[Dict[str, Any]], cur_idx: int, max_hist: int = MAX_HIST_TURNS) -> str:
    """
    Use up to `max_hist` previous turns (Sys/User pairs) plus current as context.
    Each turn format:
      [t0][SYS] ...
      [t0][USR] ...
    """
    start_idx = max(0, cur_idx - max_hist)
    lines: List[str] = []
    for t_idx in range(start_idx, cur_idx + 1):
        turn = turns[t_idx]
        sys_utt = safe_str(turn.get("system", "none"))
        usr_utt = safe_str(turn.get("user", ""))
        lines.append(f"[t{t_idx}][SYS] {sys_utt}")
        lines.append(f"[t{t_idx}][USR] {usr_utt}")
    return " ".join(lines)


# =========================
# Graph store helpers
# =========================
def build_graph_event_index(graph_turns: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Build an index from (dial_id, turn_id) -> turn_record.

    Each graph_turn record is expected to contain fields like:
    {
      "graph_id": "dial_id::turn_id",
      "dial_id": "...",
      "turn_id": 0,
      "context": {"retrieval_text": "..."},
      "slots": [...],
      "tasklabels": [...],
      "values": [...],
      "neg_slots": [...],
      "edges": [...]
    }
    """
    idx: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for rec in graph_turns:
        dial_id = safe_str(rec.get("dial_id"))
        if not dial_id:
            continue
        turn_id = int(rec.get("turn_id", 0))
        idx[(dial_id, turn_id)] = rec
    return idx


def edge_belongs_to_slot(e: Dict[str, Any], slot: str) -> bool:
    """
    Check whether this edge belongs to the given slot.
    We look at src/dst typed fields:
      - activate_slot / neg_slot / triggers_dontcare: dst is slot(id)
      - apply_label: src is tl(slot)
      - value_prev / value_curr / value_carry: dst is val(slot=...)
      - refer_to: one end is this slot
    """
    slot = safe_str(slot)
    rel = safe_str(e.get("rel"))
    src = e.get("src", {}) or {}
    dst = e.get("dst", {}) or {}

    if rel in ("activate_slot", "neg_slot", "triggers_dontcare"):
        return safe_str(dst.get("t")) == "slot" and safe_str(dst.get("id")) == slot

    if rel == "apply_label":
        return safe_str(src.get("t")) == "tl" and safe_str(src.get("slot")) == slot

    if rel in ("value_prev", "value_curr", "value_carry"):
        return safe_str(dst.get("t")) == "val" and safe_str(dst.get("slot")) == slot

    if rel == "refer_to":
        if safe_str(src.get("t")) == "slot" and safe_str(src.get("id")) == slot:
            return True
        if safe_str(dst.get("t")) == "slot" and safe_str(dst.get("id")) == slot:
            return True
        return False

    return False

def format_edges(
    edges: List[Dict[str, Any]],
    slot: str,
    force_rels: Optional[List[str]] = None,
    label_text: str = "label",
) -> List[str]:
    """
    Convert raw edge dicts into textual E-lines, only keeping allowed relations
    and only edges that belong to the given slot.

    If force_rels is provided, only those relations are kept (intersection with ALLOWED_RELS).
    label_text: the concrete tasklabel name used in label(...) node, e.g. "constrain".
    """
    lines: List[str] = []
    slot = safe_str(slot)

    # 先按槽位过滤
    filtered = [e for e in edges if edge_belongs_to_slot(e, slot)]

    for e in filtered:
        rel = safe_str(e.get("rel"))
        if rel not in ALLOWED_RELS:
            continue
        if force_rels is not None and rel not in force_rels:
            continue

        if rel == "activate_slot":
            lines.append(f"E: ctx --activate_slot--> slot({slot})")
        elif rel == "apply_label":
            # 这里用真实标签名字，而不是写死 label(label)
            lines.append(f"E: label({label_text}) --apply_label--> slot({slot})")
        elif rel == "triggers_dontcare":
            lines.append(f"E: ctx --triggers_dontcare--> slot({slot})")
        elif rel == "refer_to":
            dst = e.get("dst", {}) or {}
            ref_id = safe_str(dst.get("id", "ref_slot"))
            lines.append(f"E: slot({slot}) --refer_to--> slot({ref_id})")
        elif rel == "value_prev":
            dst = e.get("dst", {}) or {}
            val = safe_str(dst.get("value", ""))
            lines.append(f"E: slot({slot}) --value_prev--> val({slot}={val}, role=prev)")
        elif rel == "value_curr":
            dst = e.get("dst", {}) or {}
            val = safe_str(dst.get("value", ""))
            lines.append(f"E: slot({slot}) --value_curr--> val({slot}={val}, role=cur)")
        elif rel == "neg_slot":
            lines.append(f"E: ctx --neg_slot--> slot({slot})")
        elif rel == "value_carry":
            dst = e.get("dst", {}) or {}
            val = safe_str(dst.get("value", ""))
            lines.append(f"E: slot({slot}) --value_carry--> val({slot}={val}, role=carry)")

    # 去重
    unique_lines = []
    seen = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique_lines.append(line)

    # 编号 E1, E2, ...
    numbered: List[str] = []
    for i, line in enumerate(unique_lines, start=1):
        if line.startswith("E:"):
            numbered.append(f"E{i}:{line[2:]}")
        else:
            numbered.append(line)
    return numbered




# =========================
# COT building helpers
# =========================
def fill_ref_slot_placeholder(text: str, ref_slot: str) -> str:
    return text.replace("[ref_slot]", ref_slot)


LABEL_S2_S3: Dict[str, Tuple[str, str]] = {
    "constrain": (
        "S2: in [F1] the turn provides a concrete constraint/value for [F2], evoke [F3]=constrain.",
        "S3: apply [F3]=constrain to [F2]: assign the concrete value specified in this turn to [F2].",
    ),
    "change": (
        "S2: in [F1] the turn changes [F2] to a different concrete value, evoke [F3]=change.",
        "S3: apply [F3]=change to [F2]: update [F2] from the previous value to the new concrete value in this turn.",
    ),
    "dontcare": (
        "S2: in [F1] the user expresses no preference for [F2], evoke [F3]=dontcare.",
        "S3: apply [F3]=dontcare to [F2]: set [F2]=dontcare.",
    ),
    "confirm": (
        "S2: in [F1] the user confirms the proposed value for [F2], evoke [F3]=confirm.",
        "S3: apply [F3]=confirm to [F2]: infer the value from [F1]sys for [F2].",
    ),
    "switch": (
        "S2: in [F1] the dialogue switches to a new domain/task context involving [F2], evoke [F3]=switch.",
        "S3: apply [F3]=switch to [F2]: set/initialize [F2] under the new domain context.",
    ),
    "refer-clear": (
        "S2: in [F1] the turn clearly refers to another slot as the source, evoke [F3]=refer-clear for [F2].",
        "S3: apply [F3]=refer-clear to [F2]: copy the value from ref_slot=[ref_slot] into [F2].",
    ),
    "refer-implicit": (
        "S2: in [F1] there is referring/ellipsis behavior affecting [F2], evoke [F3]=refer-implicit.",
        "S3: apply [F3]=refer-implicit to [F2]: infer the intended value under implicit reference.",
    ),
    "none": (
        "S2: (none)",
        "S3: (none)",
    ),
}


def build_pos_demo_cot(
    demo_context_snippet: str,
    slot: str,
    label: str,
    prev_value: str,
    cur_value: str,
    tasklabel_des: str,
    edges: List[Dict[str, Any]],
    ref_slot: str = "",
    ref_value: str = "",
    ref_turn: Optional[int] = None,
) -> str:

    facts = [
        "(A) Facts / Nodes",
        "F1 = context (see above)",
        f"F2 = slot = {slot}",
        f"F3 = tasklabel = {label}",
    ]
    if prev_value != "":
        facts.append(f"F4 = prev_value = {prev_value}")
    facts.append(f"F5 = cur_value = {cur_value}")

    if label == "refer-clear" and ref_slot:
        if ref_turn is not None:
            ref_tag = f"ref_t{ref_turn}"
            facts.append(f"F6 = ref_slot = {ref_slot} ({ref_tag})")
        else:
            facts.append(f"F6 = ref_slot = {ref_slot}")

    pos_rels = ["activate_slot", "apply_label", "value_prev", "value_curr", "triggers_dontcare", "refer_to"]
    edge_lines = format_edges(edges, slot, force_rels=pos_rels, label_text=label)
    edges_blk = ["", "(B) Edges / Relations"]
    if edge_lines:
        edges_blk.extend(edge_lines)
    else:
        edges_blk.append(f"E1: ctx --activate_slot--> slot({slot})")
        edges_blk.append(f"E2: label({label}) --apply_label--> slot({slot})")
        if prev_value != "":
            edges_blk.append(f"E3: slot({slot}) --value_prev--> val({slot}={prev_value}, role=prev)")
        edges_blk.append(f"E4: slot({slot}) --value_curr--> val({slot}={cur_value}, role=cur)")
        if label == "dontcare":
            edges_blk.append(f"E5: ctx --triggers_dontcare--> slot({slot})")
        if label == "refer-clear" and ref_slot:
            edges_blk.append(f"E6: slot({slot}) --refer_to--> slot({ref_slot})")

    chain = ["", "(C) Chain / Path"]
    if label == "none":
        chain.append("S1: in [F1], do NOT activate [F2] (no slot-specific operation).")
    else:
        chain.append("S1: in [F1], via (activate_slot), activate [F2].")
        s2, s3 = LABEL_S2_S3.get(
            label,
            (
                f"S2: in [F1] evoke [F3]={label} for [F2].",
                f"S3: apply [F3]={label} to [F2].",
            ),
        )
        if label == "refer-clear":
            chain.append(s2)
            # 这里是你要求的 refer-clear S3：带 ref_slot[F6] + (ref_tX)
            if ref_slot:
                if ref_turn is not None:
                    chain.append(
                        f"S3: apply [F3]=refer-clear to [F2]: copy the value from ref_slot[F6] in (ref_t{ref_turn}) into [F2]."
                    )
                else:
                    chain.append(
                        "S3: apply [F3]=refer-clear to [F2]: copy the value from ref_slot[F6] into [F2]."
                    )
            else:
                chain.append(s3)
        else:
            chain.append(s3)

        if label == "change":
            chain.append("S4: update [F2] from [F4] to [F5].  (value_curr)")
        elif label == "dontcare":
            chain.append("S4: set the final value for [F2] to dontcare.")
        elif label == "confirm":
            chain.append("S4: keep the final value for [F2] as [F5].  (value_curr)")
        elif label == "refer-clear":
            chain.append("S4: copy the inferred value from the ref_slot into [F2] as [F5].")
        else:
            chain.append("S4: set the final value for [F2] to [F5].  (value_curr)")

    res = ["", "(D) Result", f"Result: {cur_value}"]
    return "\n".join([demo_context_snippet] + facts + edges_blk + chain + res).strip()




def build_none_pos_demo_cot(demo_context_snippet: str, slot: str, edges: List[Dict[str, Any]]) -> str:
    """
    Positive demo for label=none (explicit negative).
    """
    facts = [
        "(A) Facts / Nodes",
        "F1 = context (see above)",
        f"F2 = slot = {slot}",
        "F3 = tasklabel = none",
        "F4 = carry_value = none",
    ]
    edge_lines = format_edges(edges, slot, force_rels=["neg_slot", "value_carry"])
    edges_blk = ["", "(B) Edges / Relations"]
    if edge_lines:
        edges_blk.extend(edge_lines)
    else:
        edges_blk.append(f"E1: ctx --neg_slot--> slot({slot})")
        edges_blk.append(f"E2: slot({slot}) --value_carry--> val({slot}=none, role=carry)")

    chain = ["", "(C) Chain / Path"]
    chain.append("S1: in [F1], do NOT activate [F2] (neg_slot).")
    chain.append("S2: no slot-specific operation is performed for [F2].")
    chain.append("S3: treat it as negative evidence, keep value as none.")

    res = ["", "(D) Result", "Result: none"]
    return "\n".join([demo_context_snippet] + facts + edges_blk + chain + res).strip()



def build_neg_demo(
    demo_context_snippet: str,
    slot: str,
    neg_type: str,
    carry_value: str,
    edges: List[Dict[str, Any]],
) -> str:
    facts = [
        "(A) Facts / Nodes",
        "F1 = context (see above)",
        f"F2 = slot = {slot}",
        f"F3 = neg_type = {neg_type}",
        f"F4 = carry_value = {carry_value}",
    ]

    edge_lines = format_edges(edges, slot, force_rels=["neg_slot", "value_carry"])
    edges_blk = ["", "(B) Edges / Relations"]
    if edge_lines:
        edges_blk.extend(edge_lines)
    else:
        edges_blk.append(f"E1: ctx --neg_slot--> slot({slot})")
        edges_blk.append(f"E2: slot({slot}) --value_carry--> val({slot}={carry_value}, role=carry)")

    chain = ["", "(C) Chain / Path"]
    chain.append("S1: in [F1], this slot is mention-related but NOT activated. (neg_slot)")
    neg_type_l = safe_str(neg_type).strip().lower()
    if neg_type_l == "ghost":
        chain.append(
            "S2: negative type is ghost: there was a historical value, but this mention is not a real activation for [F2]."
        )
    elif neg_type_l == "void":
        chain.append("S2: negative type is void: there is no valid value for [F2] in this context.")
    else:
        chain.append("S2: negative type is [F3].")
    chain.append("S3: treat it as negative evidence: do NOT perform slot-specific operation for [F2].")

    res = ["", "(D) Result", "Result: none"]
    return "\n".join([demo_context_snippet] + facts + edges_blk + chain + res).strip()




def build_query_cot_scaffold(
    query_turn_snippet: str,
    slot: str,
    label: str,
    tasklabel_des: str,
    prev_value: str,
    candidates: List[str],
    ref_slot: str = "",
    ref_value: str = "",
    ref_turn: Optional[int] = None,
) -> str:

    if label == "none":
        facts = [
            "(A) Facts / Nodes",
            "F1 = context (see above)",
            f"F2 = slot = {slot}",
            "F3 = tasklabel = none",
            "F4 = carry_value = none",
        ]
        edges_blk = [
            "",
            "(B) Edges / Relations",
            f"E1: ctx --neg_slot--> slot({slot})",
            f"E2: slot({slot}) --value_carry--> val({slot}=none, role=carry)",
        ]
        return "\n".join(facts + edges_blk).strip()

    facts = [
        "(A) Facts / Nodes",
        "F1 = context (see above)",
        f"F2 = slot = {slot}",
        f"F3 = tasklabel = {label}",
        f"F4 = prev_value = {prev_value}",
    ]
    if candidates:
        facts.append(f"F5 = value_candidates = {json.dumps(candidates, ensure_ascii=False)}")

    if label == "refer-clear" and ref_slot:
        if ref_turn is not None:
            ref_tag = f"ref_t{ref_turn}"
            facts.append(f"F6 = ref_slot = {ref_slot} ({ref_tag})")
        else:
            facts.append(f"F6 = ref_slot = {ref_slot}")

    edges_blk = ["", "(B) Edges / Relations"]
    edges_blk.append(f"E1: ctx --activate_slot--> slot({slot})")
    edges_blk.append(f"E2: label({label}) --apply_label--> slot({slot})")
    if prev_value != "":
        edges_blk.append(f"E3: slot({slot}) --value_prev--> val({slot}={prev_value}, role=prev)")
    edges_blk.append(f"E4: slot({slot}) --value_curr--> val(?, role=cur)")
    if label == "dontcare":
        edges_blk.append(f"E5: ctx --triggers_dontcare--> slot({slot})")
    if label == "refer-clear" and ref_slot:
        edges_blk.append(f"E6: slot({slot}) --refer_to--> slot({ref_slot})")

    return "\n".join(facts + edges_blk).strip()




def build_answer_cot(
    slot: str,
    label: str,
    tasklabel_des: str,
    prev_value: str,
    cur_value: str,
    candidates: List[str],
    ref_slot: str = "",
    ref_turn: Optional[int] = None,
) -> str:
    """
    Construct training OUTPUT for the target slot.

    OUTPUT only contains:
    - (C) Chain / Path
    - (D) Result
    """
    # label==none: 直接负路径 + Result:none
    if label == "none":
        chain = [
            "(C) Chain / Path",
            "S1: in [F1], do NOT activate [F2] (no slot-specific operation).",
        ]
        res = ["", "(D) Result", "Result: none"]
        return "\n".join(chain + res).strip()

    chain = ["(C) Chain / Path"]
    chain.append("S1: in [F1], via (activate_slot), activate [F2].")

    # S2/S3 模板仍然来自 LABEL_S2_S3
    s2, s3_template = LABEL_S2_S3.get(
        label,
        (
            f"S2: in [F1] evoke [F3]={label} for [F2].",
            f"S3: apply [F3]={label} to [F2].",
        ),
    )
    chain.append(s2)

    if label == "refer-clear":
        # 这里改成你要的写法：用 ref_slot[F6] + (ref_tX)
        if ref_slot:
            if ref_turn is not None:
                chain.append(
                    f"S3: apply [F3]=refer-clear to [F2]: "
                    f"copy the value from ref_slot[F6] in (ref_t{ref_turn}) into [F2]."
                )
            else:
                # 没有 ref_turn 的兜底写法
                chain.append(
                    "S3: apply [F3]=refer-clear to [F2]: "
                    "copy the value from ref_slot[F6] into [F2]."
                )
        else:
            # 没有 ref_slot 就退回模板
            chain.append(s3_template)
    else:
        chain.append(s3_template)

    # S4 使用 F4/F5 的语义（F4=prev_value, F5=cur_value）
    if label == "change":
        chain.append("S4: update [F2] from [F4] to [F5].")
    elif label == "dontcare":
        chain.append("S4: set the final value for [F2] to dontcare.")
    elif label == "confirm":
        chain.append("S4: keep the final value for [F2] as [F5].")
    elif label == "refer-clear":
        chain.append("S4: copy the inferred value from the ref_slot into [F2] as [F5].")
    else:
        chain.append("S4: set the final value for [F2] to [F5].")

    res = ["", "(D) Result", f"Result: {cur_value}"]
    return "\n".join(chain + res).strip()




def try_infer_ref_slot_from_state(
    dial_state: Dict[str, Any],
    slot: str,
    slot2domain: Dict[str, str],
) -> Tuple[str, str]:
    """
    Heuristic: if label is refer-clear but ref_slot not provided in taklabels,
    we try to infer from current state by domain/similar slot.
    For now, this is a stub and returns ("", "") unless user later provides rules.
    """
    # User did not provide explicit heuristic; keep as no-op.
    return "", ""

def parse_retrieved_id(rid: str) -> Tuple[str, int, str]:
    """
    Parse a retrieved_topk_id of the form "dial_id::turn_id::slot" into components.
    """
    parts = safe_str(rid).split("::")
    if len(parts) < 3:
        return "", -1, ""
    dial_id = parts[0]
    try:
        turn_id = int(parts[1])
    except ValueError:
        turn_id = -1
    slot = parts[2]
    return dial_id, turn_id, slot


def extract_slot_event_from_graph(rec: Dict[str, Any], slot: str) -> Dict[str, Any]:
    slot = safe_str(slot)
    label = "none"
    from_slot = ""
    from_turn: Optional[int] = None

    # 1) 从 tasklabels 中解析 label + from_slot/from_turn
    for tl in rec.get("tasklabels", []):
        if safe_str(tl.get("slot")) == slot:
            label = safe_str(tl.get("label", "none")).lower()
            from_slot = safe_str(tl.get("from_slot", ""))
            if "from_turn" in tl:
                try:
                    from_turn = int(tl.get("from_turn"))
                except Exception:
                    from_turn = None
            break

    prev_val = ""
    curr_val = ""
    carry_val = ""

    # 2) 从 values 中解析 prev/curr/carry
    for v in rec.get("values", []):
        if safe_str(v.get("slot")) != slot:
            continue
        role = safe_str(v.get("role")).lower()
        value = safe_str(v.get("value"))
        if role == "prev" and not prev_val:
            prev_val = value
        elif role == "curr" and not curr_val:
            curr_val = value
        elif role == "carry" and not carry_val:
            carry_val = value

    # 3) 从 neg_slots 中解析 neg_type + carry_value（如果 carry 还没填）
    neg_type = ""
    for ns in rec.get("neg_slots", []):
        if safe_str(ns.get("slot")) != slot:
            continue
        neg_type = safe_str(ns.get("neg_type")).lower()
        if not carry_val:
            carry_val = safe_str(ns.get("carry_value"))
        break

    # 4) 为当前 slot 过滤 edges
    edges_for_slot = [e for e in rec.get("edges", []) if edge_belongs_to_slot(e, slot)]

    return {
        "slot": slot,
        "label": label,
        "prev_val": prev_val,
        "curr_val": curr_val,
        "neg_type": neg_type,
        "carry_value": carry_val,
        "edges": edges_for_slot,
        "from_slot": from_slot,
        "from_turn": from_turn,
    }


# =========================
# Main pipeline
# =========================
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.jsonl")

    slot_desc_map = load_slot_descriptions(SLOT_DESC_PATH)
    graph_turns = load_jsonl(GRAPH_STORE_JSONL_PATH)
    dialog_store = load_jsonl(DIALOG_STORE_JSONL_PATH)
    graph_index = build_graph_event_index(graph_turns)

    # dialog_store: 每行是一个对话，包含 turns 列表
        # dialog_store: 每行是一个对话，包含 turns 列表
    # 建立 (dial_id, turn_id) -> turn_record 的索引，方便根据 from_turn / refer_turn 找到原始回合
    dialog_idx: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for d in dialog_store:
        dial_id = safe_str(d.get("dial_id"))
        turns = d.get("turns", [])
        for t in turns:
            tid = int(t.get("turn_id", 0))
            dialog_idx[(dial_id, tid)] = t

    query_data = load_json(QUERY_JSONL_PATH)


    with open(out_path, "w", encoding="utf-8") as fout:
        total_samples = 0
        for dial in query_data:
            dial_id = safe_str(dial.get("dial_id"))
            turns = dial.get("turns", [])
            for turn in turns:
                turn_id = int(turn.get("turn_id", 0))
                sys_utt = safe_str(turn.get("system", "none"))
                usr_utt = safe_str(turn.get("user", ""))
                state = turn.get("state", {})
                taklabels = turn.get("taklabels", [])

                full_context = build_full_context(turns, turn_id, MAX_HIST_TURNS)

                for tk in taklabels:
                    slot = safe_str(tk.get("slot"))
                    if not slot:
                        continue

                    label1 = safe_str(tk.get("label-1"))
                    label2 = safe_str(tk.get("label-2"))
                    label = parse_tasklabel(label1, label2)

                    prev_val = safe_str(tk.get("prev_val"))
                    curr_val = safe_str(tk.get("curr_val"))
                    retrieved_ids = tk.get("retrieved_topk_ids", [])

                    sdesc = build_slot_desc_text(slot_desc_map.get(slot, {}))

                    # Tasklabel description (natural language) can be a simple mapping by label
                    # (User can refine; here we give a light default.)
                    if label == "constrain":
                        tasklabel_des = "The current turn provides a concrete constraint/value for the target slot."
                    elif label == "change":
                        tasklabel_des = "The current turn changes the previously set value of the target slot."
                    elif label == "dontcare":
                        tasklabel_des = "The current turn expresses no specific preference for the target slot (dontcare)."
                    elif label == "confirm":
                        tasklabel_des = "The current turn explicitly confirms/affirms the target slot value proposed by SYS."
                    elif label == "switch":
                        tasklabel_des = "The current turn switches to a new domain/task and sets constraints for the target slot."
                    elif label == "refer-clear":
                        tasklabel_des = "The current turn assigns/updates the target slot value by clearly referring to ref_slot value."
                    elif label == "refer-implicit":
                        tasklabel_des = "The current turn implicitly refers to another slot or context for the target slot value."
                    elif label == "none":
                        tasklabel_des = "The current turn does NOT activate the target slot (no slot-specific update)."
                    else:
                        tasklabel_des = f"The current turn has label={label} for the target slot."

                    # Build value candidates:
                    def build_value_candidates(prev_v: str, cur_v: str) -> List[str]:
                        """
                        候选值策略：
                        1）如果 slot_descriptions 中为该 slot 提供了 value 列表：
                            - 无论是封闭枚举（centre/east/...）还是 "(Open Slot)" 示例，都原样使用；
                            - 然后追加 "none"。
                        2）如果未配置 value 列表：退回默认的 Open Slot 示例 + "none"。
                        """
                        info = slot_desc_map.get(slot, {}) or {}
                        raw_vals = info.get("value", []) or []

                        vals_cfg: List[str] = []
                        for v in raw_vals:
                            sv = safe_str(v)
                            if sv:
                                vals_cfg.append(sv)

                        if vals_cfg:
                            # 原样使用 ontology 的配置，去重后追加 none
                            cand = list(dict.fromkeys(vals_cfg))
                            if "none" not in cand:
                                cand.append("none")
                            return cand

                        # 无配置时的兜底：老的 Open Slot 模板
                        return [
                            "(Open Slot) e.g. '17:00'",
                            "(Open Slot) e.g. '09:45'",
                            "none",
                        ]
                    cand_vals = build_value_candidates(prev_val, curr_val)
                    candidates_field = json.dumps(cand_vals, ensure_ascii=False)

                    ref_slot = safe_str(tk.get("ref_slot"))
                    ref_value = safe_str(tk.get("ref_value"))
                    if label == "refer-clear" and not ref_slot:
                        # try heuristic if not provided
                        _rslot, _rval = try_infer_ref_slot_from_state(state, slot, {})
                        if _rslot:
                            ref_slot, ref_value = _rslot, _rval
                    ref_turn = None
                    if label == "refer-clear" and ref_slot:
                        ref_turn = infer_ref_turn_from_state(
                            turns,      # 当前对话的所有轮次
                            turn_id,    # 当前 query 回合
                            ref_slot,   # 需要跟踪的 ref_slot
                        )
                    query_turn_snippet = f"[t{turn_id}][SYS] {sys_utt} [t{turn_id}][USR] {usr_utt}"

                    # Build VEC-based edges from retrieved graph events
                    pos_blocks: List[str] = []
                    neg_block: str = ""

                    pos_edges_all: List[Dict[str, Any]] = []
                    neg_edges_all: List[Dict[str, Any]] = []

                    for rid in retrieved_ids:
                        dial2, turn2, r_slot = parse_retrieved_id(rid)
                        if not dial2 or turn2 < 0:
                            continue
                        # 只保留与当前 query 槽位匹配的 demo
                        rec = graph_index.get((dial2, turn2))
                        if not rec:
                            continue

                        ev = extract_slot_event_from_graph(rec, r_slot)
                        r_label = safe_str(ev.get("label"))
                        r_prev = safe_str(ev.get("prev_val"))
                        r_cur = safe_str(ev.get("curr_val"))
                        r_edges = ev.get("edges", [])
                        r_neg_type = safe_str(ev.get("neg_type"))
                        r_carry_value = safe_str(ev.get("carry_value"))

                        if r_label == "none":
                            # 作为候选负例事件
                            neg_edges_all.append(
                                {
                                    "slot": r_slot,
                                    "neg_type": r_neg_type,
                                    "carry_value": r_carry_value,
                                    "edges": r_edges,
                                    "dial_id": dial2,
                                    "turn_id": turn2,
                                }
                            )
                        else:
                            # 正例事件
                            pos_edges_all.append(
                                {
                                    "slot": r_slot,
                                    "label": r_label,
                                    "prev_val": r_prev,
                                    "curr_val": r_cur,
                                    "edges": r_edges,
                                    "dial_id": dial2,
                                    "turn_id": turn2,
                                    "from_slot": safe_str(ev.get("from_slot", "")),
                                    "from_turn": ev.get("from_turn"),
                                }
                            )

                            # === 新增：从当前 graph 记录的 neg_slots 展开负例事件 ===
                        for ns in rec.get("neg_slots", []):
                            neg_slot = safe_str(ns.get("slot"))
                            if not neg_slot:
                                continue

                            ev_neg = extract_slot_event_from_graph(rec, neg_slot)
                            neg_type = safe_str(ev_neg.get("neg_type", "void")) or "void"
                            carry_value = safe_str(ev_neg.get("carry_value", "none")) or "none"
                            edges_neg = ev_neg.get("edges", [])

                            neg_edges_all.append({
                                "slot": neg_slot,
                                "neg_type": neg_type,
                                "carry_value": carry_value,
                                "edges": edges_neg,
                                "dial_id": dial2,
                                "turn_id": turn2,
                            })
                        # === 新增结束 ===


                    # Build POS demos (up to POS_DEMO_K)
                    for i, pe in enumerate(pos_edges_all[:POS_DEMO_K]):
                        demo_slot = pe["slot"]
                        demo_label = pe["label"]
                        demo_prev = pe["prev_val"]
                        demo_cur = pe["curr_val"]
                        demo_edges = pe["edges"]
                        demo_dial = safe_str(pe.get("dial_id", dial_id))
                        demo_turn = int(pe.get("turn_id", turn_id))
                        demo_ref_slot = safe_str(pe.get("from_slot", ""))
                        demo_ref_turn = pe.get("from_turn")

                        # 用两轮上下文构造 demo context
                        if (demo_dial, demo_turn) in dialog_idx:
                            demo_context_snippet, _ = build_demo_context_two_turns(
                                demo_dial,
                                demo_turn,
                                dialog_idx,
                                fallback=query_turn_snippet,
                            )
                        else:
                            # fall back to current query turn snippet
                            demo_context_snippet = query_turn_snippet

                        if demo_label == "none":
                            pos_block = build_none_pos_demo_cot(
                                demo_context_snippet,
                                demo_slot,
                                demo_edges,
                            )
                        else:
                            pos_block = build_pos_demo_cot(
                                demo_context_snippet,
                                demo_slot,
                                demo_label,
                                demo_prev,
                                demo_cur,
                                tasklabel_des,
                                demo_edges,
                                ref_slot=demo_ref_slot,
                                ref_value="",
                                ref_turn=demo_ref_turn,
                            )
                        pos_blocks.append(pos_block)


                    # NEG demo
                    if neg_edges_all and random.random() < NEG_PROB:
                        # pick one negative event as demo
                        ne = random.choice(neg_edges_all)
                        neg_slot = ne.get("slot", slot)
                        neg_type = safe_str(ne.get("neg_type", "void")) or "void"
                        carry_value = safe_str(ne.get("carry_value", "none")) or "none"
                        neg_edges = ne.get("edges", [])

                        neg_dial = safe_str(ne.get("dial_id", dial_id))
                        neg_turn = int(ne.get("turn_id", turn_id))
                        if (neg_dial, neg_turn) in dialog_idx:
                            demo_context_snippet, _ = build_demo_context_two_turns(
                                neg_dial,
                                neg_turn,
                                dialog_idx,
                                fallback=query_turn_snippet,
                            )
                        else:
                            demo_context_snippet = query_turn_snippet


                        neg_block = build_neg_demo(
                            demo_context_snippet,
                            neg_slot,
                            neg_type=neg_type,
                            carry_value=carry_value,
                            edges=neg_edges,
                        )
                    else:
                        neg_block = ""

                    # Build QUERY scaffold
                    query_scaffold = build_query_cot_scaffold(
                        query_turn_snippet=query_turn_snippet,
                        slot=slot,
                        label=label,
                        tasklabel_des=tasklabel_des,
                        prev_value=prev_val,
                        candidates=cand_vals,
                        ref_slot=ref_slot,
                        ref_value=ref_value,
                        ref_turn=ref_turn,
                    )


                    # Build input prompt
                    ref_slot_field = ref_slot
                    ref_value_field = ref_value

                    input_parts: List[str] = []
                    input_parts.append("[QUERY]")
                    input_parts.append("context:\n" + full_context)
                    input_parts.append(f"slot: {slot}")
                    input_parts.append(f"slot_desc: {sdesc}")
                    input_parts.append(f"tasklabel: {label}")
                    input_parts.append(f"tasklabel_des: {tasklabel_des}")
                    if prev_val:
                        input_parts.append(f"prev_state: {prev_val}")
                    if cand_vals:
                        input_parts.append(f"value_candidates: {candidates_field}")
                    if ref_slot_field:
                        input_parts.append(f"ref_slot: {ref_slot_field}")
                    if ref_value_field:
                        input_parts.append(f"ref_value: {ref_value_field}")
                    input_parts.append("\n[QUERY-COT-SCAFFOLD]\n" + query_scaffold)

                    # Positive demos: explicitly numbered, no '(exactly K)' marker
                    if len(pos_blocks) > 0:
                        input_parts.append("\n[POSITIVE DEMO 1]")
                        input_parts.append(pos_blocks[0])
                    if len(pos_blocks) > 1:
                        input_parts.append("\n[POSITIVE DEMO 2]")
                        input_parts.append(pos_blocks[1])

                    # Optional negative demo
                    if neg_block:
                        input_parts.append("\n[NEGATIVE DEMO]")
                        input_parts.append(neg_block)

                    input_text = "\n".join(input_parts).strip()

                    # Output: deterministic CoT + final `Result: <value>`
                    out_value = curr_val
                    if label == "none":
                        out_value = "none"
                    if out_value == "":
                        out_value = "none"

                    out_cot = build_answer_cot(
                        slot=slot,
                        label=label,
                        tasklabel_des=tasklabel_des,
                        prev_value=prev_val,
                        cur_value=out_value,
                        candidates=cand_vals,
                        ref_slot=ref_slot,
                        ref_turn=ref_turn,
                    )

                    record = {
                        "instruction": instruction_text,
                        "input": input_text,
                        "output": out_cot,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_samples += 1

        print(f"[INFO] Wrote {total_samples} SFT samples to {out_path}")


if __name__ == "__main__":
    main()
