#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Online inference + JGA evaluation for target-domain DST with Graph-CoT prompting.

关键要求：
1）提示构造必须和 retrievl.py 中 SFT 数据构造逻辑严格对齐：
    - instruction_text 完全一致；
    - input_text 结构：[QUERY] + context/slot/slot_desc/tasklabel/tasklabel_des/prev_state/value_candidates/ref_slot/ref_value
      + [QUERY-COT-SCAFFOLD] + [POSITIVE DEMO 1/2] + [NEGATIVE DEMO]；
    - CoT scaffold / demo 的 (A)(B)(C)(D) 文本完全一致；
    - value_candidates 使用 retrievl.py 中的 build_value_candidates（当前实现为固定 Open Slot 示例 + none）。
2）LLaMA3 模板对齐 LLaMAFactory：用 apply_chat_template(add_generation_prompt=True) 或手写：
    <|start_header_id|>user<|end_header_id|>\n\n{instruction + input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
3）prev_state 使用“预测状态中的当前槽位值”，而不是 gold。
4）金标回合选择、pred_taklabels gating、按 dial_id 批量、JGA 评估与上一版 stateinfer 保持一致。
"""

import os
import json
import re
import zlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# =========================
# 固定路径与参数（按你当前 stateinfer 文件）
# =========================
# --- Model ---
BASE_MODEL_PATH = "/home/fzus/zzp/model2/Llama-3.1-8B-Instruct"
LORA_ADAPTER_PATH = "/home/fzus/zzp2/LLaMA-Factory-main/saves/Llama-3-8B-Instruct/lora/taxi_retrievl_2026-01-26-09-07-09/checkpoint-600"
USE_BF16 = True
DEVICE_MAP = "auto"

# --- Data ---
TARGET_DOMAIN = "taxi"
SLOT_DESC_PATH = "/home/fzus/zzp2/ds_loo/ontology/slot_descriptions.json"
GRAPH_STORE_JSONL_PATH = "/home/fzus/zzp2/ds_loo/taxi/taxidata/turn_taxi.jsonl"
DEMO_DIALOG_STORE_PATH = "/home/fzus/zzp2/ds_loo/taxi/taxidata/dialog_taxi.jsonl"
QUERY_DIALOGS_PATH = "/home/fzus/zzp2/ds_loo/taxi/taxidata/infer_taxi_with_retrieval.json"
OUTPUT_DIALOGS_PATH = "/home/fzus/zzp2/ds_loo/taxi/outcome/taxi_outcome.json"

# --- Inference hyperparams ---
RNG_SEED = 7
BATCH_SIZE = 24              # batch 按 dial_id
MAX_HIST_TURNS = 4
POS_DEMO_K = 2
NEG_DEMO_PROB = 0.5
MAX_CANDIDATES = 20
FORCE_APPEND_NONE_CAND = True

MAX_NEW_TOKENS = 128          # 保持你之前设定
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0

random.seed(RNG_SEED)

# =========================
# Graph-CoT 指令与关系定义（从 retrievl.py 拷贝）
# =========================
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

# =========================
# 基础工具函数
# =========================
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def normalize_label2_tokens(label2: str) -> List[str]:
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


def load_json_or_jsonl(path: str) -> Any:
    """
    优先按 JSON 读，失败则按 JSONL 读，兼容你说的 query=JSON。
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return []
    try:
        return json.loads(txt)
    except Exception:
        out = []
        with open(path, "r", encoding="utf-8") as f2:
            for line in f2:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# =========================
# Slot 描述与 label 解析（从 retrievl.py 拷贝）
# =========================

def load_slot_descriptions(path: str) -> Dict[str, Dict[str, Any]]:
    data = load_json(path)
    slot2desc: Dict[str, Dict[str, Any]] = {}
    for slot, info in data.items():
        raw_vals = info.get("value", [])
        if not isinstance(raw_vals, list):
            raw_vals = []
        slot2desc[slot] = {
            # 用 *desc 字段，内部统一成 type / question / contras
            "type": safe_str(info.get("type_desc", "")),
            "question": safe_str(info.get("question_desc", "")),
            "contras": safe_str(info.get("contras_desc", "")),
            # 保留 value 列表，后面构造 candidates 要用
            "value": [safe_str(v) for v in raw_vals if safe_str(v).strip()],
        }
    return slot2desc



def build_slot_desc_text(desc: Dict[str, str]) -> str:
    pieces = []
    if desc.get("type"):
        pieces.append(f"type: {desc['type']}")
    if desc.get("question"):
        pieces.append(f"question: {desc['question']}")
    if desc.get("contras"):
        pieces.append(f"contras: {desc['contras']}")
    return " | ".join(pieces)

def try_infer_ref_slot_from_state(
    dial_state: Dict[str, Any],
    slot: str,
    slot2domain: Dict[str, str],
) -> Tuple[str, str]:
    """
    Heuristic: if label is refer-clear but ref_slot not provided in taklabels,
    we try to infer from current state by domain/similar slot.
    当前实现是占位符，返回 ("", "")，方便之后扩展。
    """
    return "", ""

def parse_tasklabel(label1: str, label2: str) -> str:
    """
    按你在 retrievl 中定义的 label-1/label-2 解析规则。
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

# =========================
# 上下文构造（从 retrievl.py 拷贝）
# =========================
def build_full_context(turns: List[Dict[str, Any]], cur_idx: int, max_hist: int = MAX_HIST_TURNS) -> str:
    """
    使用至多 max_hist 个历史 turn + 当前 turn 构造上下文。
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
        state = turn.get("state", {}) or {}
        slot_values = state.get("slot_values", {}) or {}
        val = safe_str(slot_values.get(ref_slot, ""))
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
    for t in cand_turns:
        dt = dialog_idx.get((dial_id, t))
        if dt is None:
            continue
        used.append(t)
        d_sys = safe_str(dt.get("system", "none"))
        d_usr = safe_str(dt.get("user", ""))
        parts.append(f"[t{t}][SYS] {d_sys} [t{t}][USR] {d_usr}")

    if not parts:
        return fallback, []

    return " ".join(parts), used


# =========================
# Graph store / Edge 格式化（从 retrievl.py 拷贝）
# =========================
def build_graph_event_index(graph_turns: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    idx: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for rec in graph_turns:
        dial_id = safe_str(rec.get("dial_id"))
        if not dial_id:
            continue
        turn_id = int(rec.get("turn_id", 0))
        idx[(dial_id, turn_id)] = rec
    return idx


def edge_belongs_to_slot(e: Dict[str, Any], slot: str) -> bool:
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
    lines: List[str] = []
    slot = safe_str(slot)

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

    unique_lines = []
    seen = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique_lines.append(line)

    numbered: List[str] = []
    for i, line in enumerate(unique_lines, start=1):
        if line.startswith("E:"):
            numbered.append(f"E{i}:{line[2:]}")
        else:
            numbered.append(line)
    return numbered


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
        f"F4 = prev_value = {prev_value}",
        f"F5 = cur_value = {cur_value}",
    ]
    ref_tag = ""
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
            if ref_tag:
                chain.append(
                    f"S3: apply [F3]=refer-clear to [F2]: "
                    f"copy the value from ref_slot[F6] in ({ref_tag}) into [F2]."
                )
            else:
                chain.append(
                    "S3: apply [F3]=refer-clear to [F2]: "
                    "copy the value from ref_slot[F6] into [F2]."
                )
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
        chain.append("S2: negative type is ghost: there was a historical value, but this mention is not a real activation for [F2].")
    elif neg_type_l == "void":
        chain.append("S2: negative type is void: there is no valid value for [F2] in this context.")
    else:
        chain.append("S2: negative type is [F3].")
    chain.append("S3: treat it as negative evidence: do NOT perform slot-specific operation for [F2].")

    res = ["", "(D) Result", "Result: none"]
    return "\n".join([demo_context_snippet] + facts + edges_blk + chain + res).strip()

# =========================
# Query CoT Scaffold（从 retrievl.py 拷贝）
# =========================
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


def parse_retrieved_id(rid: str) -> Tuple[str, int, str]:
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

    # 从 tasklabels 中解析当前 slot 对应的 label 以及 from_slot/from_turn
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

    neg_type = ""
    for ns in rec.get("neg_slots", []):
        if safe_str(ns.get("slot")) != slot:
            continue
        neg_type = safe_str(ns.get("neg_type")).lower()
        if not carry_val:
            carry_val = safe_str(ns.get("carry_value"))
        break

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
# 候选值构造（和 retrievl 当前版本保持一致）
# =========================
# =========================
# 候选值构造：用 slot_descriptions["value"]
# =========================
def build_value_candidates(
    slot: str,
    slot_desc_map: Dict[str, Dict[str, Any]],
    prev_v: str,
    cur_v: str,
) -> List[str]:
    """
    规则：
    1) 如果 slot_descriptions[slot]["value"] 是非空列表，且不含 "(Open Slot)"，
       视为封闭集合：直接把这些值作为 candidates，并追加 'none'。
    2) 如果 value 里含 "(Open Slot)"，或者根本没有配置 value，
       视为 Open Slot：给两个通用 Open Slot 示例 + 'none'，让模型从上下文里推。
    最后做一次去重和截断，长度不超过 MAX_CANDIDATES。
    """
    vals: List[str] = []

    desc = slot_desc_map.get(slot, {}) if isinstance(slot_desc_map, dict) else {}
    vlist = desc.get("value", [])
    if isinstance(vlist, list) and len(vlist) > 0:
        # 只要 ontology 中配置了 value，不管是封闭枚举还是 "(Open Slot)" 示例，都直接用
        for v in vlist:
            s = safe_str(v).strip()
            if s:
                vals.append(s)
    else:
        # 没有配置 value 时，退回默认的 Open Slot 模板
        vals.append("(Open Slot) e.g. '17:00'")
        vals.append("(Open Slot) e.g. '09:45'")


    # 补上 none
    if not any(safe_str(v).lower() == "none" for v in vals):
        vals.append("none")

    # 去重
    seen = set()
    dedup: List[str] = []
    for v in vals:
        s = safe_str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        dedup.append(s)

    # 截断到 MAX_CANDIDATES，并确保 none 不被裁掉
    if len(dedup) > MAX_CANDIDATES:
        dedup = dedup[:MAX_CANDIDATES]
        if any(safe_str(v).lower() == "none" for v in vals) and not any(
            safe_str(v).lower() == "none" for v in dedup
        ):
            dedup[-1] = "none"

    return dedup


# =========================
# Tasklabel 描述（和 retrievl 中一致）
# =========================
def tasklabel_description(label: str) -> str:
    label = safe_str(label).lower()
    if label == "constrain":
        return "The current turn provides a concrete constraint/value for the target slot."
    if label == "change":
        return "The current turn changes the previously set value of the target slot."
    if label == "dontcare":
        return "The current turn expresses no specific preference for the target slot (dontcare)."
    if label == "confirm":
        return "The current turn explicitly confirms/affirms the target slot value proposed by SYS."
    if label == "switch":
        return "The current turn switches to a new domain/task and sets constraints for the target slot."
    if label == "refer-clear":
        return "The current turn assigns/updates the target slot value by clearly referring to ref_slot value."
    if label == "refer-implicit":
        return "The current turn implicitly refers to another slot or context for the target slot value."
    if label == "none":
        return "The current turn does NOT activate the target slot (no slot-specific update)."
    return f"The current turn has label={label} for the target slot."

# =========================
# LLaMA3 模板渲染
# =========================
def render_llama3_user_prompt(tokenizer: AutoTokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_eos_token_ids(tokenizer: AutoTokenizer) -> List[int]:
    eos_ids = []
    for tok in ["<|eot_id|>", "<|eom_id|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)
        except Exception:
            pass
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    seen = set()
    out = []
    for x in eos_ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

# =========================
# 输出解析 + 归一化
# =========================
_RESULT_RE = re.compile(r"(?m)^\s*Result:\s*(.*?)\s*$")


def parse_result_value(gen_text: str) -> str:
    m = _RESULT_RE.search(gen_text)
    if m:
        val = safe_str(m.group(1)).strip()
        return val if val else "none"
    idx = gen_text.find("Result:")
    if idx >= 0:
        val = gen_text[idx + len("Result:"):].strip().splitlines()[0].strip()
        return val if val else "none"
    return "none"


def normalize_pred_value(label: str, value: str) -> str:
    label = safe_str(label).lower()
    v = safe_str(value).strip()
    if not v:
        v = "none"
    if label == "none":
        return "none"
    return v

# =========================
# 金标回合与状态累积（保留上一版逻辑）
# =========================
def extract_state_delta(turn: Dict[str, Any]) -> Dict[str, str]:
    st = turn.get("state", {})
    if isinstance(st, dict) and "slot_values" in st and isinstance(st["slot_values"], dict):
        d = st["slot_values"]
    elif isinstance(st, dict):
        d = st.get("slot_values", {}) if isinstance(st.get("slot_values", {}), dict) else {}
    else:
        d = {}
    return {safe_str(k): safe_str(v) for k, v in d.items()}


def apply_delta(full_state: Dict[str, str], delta: Dict[str, str]) -> Dict[str, str]:
    out = dict(full_state)
    for s, v in delta.items():
        v2 = safe_str(v).strip()
        if v2 == "" or v2.lower() == "none":
            out.pop(s, None)
        else:
            out[s] = v2
    return out


def compute_gold_turns_and_states(
    dialog: Dict[str, Any], target_domain: str, target_slots: List[str]
) -> Tuple[List[bool], List[Dict[str, str]]]:
    turns = dialog.get("turns", [])
    gold_full: Dict[str, str] = {}
    is_gold = [False] * len(turns)
    gold_full_target: List[Dict[str, str]] = []

    prev_gold_full = dict(gold_full)
    for t_idx, turn in enumerate(turns):
        delta = extract_state_delta(turn)
        gold_full = apply_delta(gold_full, delta)

        changed = []
        for s in target_slots:
            prev_v = safe_str(prev_gold_full.get(s, "none"))
            cur_v = safe_str(gold_full.get(s, "none"))
            if prev_v != cur_v:
                changed.append((s, prev_v, cur_v))

        if any(cur_v.strip().lower() != "none" and cur_v.strip() != "" for (_, _, cur_v) in changed):
            is_gold[t_idx] = True

        snap = {}
        for s in target_slots:
            v = gold_full.get(s, "none")
            v = safe_str(v).strip()
            snap[s] = v if v and v.lower() != "none" else "none"
        gold_full_target.append(snap)

        prev_gold_full = dict(gold_full)

    return is_gold, gold_full_target

# =========================
# 运行时结构与调度
# =========================
@dataclass
class DialogRuntime:
    dialog: Dict[str, Any]
    dial_id: str
    turns: List[Dict[str, Any]]
    gold_mask: List[bool]
    gold_states: List[Dict[str, str]]
    gold_turn_indices: List[int]

    cur_gold_ptr: int = 0
    state_pred: Dict[str, str] = field(default_factory=dict)
    pending_slots: List[Dict[str, Any]] = field(default_factory=list)
    last_filled_turn: int = -1

    turn_pred_state: List[Dict[str, str]] = field(default_factory=list)
    turn_slot_outputs: List[List[Dict[str, Any]]] = field(default_factory=list)
    turn_error_flag: List[bool] = field(default_factory=list)

    jga_hits: int = 0
    jga_total: int = 0

    def init_buffers(self):
        n = len(self.turns)
        self.turn_pred_state = [dict() for _ in range(n)]
        self.turn_slot_outputs = [[] for _ in range(n)]
        self.turn_error_flag = [False for _ in range(n)]


@dataclass
class SlotTask:
    rt: DialogRuntime
    turn_idx: int
    tk: Dict[str, Any]
    slot: str
    label: str
    prompt: str
    input_len: int = 0

# =========================
# 其他工具
# =========================
def make_deterministic_rng(*parts: str) -> random.Random:
    key = "::".join(parts)
    seed = zlib.adler32(key.encode("utf-8")) & 0xFFFFFFFF
    return random.Random(seed)


def state_snapshot_all_slots(state_active: Dict[str, str], target_slots: List[str]) -> Dict[str, str]:
    snap = {}
    for s in target_slots:
        v = safe_str(state_active.get(s, "none")).strip()
        snap[s] = v if v and v.lower() != "none" else "none"
    return snap

# =========================
# 构造单槽位任务的完整提示（不再 import retrievl）
# =========================
def build_prompt_for_task(
    rt: DialogRuntime,
    turn_idx: int,
    tk: Dict[str, Any],
    slot_desc_map: Dict[str, Dict[str, Any]],
    graph_index: Dict[Tuple[str, int], Dict[str, Any]],
    dialog_idx: Dict[Tuple[str, int], Dict[str, Any]],
    tokenizer: AutoTokenizer,
) -> str:
    turns = rt.turns
    turn = turns[turn_idx]
    sys_utt = safe_str(turn.get("system", "none"))
    usr_utt = safe_str(turn.get("user", ""))

    slot = safe_str(tk.get("slot")) or safe_str(tk.get("target_slot"))
    label = safe_str(tk.get("label"))
    if not label:
        label = parse_tasklabel(
            safe_str(tk.get("label-1")),
            safe_str(tk.get("label-2")),
        )
    label = safe_str(label).lower()

    # prev_value 用预测状态
    prev_val = rt.state_pred.get(slot, "")

    # 候选集合：根据 slot_descriptions["value"] 动态构造
    cand_vals = build_value_candidates(slot, slot_desc_map, prev_val, "")
    candidates_field = json.dumps(cand_vals, ensure_ascii=False)

    # slot 描述
    sdesc_raw = slot_desc_map.get(slot, {})
    sdesc = build_slot_desc_text(sdesc_raw)

    # tasklabel 描述
    tl_des = tasklabel_description(label)

    # 上下文（基础窗口）
    full_context = build_full_context(turns, turn_idx, MAX_HIST_TURNS)
    query_turn_snippet = f"[t{turn_idx}][SYS] {sys_utt} [t{turn_idx}][USR] {usr_utt}"

    # ref_slot/ref_value 可选
    ref_slot = safe_str(tk.get("ref_slot"))
    ref_value = safe_str(tk.get("ref_value"))

    # refer-clear 且 taklabels 没给 ref_slot 时，用预测状态做一次 heuristic 尝试
    if label == "refer-clear" and not ref_slot:
        _rslot, _rval = try_infer_ref_slot_from_state(rt.state_pred, slot, {})
        if _rslot:
            ref_slot, ref_value = _rslot, _rval

    # ---- 这里开始是新增的部分：依靠 state.slot_values 推断 ref_turn ----
    ref_turn: Optional[int] = None
    if label == "refer-clear" and ref_slot:
        ref_turn = infer_ref_turn_from_state(turns, turn_idx, ref_slot)

        # 如果 ref_turn 存在且不在当前滑窗内，就追加一个 [ref_tX] 片段
        if ref_turn is not None:
            start_idx = max(0, turn_idx - MAX_HIST_TURNS)
            # build_full_context 已经覆盖了 [start_idx..turn_idx]
            if 0 <= ref_turn < len(turns) and ref_turn < start_idx:
                ref_sys = safe_str(turns[ref_turn].get("system", "none"))
                ref_usr = safe_str(turns[ref_turn].get("user", ""))
                ref_tag = f"ref_t{ref_turn}"
                extra = f"[{ref_tag}][SYS] {ref_sys} [{ref_tag}][USR] {ref_usr}"
                full_context = full_context + " " + extra
    # ---- 新增部分到这里结束 ----


    # demos（从 graph_index 中抽）
    retrieved_ids = tk.get("retrieved_topk_ids", []) or []
    pos_edges_all: List[Dict[str, Any]] = []
    neg_edges_all: List[Dict[str, Any]] = []

    for rid in retrieved_ids:
        dial2, turn2, r_slot = parse_retrieved_id(safe_str(rid))
        if not dial2 or turn2 < 0:
            continue
        rec = graph_index.get((dial2, turn2))
        if not rec:
            continue
        ev = extract_slot_event_from_graph(rec, r_slot)
        r_label = safe_str(ev.get("label")).lower()
        r_prev = safe_str(ev.get("prev_val"))
        r_cur = safe_str(ev.get("curr_val"))
        r_edges = ev.get("edges", [])
        r_neg_type = safe_str(ev.get("neg_type")).lower()
        r_carry = safe_str(ev.get("carry_value"))
        r_from_slot = safe_str(ev.get("from_slot", ""))
        r_from_turn = ev.get("from_turn", None)

        if r_label == "none":
            neg_edges_all.append({
                "slot": r_slot,
                "neg_type": r_neg_type,
                "carry_value": r_carry,
                "edges": r_edges,
                "dial_id": dial2,
                "turn_id": turn2,
            })
        else:
            pos_edges_all.append({
                "slot": r_slot,
                "label": r_label,
                "prev_val": r_prev,
                "curr_val": r_cur,
                "edges": r_edges,
                "dial_id": dial2,
                "turn_id": turn2,
                "from_slot": r_from_slot,
                "from_turn": r_from_turn,
            })
        # --- 新增：对当前 graph 记录的 neg_slots 展开成负样本事件 ---
        for ns in rec.get("neg_slots", []):
            neg_slot = safe_str(ns.get("slot"))
            if not neg_slot:
                continue

            # 用 extractor 再跑一遍，确保 edges 只包含这个 neg_slot 的子图
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
        # --- 新增结束 ---


    pos_blocks: List[str] = []
    for pe in pos_edges_all[:POS_DEMO_K]:
        demo_dial = safe_str(pe.get("dial_id", rt.dial_id))
        demo_turn = int(pe.get("turn_id", turn_idx))

        # 两轮上下文：t=0 -> t0+t1; t>0 -> t-1+t
        demo_context_snippet, used_turns = build_demo_context_two_turns(
            demo_dial,
            demo_turn,
            dialog_idx,
            query_turn_snippet,
        )
        used_set = set(used_turns)

        demo_slot = safe_str(pe.get("slot", slot))
        demo_label = safe_str(pe.get("label")).lower()
        demo_ref_slot = safe_str(pe.get("from_slot", ""))
        demo_ref_turn: Optional[int] = None
        if "from_turn" in pe and pe.get("from_turn") is not None:
            try:
                demo_ref_turn = int(pe.get("from_turn"))
            except Exception:
                demo_ref_turn = None

        # 如果是 refer-clear 且 ref_turn 存在且不在前面两轮里，则追加 [ref_tX] 片段
        if demo_label == "refer-clear" and demo_ref_slot and demo_ref_turn is not None:
            if demo_ref_turn not in used_set:
                dt_ref = dialog_idx.get((demo_dial, demo_ref_turn))
                if dt_ref is not None:
                    ref_tag = f"ref_t{demo_ref_turn}"
                    ref_sys = safe_str(dt_ref.get("system", "none"))
                    ref_usr = safe_str(dt_ref.get("user", ""))
                    demo_context_snippet = (
                        demo_context_snippet
                        + f" [{ref_tag}][SYS] {ref_sys} [{ref_tag}][USR] {ref_usr}"
                    )

        if demo_label == "none":
            pos_blocks.append(
                build_none_pos_demo_cot(
                    demo_context_snippet,
                    demo_slot,
                    pe.get("edges", []),
                )
            )
        else:
            pos_blocks.append(
                build_pos_demo_cot(
                    demo_context_snippet,
                    demo_slot,
                    demo_label,
                    safe_str(pe.get("prev_val")),
                    safe_str(pe.get("curr_val")),
                    tasklabel_description(demo_label),
                    pe.get("edges", []),
                    ref_slot=demo_ref_slot,
                    ref_value="",
                    ref_turn=demo_ref_turn,
                )
            )

    neg_block = ""
    rng = make_deterministic_rng(rt.dial_id, str(turn_idx), slot, "NEG", str(RNG_SEED))
    if neg_edges_all and rng.random() < NEG_DEMO_PROB:
        ne = rng.choice(neg_edges_all)
        neg_dial = safe_str(ne.get("dial_id", rt.dial_id))
        neg_turn = int(ne.get("turn_id", turn_idx))

        demo_context_snippet, _ = build_demo_context_two_turns(
            neg_dial,
            neg_turn,
            dialog_idx,
            query_turn_snippet,
        )

        neg_slot = safe_str(ne.get("slot", slot))
        neg_type = safe_str(ne.get("neg_type", "void")) or "void"
        carry_value = safe_str(ne.get("carry_value", "none")) or "none"
        neg_block = build_neg_demo(
            demo_context_snippet,
            neg_slot,
            neg_type=neg_type,
            carry_value=carry_value,
            edges=ne.get("edges", []),
        )


    query_scaffold = build_query_cot_scaffold(
        query_turn_snippet=query_turn_snippet,
        slot=slot,
        label=label,
        tasklabel_des=tl_des,
        prev_value=prev_val,
        candidates=cand_vals,
        ref_slot=ref_slot,
        ref_value=ref_value,
        ref_turn=ref_turn,
    )


    input_parts: List[str] = []
    input_parts.append("[QUERY]")
    input_parts.append("context:\n" + full_context)
    input_parts.append(f"slot: {slot}")
    input_parts.append(f"slot_desc: {sdesc}")
    input_parts.append(f"tasklabel: {label}")
    input_parts.append(f"tasklabel_des: {tl_des}")
    if prev_val:
        input_parts.append(f"prev_state: {prev_val}")
    if cand_vals:
        input_parts.append(f"value_candidates: {candidates_field}")
    if ref_slot:
        input_parts.append(f"ref_slot: {ref_slot}")
    if ref_value:
        input_parts.append(f"ref_value: {ref_value}")
    input_parts.append("\n[QUERY-COT-SCAFFOLD]\n" + query_scaffold)

    if len(pos_blocks) > 0:
        input_parts.append("\n[POSITIVE DEMO 1]")
        input_parts.append(pos_blocks[0])
    if len(pos_blocks) > 1:
        input_parts.append("\n[POSITIVE DEMO 2]")
        input_parts.append(pos_blocks[1])
    if neg_block:
        input_parts.append("\n[NEGATIVE DEMO]")
        input_parts.append(neg_block)

    input_text = "\n".join(input_parts).strip()
    user_content = (instruction_text + "\n\n" + input_text).strip()
    return render_llama3_user_prompt(tokenizer, user_content)

# =========================
# 主流程
# =========================
def main() -> None:
    # slot 描述（type/question/contras），用于 slot_desc 文本
    slot_desc_map = load_slot_descriptions(SLOT_DESC_PATH)
    if not isinstance(slot_desc_map, dict):
        raise RuntimeError(f"SLOT_DESC_PATH must be a JSON dict: {SLOT_DESC_PATH}")

    target_slots = [s for s in slot_desc_map.keys() if safe_str(s).startswith(TARGET_DOMAIN + "-")]
    target_slots = sorted(target_slots)

    # graph turn + demo dialog
    graph_turns = load_jsonl(GRAPH_STORE_JSONL_PATH)
    dialog_store = load_jsonl(DEMO_DIALOG_STORE_PATH)
    graph_index = build_graph_event_index(graph_turns)

    # 将对话级 JSONL 展开成 per-turn 索引：(dial_id, turn_id) -> {"system": ..., "user": ...}
    dialog_idx: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for rec in dialog_store:
        d = safe_str(rec.get("dial_id"))
        turns = rec.get("turns", [])
        for t_rec in turns:
            t = int(t_rec.get("turn_id", 0))
            dialog_idx[(d, t)] = {
                "system": safe_str(t_rec.get("system", "none")),
                "user": safe_str(t_rec.get("user", "")),
            }


    # query dialogs
    query_data = load_json_or_jsonl(QUERY_DIALOGS_PATH)
    if not isinstance(query_data, list):
        raise RuntimeError("QUERY_DIALOGS_PATH must be a list of dialogs (JSON array) or JSONL lines.")

    dialogs = []
    for dial in query_data:
        domains = dial.get("domains", [])
        if isinstance(domains, list) and TARGET_DOMAIN in [safe_str(x) for x in domains]:
            dialogs.append(dial)

    # 模型
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16 if USE_BF16 else None,
        device_map=DEVICE_MAP,
    )
    if LORA_ADAPTER_PATH and LORA_ADAPTER_PATH.strip():
        if PeftModel is None:
            raise RuntimeError("peft is not available but LORA_ADAPTER_PATH is set.")
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    model.eval()

    eos_ids = build_eos_token_ids(tokenizer)

    # 构造每个对话的运行时
    runtimes: List[DialogRuntime] = []
    for dial in dialogs:
        dial_id = safe_str(dial.get("dial_id"))
        turns = dial.get("turns", [])
        gold_mask, gold_states = compute_gold_turns_and_states(dial, TARGET_DOMAIN, target_slots)
        gold_turn_indices = [i for i, g in enumerate(gold_mask) if g]

        rt = DialogRuntime(
            dialog=dial,
            dial_id=dial_id,
            turns=turns,
            gold_mask=gold_mask,
            gold_states=gold_states,
            gold_turn_indices=gold_turn_indices,
        )
        rt.init_buffers()
        runtimes.append(rt)
        total_gold_turns = sum(len(rt.gold_turn_indices) for rt in runtimes)
        pbar = tqdm(total=total_gold_turns, desc=f"{TARGET_DOMAIN} DST inference", ncols=100)

        # helper: prepare current gold turn slots
    def prepare_current_gold_turn(rt: DialogRuntime, pbar=None) -> bool:
        if rt.cur_gold_ptr >= len(rt.gold_turn_indices):
            # 这个对话所有 gold_turn 都处理完了，补全后续 turn 的状态快照
            for i in range(rt.last_filled_turn + 1, len(rt.turns)):
                rt.turn_pred_state[i] = state_snapshot_all_slots(rt.state_pred, target_slots)
            rt.last_filled_turn = len(rt.turns) - 1
            return False

        turn_idx = rt.gold_turn_indices[rt.cur_gold_ptr]

        # 填充中间非 gold_turn 的快照
        for i in range(rt.last_filled_turn + 1, turn_idx):
            rt.turn_pred_state[i] = state_snapshot_all_slots(rt.state_pred, target_slots)
        rt.last_filled_turn = turn_idx - 1

        turn = rt.turns[turn_idx]
        pred_tk = turn.get("pred_taklabels", [])
        if not isinstance(pred_tk, list):
            pred_tk = []

        pred_tk_td = [x for x in pred_tk if safe_str(x.get("slot")).startswith(TARGET_DOMAIN + "-")]

        if len(pred_tk_td) == 0:
            # pred_taklabels 为空：记错，清空状态，直接评估这个 gold_turn
            rt.turn_error_flag[turn_idx] = True
            rt.state_pred.clear()

            pred_snap = state_snapshot_all_slots(rt.state_pred, target_slots)
            gold_snap = rt.gold_states[turn_idx]
            hit = 1 if pred_snap == gold_snap else 0
            rt.jga_hits += hit
            rt.jga_total += 1
            if pbar is not None:
                pbar.update(1)

            rt.turn_pred_state[turn_idx] = pred_snap
            rt.last_filled_turn = turn_idx

            rt.cur_gold_ptr += 1
            rt.pending_slots = []
            return True

        rt.pending_slots = list(pred_tk_td)
        return True


    active: List[DialogRuntime] = []
    for rt in runtimes:
        if prepare_current_gold_turn(rt, pbar):
            active.append(rt)


    # 主循环：按 dial_id 组成 batch
    while True:
        batch_tasks: List[SlotTask] = []
        for rt in active:
            if not rt.pending_slots:
                continue
            tk = rt.pending_slots[0]
            slot = safe_str(tk.get("slot"))
            label = safe_str(tk.get("label")).lower()
            if not label:
                label = parse_tasklabel(
                    safe_str(tk.get("label-1")),
                    safe_str(tk.get("label-2")),
                )
                label = safe_str(label).lower()

            prompt = build_prompt_for_task(
                rt=rt,
                turn_idx=rt.gold_turn_indices[rt.cur_gold_ptr],
                tk=tk,
                slot_desc_map=slot_desc_map,
                graph_index=graph_index,
                dialog_idx=dialog_idx,
                tokenizer=tokenizer,
            )
            batch_tasks.append(
                SlotTask(
                    rt=rt,
                    turn_idx=rt.gold_turn_indices[rt.cur_gold_ptr],
                    tk=tk,
                    slot=slot,
                    label=label,
                    prompt=prompt,
                )
            )
            if len(batch_tasks) >= BATCH_SIZE:
                break

        if not batch_tasks:
            new_active = []
            progressed = False
            for rt in active:
                if rt.pending_slots:
                    new_active.append(rt)
                    continue
                # prepare next gold turn
                if prepare_current_gold_turn(rt, pbar):
                    new_active.append(rt)
                    progressed = True

            active = new_active
            if not progressed:
                break
            continue

        prompts = [t.prompt for t in batch_tasks]
        enc = tokenizer(prompts, add_special_tokens=False, return_tensors=None)
        input_ids_list = []
        lens = []
        max_ctx = getattr(model.config, "max_position_embeddings", 8192)
        max_inp = max_ctx - MAX_NEW_TOKENS - 8
        for ids in enc["input_ids"]:
            if len(ids) > max_inp:
                ids = ids[-max_inp:]
            tens = torch.tensor(ids, dtype=torch.long)
            input_ids_list.append(tens)
            lens.append(tens.numel())
        for i, task in enumerate(batch_tasks):
            input_text = tokenizer.decode(input_ids_list[i], skip_special_tokens=False)
            tqdm.write(
                f"[INPUT] dial={task.rt.dial_id} turn={task.turn_idx} "
                f"slot={task.slot} label={task.label}\n"
                f"{input_text}"
            )
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id
        )
        attention_mask = torch.zeros_like(input_ids)
        for i, L in enumerate(lens):
            attention_mask[i, :L] = 1

        device = model.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
            )

        for i, task in enumerate(batch_tasks):
            out_ids = gen[i, lens[i]:]
            gen_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            pred_val = parse_result_value(gen_text)
            pred_val = normalize_pred_value(task.label, pred_val)
            # 实时打印模型输出
            tqdm.write(
                f"[OUTPUT] dial={task.rt.dial_id} turn={task.turn_idx} "
                f"slot={task.slot} label={task.label} -> {pred_val}\n"
                f"{gen_text.strip()}"
            )

            task.rt.pending_slots.pop(0)

            if pred_val.strip().lower() == "none" or pred_val.strip() == "":
                task.rt.state_pred.pop(task.slot, None)
            else:
                task.rt.state_pred[task.slot] = pred_val.strip()

            task.rt.turn_slot_outputs[task.turn_idx].append(
                {
                    "slot": task.slot,
                    "label": task.label,
                    "pred": pred_val,
                    "raw_gen": gen_text.strip(),
                }
            )

            if not task.rt.pending_slots:
                pred_snap = state_snapshot_all_slots(task.rt.state_pred, target_slots)
                gold_snap = task.rt.gold_states[task.turn_idx]
                hit = 1 if pred_snap == gold_snap else 0
                task.rt.jga_hits += hit
                task.rt.jga_total += 1
                if pbar is not None:
                    pbar.update(1)

                task.rt.turn_pred_state[task.turn_idx] = pred_snap
                task.rt.last_filled_turn = task.turn_idx

                task.rt.cur_gold_ptr += 1
                prepare_current_gold_turn(task.rt, pbar)


        new_active = []
        for rt in active:
            if rt.cur_gold_ptr < len(rt.gold_turn_indices):
                new_active.append(rt)
            else:
                # ensure tail snapshots filled
                prepare_current_gold_turn(rt, pbar)
        active = new_active


        if not active:
            break
    pbar.close()
    # 写回输出 + 统计 JGA
    out_dialogs = []
    total_hits = 0
    total_cnt = 0

    rt_map = {rt.dial_id: rt for rt in runtimes}

    for dial in dialogs:
        dial_id = safe_str(dial.get("dial_id"))
        rt = rt_map.get(dial_id)
        if rt is None:
            out_dialogs.append(dial)
            continue

        turns = dial.get("turns", [])
        for i, turn in enumerate(turns):
            turn["pred_state_target"] = rt.turn_pred_state[i]
            turn["pred_outputs"] = rt.turn_slot_outputs[i]
            turn["gold_turn_for_eval"] = bool(rt.gold_mask[i])
            turn["eval_error_flag"] = bool(rt.turn_error_flag[i])
        out_dialogs.append(dial)

        total_hits += rt.jga_hits
        total_cnt += rt.jga_total

    write_json(OUTPUT_DIALOGS_PATH, out_dialogs)
    jga = (total_hits / total_cnt) if total_cnt > 0 else 0.0
    print(f"[DONE] dialogs_in_scope={len(dialogs)} gold_turns={total_cnt} JGA={jga:.6f}")
    print(f"[OUT] {OUTPUT_DIALOGS_PATH}")


if __name__ == "__main__":
    main()
