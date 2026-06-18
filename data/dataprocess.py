#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct-overwrite script

只做两件事：
1) 为每个对话的每一轮补上 turn_id（并且在该轮对象的**最前面**写出 turn_id）
2) 为每轮对话逐槽位打一级标签(L1)与二级标签(confirm)
   - 仅针对“发生动作”的槽位（新增/改值/变为dontcare/新领域出现）
   - L1 ∈ {constrain, change, switch, dontcare, notask}
   - switch（按你的要求）：
       只有在“之前轮次已经出现过其他领域”的前提下，
       本轮又引入了“新的领域”，该新领域里本轮新增的槽位标记为 switch；
       若这是整段对话第一次出现领域，则不记为 switch（仍按 constrain/dontcare 判）。
   - confirm（仅在 L1 ∈ {constrain, change, switch} 时）：
       当前值不在用户话语中，且出现在系统话语中；并且当前值不是 "yes" 且不属于 none 类值
3) 若某轮没有任何槽位发生动作，则在该轮增加 **turn 级别**字段：tasklabel = "notask"

输出：
  <script_dir>/processed_l1_confirm/train_processed.json
  <script_dir>/processed_l1_confirm/dev_processed.json
  <script_dir>/processed_l1_confirm/test_processed.json

注意：为保证 turn_id 显示在每轮对象最前面，写文件时使用 OrderedDict 控制键顺序。
"""

import os
import re
import json
from typing import Dict, List, Any
from collections import OrderedDict

# ======== 路径（按需修改） ========
TRAIN_PATH = "/home/fzus/zzp/data/train_dials.json"
DEV_PATH   = "/home/fzus/zzp/data/dev_dials.json"
TEST_PATH  = "/home/fzus/zzp/data/test_dials.json"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "processed_l1_confirm")

# ======== 域过滤（与 MultiWOZ 一致） ========
ALLOWED_DOMAINS = {"restaurant","attraction","hotel","train","taxi"}
BLOCKED_DOMAINS = {"hospital","police"}  # 直接剔除包含这些域的对话

# ======== 归一化辅助 ========
NONE_TOKENS = {"", "none", "notmentioned", "not mentioned", "unknown", "n/a", "na", "null", "nil"}

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # ensure_ascii=False + insertion-ordered dicts -> desired key order in output
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _norm_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "")).strip().lower()

def _norm_value(x: str) -> str:
    # 值比较/包含匹配用，去空白与常见分隔符
    return re.sub(r"[\s\-\_:]", "", _norm_text(x))

def _is_none(v: str) -> bool:
    v1, v2 = _norm_text(v), _norm_value(v)
    return (v1 in NONE_TOKENS) or (v2 in NONE_TOKENS)

def _slot_domain(slot: str) -> str:
    return slot.split("-", 1)[0] if slot else ""

def _value_in_text(val: str, text: str) -> bool:
    """数值用严格词边界，其他走子串或归一化包含。"""
    if not val: return False
    t = _norm_text(text or "")
    v = _norm_text(val)
    if not t or not v:
        return False
    if v.isdigit():
        pat = rf"(?<!\d)\b{re.escape(v)}\b(?!\d)"
        return bool(re.search(pat, t))
    if v in t:
        return True
    return _norm_value(val) in _norm_value(text or "")

# ======== L1 标注 ========
def _diff_states(prev: Dict[str,str], curr: Dict[str,str]) -> Dict[str,Any]:
    """
    计算前后状态差分（'domain-slot' -> value）。
    返回：
      added:    [(slot, curr_val), ...]
      changed:  [(slot, (prev_val, curr_val)), ...]
      dontcare: [(slot, curr_val), ...]  # 本轮值为 dontcare
      new_domains: set(本轮新增出现的 domain)
      domain_added_slots: {domain: [slot,...]}  # 本轮该 domain 新增的槽位
      had_domains_before: bool  # 之前轮次是否已出现过任意 domain（用于 switch 判定）
    """
    added, changed, dontcare = [], [], []
    prev = prev or {}; curr = curr or {}

    prev_domains = {_slot_domain(k) for k in prev}
    curr_domains = {_slot_domain(k) for k in curr}
    new_domains = curr_domains - prev_domains

    for k, v in curr.items():
        if k not in prev:
            added.append((k, v))
            if _norm_value(v) == "dontcare":
                dontcare.append((k, v))
        else:
            pv = prev[k]
            if _norm_text(pv) != _norm_text(v):
                changed.append((k, (pv, v)))
                if _norm_value(v) == "dontcare":
                    dontcare.append((k, v))

    domain_added_slots: Dict[str, List[str]] = {}
    for k, _ in added:
        d = _slot_domain(k)
        domain_added_slots.setdefault(d, []).append(k)

    return {
        "added": added,
        "changed": changed,
        "dontcare": dontcare,
        "new_domains": new_domains,
        "domain_added_slots": domain_added_slots,
        "had_domains_before": len(prev_domains) > 0,  # 关键：之前是否已有领域
    }

def _pick_l1_label(slot: str, curr_val: str, diff: Dict[str,Any]) -> str:
    """
    L1 优先级：switch > dontcare > constrain > change > notask
    switch 判定需满足：
      - had_domains_before == True（之前轮次已经出现过领域）
      - 本轮出现了新领域（slot 的 domain ∈ new_domains）
      - 该 slot 属于本轮该新领域的新增槽位（在 domain_added_slots 中）
    """
    dom = _slot_domain(slot)
    is_switch = (
        diff.get("had_domains_before", False) and
        (dom in diff["new_domains"]) and
        (slot in diff["domain_added_slots"].get(dom, []))
    )
    is_dontcare = (_norm_value(curr_val) == "dontcare")
    is_added    = any(s == slot for s,_ in diff["added"])
    is_changed  = any(s == slot for s,_ in diff["changed"])

    if is_switch:   return "switch"
    if is_dontcare: return "dontcare"
    if is_added:    return "constrain"
    if is_changed:  return "change"
    return "notask"

def _collect_taklabels(prev_state: Dict[str,str], curr_state: Dict[str,str],
                       sys_txt: str, usr_txt: str) -> List[Dict[str,Any]]:
    """
    产出本轮 taklabels，并计算 confirm：
      - 候选槽位 = added ∪ changed ∪ dontcare ∪ 新领域中的新增槽位
      - confirm 规则：仅在 L1 ∈ {constrain, change, switch} 时，
                      当前值不在用户话语、且出现在系统话语，
                      且当前值不为 "yes" 或 none 类值
    """
    diff = _diff_states(prev_state, curr_state)

    cand_slots = set()
    cand_slots.update(s for s,_ in diff["added"])
    cand_slots.update(s for s,_ in diff["changed"])
    cand_slots.update(s for s,_ in diff["dontcare"])
    for d, slots in diff["domain_added_slots"].items():
        if d in diff["new_domains"]:
            cand_slots.update(slots)

    taklabels: List[Dict[str,Any]] = []
    for slot in sorted(cand_slots, key=lambda s: (_slot_domain(s), s)):
        pv = prev_state.get(slot, "")
        cv = curr_state.get(slot, "")

        l1 = _pick_l1_label(slot, cv, diff)

        # ---- L2(confirm) ----
        l2 = ""
        if l1 in {"constrain", "change", "switch"}:
            if not _is_none(cv) and _norm_text(cv) != "yes":
                in_user   = _value_in_text(cv, usr_txt)
                in_system = _value_in_text(cv, sys_txt)
                if (not in_user) and in_system:
                    l2 = "confirm"

        taklabels.append({
            "slot": slot,
            "label-1": l1,
            "label-2": l2,
            "prev_val": pv,
            "curr_val": cv
        })

    return taklabels

# ======== 对话处理 ========
def _dialogue_domains_ok(d: Dict) -> bool:
    domains = set([_norm_text(x) for x in d.get("domains", [])])
    if domains & BLOCKED_DOMAINS:
        return False
    unknown = domains - (ALLOWED_DOMAINS | BLOCKED_DOMAINS)
    return not bool(unknown)

def _order_turn_with_turnid_first(turn_src: Dict[str, Any], turn_id: int,
                                  taklabels: List[Dict[str, Any]]) -> OrderedDict:
    """
    生成一个 OrderedDict，使 'turn_id' 作为该轮对象的第一个键。
    其他键保持原始顺序（略过旧的 turn_id / taklabels / tasklabel），
    然后把 taklabels（以及可能的 tasklabel）附在后面。
    """
    ordered = OrderedDict()
    ordered["turn_id"] = turn_id

    # 原始顺序写回（跳过可能存在的旧键）
    for k, v in turn_src.items():
        if k in ("turn_id", "taklabels", "tasklabel"):
            continue
        ordered[k] = v

    # 追加 taklabels（必有）
    ordered["taklabels"] = taklabels

    # 如果源 turn 上已有 tasklabel（比如 notask），也放在最后
    if "tasklabel" in turn_src:
        ordered["tasklabel"] = turn_src["tasklabel"]

    return ordered

def process_dialogues(dialogs: List[Dict]) -> List[Dict]:
    """为每个 turn 添加 turn_id（置于首位），并生成 taklabels（L1 + confirm）；若无动作则 turn 级别 tasklabel=notask。"""
    out = []
    for d in dialogs:
        if not _dialogue_domains_ok(d):
            continue
        turns = d.get("turns", [])
        prev_state: Dict[str,str] = {}
        new_turns: List[OrderedDict] = []
        for i, t in enumerate(turns):
            curr_state = (t.get("state") or {}).get("slot_values") or {}
            sys_txt = t.get("system", "") or ""
            usr_txt = t.get("user", "") or ""

            taklabels = _collect_taklabels(prev_state, curr_state, sys_txt, usr_txt)

            # 先把 notask（turn 级）写到临时 turn 上，再统一重新排版
            temp_turn = dict(t)  # 浅拷贝，避免污染原对象
            if len(taklabels) == 0:
                temp_turn["tasklabel"] = "notask"
            else:
                temp_turn.pop("tasklabel", None)

            # 重新组织键顺序：turn_id 放在最前
            ordered_turn = _order_turn_with_turnid_first(temp_turn, i, taklabels)
            new_turns.append(ordered_turn)

            prev_state = curr_state

        # 覆盖为新 turns（其中每个 turn 都是 OrderedDict，turn_id 在首位）
        d = dict(d)  # 拷贝，保持安全
        d["turns"] = new_turns
        out.append(d)
    return out

# ======== 主流程 ========
def main():
    print("[1/3] Loading train/dev/test ...")
    train_raw = load_json(TRAIN_PATH)
    dev_raw   = load_json(DEV_PATH)
    test_raw  = load_json(TEST_PATH)

    print("[2/3] Labeling (turn_id first + L1 + confirm + turn-level notask) ...")
    train_proc = process_dialogues(train_raw)
    dev_proc   = process_dialogues(dev_raw)
    test_proc  = process_dialogues(test_raw)

    print("[3/3] Saving ...")
    os.makedirs(OUT_DIR, exist_ok=True)
    save_json(os.path.join(OUT_DIR, "train_processed.json"), train_proc)
    save_json(os.path.join(OUT_DIR, "dev_processed.json"),   dev_proc)
    save_json(os.path.join(OUT_DIR, "test_processed.json"),  test_proc)
    print(f"✅ Done. Files saved in:\n  {OUT_DIR}")

if __name__ == "__main__":
    main()
