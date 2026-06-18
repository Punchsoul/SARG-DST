# -*- coding: utf-8 -*-
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Set, Optional

INPUT_JSON = "/home/fzus/zzp2/ds_loo/restaurant/test_restaurant_pred.json"

# 你已保证 pred_taklabels 只含目标域槽位；如果仍想强制过滤域，填 "hotel"；否则 None 自动从 pred_taklabels 推断
TARGET_DOMAIN: Optional[str] = "restaurant"

# 只评估 pred_taklabels 非空的回合
PRED_TAKLABELS_FIELD = "pred_taklabels"

# 是否跳过 skipped==true 的回合（建议 True，保持与你数据语义一致）
SKIP_SKIPPED_TURNS = True


def iter_dialogs(obj: Any):
    if isinstance(obj, list):
        for i, d in enumerate(obj):
            yield d.get("dial_id", str(i)), d
        return
    if isinstance(obj, dict):
        if "dialogs" in obj and isinstance(obj["dialogs"], list):
            for d in obj["dialogs"]:
                yield d.get("dial_id", ""), d
            return
        if "data" in obj and isinstance(obj["data"], list):
            for d in obj["data"]:
                yield d.get("dial_id", ""), d
            return
        if all(isinstance(v, dict) for v in obj.values()):
            for k, v in obj.items():
                yield k, v
            return
    raise ValueError(f"Unsupported JSON structure: {type(obj)}")


def canonicalize_slot(slot: str) -> str:
    s = (slot or "").strip()
    s = re.sub(r"\s+", " ", s)
    if "-" not in s:
        return s
    dom, rest = s.split("-", 1)
    rest = rest.replace("-", " ")
    rest = re.sub(r"\s+", " ", rest).strip()
    return f"{dom}-{rest}"


def norm_label(x: Any) -> str:
    return ("" if x is None else str(x)).strip().lower()


def infer_domain_from_pred_taklabels(pred_taklabels: List[Dict[str, Any]]) -> Optional[str]:
    # 从第一个 slot 推断域：hotel-xxx -> hotel
    for it in pred_taklabels:
        if not isinstance(it, dict):
            continue
        s = canonicalize_slot(it.get("slot", ""))
        if "-" in s:
            return s.split("-", 1)[0]
    return None


def pred_activated_slots_from_pred_taklabels(pred_taklabels: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for it in pred_taklabels:
        if not isinstance(it, dict):
            continue
        slot = canonicalize_slot(it.get("slot", ""))
        lbl = norm_label(it.get("label-1", ""))
        if not slot:
            continue
        if lbl in ("", "none", "notask"):
            continue
        out.add(slot)
    return out


def gold_activated_slots_by_state_delta(prev_sv: Dict[str, Any], cur_sv: Dict[str, Any], domain: Optional[str]) -> Set[str]:
    prev_sv = prev_sv or {}
    cur_sv = cur_sv or {}
    out: Set[str] = set()

    if domain:
        pref = domain + "-"
        keys = {k for k in prev_sv.keys() if str(k).startswith(pref)} | {k for k in cur_sv.keys() if str(k).startswith(pref)}
    else:
        keys = set(prev_sv.keys()) | set(cur_sv.keys())

    for k in keys:
        pv = prev_sv.get(k, None)
        cv = cur_sv.get(k, None)
        if pv is None and cv is None:
            continue
        if str(pv) != str(cv):
            out.add(canonicalize_slot(k))
    return out


def main():
    assert os.path.exists(INPUT_JSON), f"File not found: {INPUT_JSON}"
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        obj = json.load(f)

    evaluated_turns = 0
    correct_turns = 0
    skipped_turns = 0
    ignored_no_pred_taklabels = 0

    TP = FP = FN = 0
    err_size_cnt = Counter()
    wrong_examples = []
    MAX_KEEP = 200

    for dial_id, dial in iter_dialogs(obj):
        turns = dial.get("turns", []) or []
        prev_sv = {}

        for t in turns:
            state = t.get("state", {}) or {}
            cur_sv = state.get("slot_values", {}) or {}

            if SKIP_SKIPPED_TURNS and bool(t.get("skipped", False)):
                skipped_turns += 1
                prev_sv = cur_sv
                continue

            pred_tak = t.get(PRED_TAKLABELS_FIELD, None)
            if not isinstance(pred_tak, list) or len(pred_tak) == 0:
                ignored_no_pred_taklabels += 1
                prev_sv = cur_sv
                continue

            # domain control
            dom = TARGET_DOMAIN if TARGET_DOMAIN else infer_domain_from_pred_taklabels(pred_tak)

            pred_set = pred_activated_slots_from_pred_taklabels(pred_tak)
            gold_set = gold_activated_slots_by_state_delta(prev_sv, cur_sv, dom)

            evaluated_turns += 1

            inter = gold_set & pred_set
            tp = len(inter)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            TP += tp
            FP += fp
            FN += fn

            if fp == 0 and fn == 0:
                correct_turns += 1
            else:
                err_size_cnt[(fn, fp)] += 1
                if len(wrong_examples) < MAX_KEEP:
                    wrong_examples.append({
                        "dial_id": dial_id,
                        "turn_id": t.get("turn_id", None),
                        "domain_used": dom,
                        "gold": sorted(gold_set),
                        "pred": sorted(pred_set),
                        "missing": sorted(gold_set - pred_set),
                        "extra": sorted(pred_set - gold_set),
                    })

            prev_sv = cur_sv

    jga = correct_turns / evaluated_turns if evaluated_turns else 0.0
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) else 0.0

    summary = {
        "input": INPUT_JSON,
        "pred_taklabels_field": PRED_TAKLABELS_FIELD,
        "target_domain": TARGET_DOMAIN if TARGET_DOMAIN else "AUTO_FROM_PRED_TAKLABELS",
        "evaluated_turns": evaluated_turns,
        "correct_turns": correct_turns,
        "activation_JGA": jga,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "P": P,
        "R": R,
        "F1": F1,
        "skipped_turns_not_counted": skipped_turns,
        "ignored_turns_pred_taklabels_empty": ignored_no_pred_taklabels,
        "top_error_sizes": [{"missing": k[0], "extra": k[1], "count": c} for k, c in err_size_cnt.most_common(10)],
        "wrong_examples_kept": len(wrong_examples),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_dir = os.path.dirname(INPUT_JSON) or "."
    out_path = os.path.join(out_dir, "activation_jga_only_pred_taklabels_turns_errors_sample.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wrong_examples, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
