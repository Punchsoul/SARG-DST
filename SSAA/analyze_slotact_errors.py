# -*- coding: utf-8 -*-
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Any, List, Set


# =========================
# 只改这里：默认输入/输出/领域
# =========================
DEFAULT_IN_PATH = "/home/fzus/zzp2/ds_loo/restaurant/restaurant_oytcome.json"
DEFAULT_OUT_DIR = "/home/fzus/zzp2/ds_loo/restaurant"
DEFAULT_TARGET_DOMAIN = "restaurant"
DEFAULT_TOPK = 30
DEFAULT_SKIP_EMPTY_GOLD = True


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_csv(path: str, header: List[str], rows: List[List[Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def slot_to_domain(slot: str) -> str:
    if not isinstance(slot, str):
        return ""
    return slot.split("-", 1)[0] if "-" in slot else ""


def get_gold_slots(turn: Dict[str, Any], target_domain: str) -> Set[str]:
    gold = set()
    for obj in turn.get("taklabels", []) or []:
        if isinstance(obj, dict) and "slot" in obj:
            s = str(obj["slot"])
            if slot_to_domain(s) == target_domain:
                gold.add(s)
    return gold


def get_pred_slots(turn: Dict[str, Any], target_domain: str) -> Set[str]:
    """
    ✅ 关键修正：你的原始数据预测字段就是 turn["pred"]（list[str]）
    原来读 pred_activated_slots 会导致 pred 读不到 -> 全空 -> 指标全 0
    """
    pred = set()
    for s in turn.get("pred", []) or []:   # ← 就改这一处：pred_activated_slots -> pred
        s = str(s)
        if slot_to_domain(s) == target_domain:
            pred.add(s)
    return pred


def build_detail_map(turn: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    turn 里可能叫：
      - pred_slotact_detail
      - slotact_raw
    这里统一抽成: slot -> {raw_gen, pred, ...}
    """
    detail = {}
    candidates = None
    if isinstance(turn.get("pred_slotact_detail", None), list):
        candidates = turn["pred_slotact_detail"]
    elif isinstance(turn.get("slotact_raw", None), list):
        candidates = turn["slotact_raw"]

    if candidates:
        for item in candidates:
            if isinstance(item, dict) and "slot" in item:
                detail[str(item["slot"])] = item
    return detail


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    # ✅ 改动：把 required=True 变成 default=...，方便你只改文件顶部
    ap.add_argument("--in_path", type=str, default=DEFAULT_IN_PATH, help="writeback后的测试集json路径")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="输出目录")
    ap.add_argument("--target_domain", type=str, default=DEFAULT_TARGET_DOMAIN)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="输出topK槽位/错误样本")

    # ✅ 改动：skip_empty_gold 默认值写在文件里；命令行也可覆盖
    ap.add_argument("--skip_empty_gold", dest="skip_empty_gold", action="store_true",
                    help="与当前eval对齐：跳过gold为空的轮次（建议开启）")
    ap.add_argument("--no_skip_empty_gold", dest="skip_empty_gold", action="store_false",
                    help="不跳过gold为空的轮次")
    ap.set_defaults(skip_empty_gold=DEFAULT_SKIP_EMPTY_GOLD)

    args = ap.parse_args()

    # 打印配置，方便确认
    print(f"[CFG] in_path={args.in_path}")
    print(f"[CFG] out_dir={args.out_dir}")
    print(f"[CFG] target_domain={args.target_domain} skip_empty_gold={args.skip_empty_gold} topk={args.topk}")

    data = load_json(args.in_path)
    td = args.target_domain

    evaluated_turns = 0
    correct_turns = 0
    tp = fp = fn = 0

    # per-slot stats
    slot_tp = Counter()
    slot_fp = Counter()
    slot_fn = Counter()
    slot_gold_cnt = Counter()
    slot_pred_cnt = Counter()

    error_type_cnt = Counter()
    error_size_cnt = Counter()  # (missing_count, extra_count)

    error_rows = []

    for dial in data:
        dial_id = dial.get("dial_id", "")
        turns = dial.get("turns", []) or []
        for turn in turns:
            turn_id = turn.get("turn_id", None)
            sys_uttr = turn.get("system", "none")
            usr_uttr = turn.get("user", "")

            gold = get_gold_slots(turn, td)
            pred = get_pred_slots(turn, td)

            if args.skip_empty_gold and len(gold) == 0:
                continue

            # 与你当前评估更贴近：只要 gold/pred 有一个非空就算评估轮次
            if len(gold) == 0 and len(pred) == 0:
                continue

            evaluated_turns += 1
            if gold == pred:
                correct_turns += 1

            # slot-level counts
            for s in gold:
                slot_gold_cnt[s] += 1
            for s in pred:
                slot_pred_cnt[s] += 1

            inter = gold & pred
            miss = sorted(list(gold - pred))
            extra = sorted(list(pred - gold))

            tp += len(inter)
            fp += len(extra)
            fn += len(miss)

            for s in inter:
                slot_tp[s] += 1
            for s in extra:
                slot_fp[s] += 1
            for s in miss:
                slot_fn[s] += 1

            if gold != pred:
                if len(miss) > 0 and len(extra) == 0:
                    et = "miss_only"
                elif len(miss) == 0 and len(extra) > 0:
                    et = "extra_only"
                else:
                    et = "miss_and_extra"
                error_type_cnt[et] += 1
                error_size_cnt[(len(miss), len(extra))] += 1

                # 把关键信息写出来（并附上该turn里对应slot的raw_gen）
                detail_map = build_detail_map(turn)
                miss_details = []
                extra_details = []
                for s in miss:
                    d = detail_map.get(s, {})
                    miss_details.append({
                        "slot": s,
                        "pred": d.get("pred", ""),
                        "raw_gen": d.get("raw_gen", ""),
                    })
                for s in extra:
                    d = detail_map.get(s, {})
                    extra_details.append({
                        "slot": s,
                        "pred": d.get("pred", ""),
                        "raw_gen": d.get("raw_gen", ""),
                    })

                error_rows.append({
                    "dial_id": dial_id,
                    "turn_id": turn_id,
                    "system": sys_uttr,
                    "user": usr_uttr,
                    "gold": sorted(list(gold)),
                    "pred": sorted(list(pred)),
                    "missing": miss,
                    "extra": extra,
                    "missing_details": miss_details,
                    "extra_details": extra_details,
                })

    class_jga = correct_turns / evaluated_turns if evaluated_turns > 0 else 0.0
    P, R, F1 = prf(tp, fp, fn)

    # ===== 输出总体总结 =====
    summary = {
        "target_domain": td,
        "evaluated_turns": evaluated_turns,
        "correct_turns": correct_turns,
        "classJGA": round(class_jga, 6),
        "TP": tp, "FP": fp, "FN": fn,
        "P": round(P, 6), "R": round(R, 6), "F1": round(F1, 6),
        "error_turns": evaluated_turns - correct_turns,
        "error_type_cnt": dict(error_type_cnt),
        "top_error_sizes": [
            {"missing": k[0], "extra": k[1], "count": v}
            for k, v in error_size_cnt.most_common(20)
        ],
    }

    # ===== 每槽位指标 =====
    per_slot_rows = []
    all_slots = set(slot_gold_cnt.keys()) | set(slot_pred_cnt.keys())
    for s in sorted(all_slots):
        stp = slot_tp[s]
        sfp = slot_fp[s]
        sfn = slot_fn[s]
        sp, sr, sf1 = prf(stp, sfp, sfn)
        per_slot_rows.append({
            "slot": s,
            "gold_cnt": slot_gold_cnt[s],
            "pred_cnt": slot_pred_cnt[s],
            "tp": stp, "fp": sfp, "fn": sfn,
            "P": round(sp, 6), "R": round(sr, 6), "F1": round(sf1, 6),
        })

    # 重点关注：FN多 / FP多 的槽位
    top_fn = slot_fn.most_common(args.topk)
    top_fp = slot_fp.most_common(args.topk)

    summary["top_FN_slots"] = [{"slot": s, "fn": c} for s, c in top_fn]
    summary["top_FP_slots"] = [{"slot": s, "fp": c} for s, c in top_fp]

    # ===== 写文件 =====
    out_summary = os.path.join(args.out_dir, "summary.json")
    out_errors = os.path.join(args.out_dir, "errors.jsonl")
    out_per_slot = os.path.join(args.out_dir, "per_slot_metrics.csv")

    save_json(out_summary, summary)
    save_jsonl(out_errors, error_rows)

    header = ["slot", "gold_cnt", "pred_cnt", "tp", "fp", "fn", "P", "R", "F1"]
    rows = [
        [r["slot"], r["gold_cnt"], r["pred_cnt"], r["tp"], r["fp"], r["fn"], r["P"], r["R"], r["F1"]]
        for r in per_slot_rows
    ]
    save_csv(out_per_slot, header, rows)

    print(f"[OK] summary -> {out_summary}")
    print(f"[OK] errors  -> {out_errors}  (#{len(error_rows)})")
    print(f"[OK] per_slot-> {out_per_slot}")
    print(f"[EVAL] target_domain={td}")
    print(f"[EVAL] evaluated_turns={evaluated_turns} correct_turns={correct_turns} classJGA={class_jga:.4f}")
    print(f"[EVAL] P={P:.4f} R={R:.4f} F1={F1:.4f}")


if __name__ == "__main__":
    main()
