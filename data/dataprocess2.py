#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_refer_clear_implicit_v1.py — 完整替换版

实现：
- refer-clear / refer-implicit 两种标注（不再写 refer-n / refer-o）。
- src_turn_id == -1 才触发 future 搜索；否则只在当前回合同值匹配。
- 只在“发生指代的目标槽” taklabel 上写入（refer-clear 时附加 refered-slot / refered-value）。
- 不修改 slot_values；目标槽不存在时，refer-implicit 可在占位槽上落地。
"""

import json, os, re
from collections import defaultdict

# ====== 路径 ======
OUR_TRAIN = "/home/fzus/zzp/data/code/dategen/processed_l1_confirm/train_processed.json"
OUR_DEV   = "/home/fzus/zzp/data/code/dategen/processed_l1_confirm/dev_processed.json"
OUR_TEST  = "/home/fzus/zzp/data/code/dategen/processed_l1_confirm/test_processed.json"
MWOZ23_PATH = "/home/fzus/zzp/data/MultiWOZ-coref-main/MultiWOZ2_3/data.json"

def out_path_with_refer(path):
    b,e = os.path.splitext(path)
    return f"{b}_with_refer{e}" if e else f"{b}_with_refer.json"

def out_path_missed(path):
    b,e = os.path.splitext(path)
    return f"{b}_missed_refer{e}" if e else f"{b}_missed_refer.json"

# 发生指代的目标槽（被指代槽不限名/域）
ALLOWED_SLOTS = {"departure","area","destination","day","pricerange","people","arriveby"}

# ====== 值归一化 ======
NUM_WORDS = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10"
}
PUNCT_SP = re.compile(r"[^\w\s:/]+")  # 保留字母数字空格 : /
SPACES   = re.compile(r"\s+")
TIME_SIMPLE = re.compile(r"^\s*(\d{1,2})(?:[:\.](\d{2}))?\s*(am|pm)?\s*$")
ONLY_DIGITS = re.compile(r"^\s*\d+\s*$")

def normalize_time(s: str) -> str:
    ss = s.strip().lower()
    m = TIME_SIMPLE.match(ss)
    if not m:
        if re.match(r"^\d{1,2}:\d{2}$", ss):
            try:
                h, m2 = ss.split(":")
                hi, mi = int(h), int(m2)
                if 0 <= hi <= 23 and 0 <= mi <= 59:
                    return f"{hi:02d}:{mi:02d}"
            except:
                pass
        return ss
    h = int(m.group(1)); mm = m.group(2) or "00"; ap = m.group(3)
    if ap == "pm" and 1 <= h <= 11: h += 12
    if ap == "am" and h == 12: h = 0
    if 0 <= h <= 23:
        return f"{h:02d}:{int(mm):02d}"
    return ss

def normalize_value(s: str) -> str:
    if s is None: return ""
    x = str(s).strip().lower()
    # 纯数字兜底
    if ONLY_DIGITS.match(x):
        return str(int(x))
    # 常见替换
    x = x.replace("&", " and ")
    x = x.replace(" b&b", " bed and breakfast").replace(" b & b", " bed and breakfast")
    # 数词 -> 数字
    toks = SPACES.split(x)
    toks = [NUM_WORDS.get(t, t) for t in toks]
    x = " ".join(toks)
    # 时间标准化
    if any(k in x for k in ["am","pm",":"]) and len(x) <= 10:
        x = normalize_time(x)
    # 去符号与空白
    x = PUNCT_SP.sub(" ", x)
    x = SPACES.sub(" ", x).strip()
    # 再做数字兜底
    if ONLY_DIGITS.match(x):
        return str(int(x))
    return x

# ====== 槽名解析 ======
SLOT_SPLIT = re.compile(r"^([^-\s]+)-\s*(.*)$")
def parse_ours_slot(slot_str: str):
    m = SLOT_SPLIT.match((slot_str or "").strip().lower())
    if not m: return None, None, None
    dom = m.group(1)
    tail = m.group(2)
    raw_tail = tail
    tail = re.sub(r"^book\s+", "", tail).strip()
    tail = tail.replace("_"," ").replace("-"," ")
    if tail in ("dest","destination"): cs = "destination"
    elif tail in ("depart","departure"): cs = "departure"
    elif tail in ("arrive","arrive by","arriveby"): cs = "arriveby"
    elif tail in ("price","price range","pricerange"): cs = "pricerange"
    elif tail in ("area",): cs = "area"
    elif tail in ("day",): cs = "day"
    elif tail in ("people","persons","party","group"): cs = "people"
    else: cs = tail.replace(" ","")
    return dom, cs, raw_tail

# ====== turn 对齐 ======
PUNCT_RE = re.compile(r"[^\w\s]+")

def norm_text(s: str) -> str:
    s = (s or "").lower().replace("\u00a0"," ")
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

def token_set(s: str):
    return set([t for t in norm_text(s).split() if t])

def text_sim(a: str, b: str) -> float:
    na, nb = norm_text(a), norm_text(b)
    if not na or not nb: return 0.0
    if na == nb: return 1.0
    if na in nb or nb in na: return 0.95
    A, B = token_set(a), token_set(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def best_turn_match(mwoz_text: str, our_turns: list, th_loose=0.5):
    nt = norm_text(mwoz_text)
    for i, t in enumerate(our_turns):
        for which in ("system","user"):
            cand = norm_text(t.get(which,""))
            if cand and (nt == cand or nt in cand or cand in nt):
                return i
    bi, bs = None, -1.0
    for i, t in enumerate(our_turns):
        s = max(text_sim(mwoz_text, t.get("system","")), text_sim(mwoz_text, t.get("user","")))
        if s > bs: bi, bs = i, s
    return bi if bs >= th_loose else None

# ====== slot_values 索引 ======
def build_slotvalue_index(turn_obj):
    """
    返回：
      items: [(slot_key, domain, canon_slot, value_str, norm_value_str)]
      by_pair: {(domain, canon_slot): (slot_key, value_str, norm_value_str)} —— 多键优先含 book*
    """
    sv = (turn_obj.get("state",{}) or {}).get("slot_values",{}) or {}
    temp = defaultdict(list)
    items = []
    for k, v in sv.items():
        dom, cs, raw = parse_ours_slot(k)
        if not dom or not cs: continue
        vs = "" if v is None else str(v)
        items.append((k, dom, cs, vs, normalize_value(vs)))
        temp[(dom, cs)].append((k, vs, normalize_value(vs), raw))
    by_pair = {}
    for key, lst in temp.items():
        lst.sort(key=lambda x: (0 if (x[3] and x[3].startswith("book")) else 1))
        k, vv, nv, _ = lst[0]
        by_pair[key] = (k, vv, nv)
    return items, by_pair

def get_prev_value(turns, idx, dom, cs):
    if idx <= 0: return ""
    _, prev_map = build_slotvalue_index(turns[idx-1])
    return prev_map.get((dom, cs), ("","", ""))[1]

# ====== taklabel 工具 ======
def to_list_label2(val):
    if val is None or val == "": return []
    if isinstance(val, list): return [str(x).strip() for x in val if str(x).strip()]
    return [p.strip() for p in str(val).split("|") if p.strip()]

def merge_label2_list(tl: dict, tags: list) -> bool:
    """把 tags（去重）合并入 tl['label-2']；返回是否有新增。"""
    cur = set(to_list_label2(tl.get("label-2")))
    before = len(cur)
    for t in tags:
        t = str(t).strip()
        if t: cur.add(t)
    tl["label-2"] = list(cur)
    return len(cur) > before

def find_taklabel(tl_list, dom, cs):
    for i, tl in enumerate(tl_list):
        d2, c2, _ = parse_ours_slot(tl.get("slot",""))
        if d2 == dom and c2 == cs: return i, tl
    return None, None

def make_taklabel(slot_key, curr_val, prev_val, label2_tags):
    tags = label2_tags if isinstance(label2_tags, list) else [label2_tags]
    return {"slot": slot_key, "label-1": "", "label-2": tags, "prev_val": prev_val, "curr_val": curr_val}

def construct_placeholder_key(domain: str, cs: str) -> str:
    # 占位槽：不含 book（例：train-day）
    return f"{domain}-{cs}"

# ====== 同回合：找“与目标槽同值”的其它槽位（双锚）======
def find_other_same_value_in_turn(turn_obj, target_slot_key, tnorm, cnorm=None):
    items, _ = build_slotvalue_index(turn_obj)
    # 先用 tnorm 严格
    if tnorm:
        for k, dom, cs, v, nv in items:
            if k == target_slot_key: continue
            if nv == tnorm:
                return dom, cs, k, v
        # 再包含
        for k, dom, cs, v, nv in items:
            if k == target_slot_key: continue
            if tnorm in nv or nv in tnorm:
                return dom, cs, k, v
    # 再用 cnorm（若有 true_val）
    if cnorm:
        for k, dom, cs, v, nv in items:
            if k == target_slot_key: continue
            if nv == cnorm:
                return dom, cs, k, v
        for k, dom, cs, v, nv in items:
            if k == target_slot_key: continue
            if cnorm in nv or nv in cnorm:
                return dom, cs, k, v
    return None

# ====== 仅在 src==-1 时：从 tgt_turn 起扫到末尾，找“首次同回合 目标槽 + 任意同值其它槽” ======
def future_find_first_clear_turn(turns, start_idx, domain, cs, cnorm=None):
    n = len(turns)
    first_occ_idx = None
    for r in range(start_idx, n):
        items, by_pair = build_slotvalue_index(turns[r])
        if (domain, cs) in by_pair and first_occ_idx is None:
            first_occ_idx = r
        if (domain, cs) not in by_pair:
            continue
        tgt_key, tgt_val, tgt_norm = by_pair[(domain, cs)]
        cand = find_other_same_value_in_turn(turns[r], tgt_key, tgt_norm, cnorm=cnorm)
        if cand:
            src_dom, src_cs, src_key, src_val = cand
            return ("clear", r, tgt_key, tgt_val, src_dom, src_cs, src_key, src_val)
    if first_occ_idx is not None:
        return ("implicit", first_occ_idx, None, None, None, None, None, None)
    return ("implicit_no_occurrence", None, None, None, None, None, None, None)

# ====== 主流程 ======
def process_split(our_path, coref_db, stats):
    with open(our_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "dial_id" in data:
        data = [data]
    by_id = { d.get("dial_id"): d for d in data if d.get("dial_id") }

    missed = []

    for did, cobj in coref_db.items():
        if did not in by_id:
            continue
        ours = by_id[did]; turns = ours.get("turns", [])
        mlog = cobj.get("log", [])

        # turn 对齐
        turn_map = {}
        for k, item in enumerate(mlog):
            turn_map[k] = best_turn_match(item.get("text",""), turns)

        for k, item in enumerate(mlog):
            coref = item.get("coreference")
            if not coref:
                continue
            tgt_turn = turn_map.get(k, None)
            tgt_text = None
            if tgt_turn is not None and 0 <= tgt_turn < len(turns):
                tgt_text = (turns[tgt_turn].get("user","") + " ||| " + turns[tgt_turn].get("system",""))

            for dom_act, arr in coref.items():
                domain = (dom_act.split("-",1)[0] or "").strip().lower()
                for entry in arr:
                    if not isinstance(entry, list) or len(entry) < 5:
                        continue
                    slot_raw, expr, true_val, src_turn_id, evidence = entry[0], entry[1], entry[2], entry[3], entry[4]

                    # 目标槽解析
                    s = (str(slot_raw) if slot_raw is not None else "").strip().lower()
                    if s in ("dest","destination"): cs = "destination"
                    elif s in ("depart","departure"): cs = "departure"
                    elif s in ("arrive","arriveby","arrive by"): cs = "arriveby"
                    elif s in ("price","pricerange","price-range","price_range"): cs = "pricerange"
                    elif s in ("area",): cs = "area"
                    elif s in ("day",): cs = "day"
                    elif s in ("people","persons","party","group"): cs = "people"
                    else: cs = s
                    if cs not in ALLOWED_SLOTS:
                        continue

                    if tgt_turn is None or not (0 <= tgt_turn < len(turns)):
                        stats["miss_target_turn"] += 1
                        continue

                    # 当前回合索引与锚
                    items_t, by_pair_t = build_slotvalue_index(turns[tgt_turn])
                    tl_list_t = turns[tgt_turn].setdefault("taklabels", [])
                    cnorm = normalize_value("" if true_val is None else str(true_val))

                    # ===== 情况 A：src_turn_id >= 0 —— 只在“当前回合”判定 clear/implicit =====
                    if isinstance(src_turn_id, int) and src_turn_id >= 0:
                        if (domain, cs) in by_pair_t:
                            tgt_key, tgt_val, tgt_norm = by_pair_t[(domain, cs)]
                            # 查找同值其它槽
                            cand = find_other_same_value_in_turn(turns[tgt_turn], tgt_key, tgt_norm, cnorm=cnorm)
                            i_found, tl_n = find_taklabel(tl_list_t, domain, cs)
                            if cand:
                                src_dom, src_cs, src_key, src_val = cand
                                # refer-clear：只在目标槽上写
                                if i_found is None:
                                    prev_val = get_prev_value(turns, tgt_turn, domain, cs)
                                    tl_list_t.append(
                                        make_taklabel(
                                            tgt_key, tgt_val, prev_val,
                                            ["refer-clear", f"refered-slot:{src_key}", f"refered-value:{src_val}"]
                                        )
                                    )
                                else:
                                    merge_label2_list(tl_n, ["refer-clear", f"refered-slot:{src_key}", f"refered-value:{src_val}"])
                                stats["write_refer_clear"] += 1
                            else:
                                # refer-implicit：只在目标槽上写
                                if i_found is None:
                                    prev_val = get_prev_value(turns, tgt_turn, domain, cs)
                                    tl_list_t.append(make_taklabel(tgt_key, tgt_val, prev_val, ["refer-implicit"]))
                                else:
                                    merge_label2_list(tl_n, ["refer-implicit"])
                                stats["write_refer_implicit"] += 1
                        else:
                            # 当前回合没有目标槽：在占位槽写 refer-implicit
                            placeholder_key = construct_placeholder_key(domain, cs)
                            i_found, tl_n = find_taklabel(tl_list_t, domain, cs)
                            if i_found is None:
                                prev_val = get_prev_value(turns, tgt_turn, domain, cs)
                                tl_list_t.append(make_taklabel(placeholder_key, "", prev_val, ["refer-implicit"]))
                            else:
                                merge_label2_list(tl_n, ["refer-implicit"])
                            stats["write_refer_implicit"] += 1

                    # ===== 情况 B：src_turn_id == -1 —— 触发 future 搜索（至结尾）=====
                    elif isinstance(src_turn_id, int) and src_turn_id == -1:
                        status, idx, tgt_key_f, tgt_val_f, src_dom, src_cs, src_key, src_val = \
                            future_find_first_clear_turn(turns, tgt_turn, domain, cs, cnorm=cnorm)

                        if status == "clear":
                            tl_list_r = turns[idx].setdefault("taklabels", [])
                            i_found_r, tl_n_r = find_taklabel(tl_list_r, domain, cs)
                            if i_found_r is None:
                                prev_t = get_prev_value(turns, idx, domain, cs)
                                tl_list_r.append(
                                    make_taklabel(
                                        tgt_key_f, tgt_val_f, prev_t,
                                        ["refer-clear", f"refered-slot:{src_key}", f"refered-value:{src_val}"]
                                    )
                                )
                            else:
                                merge_label2_list(tl_n_r, ["refer-clear", f"refered-slot:{src_key}", f"refered-value:{src_val}"])
                            stats["write_refer_clear_future"] += 1
                        elif status == "implicit":
                            # 在“目标槽首次出现”的回合写 refer-implicit
                            tl_list_r = turns[idx].setdefault("taklabels", [])
                            i_found_r, tl_n_r = find_taklabel(tl_list_r, domain, cs)
                            if i_found_r is None:
                                # 这里必然有目标槽（因为 idx 是首次出现的回合），找出 key/val
                                _, by_pair_r = build_slotvalue_index(turns[idx])
                                key_r, val_r, _ = by_pair_r[(domain, cs)]
                                prev_t = get_prev_value(turns, idx, domain, cs)
                                tl_list_r.append(make_taklabel(key_r, val_r, prev_t, ["refer-implicit"]))
                            else:
                                merge_label2_list(tl_n_r, ["refer-implicit"])
                            stats["write_refer_implicit_future"] += 1
                        else:  # implicit_no_occurrence
                            # 到结尾都没出现目标槽：在目标回合占位槽上落 refer-implicit
                            tl_list_t = turns[tgt_turn].setdefault("taklabels", [])
                            placeholder_key = construct_placeholder_key(domain, cs)
                            i_found_t, tl_n_t = find_taklabel(tl_list_t, domain, cs)
                            if i_found_t is None:
                                prev_val = get_prev_value(turns, tgt_turn, domain, cs)
                                tl_list_t.append(make_taklabel(placeholder_key, "", prev_val, ["refer-implicit"]))
                            else:
                                merge_label2_list(tl_n_t, ["refer-implicit"])
                            stats["write_refer_implicit_future_placeholder"] += 1
                    else:
                        # 异常 src_turn_id
                        stats["invalid_src_turn"] += 1
                        missed.append({
                            "dialog_id": did,
                            "mwoz_turn_id": k,
                            "mwoz_text": item.get("text",""),
                            "dom_act": dom_act,
                            "coref_item": entry,
                            "our_matched_target_turn": tgt_turn,
                            "our_target_turn_text": tgt_text,
                            "reason": "invalid_src_turn_field"
                        })

    # 写回
    outp = out_path_with_refer(our_path)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    out_missed = out_path_missed(our_path)
    with open(out_missed, "w", encoding="utf-8") as f:
        json.dump({"missed": missed}, f, ensure_ascii=False, indent=2)

    return outp, out_missed

def main():
    with open(MWOZ23_PATH, "r", encoding="utf-8") as f:
        mwoz23 = json.load(f)

    stats = defaultdict(int)
    outs, missed_files = [], []
    for p in [OUR_TRAIN, OUR_DEV, OUR_TEST]:
        if not os.path.isfile(p):
            print(f"[WARN] Not found: {p}")
            continue
        outp, out_missed = process_split(p, mwoz23, stats)
        outs.append(outp); missed_files.append(out_missed)

    print("\n==== refer 统计（refer-clear / refer-implicit）====")
    for k, v in stats.items():
        print(f"{k:36s}: {v}")
    if outs:
        print("\nGenerated files:")
        for x in outs:
            print(" -", x)
    if missed_files:
        print("\nMissed-case files:")
        for x in missed_files:
            print(" -", x)

if __name__ == "__main__":
    main()








