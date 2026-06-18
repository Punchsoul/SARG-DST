import os
import json
from typing import Dict, Tuple, List, Any
from collections import Counter, defaultdict

# ===================== 你提供的路径（已写死） =====================
TARGET_DOMAIN = "train"
CAND_TRAIN = "/home/zzp/model2/candidates_stage1_train/train_candidates_top10.jsonl"
CAND_DEV   = "/home/zzp/model2/candidates_stage1_train/dev_candidates_top10.jsonl"
CAND_TEST  = "/home/zzp/model2/candidates_stage1_train/test_candidates_top10.jsonl"

DATA_TRAIN = "/home/zzp/ds_loo/train/train_train.json"
DATA_DEV   = "/home/zzp/ds_loo/train/train_dev.json"
DATA_TEST  = "/home/zzp/ds_loo/train/train_test.json"

OUT_DIR    = "/home/zzp/model2/datasets_with_candidates_stage1_train"
FIELD_NAME = "candidates"   # 写回数据集的字段名
# ================================================================

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_cand_map(cand_jsonl: str) -> Dict[Tuple[str, int], List[str]]:
    mp: Dict[Tuple[str, int], List[str]] = {}
    with open(cand_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            dial_id = x["dial_id"]
            turn_id = int(x["turn_id"])
            cands = x.get("candidates", [])
            mp[(dial_id, turn_id)] = cands
    return mp

def get_turn_id(turn: Dict[str, Any], fallback_i: int) -> int:
    return int(turn.get("turn_id", fallback_i))

def filter_test_dialogs_by_domain(dialogs: List[Dict[str, Any]], target_domain: str) -> List[Dict[str, Any]]:
    out = []
    for dlg in dialogs:
        domains = dlg.get("domains", []) or []
        if target_domain in domains:
            out.append(dlg)
    return out



def analyze_missing(dialogs, cand_map, n_show=10):
    cand_dial_ids = set(d for (d, _) in cand_map.keys())
    cand_turn_ids_by_dial = defaultdict(set)
    for (d, t) in cand_map.keys():
        cand_turn_ids_by_dial[d].add(t)

    cnt = Counter()
    samples = []

    for dlg in dialogs:
        did = dlg.get("dial_id", None)
        if not did:
            cnt["dialog_missing_dial_id"] += 1
            # 这类对话下面所有 turns 基本都会 missing（因为 did=None/""）
        turns = dlg.get("turns", []) or []
        for i, turn in enumerate(turns):
            tid = int(turn.get("turn_id", i))
            key = ((did or ""), tid)

            if key in cand_map:
                cnt["hit"] += 1
                continue

            # miss：归因
            if not did:
                reason = "no_dial_id_in_dataset"
            elif did not in cand_dial_ids:
                reason = "dial_id_not_in_candidates"
            else:
                reason = "turn_id_not_in_candidates_for_this_dial"

            cnt[reason] += 1
            if len(samples) < n_show:
                # 顺便展示：该 dial 在候选里到底有哪些 turn_id，便于判断是否整体偏移
                cand_tids = sorted(list(cand_turn_ids_by_dial.get(did, [])))
                samples.append({
                    "dial_id": did,
                    "dataset_turn_index_i": i,
                    "dataset_turn_id_used": tid,
                    "reason": reason,
                    "cand_turn_ids_head": cand_tids[:15],
                })

    return cnt, samples

def attach_candidates_inplace(
    dialogs: List[Dict[str, Any]],
    cand_map: Dict[Tuple[str, int], List[str]],
    field_name: str,
    *,
    filter_to_target_domain: bool = False,
    target_domain: str = ""
) -> Dict[str, int]:
    """
    给每个 turn 增加 field_name 字段。
    test 要求：只写入目标领域 candidates（prefix 过滤）
    """
    pref = (target_domain + "-") if target_domain else ""
    stats = {
        "dialogs": len(dialogs),
        "turns_total": 0,
        "turns_has_key": 0,
        "turns_missing_key": 0,
    }

    for dlg in dialogs:
        dial_id = dlg.get("dial_id", "")
        turns = dlg.get("turns", []) or []
        for i, t in enumerate(turns):
            stats["turns_total"] += 1
            tid = get_turn_id(t, i)
            key = (dial_id, tid)

            if key not in cand_map:
                stats["turns_missing_key"] += 1
                t[field_name] = []
                continue

            stats["turns_has_key"] += 1
            cands = cand_map[key]

            if filter_to_target_domain:
                cands = [s for s in cands if isinstance(s, str) and s.startswith(pref)]

            t[field_name] = cands

    return stats

def main():
    # load datasets
    train_dialogs = load_json(DATA_TRAIN)
    dev_dialogs   = load_json(DATA_DEV)
    test_dialogs  = load_json(DATA_TEST)

    # load candidates
    cmap_train = load_cand_map(CAND_TRAIN)
    cmap_dev   = load_cand_map(CAND_DEV)
    cmap_test  = load_cand_map(CAND_TEST)

    # ===== DEBUG: analyze why train keys are missing =====
    cnt, samples = analyze_missing(train_dialogs, cmap_train, n_show=20)
    print("[train missing analysis]", dict(cnt))
    print("[train missing samples]")
    for s in samples:
        print(s)

    # attach to train/dev (不做领域过滤：原样写入)
    st_train = attach_candidates_inplace(train_dialogs, cmap_train, FIELD_NAME, filter_to_target_domain=False)
    st_dev   = attach_candidates_inplace(dev_dialogs,   cmap_dev,   FIELD_NAME, filter_to_target_domain=False)

    # merge train+dev -> train
    merged_train = train_dialogs + dev_dialogs

    # test: only keep dialogs containing target domain
    test_filtered = filter_test_dialogs_by_domain(test_dialogs, TARGET_DOMAIN)

    # attach to test (只写入目标领域 candidates)
    st_test = attach_candidates_inplace(
        test_filtered,
        cmap_test,
        FIELD_NAME,
        filter_to_target_domain=True,
        target_domain=TARGET_DOMAIN
    )

    # save
    out_train = os.path.join(OUT_DIR, f"train_merged_with_{FIELD_NAME}_{TARGET_DOMAIN}.json")
    out_test  = os.path.join(OUT_DIR, f"test_{TARGET_DOMAIN}_with_{FIELD_NAME}_only_{TARGET_DOMAIN}.json")

    save_json(merged_train, out_train)
    save_json(test_filtered, out_test)

    print("==== merge candidates into dataset DONE ====")
    print("[train] stats:", st_train)
    print("[dev]   stats:", st_dev)
    print("[train merged] dialogs:", len(merged_train))
    print("[test filtered] dialogs:", len(test_filtered))
    print("[test]  stats:", st_test)
    print("saved:")
    print(" ", out_train)
    print(" ", out_test)

if __name__ == "__main__":
    main()
