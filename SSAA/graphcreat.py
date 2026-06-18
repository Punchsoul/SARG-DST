# -*- coding: utf-8 -*-
"""
Step-1: Build turn-level "graph-like" DST training store with improved negative evidence:
  - [SYS]/[USR] role tags in retrieval_text
  - refer-clear adds from_turn (source evidence index)
  - Negative evidence with:
      (1) Confusion Pairs (same-domain mutual exclusives)
      (2) Cross-domain homonyms (e.g., taxi-* vs hotel-name/attraction-name)
      (3) Value state tiers: Ghost (carry value) / Void (explicit none for top FP slots)

Input:
  - train_merged_with_candidates_*.json  (main training file)
  - slot_descriptions.json              (external; we DO NOT embed it into graphs)

Output (5 files):
  - dialog_store_*.jsonl                               (context bank; full)
  - turn_graph_store_*.jsonl                           (FULL query graph DB for inference-time retrieval)
  - turn_graph_store_*_partial.jsonl                   (PARTIAL query graph DB for instruction DB)
  - dialogs_*_partial.json                             (PARTIAL raw dialogs; original dataset format; for building instruction DB)
  - dialogs_*_nondb.json                               (NON-DB raw dialogs; original dataset format; NOT used as retrieval DB,
                                                       used later as finetune query input)
"""

import os
import json
import random
from typing import Any, Dict, List, Tuple, Optional, Set
from tqdm import tqdm


# =========================
# Hard-coded parameters here
# =========================
SLOT_DESC_PATH = "/home/fzus/zzp2/ds_loo/ontology/slot_descriptions.json"
INPUT_PATH = "/home/fzus/zzp2/ds_loo/hotel/train_merged_with_candidates_hotel.json"

OUT_DIALOG_STORE = "/home/fzus/zzp2/ds_loo/hotel/hoteldata/dialog_hotel.jsonl"
OUT_TURN_GRAPH_STORE = "/home/fzus/zzp2/ds_loo/hotel/hoteldata/turn_hotel.jsonl"

# NEW outputs for instruction DB (PARTIAL)
OUT_PARTIAL_TURN_GRAPH_STORE = "/home/fzus/zzp2/ds_loo/hotel/hoteldata/turn_hotel_partial.jsonl"
OUT_PARTIAL_DIALOGS = "/home/fzus/zzp2/ds_loo/hotel/hoteldata/dialogs_hotel_partial.json"

# NEW output: dialogs NOT used as retrieval DB (for finetune query input)
OUT_NONDB_DIALOGS = "/home/fzus/zzp2/ds_loo/hotel/hoteldata/dialogs_hotel_nondb.json"

WINDOW_K = 2
RANDOM_SEED = 13

# Dialog-level sampling ratio for PARTIAL DB
PARTIAL_DIAL_RATIO = 0.35

# For each eligible turn, ALWAYS inject negatives (coverage=100%).
# How many negative slots per turn:
NEG_NUM_CHOICES = (1, 2)

# Confusion pairs by suffix (domain-agnostic suffix)
CONFUSION_MAP: Dict[str, List[str]] = {
    "departure": ["destination"],
    "destination": ["departure"],
    "leaveat": ["arriveby"],
    "arriveby": ["leaveat"],
    "area": ["name"],
    "name": ["area","type"],
    "type": ["name"],
}

# Cross-domain homonyms: if pos contains taxi-(departure|destination), add these as hard negatives
TAXI_HOMONYM_NEGS = ["hotel-name", "attraction-name"]

# When filling remaining neg slots by weighted sampling, boost "enemies" probability
CONFUSION_BOOST = 3.0

# FP-based priors (weights)
FP_WEIGHTS: Dict[str, Dict[str, int]] = {
    "attraction": {"attraction-area": 43, "attraction-type": 18, "attraction-name": 4},
    "hotel": {
        "hotel-type": 133, "hotel-pricerange": 125, "hotel-area": 85, "hotel-name": 65,
        "hotel-stars": 55, "hotel-internet": 34, "hotel-parking": 24,
        "hotel-book day": 23, "hotel-book people": 18, "hotel-book stay": 17
    },
    "restaurant": {
        "restaurant-name": 53, "restaurant-area": 28, "restaurant-food": 26,
        "restaurant-book people": 15, "restaurant-pricerange": 14,
        "restaurant-book day": 13, "restaurant-book time": 11
    },
    "taxi": {"taxi-departure": 24, "taxi-destination": 23, "taxi-arriveby": 12, "taxi-leaveat": 7},
    "train": {
        "train-leaveat": 37, "train-departure": 32, "train-book people": 32,
        "train-arriveby": 29, "train-destination": 28, "train-day": 19
    }
}


def ensure_dir(p: str) -> None:
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def safe_get_slot_values(turn: Dict[str, Any]) -> Dict[str, str]:
    return (turn.get("state") or {}).get("slot_values") or {}


def build_retrieval_text(turns: List[Dict[str, Any]], t_idx: int, k: int) -> str:
    start = max(0, t_idx - k)
    parts: List[str] = []
    for i in range(start, t_idx + 1):
        sys_utt = turns[i].get("system", "")
        usr_utt = turns[i].get("user", "")
        parts.append(f"[t{i}][SYS] {sys_utt}")
        parts.append(f"[t{i}][USR] {usr_utt}")
    return "\n".join(parts)


def parse_label_and_from_slot(label1: str, label2: Any) -> Tuple[str, str]:
    """
    Your rule:
      - If label-2 non-empty:
          - contain refer-clear -> refer-clear + refered-slot
          - contain refer-implicit -> refer-implicit
          - else -> confirm
      - Else -> label-1
    """
    from_slot = ""

    def tokens_from_label2(l2: Any) -> List[str]:
        if isinstance(l2, list) and len(l2) > 0:
            return [str(x).strip() for x in l2 if str(x).strip()]
        if isinstance(l2, str) and l2.strip():
            s = l2.strip()
            return [x.strip() for x in s.split(",")] if "," in s else [s]
        return []

    tokens = tokens_from_label2(label2)
    if tokens:
        if "refer-clear" in tokens:
            for t in tokens:
                if t.startswith("refered-slot:"):
                    from_slot = t.split("refered-slot:", 1)[1].strip()
                    break
            return "refer-clear", from_slot
        if "refer-implicit" in tokens:
            return "refer-implicit", ""
        return "confirm", ""

    return (label1 or "").strip(), ""


def parse_refer_value(label2: Any) -> str:
    if isinstance(label2, list):
        for t in label2:
            s = str(t).strip()
            if s.startswith("refered-value:"):
                return s.split("refered-value:", 1)[1].strip()
    elif isinstance(label2, str) and label2.strip():
        s = label2.strip()
        if "refered-value:" in s:
            return s.split("refered-value:", 1)[1].split(",")[0].strip()
    return ""


def find_from_turn_by_state_diff(turns: List[Dict[str, Any]], t_idx: int, from_slot: str, refer_value: str) -> int:
    if t_idx <= 0 or not from_slot or not refer_value:
        return -1

    for j in range(t_idx - 1, -1, -1):
        sv_j = safe_get_slot_values(turns[j])
        if sv_j.get(from_slot, "") != refer_value:
            continue
        if j == 0:
            return 0
        sv_prev = safe_get_slot_values(turns[j - 1])
        if sv_prev.get(from_slot, "") != refer_value:
            return j

    for j in range(t_idx - 1, -1, -1):
        sv_j = safe_get_slot_values(turns[j])
        if sv_j.get(from_slot, "") == refer_value:
            return j

    return -1


def add_edge(edges: List[Dict[str, Any]], seen: set, src: Dict[str, Any], rel: str, dst: Dict[str, Any]) -> None:
    e = {"src": src, "rel": rel, "dst": dst}
    key = json.dumps(e, sort_keys=True, ensure_ascii=False)
    if key not in seen:
        edges.append(e)
        seen.add(key)


def build_domain_suffix_index(slot_keys: Set[str]) -> Dict[str, Dict[str, List[str]]]:
    idx: Dict[str, Dict[str, List[str]]] = {}
    for s in slot_keys:
        if "-" not in s:
            continue
        dom, suf = s.split("-", 1)
        idx.setdefault(dom, {}).setdefault(suf, []).append(s)
    return idx


def fp_weight_of(slot: str) -> int:
    if "-" not in slot:
        return 1
    dom = slot.split("-", 1)[0]
    return FP_WEIGHTS.get(dom, {}).get(slot, 1)


def build_priority_neg_candidates(
    pos_slots: Set[str],
    domain_suffix_index: Dict[str, Dict[str, List[str]]],
    slot_desc_keys: Set[str]
) -> Tuple[List[str], Set[str]]:
    priority: List[str] = []
    enemy_slots: Set[str] = set()

    def push(s: str):
        if s and (s not in pos_slots) and (s not in priority):
            priority.append(s)

    # confusion pairs
    for s_pos in pos_slots:
        if "-" not in s_pos:
            continue
        dom, suf = s_pos.split("-", 1)
        if suf not in CONFUSION_MAP:
            continue
        for enemy_suf in CONFUSION_MAP[suf]:
            cand_list = domain_suffix_index.get(dom, {}).get(enemy_suf, [])
            for s_neg in cand_list:
                if s_neg in slot_desc_keys and s_neg not in pos_slots:
                    push(s_neg)
                    enemy_slots.add(s_neg)

    # taxi homonyms
    for s_pos in pos_slots:
        if s_pos in ("taxi-departure", "taxi-destination"):
            for s_neg in TAXI_HOMONYM_NEGS:
                if s_neg in slot_desc_keys:
                    push(s_neg)

    return priority, enemy_slots


def weighted_sample_from_domain(
    domain: str,
    forbid: Set[str],
    rng: random.Random,
    domain_slots: List[str],
    enemy_slots: Set[str],
) -> Optional[str]:
    def ok(s: str) -> bool:
        return s not in forbid

    if domain in FP_WEIGHTS and FP_WEIGHTS[domain]:
        slots = [s for s in FP_WEIGHTS[domain].keys() if ok(s)]
        if slots:
            weights = []
            for s in slots:
                w = FP_WEIGHTS[domain].get(s, 1)
                if s in enemy_slots:
                    w = w * CONFUSION_BOOST
                weights.append(w)
            return rng.choices(slots, weights=weights, k=1)[0]

    cand = [s for s in domain_slots if ok(s)]
    if not cand:
        return None

    boosted = []
    for s in cand:
        boosted.append(s)
        if s in enemy_slots:
            boosted.extend([s] * int(max(1, round(CONFUSION_BOOST))))
    return rng.choice(boosted) if boosted else None


def main() -> None:
    rng_graph = random.Random(RANDOM_SEED)
    rng_split = random.Random(RANDOM_SEED)

    ensure_dir(OUT_DIALOG_STORE)
    ensure_dir(OUT_TURN_GRAPH_STORE)
    ensure_dir(OUT_PARTIAL_TURN_GRAPH_STORE)
    ensure_dir(OUT_PARTIAL_DIALOGS)
    ensure_dir(OUT_NONDB_DIALOGS)

    # load slot descriptions
    with open(SLOT_DESC_PATH, "r", encoding="utf-8") as f:
        slot_desc = json.load(f)
    slot_desc_keys = set(slot_desc.keys())

    domain_suffix_index = build_domain_suffix_index(slot_desc_keys)

    domain_slots_from_desc: Dict[str, List[str]] = {}
    for s in slot_desc_keys:
        dom = s.split("-", 1)[0] if "-" in s else ""
        domain_slots_from_desc.setdefault(dom, []).append(s)

    # load dataset
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        dialogs = json.load(f)
    assert isinstance(dialogs, list), "Expected a list of dialogs"

    total_turns = sum(len(d.get("turns", [])) for d in dialogs)

    # partial sampling by dialog
    idxs = list(range(len(dialogs)))
    rng_split.shuffle(idxs)
    n_partial = int(round(len(dialogs) * PARTIAL_DIAL_RATIO))
    partial_idxs = set(idxs[:n_partial])

    dialog_out = open(OUT_DIALOG_STORE, "w", encoding="utf-8")
    graph_out = open(OUT_TURN_GRAPH_STORE, "w", encoding="utf-8")
    graph_partial_out = open(OUT_PARTIAL_TURN_GRAPH_STORE, "w", encoding="utf-8")

    # partial raw dialogs (json list)
    partial_dialogs_fp = open(OUT_PARTIAL_DIALOGS, "w", encoding="utf-8")
    partial_dialogs_fp.write("[\n")
    partial_first = True

    # NON-DB raw dialogs (json list)
    nondb_dialogs_fp = open(OUT_NONDB_DIALOGS, "w", encoding="utf-8")
    nondb_dialogs_fp.write("[\n")
    nondb_first = True

    # stats
    eligible_turns = 0
    total_neg_slots = 0
    missing_slot_desc_cnt = 0
    void_neg_cnt = 0
    ghost_neg_cnt = 0
    num_graphs = 0
    num_graphs_partial = 0
    partial_dialogs_cnt = 0
    nondb_dialogs_cnt = 0

    pbar = tqdm(total=total_turns, desc="Building turn graphs", ncols=100)
    try:
        for di, d in enumerate(dialogs):
            dial_id = d.get("dial_id", "")
            turns = d.get("turns", [])
            if not isinstance(turns, list):
                continue

            is_partial = di in partial_idxs

            # write raw dialogs into two disjoint sets:
            #   partial dialogs -> retrieval DB raw pool
            #   nondb dialogs   -> not used as retrieval DB; used later as finetune query input
            if is_partial:
                partial_dialogs_cnt += 1
                if not partial_first:
                    partial_dialogs_fp.write(",\n")
                partial_dialogs_fp.write(json.dumps(d, ensure_ascii=False))
                partial_first = False
            else:
                nondb_dialogs_cnt += 1
                if not nondb_first:
                    nondb_dialogs_fp.write(",\n")
                nondb_dialogs_fp.write(json.dumps(d, ensure_ascii=False))
                nondb_first = False

            # context bank (FULL)
            dialog_line = {
                "dial_id": dial_id,
                "turns": [{"turn_id": t.get("turn_id"), "system": t.get("system", ""), "user": t.get("user", "")} for t in turns]
            }
            dialog_out.write(json.dumps(dialog_line, ensure_ascii=False) + "\n")

            for ti, turn in enumerate(turns):
                pbar.update(1)

                taklabels = turn.get("taklabels", [])
                if not taklabels:
                    continue

                eligible_turns += 1

                retrieval_text = build_retrieval_text(turns, ti, WINDOW_K)
                prev_sv = safe_get_slot_values(turns[ti - 1]) if ti > 0 else {}

                pos_slots: List[str] = []
                tasklabel_nodes: List[Dict[str, Any]] = []
                labels_for_retrieval: Set[str] = set()

                value_nodes_set = set()
                value_nodes: List[Dict[str, Any]] = []

                edges: List[Dict[str, Any]] = []
                seen_edges = set()
                ctx_ep = {"t": "ctx"}

                # positives
                for e in taklabels:
                    slot = (e.get("slot") or "").strip()
                    if not slot:
                        continue
                    pos_slots.append(slot)
                    if slot not in slot_desc_keys:
                        missing_slot_desc_cnt += 1

                    label1 = (e.get("label-1") or "").strip()
                    label2 = e.get("label-2", "")
                    label, from_slot = parse_label_and_from_slot(label1, label2)

                    from_turn = -1
                    refer_value = ""
                    if label == "refer-clear" and from_slot:
                        refer_value = parse_refer_value(label2) or (e.get("curr_val") or "").strip()
                        if refer_value:
                            from_turn = find_from_turn_by_state_diff(turns, ti, from_slot, refer_value)

                    tasklabel_nodes.append({
                        "slot": slot,
                        "label": label,
                        "from_slot": from_slot or "",
                        "from_turn": from_turn
                    })

                    if label:
                        labels_for_retrieval.add(label)

                    add_edge(edges, seen_edges, ctx_ep, "activate_slot", {"t": "slot", "id": slot})
                    add_edge(edges, seen_edges, {"t": "tl", "slot": slot}, "apply_label", {"t": "slot", "id": slot})

                    if label == "dontcare":
                        add_edge(edges, seen_edges, ctx_ep, "triggers_dontcare", {"t": "tl", "slot": slot})

                    if label == "refer-clear" and from_slot:
                        add_edge(edges, seen_edges, {"t": "slot", "id": slot}, "refer_to", {"t": "slot", "id": from_slot})

                    prev_val = (e.get("prev_val") or "").strip()
                    curr_val = (e.get("curr_val") or "").strip()

                    if prev_val and prev_val != "none":
                        key = (slot, prev_val, "prev")
                        if key not in value_nodes_set:
                            value_nodes_set.add(key)
                            value_nodes.append({"slot": slot, "value": prev_val, "role": "prev"})
                        add_edge(edges, seen_edges,
                                 {"t": "slot", "id": slot}, "value_prev",
                                 {"t": "val", "slot": slot, "value": prev_val, "role": "prev"})

                    if curr_val and curr_val != "none":
                        key = (slot, curr_val, "curr")
                        if key not in value_nodes_set:
                            value_nodes_set.add(key)
                            value_nodes.append({"slot": slot, "value": curr_val, "role": "curr"})
                        add_edge(edges, seen_edges,
                                 {"t": "slot", "id": slot}, "value_curr",
                                 {"t": "val", "slot": slot, "value": curr_val, "role": "curr"})

                pos_slot_set = set(pos_slots)

                # =========================
                # NEGATIVE EVIDENCE (100% coverage)
                # =========================
                neg_slots: List[Dict[str, Any]] = []

                # Choose k neg slots (>=1)
                k = rng_graph.choice(list(NEG_NUM_CHOICES))
                if k <= 0:
                    k = 1

                forbid = set(pos_slot_set)

                # domains involved in positives
                pos_domains = sorted({s.split("-", 1)[0] for s in pos_slot_set if "-" in s})

                # priority candidates + enemy slots
                priority_list, enemy_slots = build_priority_neg_candidates(
                    pos_slot_set, domain_suffix_index, slot_desc_keys
                )

                chosen: List[str] = []
                chosen_set: Set[str] = set()

                # 1) take from priority first (weighted)
                if priority_list:
                    pr = sorted(priority_list, key=lambda s: fp_weight_of(s), reverse=True)
                    while pr and len(chosen) < k:
                        weights = [max(1, fp_weight_of(s)) for s in pr]
                        pick = rng_graph.choices(pr, weights=weights, k=1)[0]
                        pr.remove(pick)
                        if pick not in forbid and pick not in chosen_set:
                            chosen.append(pick)
                            chosen_set.add(pick)

                # 2) fill remaining by weighted sampling in positive domains
                while len(chosen) < k and pos_domains:
                    dom = rng_graph.choice(pos_domains)
                    s_neg = weighted_sample_from_domain(
                        domain=dom,
                        forbid=forbid.union(chosen_set),
                        rng=rng_graph,
                        domain_slots=domain_slots_from_desc.get(dom, []),
                        enemy_slots=enemy_slots,
                    )
                    if not s_neg:
                        break
                    chosen.append(s_neg)
                    chosen_set.add(s_neg)

                # 3) hard fallback to guarantee at least 1 negative slot
                if not chosen:
                    global_cand = [s for s in slot_desc_keys if s not in forbid]
                    if global_cand:
                        chosen.append(rng_graph.choice(global_cand))
                        chosen_set.add(chosen[0])

                # materialize chosen negatives (NO empty: use carry if exists else "none")
                for s_neg in chosen:
                    total_neg_slots += 1
                    add_edge(edges, seen_edges, ctx_ep, "neg_slot", {"t": "slot", "id": s_neg})

                    carry_val = (prev_sv.get(s_neg, "") or "").strip()
                    if carry_val and carry_val != "none":
                        ghost_neg_cnt += 1
                        neg_slots.append({"slot": s_neg, "neg_type": "ghost", "carry_value": carry_val})

                        key = (s_neg, carry_val, "carry")
                        if key not in value_nodes_set:
                            value_nodes_set.add(key)
                            value_nodes.append({"slot": s_neg, "value": carry_val, "role": "carry"})
                        add_edge(edges, seen_edges,
                                 {"t": "slot", "id": s_neg}, "value_carry",
                                 {"t": "val", "slot": s_neg, "value": carry_val, "role": "carry"})
                    else:
                        void_neg_cnt += 1
                        neg_slots.append({"slot": s_neg, "neg_type": "void", "carry_value": "none"})

                        key = (s_neg, "none", "carry")
                        if key not in value_nodes_set:
                            value_nodes_set.add(key)
                            value_nodes.append({"slot": s_neg, "value": "none", "role": "carry"})
                        add_edge(edges, seen_edges,
                                 {"t": "slot", "id": s_neg}, "value_carry",
                                 {"t": "val", "slot": s_neg, "value": "none", "role": "carry"})

                # collect slots: pos + neg + refer sources
                all_slots = set(pos_slot_set)
                for ns in neg_slots:
                    all_slots.add(ns["slot"])
                for tl in tasklabel_nodes:
                    if tl.get("label") == "refer-clear" and tl.get("from_slot"):
                        all_slots.add(tl["from_slot"])

                # ORIGINAL schema preserved + only one new top-level field
                graph_line = {
                    "labels_for_retrieval": sorted(labels_for_retrieval),
                    "graph_id": f"{dial_id}::{turn.get('turn_id')}",
                    "dial_id": dial_id,
                    "turn_id": turn.get("turn_id"),
                    "context": {"retrieval_text": retrieval_text},
                    "slots": sorted(all_slots),
                    "tasklabels": tasklabel_nodes,
                    "values": value_nodes,
                    "neg_slots": neg_slots,
                    "edges": edges
                }

                graph_out.write(json.dumps(graph_line, ensure_ascii=False) + "\n")
                num_graphs += 1

                if is_partial:
                    graph_partial_out.write(json.dumps(graph_line, ensure_ascii=False) + "\n")
                    num_graphs_partial += 1

    finally:
        pbar.close()
        dialog_out.close()
        graph_out.close()
        graph_partial_out.close()

        partial_dialogs_fp.write("\n]\n")
        partial_dialogs_fp.close()

        nondb_dialogs_fp.write("\n]\n")
        nondb_dialogs_fp.close()

    print("[DONE]")
    print(f"Input dialogs: {len(dialogs)}")
    print(f"Partial dialogs (used as retrieval DB raw pool): {partial_dialogs_cnt} (ratio={partial_dialogs_cnt / max(1, len(dialogs)):.3f})")
    print(f"Non-DB dialogs (for finetune query input): {nondb_dialogs_cnt} (ratio={nondb_dialogs_cnt / max(1, len(dialogs)):.3f})")
    print(f"Total turns: {total_turns}")
    print(f"Eligible turns (taklabels non-empty): {eligible_turns}")
    print(f"Graphs written (FULL): {num_graphs}")
    print(f"Graphs written (PARTIAL): {num_graphs_partial}")
    print(f"Total neg slots injected: {total_neg_slots} (coverage=100% over eligible turns)")
    print(f"Neg types: ghost={ghost_neg_cnt}, void={void_neg_cnt}")
    print(f"Missing slot descriptions (count): {missing_slot_desc_cnt}")
    print(f"[OUT] Dialog store: {OUT_DIALOG_STORE}")
    print(f"[OUT] FULL graph store: {OUT_TURN_GRAPH_STORE}")
    print(f"[OUT] PARTIAL graph store: {OUT_PARTIAL_TURN_GRAPH_STORE}")
    print(f"[OUT] PARTIAL raw dialogs: {OUT_PARTIAL_DIALOGS}")
    print(f"[OUT] NON-DB raw dialogs: {OUT_NONDB_DIALOGS}")


if __name__ == "__main__":
    main()
