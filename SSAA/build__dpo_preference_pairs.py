# -*- coding: utf-8 -*-
"""
Build SSAA DPO preference pairs for SARG-DST.

This script implements the three preference-pair sources described in the
manuscript:

P_A: false-activation suppression
  - P_A1 surface mention: chosen=NONE, rejected=CONSTRAIN
  - P_A2 memory interference: chosen=NONE, rejected=latest historical action

P_B: fine-grained label confusion
  - chosen=gold semantic action
  - rejected=Phi(gold semantic action)

P_C: empirical hard negatives from SFT prediction errors
  - chosen=gold semantic action
  - rejected=SFT wrong prediction

The output is a raw JSONL file. Use convert_ssaa_dpo_to_llamafactory.py to
create the LLaMA-Factory DPO training file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from dynamic_slot_memory import DynamicSlotMemory, get_domain, normalize_action, parse_event_action


DOMAINS = ["hotel", "restaurant", "attraction", "train", "taxi"]

ACTIVE_ACTIONS = {
    "CONSTRAIN",
    "CHANGE",
    "SWITCH",
    "DONTCARE",
    "CONFIRM",
    "REF-EXPLICIT",
    "REF-IMPLICIT",
}

CONFUSION_MAP = {
    "CHANGE": "CONSTRAIN",
    "CONSTRAIN": "CHANGE",
    "CONFIRM": "CONSTRAIN",
    "SWITCH": "CONSTRAIN",
    "DONTCARE": "NONE",
    "REF-IMPLICIT": "REF-EXPLICIT",
    "REF-EXPLICIT": "CONSTRAIN",
    "NONE": "CONSTRAIN",
}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def hash_float(text: str) -> float:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(0xFFFFFFFF)


def normalize_text(x: Any) -> str:
    return str(x or "").strip()


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def slot_type(slot: str) -> str:
    return slot.split("-", 1)[1].strip() if "-" in slot else slot.strip()


def mask_slot(slot: str) -> str:
    typ = slot_type(slot)
    return f"[domain]-{typ}" if typ else "[domain]"


def mask_domain_text(text: str) -> str:
    if not text:
        return ""
    pat = r"\b(" + "|".join(map(re.escape, DOMAINS)) + r")-"
    text = re.sub(pat, "[domain]-", text)
    for domain in DOMAINS:
        text = re.sub(rf"\b{re.escape(domain)}\b", "[domain]", text, flags=re.I)
    return text


def maybe_mask(text: str, enabled: bool) -> str:
    return mask_domain_text(text) if enabled else text


def slot_desc_text(slot: str, slot_desc: Mapping[str, Any]) -> str:
    obj = slot_desc.get(slot, {})
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, Mapping):
        return ""
    parts: List[str] = []
    for key in ("concise_desc", "contras_desc", "type_desc", "description"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    vals = obj.get("value") or obj.get("values") or obj.get("value_examples")
    if isinstance(vals, list):
        parts.extend(str(v) for v in vals[:20])
    elif isinstance(vals, str):
        parts.append(vals)
    return " ".join(parts)


def lexical_similarity(slot: str, context_text: str, slot_desc: Mapping[str, Any]) -> float:
    slot_tokens = tokenize(slot.replace("-", " ") + " " + slot_desc_text(slot, slot_desc))
    ctx_tokens = tokenize(context_text)
    if not slot_tokens or not ctx_tokens:
        return 0.0
    inter = len(slot_tokens & ctx_tokens)
    return inter / math.sqrt(len(slot_tokens) * len(ctx_tokens))


def get_dialog_id(dialog: Mapping[str, Any], fallback: str) -> str:
    return str(dialog.get("dial_id") or dialog.get("dialogue_id") or dialog.get("dialog_id") or fallback)


def get_turns(dialog: Mapping[str, Any]) -> List[Dict[str, Any]]:
    turns = dialog.get("turns")
    if not isinstance(turns, list):
        turns = dialog.get("dialog")
    return turns if isinstance(turns, list) else []


def get_turn_id(turn: Mapping[str, Any], idx: int) -> int:
    try:
        return int(turn.get("turn_id", idx))
    except Exception:
        return idx


def get_turn_text(turns: Sequence[Mapping[str, Any]], idx: int, window: int = 1) -> str:
    start = max(0, idx - window)
    parts: List[str] = []
    for j in range(start, idx + 1):
        t = turns[j]
        parts.append(str(t.get("system", "") or t.get("sys", "")))
        parts.append(str(t.get("user", "") or t.get("usr", "")))
    return "\n".join(x for x in parts if x)


def event_key(slot: str, action: str) -> str:
    return f"{slot}\t{normalize_action(action)}"


def gold_events_by_slot(
    turn: Mapping[str, Any],
    event_field: str,
    use_label2_refer_implicit: bool,
) -> Dict[str, Dict[str, Any]]:
    events = turn.get(event_field)
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(events, list):
        return out
    for ev in events:
        if not isinstance(ev, Mapping):
            continue
        slot = normalize_text(ev.get("slot"))
        if not slot:
            continue
        action = parse_event_action(ev, use_label2_refer_implicit=use_label2_refer_implicit)
        copied = dict(ev)
        copied["_action"] = action
        out[slot] = copied
    return out


def all_slots_from_desc(slot_desc: Mapping[str, Any], domains: Optional[Set[str]]) -> List[str]:
    slots = sorted(str(k) for k in slot_desc.keys())
    if domains:
        slots = [s for s in slots if get_domain(s) in domains]
    return slots


def make_pair(
    pair_type: str,
    dial_id: str,
    turn_id: int,
    turn_index: int,
    slot: str,
    chosen: str,
    rejected: str,
    dsm: DynamicSlotMemory,
    slot_desc: Mapping[str, Any],
    ref_slot: str = "",
    value: str = "",
    masked: bool = False,
    reason: str = "",
) -> Dict[str, Any]:
    ctx = dsm.build_context(slot, turn_id, ref_slot=ref_slot).as_dict()
    desc_obj = slot_desc.get(slot, {})
    if not isinstance(desc_obj, Mapping):
        desc_obj = {}
    contras_desc = str(desc_obj.get("contras_desc", "") or desc_obj.get("description", "") or "")
    type_desc = str(desc_obj.get("type_desc", "") or desc_obj.get("concise_desc", "") or "")
    return {
        "pair_type": pair_type,
        "dial_id": dial_id,
        "turn_id": turn_id,
        "turn_index": turn_index,
        "slot": slot,
        "target_slot": mask_slot(slot) if masked else slot,
        "chosen": normalize_action(chosen),
        "rejected": normalize_action(rejected),
        "value": value,
        "ref_slot": ref_slot,
        "masked": masked,
        "reason": reason,
        "dsm_context": {
            k: maybe_mask(v, masked) for k, v in ctx.items()
        },
        "slot_descriptions": {
            "contras_desc": maybe_mask(contras_desc, masked),
            "type_desc": maybe_mask(type_desc, masked),
        },
    }


def load_sft_predictions(path: str) -> Dict[Tuple[str, int, str], str]:
    """Load optional SFT errors for P_C.

    Accepted JSONL/JSON record shapes:
      {"dial_id": "...", "turn_id": 1, "slot": "...", "pred": "CHANGE"}
      {"dial_id": "...", "turn_id": 1, "pred_detail": [{"slot": ..., "pred": ...}]}
    """

    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    rows: List[Any]
    if p.suffix.lower() == ".jsonl":
        rows = list(iter_jsonl(str(p)))
    else:
        obj = load_json(str(p))
        rows = obj if isinstance(obj, list) else list(obj.values()) if isinstance(obj, dict) else []

    out: Dict[Tuple[str, int, str], str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        dial_id = str(row.get("dial_id") or row.get("dialogue_id") or "")
        if "slot" in row and "pred" in row:
            out[(dial_id, int(row.get("turn_id", 0)), str(row["slot"]))] = normalize_action(row["pred"])
        details = row.get("pred_detail")
        if isinstance(details, list):
            turn_id = int(row.get("turn_id", 0))
            for item in details:
                if isinstance(item, Mapping) and "slot" in item:
                    pred = item.get("pred_action") or item.get("pred_label") or item.get("pred")
                    out[(dial_id, turn_id, str(item["slot"]))] = normalize_action(pred)
    return out


def build_pairs(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    data = load_json(args.train_path)
    slot_desc = load_json(args.slot_desc_path)
    domains = set(args.domains.split(",")) if args.domains else None
    all_slots = all_slots_from_desc(slot_desc, domains)
    sft_preds = load_sft_predictions(args.sft_pred_path) if args.sft_pred_path else {}

    if isinstance(data, Mapping):
        dialogs = list(data.items())
    elif isinstance(data, list):
        dialogs = [(str(i), d) for i, d in enumerate(data)]
    else:
        raise ValueError("train_path must be a JSON list or dict of dialogues.")

    seen_pairs: Set[Tuple[str, int, str, str, str]] = set()

    for fallback_id, dialog in dialogs:
        if not isinstance(dialog, Mapping):
            continue
        dial_id = get_dialog_id(dialog, fallback_id)
        turns = get_turns(dialog)
        dsm = DynamicSlotMemory(turns=turns, history_window=args.history_window)

        for t_idx, turn in enumerate(turns):
            if not isinstance(turn, Mapping):
                continue
            turn_id = get_turn_id(turn, t_idx)
            gold_by_slot = gold_events_by_slot(
                turn,
                event_field=args.event_field,
                use_label2_refer_implicit=args.use_label2_refer_implicit,
            )
            active_slots = set(gold_by_slot.keys())
            context_text = get_turn_text(turns, t_idx, window=args.surface_window)
            masked = hash_float(f"{args.seed}|mask|{dial_id}|{turn_id}") < args.mask_prob

            def emit(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                key = (
                    row["dial_id"],
                    int(row["turn_id"]),
                    row["slot"],
                    row["chosen"],
                    row["rejected"],
                )
                if row["chosen"] == row["rejected"] or key in seen_pairs:
                    return None
                seen_pairs.add(key)
                return row

            # P_B and P_C for gold activated slots.
            for slot, ev in gold_by_slot.items():
                gold_action = normalize_action(ev.get("_action"))
                if gold_action not in ACTIVE_ACTIONS:
                    continue
                ref_slot = normalize_text(ev.get("ref_slot") or ev.get("refer_slot"))
                value = normalize_text(ev.get("value") or ev.get("val"))

                rejected = CONFUSION_MAP.get(gold_action, "CONSTRAIN")
                row = emit(make_pair(
                    pair_type="P_B_label_confusion",
                    dial_id=dial_id,
                    turn_id=turn_id,
                    turn_index=t_idx,
                    slot=slot,
                    chosen=gold_action,
                    rejected=rejected,
                    dsm=dsm,
                    slot_desc=slot_desc,
                    ref_slot=ref_slot,
                    value=value,
                    masked=masked,
                    reason=f"Phi({gold_action})={rejected}",
                ))
                if row:
                    yield row

                pred = sft_preds.get((dial_id, turn_id, slot))
                if pred and pred != gold_action and pred != rejected:
                    row = emit(make_pair(
                        pair_type="P_C_sft_error",
                        dial_id=dial_id,
                        turn_id=turn_id,
                        turn_index=t_idx,
                        slot=slot,
                        chosen=gold_action,
                        rejected=pred,
                        dsm=dsm,
                        slot_desc=slot_desc,
                        ref_slot=ref_slot,
                        value=value,
                        masked=masked,
                        reason="SFT predicted wrong action",
                    ))
                    if row:
                        yield row

            inactive_slots = [s for s in all_slots if s not in active_slots]

            # P_A1: highest lexical-similarity inactive slot in this turn.
            if inactive_slots:
                ranked = sorted(
                    inactive_slots,
                    key=lambda s: (
                        lexical_similarity(s, context_text, slot_desc),
                        -hash_float(f"{args.seed}|surface|{dial_id}|{turn_id}|{s}"),
                    ),
                    reverse=True,
                )
                for slot in ranked[: args.max_surface_pairs_per_turn]:
                    if lexical_similarity(slot, context_text, slot_desc) < args.min_surface_score:
                        continue
                    row = emit(make_pair(
                        pair_type="P_A1_surface_mention",
                        dial_id=dial_id,
                        turn_id=turn_id,
                        turn_index=t_idx,
                        slot=slot,
                        chosen="NONE",
                        rejected="CONSTRAIN",
                        dsm=dsm,
                        slot_desc=slot_desc,
                        masked=masked,
                        reason="inactive slot with highest context similarity",
                    ))
                    if row:
                        yield row

            # P_A2: inactive slot with non-empty memory; reject latest historical action.
            memory_slots = [
                s for s in inactive_slots
                if dsm.slot_memory.get(s)
            ]
            for slot in memory_slots[: args.max_memory_pairs_per_turn]:
                latest = dsm.slot_memory[slot][-1]
                row = emit(make_pair(
                    pair_type="P_A2_memory_interference",
                    dial_id=dial_id,
                    turn_id=turn_id,
                    turn_index=t_idx,
                    slot=slot,
                    chosen="NONE",
                    rejected=latest.action,
                    dsm=dsm,
                    slot_desc=slot_desc,
                    masked=masked,
                    reason="inactive slot has historical memory",
                ))
                if row:
                    yield row

            # Update DSM after all current-turn pairs are constructed.
            for ev in gold_by_slot.values():
                dsm.update_from_event(
                    ev,
                    turn_id=turn_id,
                    use_label2_refer_implicit=args.use_label2_refer_implicit,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--slot-desc-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--event-field", default="taklabels")
    parser.add_argument("--sft-pred-path", default="")
    parser.add_argument("--domains", default="", help="Comma-separated domain filter, e.g. hotel,train")
    parser.add_argument("--seed", type=int, default=20251218)
    parser.add_argument("--mask-prob", type=float, default=0.3)
    parser.add_argument("--history-window", type=int, default=1)
    parser.add_argument("--surface-window", type=int, default=1)
    parser.add_argument("--min-surface-score", type=float, default=0.01)
    parser.add_argument("--max-surface-pairs-per-turn", type=int, default=1)
    parser.add_argument("--max-memory-pairs-per-turn", type=int, default=2)
    parser.add_argument("--use-label2-refer-implicit", action="store_true")
    args = parser.parse_args()

    count = write_jsonl(args.out_path, build_pairs(args))
    print(f"[DONE] wrote {count} raw DPO preference pairs -> {args.out_path}")


if __name__ == "__main__":
    main()
