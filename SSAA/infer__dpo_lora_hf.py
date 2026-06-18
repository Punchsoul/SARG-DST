# -*- coding: utf-8 -*-
"""
Infer SSAA semantic action labels with a DPO-aligned LoRA model.

The script loads a base model plus LoRA adapter, builds Dynamic Slot Memory
prompts for each target slot, predicts one semantic action label, and updates
DSM with rolling predictions. The output dialogue file receives:

turn["pred_taklabels"] = [
  {"slot": "...", "label": "CONSTRAIN", "value": "", "ref_slot": "", "raw_gen": "..."},
  ...
]

By default NONE predictions are omitted from pred_taklabels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from dynamic_slot_memory import DynamicSlotMemory, get_domain, normalize_action
from convert_ssaa_dpo_to_llamafactory import SYSTEM_TEXT, build_user_prompt

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


LABELS = [
    "CONSTRAIN",
    "SWITCH",
    "CHANGE",
    "DONTCARE",
    "CONFIRM",
    "REF-EXPLICIT",
    "REF-IMPLICIT",
    "NONE",
]

LABEL_RE = re.compile(
    r"\b(CONSTRAIN|SWITCH|CHANGE|DONTCARE|CONFIRM|REF[-_ ]?EXPLICIT|REF[-_ ]?IMPLICIT|NONE)\b",
    re.I,
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def target_slots_from_desc(slot_desc: Mapping[str, Any], target_domain: str = "", slot_file: str = "") -> List[str]:
    if slot_file:
        obj = load_json(slot_file)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, Mapping):
            return [str(x) for x in obj.keys()]
        raise ValueError("--slot-file must be a JSON list or dict.")

    slots = sorted(str(k) for k in slot_desc.keys())
    if target_domain:
        slots = [s for s in slots if get_domain(s) == target_domain]
    return slots


def parse_label(text: str) -> str:
    match = LABEL_RE.search(text or "")
    if not match:
        return "NONE"
    return normalize_action(match.group(1).replace("_", "-").replace(" ", "-"))


def render_chat(tokenizer: Any, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_batch(
    model: Any,
    tokenizer: Any,
    rendered_prompts: Sequence[str],
    max_new_tokens: int,
    max_length: int,
) -> List[str]:
    import torch

    with torch.no_grad():
        enc = tokenizer(
            list(rendered_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        return tokenizer.batch_decode(gen, skip_special_tokens=True)


def gold_action_for_eval(turn: Mapping[str, Any], slot: str, event_field: str) -> str:
    events = turn.get(event_field)
    if not isinstance(events, list):
        return "NONE"
    for ev in events:
        if isinstance(ev, Mapping) and str(ev.get("slot", "")) == slot:
            from dynamic_slot_memory import parse_event_action

            return parse_event_action(ev, use_label2_refer_implicit=True)
    return "NONE"


def make_prompt_pair(
    dsm: DynamicSlotMemory,
    slot: str,
    turn_id: int,
    slot_desc: Mapping[str, Any],
    ref_slot: str = "",
) -> Dict[str, Any]:
    ctx = dsm.build_context(slot, turn_id, ref_slot=ref_slot).as_dict()
    desc_obj = slot_desc.get(slot, {})
    if not isinstance(desc_obj, Mapping):
        desc_obj = {}
    return {
        "slot": slot,
        "target_slot": slot,
        "chosen": "NONE",
        "rejected": "CONSTRAIN",
        "dsm_context": ctx,
        "slot_descriptions": {
            "contras_desc": str(desc_obj.get("contras_desc", "") or desc_obj.get("description", "") or "None"),
            "type_desc": str(desc_obj.get("type_desc", "") or desc_obj.get("concise_desc", "") or "None"),
        },
    }


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and args.bf16 else torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    return model, tokenizer


def run_inference(args: argparse.Namespace) -> List[Dict[str, Any]]:
    slot_desc = load_json(args.slot_desc_path)
    target_slots = target_slots_from_desc(slot_desc, target_domain=args.target_domain, slot_file=args.slot_file)
    if args.slot_topk is not None:
        target_slots = target_slots[: args.slot_topk]
    if not target_slots:
        raise RuntimeError("No target slots found.")

    data = load_json(args.input_path)
    if isinstance(data, Mapping):
        dialogs = list(data.values())
    elif isinstance(data, list):
        dialogs = data
    else:
        raise ValueError("--input-path must be a JSON list or dict of dialogues.")

    model, tokenizer = load_model_and_tokenizer(args)

    total_calls = 0
    eval_total = 0
    eval_correct = 0

    for didx, dialog in enumerate(tqdm(dialogs, desc="Infer SSAA-DPO")):
        if not isinstance(dialog, Mapping):
            continue
        turns = get_turns(dialog)
        dsm = DynamicSlotMemory(turns=turns, history_window=args.history_window)
        dial_id = get_dialog_id(dialog, str(didx))

        for t_idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            turn_id = get_turn_id(turn, t_idx)

            rendered_inputs: List[str] = []
            metas: List[Dict[str, Any]] = []

            slots_this_turn = target_slots
            if args.candidate_field:
                cands = turn.get(args.candidate_field)
                if isinstance(cands, list) and cands:
                    cand_set = set(str(x) for x in cands)
                    slots_this_turn = [s for s in target_slots if s in cand_set]

            for slot in slots_this_turn:
                pair_like = make_prompt_pair(dsm, slot, turn_id, slot_desc)
                user_prompt = build_user_prompt(pair_like, include_label_definitions=not args.no_label_definitions)
                rendered_inputs.append(render_chat(tokenizer, SYSTEM_TEXT, user_prompt))
                metas.append({"slot": slot})

            details: List[Dict[str, Any]] = []
            pred_events: List[Dict[str, Any]] = []

            for start in range(0, len(rendered_inputs), args.batch_size):
                batch_inputs = rendered_inputs[start:start + args.batch_size]
                batch_metas = metas[start:start + args.batch_size]
                texts = generate_batch(
                    model,
                    tokenizer,
                    batch_inputs,
                    max_new_tokens=args.max_new_tokens,
                    max_length=args.max_length,
                )
                total_calls += len(batch_inputs)

                for text, meta in zip(texts, batch_metas):
                    slot = meta["slot"]
                    label = parse_label(text)
                    detail = {
                        "slot": slot,
                        "pred_label": label,
                        "raw_gen": str(text).strip(),
                    }
                    if args.eval_event_field:
                        gold = gold_action_for_eval(turn, slot, args.eval_event_field)
                        detail["gold_label"] = gold
                        eval_total += 1
                        eval_correct += int(gold == label)
                    details.append(detail)

                    if label != "NONE" or args.keep_none:
                        pred_events.append({
                            "slot": slot,
                            "label": label,
                            "value": "",
                            "ref_slot": "",
                            "raw_gen": str(text).strip(),
                        })

            turn[args.output_field] = pred_events
            turn[f"{args.output_field}_detail"] = details

            # Rolling DSM update uses predicted active labels only.
            for ev in pred_events:
                if ev["label"] != "NONE":
                    dsm.update(
                        slot=ev["slot"],
                        turn_id=turn_id,
                        action=ev["label"],
                        value=ev.get("value", ""),
                        ref_slot=ev.get("ref_slot", ""),
                    )

        if isinstance(dialog, dict):
            dialog["_ssaa_dpo_infer"] = {
                "dial_id": dial_id,
                "target_domain": args.target_domain,
                "num_target_slots": len(target_slots),
            }

    print(f"[INFO] total_model_calls={total_calls}")
    if eval_total:
        print(f"[EVAL] label_acc={eval_correct / eval_total:.4f} ({eval_correct}/{eval_total})")
    return dialogs  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--slot-desc-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--target-domain", default="")
    parser.add_argument("--slot-file", default="")
    parser.add_argument("--candidate-field", default="")
    parser.add_argument("--output-field", default="pred_taklabels")
    parser.add_argument("--eval-event-field", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--history-window", type=int, default=1)
    parser.add_argument("--slot-topk", type=int, default=None)
    parser.add_argument("--keep-none", action="store_true")
    parser.add_argument("--no-label-definitions", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    dialogs = run_inference(args)
    save_json(args.out_path, dialogs)
    print(f"[DONE] saved inferred labels -> {args.out_path}")


if __name__ == "__main__":
    main()
