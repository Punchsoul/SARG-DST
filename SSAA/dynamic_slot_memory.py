# -*- coding: utf-8 -*-
"""
Dynamic Slot Memory (DSM) for SARG-DST.

This module implements the mechanism described in the manuscript:

1. Maintain one chronological memory chain for each slot:
      M(slot) = [(turn_id, action, value, ref_slot), ...]
2. Maintain one domain-level latest-activation mapping:
      M_domain(domain) = {slot: latest_record}
3. Before predicting the action of a target slot at turn t, build a
   streamlined context from:
      - target-slot memory turns
      - current local turn context
      - same-domain latest slot activations
      - optional reference-slot memory
4. After the semantic action is predicted, update memory if action != NONE.

The code is dependency-free and can be plugged into the existing
research.py / retrivil.py data flow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


NONE_ACTIONS = {"", "NONE", "none", None}


ACTION_ALIASES = {
    "none": "NONE",
    "constrain": "CONSTRAIN",
    "constraint": "CONSTRAIN",
    "change": "CHANGE",
    "switch": "SWITCH",
    "dontcare": "DONTCARE",
    "don't care": "DONTCARE",
    "do not care": "DONTCARE",
    "confirm": "CONFIRM",
    "ref-explicit": "REF-EXPLICIT",
    "refer-explicit": "REF-EXPLICIT",
    "refer-clear": "REF-EXPLICIT",
    "ref-clear": "REF-EXPLICIT",
    "ref-implicit": "REF-IMPLICIT",
    "refer-implicit": "REF-IMPLICIT",
}


@dataclass
class Turn:
    """One dialogue turn."""

    turn_id: int
    system: str = ""
    user: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    """One activated slot memory item."""

    turn_id: int
    action: str
    value: str = ""
    ref_slot: str = ""


@dataclass
class DSMContext:
    """Prompt-ready DSM context sections."""

    memory_turns: str
    current_turn: str
    domain_context: str
    ref_slot_memory: str
    target_slot: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)

    def render(self) -> str:
        return (
            "[Streamlined Dialogue Context]\n"
            "[1] Memory Turns for Target Slot\n"
            f"{self.memory_turns}\n"
            "[2] Current Turn:\n"
            f"{self.current_turn}\n"
            "[3] Domain Context:\n"
            f"{self.domain_context}\n"
            "[4] Reference Slot Memory:\n"
            f"{self.ref_slot_memory}\n"
            "[Target Slot]\n"
            f"{self.target_slot}"
        )


def get_domain(slot: str) -> str:
    return slot.split("-", 1)[0] if "-" in slot else ""


def normalize_action(action: Any) -> str:
    if action is None:
        return "NONE"
    text = str(action).strip()
    if not text:
        return "NONE"
    lowered = text.lower()
    return ACTION_ALIASES.get(lowered, text.upper())


def is_active_action(action: Any) -> bool:
    return normalize_action(action) not in {"NONE"}


def label2_tokens(label2: Any) -> List[str]:
    if isinstance(label2, list):
        return [str(x).strip() for x in label2 if str(x).strip()]
    if isinstance(label2, str) and label2.strip():
        text = label2.strip()
        return [x.strip() for x in text.split(",")] if "," in text else [text]
    return []


def parse_event_action(
    event: Mapping[str, Any],
    use_label2_refer_implicit: bool = True,
) -> str:
    """Parse an event with label-1 / label-2 into a canonical DSM action.

    Priority follows the manuscript and your retrieval scripts:
      REF-EXPLICIT / refer-clear > CONFIRM > REF-IMPLICIT.
    If label-2 is non-empty and has no explicit reference, it becomes CONFIRM
    unless use_label2_refer_implicit is enabled and label-2 contains
    refer-implicit.
    """

    label1 = event.get("label-1") or event.get("label1") or event.get("label")
    toks = [normalize_action(x) for x in label2_tokens(event.get("label-2", ""))]

    if toks:
        if "REF-EXPLICIT" in toks:
            return "REF-EXPLICIT"
        if "CONFIRM" in toks:
            return "CONFIRM"
        if use_label2_refer_implicit and "REF-IMPLICIT" in toks:
            return "REF-IMPLICIT"
        return "CONFIRM"

    return normalize_action(label1)


class DynamicSlotMemory:
    """Dynamic Slot Memory manager.

    The object should be reset at the beginning of each dialogue. During
    training, call update_from_event/update after gold labels are known.
    During inference, call update after the model predicts a semantic action.
    """

    def __init__(
        self,
        turns: Optional[Sequence[Mapping[str, Any]]] = None,
        history_window: int = 1,
    ) -> None:
        self.history_window = max(0, int(history_window))
        self.turns: List[Turn] = []
        self.slot_memory: Dict[str, List[MemoryRecord]] = {}
        self.domain_memory: Dict[str, Dict[str, MemoryRecord]] = {}
        if turns is not None:
            self.set_turns(turns)

    def reset(self) -> None:
        self.slot_memory.clear()
        self.domain_memory.clear()

    def set_turns(self, turns: Sequence[Mapping[str, Any]]) -> None:
        self.turns = []
        for idx, item in enumerate(turns):
            turn_id = int(item.get("turn_id", idx))
            self.turns.append(
                Turn(
                    turn_id=turn_id,
                    system=str(item.get("system", "") or item.get("sys", "")),
                    user=str(item.get("user", "") or item.get("usr", "")),
                    raw=dict(item),
                )
            )

    def update(
        self,
        slot: str,
        turn_id: int,
        action: Any,
        value: Any = "",
        ref_slot: Any = "",
    ) -> None:
        action_text = normalize_action(action)
        slot = str(slot or "").strip()
        if not slot or action_text == "NONE":
            return

        record = MemoryRecord(
            turn_id=int(turn_id),
            action=action_text,
            value=str(value or "").strip(),
            ref_slot=str(ref_slot or "").strip(),
        )
        self.slot_memory.setdefault(slot, []).append(record)

        domain = get_domain(slot)
        if domain:
            self.domain_memory.setdefault(domain, {})[slot] = record

    def update_from_event(
        self,
        event: Mapping[str, Any],
        turn_id: int,
        use_label2_refer_implicit: bool = True,
    ) -> None:
        slot = str(event.get("slot", "") or "").strip()
        action = parse_event_action(event, use_label2_refer_implicit)
        value = event.get("value", event.get("val", ""))
        ref_slot = event.get("ref_slot", event.get("refer_slot", ""))
        self.update(slot=slot, turn_id=turn_id, action=action, value=value, ref_slot=ref_slot)

    def build_context(
        self,
        target_slot: str,
        turn_id: int,
        ref_slot: str = "",
    ) -> DSMContext:
        target_slot = str(target_slot or "").strip()
        turn_id = int(turn_id)
        return DSMContext(
            memory_turns=self._format_slot_memory(target_slot),
            current_turn=self._format_local_turns(turn_id, label="Current turn"),
            domain_context=self._format_domain_context(target_slot),
            ref_slot_memory=self._format_ref_memory(ref_slot),
            target_slot=target_slot,
        )

    def build_sft_prompt(
        self,
        target_slot: str,
        turn_id: int,
        contras_desc: str = "",
        type_desc: str = "",
        gold_semantic_action: str = "",
        ref_slot: str = "",
    ) -> str:
        context = self.build_context(target_slot, turn_id, ref_slot=ref_slot)
        return (
            "Instruction:\n"
            "A. You are a slot-level semantic action classifier for Dialogue State Tracking. "
            "Given the following information:\n"
            "(1) the streamlined dialogue context constructed by dynamic slot memory;\n"
            "(2) one target slot;\n"
            "(3) the semantic descriptions of the target slot;\n"
            "predict only one semantic action label of the target slot in the current turn only.\n"
            "Output_Format:\n"
            "{CONSTRAIN}, {SWITCH}, {CHANGE}, {DONTCARE}, {CONFIRM}, "
            "{REF-EXPLICIT}, {REF-IMPLICIT}, {NONE}\n"
            "If and only if the predicted label is \"REF-EXPLICIT\", an additional key "
            "\"ref_slot\" must be included.\n"
            "Input:\n"
            f"{context.render()}\n"
            "[Slot Descriptions]\n"
            f"- Contrastive description: {contras_desc}\n"
            f"- Type description: {type_desc}\n"
            "Output:\n"
            f"{normalize_action(gold_semantic_action) if gold_semantic_action else ''}"
        )

    def build_dpo_prompt(
        self,
        target_slot: str,
        turn_id: int,
        preferred_semantic_action: str,
        rejected_semantic_action: str,
        contras_desc: str = "",
        type_desc: str = "",
        ref_slot: str = "",
    ) -> str:
        context = self.build_context(target_slot, turn_id, ref_slot=ref_slot)
        return (
            "Instruction:\n"
            "A. You are a slot-level semantic action classifier for Dialogue State Tracking. "
            "Given the streamlined dialogue context, target slot, and slot descriptions, "
            "predict only one semantic action label of the target slot in the current turn only.\n"
            "Decision Focus:\n"
            "- Do not treat a surface-level mention as a real slot activation.\n"
            "- Do not copy historical actions from memory unless the target slot is triggered again "
            "in the current turn.\n"
            "Priority Rules:\n"
            "REF-EXPLICIT > CONFIRM > REF-IMPLICIT ; "
            "NONE > DONTCARE > SWITCH > CHANGE > CONSTRAIN\n"
            "Input:\n"
            f"{context.render()}\n"
            "[Slot Descriptions]\n"
            f"- Contrastive description: {contras_desc}\n"
            f"- Type description: {type_desc}\n"
            "Chosen Response:\n"
            f"{normalize_action(preferred_semantic_action)}\n"
            "Rejected Response:\n"
            f"{normalize_action(rejected_semantic_action)}"
        )

    def _format_slot_memory(self, slot: str) -> str:
        records = self.slot_memory.get(slot, [])
        if not records:
            return "None"

        chunks: List[str] = []
        for record in records:
            chunks.append(
                self._format_local_turns(
                    record.turn_id,
                    label="Memory turn",
                    action=record.action,
                    value=record.value,
                )
            )
        return "\n\n".join(chunks)

    def _format_ref_memory(self, ref_slot: str) -> str:
        ref_slot = str(ref_slot or "").strip()
        if not ref_slot:
            return "None"
        records = self.slot_memory.get(ref_slot, [])
        if not records:
            return f"[Reference slot] {ref_slot}\nNone"

        chunks = [f"[Reference slot] {ref_slot}"]
        for record in records:
            chunks.append(
                self._format_local_turns(
                    record.turn_id,
                    label="Ref turn",
                    action=record.action,
                    value=record.value,
                )
            )
        return "\n\n".join(chunks)

    def _format_domain_context(self, target_slot: str) -> str:
        domain = get_domain(target_slot)
        latest = self.domain_memory.get(domain, {})
        rows: List[str] = []
        for slot, record in sorted(latest.items()):
            if slot == target_slot:
                continue
            value_part = f", value={record.value}" if record.value else ""
            rows.append(
                f"- slot={slot}, latest_turn={record.turn_id}, action={record.action}{value_part}"
            )

        if rows:
            return "\n".join(rows)

        other_domains = [
            d for d, slot_map in sorted(self.domain_memory.items()) if d != domain and slot_map
        ]
        if other_domains:
            return "No prior activation in target domain. Activated other domains: " + ", ".join(other_domains)
        return "None"

    def _format_local_turns(
        self,
        turn_id: int,
        label: str,
        action: str = "",
        value: str = "",
    ) -> str:
        indices = self._turn_indices_for_window(turn_id)
        if not indices:
            return f"[{label} {turn_id}] None"

        start = self.turns[indices[0]].turn_id
        end = self.turns[indices[-1]].turn_id
        meta = f" (action: {action}" if action else ""
        if value:
            meta += f", value: {value}"
        if meta:
            meta += ")"

        lines = [f"[{label} {start}-{end}]{meta}"]
        for idx in indices:
            turn = self.turns[idx]
            if turn.system:
                lines.append(f"[t{turn.turn_id}][SYS] {turn.system}")
            if turn.user:
                lines.append(f"[t{turn.turn_id}][USR] {turn.user}")
        return "\n".join(lines)

    def _turn_indices_for_window(self, turn_id: int) -> List[int]:
        if not self.turns:
            return []

        index_by_turn_id = {turn.turn_id: i for i, turn in enumerate(self.turns)}
        if turn_id not in index_by_turn_id:
            return []
        end = index_by_turn_id[turn_id]
        start = max(0, end - self.history_window)
        return list(range(start, end + 1))


def attach_dsm_contexts_to_dialog(
    dialog: Mapping[str, Any],
    event_key: str = "taklabels",
    context_key: str = "dsm_context",
    turns_key: str = "turns",
    use_label2_refer_implicit: bool = True,
    history_window: int = 1,
) -> Dict[str, Any]:
    """Attach DSM context sections to each event in a dialogue.

    This helper is useful for building SSAA training data. It uses gold event
    labels to update memory after constructing all contexts for the current
    turn, so each context only sees historical memory from turns < t.
    """

    out = dict(dialog)
    turns = [dict(t) for t in out.get(turns_key, [])]
    out[turns_key] = turns

    dsm = DynamicSlotMemory(turns=turns, history_window=history_window)
    for idx, turn in enumerate(turns):
        turn_id = int(turn.get("turn_id", idx))
        events = turn.get(event_key)
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue
            slot = str(event.get("slot", "") or "").strip()
            ref_slot = str(event.get("ref_slot", event.get("refer_slot", "")) or "").strip()
            event[context_key] = dsm.build_context(slot, turn_id, ref_slot=ref_slot).as_dict()

        for event in events:
            if isinstance(event, dict):
                dsm.update_from_event(
                    event,
                    turn_id=turn_id,
                    use_label2_refer_implicit=use_label2_refer_implicit,
                )

    return out


def iter_dialogs_with_dsm_contexts(
    dialogs: Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> Iterable[Dict[str, Any]]:
    for dialog in dialogs:
        yield attach_dsm_contexts_to_dialog(dialog, **kwargs)


if __name__ == "__main__":
    demo_dialog = {
        "dial_id": "demo-1",
        "turns": [
            {
                "turn_id": 0,
                "system": "",
                "user": "I need a hotel in the west.",
                "taklabels": [
                    {"slot": "hotel-area", "label-1": "constrain", "value": "west"},
                ],
            },
            {
                "turn_id": 1,
                "system": "I found several hotels in the west.",
                "user": "Book it for Wednesday.",
                "taklabels": [
                    {"slot": "hotel-book day", "label-1": "constrain", "value": "wednesday"},
                ],
            },
        ],
    }
    annotated = attach_dsm_contexts_to_dialog(demo_dialog)
    print(annotated["turns"][1]["taklabels"][0]["dsm_context"]["domain_context"])
