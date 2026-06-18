# -*- coding: utf-8 -*-
"""
Retrieval module (ID-only return) for CROSS-DOMAIN ZERO-SHOT DST.

Key changes (as per your latest requirement):
1) FORCE query events to come ONLY from `taklabels` (no tasklabels fallback).
2) Label parsing for retrieval MUST align with graph construction:
   - If label-2 is non-empty: prefer label-2
   - Priority: refer-clear > confirm > refer-implicit
     Concretely:
       if label-2 non-empty:
          if contains refer-clear -> refer-clear
          elif contains refer-implicit -> (LOWEST; only used if we *explicitly* decide to)
          else -> confirm
       else:
          use label-1
   Because you stated refer-clear > confirm > refer-implicit, this implementation:
     - returns refer-clear if present
     - otherwise returns confirm whenever label-2 is non-empty
     - returns refer-implicit ONLY when label-2 is empty AND label-1 == refer-implicit (rare, but keeps the label reachable)
   If you want refer-implicit to be used when label-2 explicitly contains it (despite your priority), set:
      USE_LABEL2_REFER_IMPLICIT = True

Other design choices unchanged:
- Scheme B: build SimSlot and Alpha matrices IN MEMORY (no files saved).
- Two switches:
    EXCEPT_DOMAIN, MODE ("finetune" or "infer")
- Filter by label first, then within each candidate turn use only same-label slots.
- Alpha mapping: alpha = clip(0.2 + 0.3*sim_slot, 0.2, 0.5), w_ctx=1-alpha, w_slot=alpha
- Return topK IDs only: ret_id = f"{db_graph_id}::{matched_slot}"
- Write a NEW file A' with per-taklabel field: retrieved_topk_ids
"""

import os
import json
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# =========================
# User-editable run params
# =========================
SLOT_DESC_PATH = "/home/zzp/ds_loo/ontology/slot_descriptionsx.json"
DB_GRAPH_PATH  = "/home/zzp/ds_loo/research/turn_attraction_partial.jsonl"
QUERY_A_PATH   = "/home/zzp/ds_loo/research/dialogs_attraction_nondb.json"  # can be JSONL of turns; see reader below
LOCAL_MODEL_PATH = "/home/zzp/senbert"

# Output file (A')
OUTPUT_A_PATH = "/home/zzp/ds_loo/research/turn_attraction_with_retrieval.jsonl"

# Two switches
EXCEPT_DOMAIN = "attraction"      # <-- change to your target domain
MODE = "finetune"                 # "finetune" or "infer"

TOPK = 3
BATCH_SIZE_DB_CTX = 256
BATCH_SIZE_QUERY_CTX = 256
BATCH_SIZE_SLOT_DESC = 128

# Optional safety: forbid retrieving turns from the same dial_id as the query
FORBID_SAME_DIAL = True
TIEBREAK_BY_GRAPH_ID = True

# NEW: strictly follow your stated priority refer-clear > confirm > refer-implicit for label-2.
# Default False means: if label-2 contains refer-implicit but no refer-clear, we still return confirm.
USE_LABEL2_REFER_IMPLICIT = False

# If query file is a JSON list of dialogs (not JSONL), we can build ctx on the fly.
# For JSONL without context, we can't do multi-turn window.
WINDOW_K_FOR_DIALOG_JSON = 2


# =========================
# Helpers
# =========================
def get_domain(slot: str) -> str:
    return slot.split("-", 1)[0] if "-" in slot else ""


def serialize_slot_desc(desc_obj: Any) -> str:
    if desc_obj is None:
        return ""
    if isinstance(desc_obj, str):
        return desc_obj.strip()

    if isinstance(desc_obj, dict):
        skip_keys = {"is_transferable"}  # meta only
        parts: List[str] = []
        for k in sorted(desc_obj.keys()):
            if k in skip_keys:
                continue
            v = desc_obj[k]
            if v is None:
                continue
            if isinstance(v, str):
                vv = v.strip()
                if vv:
                    parts.append(f"{k}: {vv}")
            elif isinstance(v, (list, tuple)):
                vs = [str(x).strip() for x in v if str(x).strip()]
                if vs:
                    if len(vs) > 20:
                        vs = vs[:20]
                    parts.append(f"{k}: " + " | ".join(vs))
            else:
                vv = str(v).strip()
                if vv:
                    parts.append(f"{k}: {vv}")
        return "\n".join(parts).strip()

    return str(desc_obj).strip()


def clip_alpha(sim: np.ndarray) -> np.ndarray:
    alpha = 0.2 + 0.3 * sim
    return np.clip(alpha, 0.2, 0.5).astype(np.float32)


def _label2_tokens(label2: Any) -> List[str]:
    if isinstance(label2, list) and len(label2) > 0:
        return [str(x).strip() for x in label2 if str(x).strip()]
    if isinstance(label2, str) and label2.strip():
        s = label2.strip()
        return [x.strip() for x in s.split(",")] if "," in s else [s]
    return []


def parse_label_for_retrieval_from_taklabel(ev: Dict[str, Any]) -> str:
    """
    STRICT (as requested):
    - If label-2 non-empty: prefer label-2 with priority refer-clear > confirm > refer-implicit
    - Else: use label-1

    Implementation:
    - If label-2 has "refer-clear" => refer-clear
    - Else if label-2 non-empty:
         - if USE_LABEL2_REFER_IMPLICIT and label-2 has "refer-implicit" => refer-implicit
         - else => confirm
    - Else => label-1
    """
    label1 = (ev.get("label-1") or "").strip()
    label2 = ev.get("label-2", "")
    toks = _label2_tokens(label2)

    if toks:  # label-2 non-empty => prefer label-2
        if "refer-clear" in toks:
            return "refer-clear"
        # refer-clear > confirm > refer-implicit (your requirement)
        if USE_LABEL2_REFER_IMPLICIT and ("refer-implicit" in toks):
            return "refer-implicit"
        return "confirm"

    # label-2 empty => use label-1
    return label1


def is_json_array_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return False
            if ch.isspace():
                continue
            return ch == "["


def build_ctx_from_dialog_turns(turns: List[Dict[str, Any]], t_idx: int, k: int) -> str:
    start = max(0, t_idx - k)
    parts: List[str] = []
    for i in range(start, t_idx + 1):
        sys_utt = turns[i].get("system", "")
        usr_utt = turns[i].get("user", "")
        parts.append(f"[t{i}][SYS] {sys_utt}")
        parts.append(f"[t{i}][USR] {usr_utt}")
    return "\n".join(parts).strip()


# =========================
# Load slot descriptions & build slot sets
# =========================
print(f"[LOAD] slot descriptions: {SLOT_DESC_PATH}")
with open(SLOT_DESC_PATH, "r", encoding="utf-8") as f:
    slot_desc = json.load(f)
all_slots = sorted(list(slot_desc.keys()))

tgt_slots = [s for s in all_slots if get_domain(s) == EXCEPT_DOMAIN]
non_slots = [s for s in all_slots if get_domain(s) != EXCEPT_DOMAIN]

if MODE not in ("finetune", "infer"):
    raise ValueError(f"MODE must be 'finetune' or 'infer', got: {MODE}")

if MODE == "finetune":
    q_slots = non_slots
    db_slots = non_slots
else:
    q_slots = tgt_slots
    db_slots = non_slots

print(f"[SLOTS] total={len(all_slots)} | tgt({EXCEPT_DOMAIN})={len(tgt_slots)} | non={len(non_slots)}")
print(f"[SLOTS] MODE={MODE} => q_slots={len(q_slots)} | db_slots={len(db_slots)}")

q_slot2idx = {s: i for i, s in enumerate(q_slots)}
db_slot2idx = {s: i for i, s in enumerate(db_slots)}


# =========================
# Load model
# =========================
print(f"[MODEL] loading SBERT from: {LOCAL_MODEL_PATH}")
model = SentenceTransformer(LOCAL_MODEL_PATH)

# =========================
# Compute slot embeddings (only needed slots)
# =========================
needed_slots = sorted(set(q_slots).union(set(db_slots)))
slot_texts = [serialize_slot_desc(slot_desc.get(s)) for s in needed_slots]

print(f"[EMB] encoding slot_desc for {len(needed_slots)} slots ...")
slot_emb = model.encode(
    slot_texts,
    batch_size=BATCH_SIZE_SLOT_DESC,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
slot2emb = {s: slot_emb[i].astype(np.float32) for i, s in enumerate(needed_slots)}

# normalized => cosine == dot
q_emb = np.stack([slot2emb[s] for s in q_slots], axis=0) if q_slots else np.zeros((0, 768), dtype=np.float32)
db_emb = np.stack([slot2emb[s] for s in db_slots], axis=0) if db_slots else np.zeros((0, 768), dtype=np.float32)

print("[SIM] building SimSlot and Alpha matrices in memory ...")
sim_slot_mat = (q_emb @ db_emb.T).astype(np.float32) if (len(q_slots) and len(db_slots)) else np.zeros((len(q_slots), len(db_slots)), dtype=np.float32)
alpha_mat = clip_alpha(sim_slot_mat)  # currently not directly used; kept for clarity


# =========================
# Load DB graphs and build label buckets / per-turn label->slots
# =========================
print(f"[LOAD] DB graph store: {DB_GRAPH_PATH}")
db_graph_ids: List[str] = []
db_dial_ids: List[str] = []
db_ctx_texts: List[str] = []
turn_label_slots: List[Dict[str, List[int]]] = []  # idx -> {label: [db_slot_idx...]}

label_bucket: Dict[str, List[int]] = {}

with open(DB_GRAPH_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        graph_id = obj.get("graph_id") or f"{obj.get('dial_id','')}::{obj.get('turn_id','')}"
        dial_id = obj.get("dial_id") or ""

        ctx = obj.get("context") or {}
        ctx_text = (ctx.get("retrieval_text") or "").strip()

        tls = obj.get("tasklabels") or []
        if not isinstance(tls, list):
            tls = []

        labels = obj.get("labels_for_retrieval")
        if not isinstance(labels, list) or not labels:
            labels = sorted({(tl.get("label") or "").strip() for tl in tls if (tl.get("label") or "").strip()})

        idx = len(db_graph_ids)
        db_graph_ids.append(graph_id)
        db_dial_ids.append(dial_id)
        db_ctx_texts.append(ctx_text)

        # build per-turn label -> db_slot_idx list
        l2slots: Dict[str, List[int]] = {}
        for tl in tls:
            lab = (tl.get("label") or "").strip()
            s = (tl.get("slot") or "").strip()
            if not lab or not s:
                continue
            if s not in db_slot2idx:
                continue
            l2slots.setdefault(lab, []).append(db_slot2idx[s])

        for lab in list(l2slots.keys()):
            l2slots[lab] = sorted(list(set(l2slots[lab])))

        turn_label_slots.append(l2slots)

        for lab in labels:
            lab = str(lab).strip()
            if not lab:
                continue
            label_bucket.setdefault(lab, []).append(idx)

print(f"[DB] turns loaded: {len(db_graph_ids)}")
print(f"[DB] labels in bucket: {len(label_bucket)}")


# =========================
# Encode DB ctx embeddings
# =========================
print("[EMB] encoding DB ctx embeddings ...")
db_ctx_emb = model.encode(
    db_ctx_texts,
    batch_size=BATCH_SIZE_DB_CTX,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)


# =========================
# Precompute (per label) best_sim / best_slot for each query slot
# =========================
print("[PRECOMP] building per-label best_sim/best_slot tables ...")
Q = len(q_slots)

label_tables: Dict[str, Dict[str, Any]] = {}

for lab, cand in tqdm(label_bucket.items(), desc="Precompute by label", ncols=100):
    cand_idx = np.array(cand, dtype=np.int32)
    N = cand_idx.shape[0]
    if Q == 0 or N == 0:
        continue

    best_sim = np.full((Q, N), -1.0, dtype=np.float32)
    best_slot = np.full((Q, N), -1, dtype=np.int32)

    for col in range(N):
        db_i = int(cand_idx[col])
        sidxs = turn_label_slots[db_i].get(lab, [])
        if not sidxs:
            continue
        sidxs_arr = np.array(sidxs, dtype=np.int32)

        sims = sim_slot_mat[:, sidxs_arr]  # [Q, M]
        arg = np.argmax(sims, axis=1)      # [Q]
        maxv = sims[np.arange(Q), arg]     # [Q]
        best_sim[:, col] = maxv
        best_slot[:, col] = sidxs_arr[arg]

    best_alpha = clip_alpha(best_sim)

    label_tables[lab] = {
        "cand_idx": cand_idx,
        "best_sim": best_sim,
        "best_slot": best_slot,
        "best_alpha": best_alpha,
    }

print(f"[PRECOMP] label_tables built: {len(label_tables)}")


# =========================
# Retrieval function
# =========================
def retrieve_topk_ids(
    q_dial_id: str,
    q_label: str,
    q_slot: str,
    q_ctx_emb: np.ndarray,
) -> List[str]:
    q_label = (q_label or "").strip()
    q_slot = (q_slot or "").strip()
    if not q_label or not q_slot:
        return []

    if q_slot not in q_slot2idx:
        return []

    tbl = label_tables.get(q_label)
    if not tbl:
        return []

    q_idx = q_slot2idx[q_slot]
    cand_idx = tbl["cand_idx"]              # [N]
    best_sim_vec = tbl["best_sim"][q_idx]   # [N]
    best_slot_vec = tbl["best_slot"][q_idx] # [N]
    alpha_vec = tbl["best_alpha"][q_idx]    # [N]

    valid = (best_slot_vec >= 0)
    if not np.any(valid):
        return []

    sim_ctx_vec = db_ctx_emb[cand_idx] @ q_ctx_emb  # [N]
    score = (1.0 - alpha_vec) * sim_ctx_vec + alpha_vec * best_sim_vec

    if FORBID_SAME_DIAL and q_dial_id:
        same = np.array([db_dial_ids[int(i)] == q_dial_id for i in cand_idx], dtype=bool)
        score[same] = -1e9

    score[~valid] = -1e9

    N = score.shape[0]
    k = min(TOPK, N)
    if k <= 0:
        return []

    top_idx = np.argpartition(score, -k)[-k:]
    if TIEBREAK_BY_GRAPH_ID:
        top_idx = sorted(
            top_idx.tolist(),
            key=lambda j: (float(score[j]), db_graph_ids[int(cand_idx[j])]),
            reverse=True
        )
        top_idx = np.array(top_idx, dtype=np.int32)
    else:
        top_idx = top_idx[np.argsort(score[top_idx])[::-1]]

    out: List[str] = []
    for j in top_idx:
        db_i = int(cand_idx[int(j)])
        matched_db_slot_idx = int(best_slot_vec[int(j)])
        matched_slot = db_slots[matched_db_slot_idx] if 0 <= matched_db_slot_idx < len(db_slots) else ""
        out.append(f"{db_graph_ids[db_i]}::{matched_slot}")
    return out


# =========================
# Query file processing (FORCE taklabels)
# =========================
print(f"[RUN] annotate query file A: {QUERY_A_PATH}")
print(f"[OUT]  write annotated A': {OUTPUT_A_PATH}")
os.makedirs(os.path.dirname(OUTPUT_A_PATH), exist_ok=True)


def flush_batch_turn_jsonl(out_fp, buffer_objs, buffer_ctx_texts):
    if not buffer_objs:
        return

    q_ctx_embs = model.encode(
        buffer_ctx_texts,
        batch_size=BATCH_SIZE_QUERY_CTX,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    for obj, q_emb in zip(buffer_objs, q_ctx_embs):
        q_dial_id = obj.get("dial_id") or ""

        # FORCE taklabels only
        events = obj.get("taklabels")
        if not isinstance(events, list):
            events = []

        for ev in events:
            if not isinstance(ev, dict):
                continue
            q_slot = (ev.get("slot") or "").strip()
            q_label = parse_label_for_retrieval_from_taklabel(ev)
            ev["retrieved_topk_ids"] = retrieve_topk_ids(q_dial_id=q_dial_id, q_label=q_label, q_slot=q_slot, q_ctx_emb=q_emb)

        out_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

    buffer_objs.clear()
    buffer_ctx_texts.clear()


def annotate_jsonl_turn_file():
    buffer_objs: List[Dict[str, Any]] = []
    buffer_ctx_texts: List[str] = []

    with open(QUERY_A_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_A_PATH, "w", encoding="utf-8") as fout:
        pbar = tqdm(desc="Annotating JSONL turns with retrieval IDs", ncols=110)
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            ctx = obj.get("context") or {}
            ctx_text = (ctx.get("retrieval_text") or "").strip()

            # If JSONL lacks context, we can only fall back to current turn (no history here).
            if not ctx_text:
                sys_utt = obj.get("system", "")
                usr_utt = obj.get("user", "")
                ctx_text = f"[SYS] {sys_utt}\n[USR] {usr_utt}".strip()

            buffer_objs.append(obj)
            buffer_ctx_texts.append(ctx_text)
            pbar.update(1)

            if len(buffer_objs) >= BATCH_SIZE_QUERY_CTX:
                flush_batch_turn_jsonl(fout, buffer_objs, buffer_ctx_texts)

        flush_batch_turn_jsonl(fout, buffer_objs, buffer_ctx_texts)
        pbar.close()


def annotate_dialog_json_file():
    # JSON list of dialogs: keep original structure; add retrieved_topk_ids into each turn's taklabels
    with open(QUERY_A_PATH, "r", encoding="utf-8") as fin:
        dialogs = json.load(fin)
    assert isinstance(dialogs, list), "QUERY_A_PATH is JSON but not a list."

    with open(OUTPUT_A_PATH, "w", encoding="utf-8") as fout:
        fout.write("[\n")
        first_dialog = True

        for d in tqdm(dialogs, desc="Annotating dialog JSON with retrieval IDs", ncols=110):
            dial_id = d.get("dial_id") or ""
            turns = d.get("turns") or []
            if not isinstance(turns, list):
                turns = []

            # build ctx for turns that have taklabels
            ctx_texts: List[str] = []
            turn_indices: List[int] = []

            for ti, t in enumerate(turns):
                evs = t.get("taklabels")
                if not isinstance(evs, list) or len(evs) == 0:
                    continue
                ctx_text = ""
                # if your turn already has context.retrieval_text, use it
                ctx = t.get("context") or {}
                if isinstance(ctx, dict):
                    ctx_text = (ctx.get("retrieval_text") or "").strip()
                if not ctx_text:
                    ctx_text = build_ctx_from_dialog_turns(turns, ti, WINDOW_K_FOR_DIALOG_JSON)
                ctx_texts.append(ctx_text)
                turn_indices.append(ti)

            # encode ctx in batches
            if ctx_texts:
                ctx_embs = model.encode(
                    ctx_texts,
                    batch_size=BATCH_SIZE_QUERY_CTX,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32)

                for (ti, emb) in zip(turn_indices, ctx_embs):
                    evs = turns[ti].get("taklabels")
                    if not isinstance(evs, list):
                        continue
                    for ev in evs:
                        if not isinstance(ev, dict):
                            continue
                        q_slot = (ev.get("slot") or "").strip()
                        q_label = parse_label_for_retrieval_from_taklabel(ev)
                        ev["retrieved_topk_ids"] = retrieve_topk_ids(
                            q_dial_id=dial_id, q_label=q_label, q_slot=q_slot, q_ctx_emb=emb
                        )

            # stream write dialog
            if not first_dialog:
                fout.write(",\n")
            fout.write(json.dumps(d, ensure_ascii=False))
            first_dialog = False

        fout.write("\n]\n")


if is_json_array_file(QUERY_A_PATH):
    annotate_dialog_json_file()
else:
    annotate_jsonl_turn_file()

print("[DONE]")
print(f"Annotated file written to: {OUTPUT_A_PATH}")
print(f"MODE={MODE}, EXCEPT_DOMAIN={EXCEPT_DOMAIN}, TOPK={TOPK}, FORBID_SAME_DIAL={FORBID_SAME_DIAL}, USE_LABEL2_REFER_IMPLICIT={USE_LABEL2_REFER_IMPLICIT}")
