import os
import json
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm


# ================== 默认路径/配置（可按需改） ==================
ROOT_DIR = "/home/zzp"

TRAIN_PATH = os.path.join(ROOT_DIR, "ds_loo/train/train_train.json")
DEV_PATH   = os.path.join(ROOT_DIR, "ds_loo/train/train_dev.json")
TEST_PATH  = os.path.join(ROOT_DIR, "ds_loo/TEST/TEST_sum.json")

SLOT_DESC_PATH = os.path.join(ROOT_DIR, "ds_loo/ontology/slot_descriptions.json")
SENBERT_PATH   = os.path.join(ROOT_DIR, "senbert")

CANDIDATES_DIR = os.path.join(ROOT_DIR, "model2/candidates_stage1_train")
os.makedirs(CANDIDATES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 6
LR_CTX = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 3
CLIP_NORM = 1.0

MAX_LEN_TEXT = 256
MAX_LEN_SLOT = 96  # 你的 NOT/contrast 长的话建议 96 或 128

TEMPERATURE = 0.07

# 你现在要 top9
TOP_K = 10
USE_CONTEXT_DEFAULT = False

# Top-K 友好的 ranking loss（弱正例 / hardest negatives）
LAMBDA_RANK = 0.8
RANK_MARGIN = 0.20
HARDNEG_H   = 64

TEST_DOMAIN_DEFAULT = "train"


# ================== 数据读取 ==================
def load_dialogs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_sys_usr(turn: Dict[str, Any]) -> Tuple[str, str]:
    sys_txt = turn.get("sys", "") or turn.get("system", "") or ""
    usr_txt = turn.get("user", "") or turn.get("usr", "") or ""
    return sys_txt, usr_txt

def build_turn_samples(
    dialogs: List[Dict[str, Any]],
    split: str,
    test_domain: str,
    use_context: bool = True
) -> List[Dict[str, Any]]:
    """
    严格版：
      - train/dev：剔除包含 test_domain 的对话（保证训练阶段连目标域对话都不碰）
      - test：只保留包含 test_domain 的对话

    样本文本：
      - 前两轮 Fine_summary 作为上下文（CTX-2/CTX-1）
      - 当前轮 sys/user 原文（TURN-0）
    gold 槽位：
      - turn["taklabels"] 中的 slot 列表
    """
    samples = []
    for dlg in dialogs:
        dial_id = dlg.get("dial_id", "")
        domains = dlg.get("domains", []) or []

        if split in ("train", "dev"):
            if test_domain in domains:
                continue  # 严格：训练阶段不看目标域对话
        elif split == "test":
            if test_domain not in domains:
                continue
        else:
            raise ValueError(f"unknown split: {split}")

        turns = dlg.get("turns", [])
        for i, turn in enumerate(turns):
                        # 前两轮上下文改为“原始对话”（system+user），不再用摘要
            ctx2_text = ""
            ctx1_text = ""
            if i - 2 >= 0:
                prev2_sys, prev2_usr = _get_sys_usr(turns[i - 2])
                parts2 = []
                if prev2_sys.strip():
                    parts2.append(f"System: {prev2_sys.strip()}")
                if prev2_usr.strip():
                    parts2.append(f"User: {prev2_usr.strip()}")
                ctx2_text = " ".join(parts2).strip()
            if i - 1 >= 0:
                prev1_sys, prev1_usr = _get_sys_usr(turns[i - 1])
                parts1 = []
                if prev1_sys.strip():
                    parts1.append(f"System: {prev1_sys.strip()}")
                if prev1_usr.strip():
                    parts1.append(f"User: {prev1_usr.strip()}")
                ctx1_text = " ".join(parts1).strip()

            sys_txt, usr_txt = _get_sys_usr(turn)
            cur_parts = []
            if sys_txt.strip():
                cur_parts.append(f"System: {sys_txt.strip()}")
            if usr_txt.strip():
                cur_parts.append(f"User: {usr_txt.strip()}")
            cur_text = " ".join(cur_parts).strip()

            pieces = []
            if use_context:
                if ctx2_text:
                    pieces.append(f"[CTX-2] {ctx2_text}")
                if ctx1_text:
                    pieces.append(f"[CTX-1] {ctx1_text}")

            if cur_text:
                pieces.append(f"[TURN-0] {cur_text}")


            text = " ".join(pieces).strip()

            taklabels = turn.get("taklabels", [])
            active_slots = [lab["slot"] for lab in taklabels if isinstance(lab, dict) and lab.get("slot")]

            samples.append({
                "dial_id": dial_id,
                "turn_id": int(turn.get("turn_id", i)),
                "domains": domains,
                "text": text,
                "active_slots": active_slots,
            })

    return samples


# ================== Slot 描述（四字段结构化拼接） ==================
def load_slot_descriptions(path: str) -> Dict[str, str]:
    """
    你的字段：
      - type_desc
      - concise_desc
      - question_desc
      - contras_desc

    SBERT 更稳：结构化标签 + 分隔符
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    slot_texts: Dict[str, str] = {}
    for slot, desc_obj in raw.items():
        if isinstance(desc_obj, dict):
            type_desc = desc_obj.get("type_desc", "") or ""
            concise   = desc_obj.get("concise_desc", "") or ""
            question  = desc_obj.get("question_desc", "") or ""
            contras   = desc_obj.get("contras_desc", "") or ""

            chunks = []
            if type_desc.strip():
                chunks.append(f"TYPE: {type_desc.strip()}")
            if concise.strip():
                chunks.append(f"DESC: {concise.strip()}")
            if question.strip():
                chunks.append(f"Q: {question.strip()}")
            if contras.strip():
                chunks.append(f"NOT: {contras.strip()}")

            txt = " | ".join(chunks) if chunks else json.dumps(desc_obj, ensure_ascii=False)
        else:
            txt = str(desc_obj)

        slot_texts[slot] = txt

    return slot_texts


# ================== 模型 ==================
class SenBERTDualEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, hidden_size: int = 768):
        super().__init__()
        self.ctx_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.slot_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.proj_ctx = nn.Linear(self.ctx_encoder.config.hidden_size, hidden_size)
        self.proj_slot = nn.Linear(self.slot_encoder.config.hidden_size, hidden_size)

    @staticmethod
    def mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def encode_context(self, input_ids, attention_mask):
        out = self.ctx_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(out.last_hidden_state, attention_mask)
        return self.proj_ctx(pooled)

    def encode_slots(self, input_ids, attention_mask):
        out = self.slot_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(out.last_hidden_state, attention_mask)
        return self.proj_slot(pooled)


# ================== Dataset / Collate ==================
class TurnDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx: int):
        return self.samples[idx]

def collate_turns(batch: List[Dict[str, Any]], tokenizer, max_len: int):
    texts = [b["text"] for b in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "batch_meta": batch
    }


# ================== Dev coverage@K（类JGA） ==================
def eval_coverage_at_k(
    model: SenBERTDualEncoder,
    dev_samples: List[Dict[str, Any]],
    tokenizer,
    slot_vecs_norm: torch.Tensor,  # [S, H]
    slots: List[str],
    slot2idx: Dict[str, int],
    k: int
) -> float:
    """
    turn-level coverage@k：
      - 只统计 active_slots 非空的轮
      - gold_set ⊆ topk_set -> correct
    """
    model.eval()
    loader = DataLoader(
        TurnDataset(dev_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_turns(b, tokenizer, MAX_LEN_TEXT)
    )

    correct_turns = 0
    pos_turns = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[Stage1][Dev] coverage@{k}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            metas = batch["batch_meta"]

            ctx = model.encode_context(input_ids, attention_mask)
            ctx = torch.nn.functional.normalize(ctx, dim=-1)

            sims = torch.matmul(ctx, slot_vecs_norm.t())  # [B, S]
            _, topk_idx = torch.topk(sims, k=k, dim=-1)
            topk_idx = topk_idx.cpu().tolist()

            for meta, idxs in zip(metas, topk_idx):
                gold_slots = [s for s in meta["active_slots"] if s in slot2idx]
                if not gold_slots:
                    continue
                pos_turns += 1
                topk_slots = [slots[j] for j in idxs]
                if set(gold_slots).issubset(set(topk_slots)):
                    correct_turns += 1

    if pos_turns == 0:
        print(f"[Stage1][Dev] WARNING: no positive turns, coverage@{k}=0.0")
        return 0.0
    cov = correct_turns / pos_turns
    print(f"[Stage1][Dev] coverage@{k} = {cov:.4f} ({correct_turns}/{pos_turns})")
    return cov


# ================== 训练 + 候选生成（严格版） ==================
def main(test_domain: str, use_context: bool = USE_CONTEXT_DEFAULT):
    print(f"[Stage1 strict] test_domain={test_domain} TOP_K={TOP_K} device={DEVICE} use_context={use_context}")


    # --- samples ---
    train_samples = build_turn_samples(load_dialogs(TRAIN_PATH), split="train", test_domain=test_domain, use_context=use_context)
    dev_samples   = build_turn_samples(load_dialogs(DEV_PATH),   split="dev",   test_domain=test_domain, use_context=use_context)
    test_samples  = build_turn_samples(load_dialogs(TEST_PATH),  split="test",  test_domain=test_domain, use_context=use_context)


    print(f"[Stage1 strict] #train_turns={len(train_samples)} #dev_turns={len(dev_samples)} #test_turns={len(test_samples)}")

    # --- slot texts ---
    slot_texts_all = load_slot_descriptions(SLOT_DESC_PATH)
    slots_all = sorted(slot_texts_all.keys())

    # 严格：训练阶段 slots_train 不包含 test_domain 槽位
    prefix = f"{test_domain}-"
    slots_train = [s for s in slots_all if not s.startswith(prefix)]
    slot_texts_train = {s: slot_texts_all[s] for s in slots_train}

    print(f"[Stage1 strict] #slots_all={len(slots_all)} #slots_train(exclude {test_domain})={len(slots_train)}")

    tokenizer = AutoTokenizer.from_pretrained(SENBERT_PATH)
    model = SenBERTDualEncoder(SENBERT_PATH).to(DEVICE)

    # 冻结 slot side（slot encoder 绝对不训练）
    for p in model.slot_encoder.parameters():
        p.requires_grad = False
    for p in model.proj_slot.parameters():
        p.requires_grad = False

    # --- encode train slots (only) ---
    slot_enc_train = tokenizer(
        [slot_texts_train[s] for s in slots_train],
        padding=True,
        truncation=True,
        max_length=MAX_LEN_SLOT,
        return_tensors="pt"
    )
    slot_train_input_ids = slot_enc_train["input_ids"].to(DEVICE)
    slot_train_attention = slot_enc_train["attention_mask"].to(DEVICE)

    slot2idx_train = {s: i for i, s in enumerate(slots_train)}

    # 预计算 train slot vectors（冻结后不变）
    model.eval()
    with torch.no_grad():
        slot_vecs_train = model.encode_slots(slot_train_input_ids, slot_train_attention)
        slot_vecs_train = torch.nn.functional.normalize(slot_vecs_train, dim=-1).detach()
    model.train()

    # --- train loader ---
    train_loader = DataLoader(
        TurnDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_turns(b, tokenizer, MAX_LEN_TEXT)
    )

    optimizer = torch.optim.AdamW([
        {"params": model.ctx_encoder.parameters(), "lr": LR_CTX},
        {"params": model.proj_ctx.parameters(), "lr": LR_CTX},
    ], weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps
    )

    best = 0.0
    patience = 0
    best_path = os.path.join(CANDIDATES_DIR, f"stage1_strict_best_{test_domain}_top{TOP_K}.pt")

    # ================== train ==================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss_weighted = 0.0
        total_pos_turns = 0

        pbar = tqdm(train_loader, desc=f"[Stage1 strict][Epoch {epoch}]")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            metas = batch["batch_meta"]

            ctx = model.encode_context(input_ids, attention_mask)
            ctx = torch.nn.functional.normalize(ctx, dim=-1)

            scores = torch.matmul(ctx, slot_vecs_train.t())          # cosine [B, S_train]
            logits = scores / TEMPERATURE                             # for InfoNCE

            # --- Top-K aware weakest-positive ranking loss (boundary negatives) ---
            rank_loss = 0.0
            rank_turns = 0
            BOUNDARY_M = 3  # 取边界附近几个负例做平均，更稳定；可设 1/3/5

            for i, meta in enumerate(metas):
                pos_slots = meta["active_slots"]
                if not pos_slots:
                    continue

                pos_idx = [slot2idx_train[s] for s in pos_slots if s in slot2idx_train]
                if not pos_idx:
                    continue

                s_i = scores[i]  # [S_train]
                pos_min = s_i[pos_idx].min()  # turn-level 关键：最差正例必须过线

                neg_mask = torch.ones_like(s_i, dtype=torch.bool)
                neg_mask[pos_idx] = False
                neg_scores = s_i[neg_mask]
                if neg_scores.numel() == 0:
                    continue

                # 只关心“topK 边界”负例：它们决定正例会不会被挤出 topK
                k_eff = min(TOP_K, neg_scores.numel())
                topk_neg, _ = torch.topk(neg_scores, k=k_eff)  # [k_eff] from high to low

                # 取边界邻域（最后几个）：K-BOUNDARY_M+1 ... K
                m_eff = min(BOUNDARY_M, topk_neg.numel())
                boundary_negs = topk_neg[-m_eff:]  # [m_eff] 这是最关键的挤位负例

                hinge = torch.relu(RANK_MARGIN - (pos_min - boundary_negs)).mean()
                rank_loss = rank_loss + hinge
                rank_turns += 1

            rank_loss = (rank_loss / rank_turns) if rank_turns > 0 else 0.0


            # --- InfoNCE (multi-positive) ---
            log_probs = torch.log_softmax(logits, dim=-1)
            info_loss = 0.0
            pos_turns = 0

            for i, meta in enumerate(metas):
                pos_slots = meta["active_slots"]
                if not pos_slots:
                    continue
                pos_idx = [slot2idx_train[s] for s in pos_slots if s in slot2idx_train]
                if not pos_idx:
                    continue
                pos_turns += 1
                info_loss = info_loss + (-log_probs[i, pos_idx].mean())

            if pos_turns == 0:
                continue

            info_loss = info_loss / pos_turns
            loss = info_loss + LAMBDA_RANK * rank_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            scheduler.step()

            total_loss_weighted += loss.item() * pos_turns
            total_pos_turns += pos_turns
            avg_loss = total_loss_weighted / max(total_pos_turns, 1)

            pbar.set_postfix({
                "avg_loss": f"{avg_loss:.4f}",
                "pos_turns": total_pos_turns
            })

        print(f"[Stage1 strict] Epoch {epoch} avg_loss={avg_loss:.4f}")

        # dev coverage@K（严格版 dev 也只在 slots_train 上评估）
        dev_cov = eval_coverage_at_k(
            model=model,
            dev_samples=dev_samples,
            tokenizer=tokenizer,
            slot_vecs_norm=slot_vecs_train,
            slots=slots_train,
            slot2idx=slot2idx_train,
            k=TOP_K
        )

        if dev_cov > best:
            best = dev_cov
            patience = 0
            torch.save({"model_state_dict": model.state_dict()}, best_path)
            print(f"[Stage1 strict] New best: dev coverage@{TOP_K}={best:.4f} -> {best_path}")
        else:
            patience += 1
            print(f"[Stage1 strict] Not improved, patience={patience}/{PATIENCE}")
            if patience >= PATIENCE:
                print("[Stage1 strict] Early stopping.")
                break

    # load best
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Stage1 strict] Loaded best ckpt: {best_path}")

    # ================== 候选生成 ==================
    model.eval()

    # 训练/开发：只用 slots_train（严格训练槽位集合）
    def generate_candidates(split_name: str, samples: List[Dict[str, Any]], slots: List[str], slot_texts: Dict[str, str]):
        out_path = os.path.join(CANDIDATES_DIR, f"{split_name}_candidates_top{TOP_K}.jsonl")
        print(f"[Stage1 strict] Generate {split_name} -> {out_path}")

        # encode slots for this split
        slot_enc = tokenizer(
            [slot_texts[s] for s in slots],
            padding=True,
            truncation=True,
            max_length=MAX_LEN_SLOT,
            return_tensors="pt"
        )
        slot_ids = slot_enc["input_ids"].to(DEVICE)
        slot_att = slot_enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            slot_vecs = model.encode_slots(slot_ids, slot_att)
            slot_vecs = torch.nn.functional.normalize(slot_vecs, dim=-1)

        loader = DataLoader(
            TurnDataset(samples),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda b: collate_turns(b, tokenizer, MAX_LEN_TEXT)
        )

        fout = open(out_path, "w", encoding="utf-8")
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"[Stage1 strict][{split_name}]"):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                metas = batch["batch_meta"]

                ctx = model.encode_context(input_ids, attention_mask)
                ctx = torch.nn.functional.normalize(ctx, dim=-1)
                sims = torch.matmul(ctx, slot_vecs.t())  # [B, S]

                _, topk_idx = torch.topk(sims, k=TOP_K, dim=-1)
                topk_idx = topk_idx.cpu().tolist()

                for meta, idxs in zip(metas, topk_idx):
                    cand = [slots[j] for j in idxs]
                    fout.write(json.dumps({
                        "dial_id": meta["dial_id"],
                        "turn_id": int(meta["turn_id"]),
                        "domains": meta["domains"],
                        "candidates": cand
                    }, ensure_ascii=False) + "\n")

        fout.close()

    # train/dev：严格只用 slots_train
    generate_candidates("train", train_samples, slots_train, slot_texts_train)
    generate_candidates("dev",   dev_samples,   slots_train, slot_texts_train)

    # test：部署阶段允许拿到全量 schema（slots_all，包含 test_domain）
    generate_candidates("test",  test_samples,  slots_all,   slot_texts_all)

    print("[Stage1 strict] Done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_domain", type=str, default=TEST_DOMAIN_DEFAULT)
    ap.add_argument("--use_context", type=int, default=None,
                    help="1/0 override. If not set, use USE_CONTEXT_DEFAULT in code.")
    args = ap.parse_args()

    use_context = USE_CONTEXT_DEFAULT if args.use_context is None else bool(args.use_context)
    main(args.test_domain, use_context=use_context)


