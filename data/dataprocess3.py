#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_ds_loo_splits_folders.py

在 5 个领域 (attraction, hotel, restaurant, taxi, train) 上做 DS 留一法：
- 训练/验证：剔除包含被留出领域的对话
- 测试：只保留在 5 域交集 == {该领域} 的对话

输出结构：
<base_dir>/ds_loo/
  ├── attraction/
  │     ├── attraction_train.json
  │     ├── attraction_dev.json
  │     └── attraction_test.json
  ├── hotel/
  │     ├── hotel_train.json
  │     ├── hotel_dev.json
  │     └── hotel_test.json
  └── ...

base_dir 取自输入文件所在目录（与输入同级）。
"""

import os
import json

# ====== 输入路径 ======
DEV_PATH   = "/home/fzus/zzp/data/code/dategen/processed_l1_confirm/dev_processed_with_refer.json"
TEST_PATH  = "/home/fzus/zzp/data/code/dategen/processed_l1_confirm/test_processed_with_refer.json"
TRAIN_PATH = "/home/fzus/zzp/data/code/dategen/processed_l1_confirm/train_processed_with_refer.json"

# 目标领域
TARGET_DOMAINS = ["attraction", "hotel", "restaurant", "taxi", "train"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "dial_id" in data:
        data = [data]
    return data


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_domains(dialog):
    ds = dialog.get("domains", [])
    if not isinstance(ds, list):
        return set()
    return set([str(x).strip().lower() for x in ds if str(x).strip()])


def ds_loo_filter(train_set, dev_set, test_set, left_out: str):
    left_out = left_out.lower()
    targets = set(TARGET_DOMAINS)

    def for_train_or_dev(dset):
        out = []
        for d in dset:
            doms = normalize_domains(d)
            if left_out in doms:
                continue
            out.append(d)
        return out

    def for_test(dset):
        out = []
        for d in dset:
            doms = normalize_domains(d)
            inter = doms & targets
            if inter == {left_out}:
                out.append(d)
        return out

    return for_train_or_dev(train_set), for_train_or_dev(dev_set), for_test(test_set)


def main():
    # 读入数据
    train_all = load_json(TRAIN_PATH)
    dev_all   = load_json(DEV_PATH)
    test_all  = load_json(TEST_PATH)
    print(f"Loaded: train={len(train_all)}, dev={len(dev_all)}, test={len(test_all)}")

    # 输出根目录（与输入同级）
    base_dir = os.path.dirname(TRAIN_PATH)
    out_root = os.path.join(base_dir, "ds_loo")
    os.makedirs(out_root, exist_ok=True)

    # 每个领域生成子文件夹与文件
    for dom in TARGET_DOMAINS:
        tr, dv, te = ds_loo_filter(train_all, dev_all, test_all, left_out=dom)

        subdir = os.path.join(out_root, dom)
        os.makedirs(subdir, exist_ok=True)

        out_train = os.path.join(subdir, f"{dom}_train.json")
        out_dev   = os.path.join(subdir, f"{dom}_dev.json")
        out_test  = os.path.join(subdir, f"{dom}_test.json")

        save_json(out_train, tr)
        save_json(out_dev, dv)
        save_json(out_test, te)

        print(f"\n[LOO for '{dom}']")
        print(f"  train (no '{dom}') : {len(tr)} -> {out_train}")
        print(f"  dev   (no '{dom}') : {len(dv)} -> {out_dev}")
        print(f"  test  (only '{dom}'): {len(te)} -> {out_test}")

    print("\nAll done. Output root:", out_root)


if __name__ == "__main__":
    main()
