# SARG-DST

This project implements a zero-shot dialogue state tracking framework based on
Slot Semantic Action Awareness (SSAA) and Retrieval-Augmented Graph Reasoning
(SARG).

## Directory Overview

```text
data/
  Data preprocessing, ontology conversion, and MultiWOZ/SGD format building.

SSAA/
  Slot Semantic Action Awareness module. This part predicts fine-grained slot
  action labels such as CONSTRAIN, CHANGE, SWITCH, CONFIRM, REF-EXPLICIT,
  REF-IMPLICIT, DONTCARE, and NONE.

SARG/
  Retrieval-Augmented Graph Reasoning module. This part retrieves relevant
  graph instances and uses graph-based reasoning prompts to infer slot values.
```

## Overall Pipeline

### 1. Data Preprocessing

Use the scripts in `data/` to prepare dialogue data and ontology files.

Main scripts include:

- `dataprocess.py`, `dataprocess2.py`, `dataprocess3.py`
- `create_mwoz.py`, `create_mwoz_2_1.py`
- `create_sgd_mwoz_2_1_format.py`
- `sgd_data.py`

These scripts convert raw dialogue data into processed dialogue files with
turn IDs, dialogue states, slot labels, and ontology descriptions.

### 2. Slot Semantic Action Awareness

The SSAA module predicts whether a slot is activated in the current turn and
which semantic action label it belongs to.

Typical steps:

1. Build candidate slots:

```bash
python slotact_stage1_senbert.py
```

2. Merge candidates into the training data:

```bash
python merge_stage1_candidates_into_dataset.py
```

3. Build SFT data for slot action prediction:

```bash
python build_slotact_sft_openai.py
```

4. Train the SFT model with LLaMA-Factory.

5. Run SFT inference:

```bash
python infer_slotact_sft_lora_hf.py
```

6. Build DPO preference pairs:

```bash
python build__dpo_preference_pairs.py
```

7. Convert DPO pairs to LLaMA-Factory format:

```bash
python convert__dpo_to_llamafactory.py
```

8. Train the DPO model and run DPO inference:

```bash
python infer__dpo_lora_hf.py
```

The `dynamic_slot_memory.py` script maintains slot-level memory, domain-level
memory, and reference-slot memory for constructing compact SSAA prompts.

### 3. Graph Data Construction

After semantic action labels are available, build graph-style training and
retrieval stores:

```bash
python SSAA/graphcreat.py
```

This step produces turn-level graph stores, partial retrieval databases, and
non-database query files used by the retrieval module.

### 4. Retrieval-Augmented Graph Reasoning

Use the scripts in `SARG/` to retrieve graph examples and train/infer the final
state prediction model.

Typical steps:

1. Retrieve graph IDs for query turns:

```bash
python SARG/research.py
# or
python SARG/retrivil.py
```

2. Build Graph-CoT SFT data:

```bash
python SARG/RG_SFT.py
```

3. Train the graph reasoning model with LLaMA-Factory.

4. Run final state inference:

```bash
python SARG/stateinfer.py
```

5. Optionally recheck predicted slots:

```bash
python SARG/slot_recheck.py
```

## Notes

- Most scripts contain hard-coded paths and should be edited before running.
- The corresponding paths, runtime environment, model checkpoints, dataset
  names, and small implementation details should be adjusted according to your
  own machine and experimental setting.
- The expected model training backend is LLaMA-Factory.
- The pipeline is usually run domain by domain, such as `hotel`, `restaurant`,
  `attraction`, `train`, or `taxi`.
- SSAA provides semantic action labels, while SARG uses those labels and
  retrieved graph instances to predict final dialogue states.
