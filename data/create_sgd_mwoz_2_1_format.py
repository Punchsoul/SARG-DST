import argparse
import glob
import json
import os
from collections import OrderedDict


SPLITS = ("train", "dev", "test")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def list_dialogue_files(split_dir):
    return sorted(glob.glob(os.path.join(split_dir, "dialogue*.json")))


def normalize_domain(service):
    return service.lower().split("_", 1)[0]


def normalize_slot(slot):
    return " ".join(slot.lower().split("_"))


def choose_value(values, utterance):
    if not values:
        return ""

    utterance = (utterance or "").lower()
    value = str(values[0])
    for item in values:
        item = str(item)
        if item.lower() in utterance:
            value = item
            break
    return value


def extract_user_state(turn):
    slot_values = OrderedDict()
    domains = set()
    active_intents = []

    for frame in turn.get("frames", []):
        service = frame.get("service", "")
        if not service:
            continue

        domain = normalize_domain(service)
        domains.add(domain)

        state = frame.get("state", {})
        active_intent = state.get("active_intent", "none")
        if active_intent and active_intent != "NONE":
            active_intents.append(active_intent)

        for slot, values in state.get("slot_values", {}).items():
            value = choose_value(values, turn.get("utterance", ""))
            if not value or value.lower() == "none":
                continue

            slot_name = f"{domain}-{normalize_slot(slot)}"
            slot_values[slot_name] = value

    active_intent = active_intents[0] if active_intents else "none"
    return domains, active_intent, slot_values


def convert_dialogue(dialogue):
    dial_id = dialogue["dialogue_id"]
    domains = set()
    turns = []
    cumulative_state = OrderedDict()
    last_system = "none"

    for turn in dialogue.get("turns", []):
        speaker = turn.get("speaker", "").upper()
        utterance = turn.get("utterance", "")

        if speaker == "SYSTEM":
            last_system = utterance
            continue

        if speaker != "USER":
            continue

        turn_domains, active_intent, turn_state = extract_user_state(turn)
        domains.update(turn_domains)
        cumulative_state.update(turn_state)

        turns.append(
            OrderedDict(
                [
                    ("system", last_system),
                    ("user", utterance),
                    (
                        "state",
                        OrderedDict(
                            [
                                ("active_intent", active_intent),
                                ("slot_values", OrderedDict(cumulative_state)),
                            ]
                        ),
                    ),
                ]
            )
        )

    return OrderedDict(
        [
            ("dial_id", dial_id),
            ("domains", sorted(domains)),
            ("turns", turns),
        ]
    )


def build_ontology(dialogues):
    ontology = OrderedDict()

    for dialogue in dialogues:
        for turn in dialogue["turns"]:
            for slot, value in turn["state"]["slot_values"].items():
                ontology.setdefault(slot, [])
                if value not in ontology[slot]:
                    ontology[slot].append(value)

    return ontology


def convert_split(data_dir, split):
    converted = []
    split_dir = os.path.join(data_dir, split)

    for path in list_dialogue_files(split_dir):
        for dialogue in read_json(path):
            converted.append(convert_dialogue(dialogue))

    return converted


def convert_dataset(data_dir, target_path):
    os.makedirs(target_path, exist_ok=True)

    split_data = {split: convert_split(data_dir, split) for split in SPLITS}
    ontology = build_ontology(
        split_data["train"] + split_data["dev"] + split_data["test"]
    )

    save_json(os.path.join(target_path, "train_dials.json"), split_data["train"])
    save_json(os.path.join(target_path, "dev_dials.json"), split_data["dev"])
    save_json(os.path.join(target_path, "test_dials.json"), split_data["test"])
    save_json(os.path.join(target_path, "ontology.json"), ontology)

    with open(os.path.join(target_path, "trainListFile"), "w", encoding="utf-8") as f:
        for dialogue in split_data["train"]:
            f.write(dialogue["dial_id"] + "\n")

    print(
        "# of dialogues: Train {}, Val {}, Test {}".format(
            len(split_data["train"]), len(split_data["dev"]), len(split_data["test"])
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./sourcedata/dstc8-schema-guided-dialogue/",
    )
    parser.add_argument("--target_path", type=str, default="./data/sgd_mwoz2.1")
    args = parser.parse_args()

    convert_dataset(args.data_dir, args.target_path)


if __name__ == "__main__":
    main()
