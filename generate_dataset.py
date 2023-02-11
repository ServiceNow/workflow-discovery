import sys
import argparse
import logging

import random
import re

import jsonlines
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from urllib.request import urlretrieve
from tqdm import tqdm as progress_bar
from datasets import load_dataset

import src.data.multiwoz_24_nlp_utils as multiwoz_24_nlp_utils
import src.data.abcd_utils as abcd_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def read_action_mapping_file(data_path: Path, prefix: str):
    action_mapping_file_path = data_path / f"{prefix}_action_mappings.json"
    with action_mapping_file_path.open() as f:
        action_mappings = json.load(f)
    return action_mappings


def parse_abcd_dataset_for_workflow_discovery(data: List, action_mappings: Dict, value_mappings: Dict):
    parsed_samples = []
    for sample in data:
        original_dialogue = []
        workflow = []

        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]

                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    target_values = [value_mappings.get(v, v) for v in target_values if v.lower() != "n/a"]

                target_action = action_mappings[target_action]

                step_data = [target_action, target_values]

                if step_data not in workflow:  # Skipping annotation duplicates
                    workflow.append(step_data)

            else:
                # We use the original dialogue, since the delexed version has anonymized slot values (e.g., [username])
                original_dialogue.append(sample["original"][idx])

        parsed_samples.append({"original_dialogue": original_dialogue, "workflow": workflow})

    return parsed_samples


def convert_workflow_to_str(workflow: List):
    workflow_str_parts = []
    for action, values in workflow:
        workflow_str_parts.append(f"{action} [{', '.join(values)}]")

    workflow_str = "; ".join(workflow_str_parts)
    return workflow_str


def convert_dialogue_to_str(dialogue: List):
    dialogue_str_parts = []
    for speaker, utterance in dialogue:
        dialogue_str_parts.append(utterance)

    dialogue_str = "Dialogue: " + " ".join(dialogue_str_parts)
    return dialogue_str


def create_input_w_possible_actions(dialogue_str: str, all_actions: List):
    possible_actions = all_actions
    return dialogue_str + " Actions: " + ", ".join(possible_actions)


def create_input_w_possible_actions_plus(dialogue_str: str, all_actions: List, workflow: List, r_min: int = 10):
    random_actions = random.sample(all_actions, random.randint(r_min, len(all_actions)))
    actions = [a[0] for a in workflow]

    possible_actions_plus = list(set(random_actions + actions))
    random.shuffle(possible_actions_plus)

    return dialogue_str + " Actions: " + ", ".join(possible_actions_plus)


def create_workflow_discovery_split_dataset(parsed_data, all_actions):
    wd_split_data = []
    for idx, sample in enumerate(parsed_data):
        workflow = sample["workflow"]

        workflow_str = convert_workflow_to_str(workflow)
        dialogue_str = convert_dialogue_to_str(sample["original_dialogue"])
        input_w_possible_actions = create_input_w_possible_actions(dialogue_str, all_actions)
        input_w_possible_actions_plus = create_input_w_possible_actions_plus(dialogue_str, all_actions, workflow)

        wd_sample = {
            "sample_id": len(wd_split_data),
            "target": workflow_str,
            "input": dialogue_str,
            "input_w_possible_actions": input_w_possible_actions,  # i.e., w/ Possible Actions in paper
            "input_w_possible_actions_plus": input_w_possible_actions_plus,  # i.e., w/ Possible Actions+ in paper
            "target_data": json.dumps(workflow),  # Used during metrics evaluation
        }

        wd_split_data.append(wd_sample)

    return wd_split_data


def get_workflow_discovery_data_from_abcd(data_path: Path):
    raw_data = abcd_utils.read_abcd_raw_data(data_path)
    value_mappings = abcd_utils.get_value_mappings(data_path)
    action_mappings = read_action_mapping_file(data_path, "abcd")

    parsed_data = {}
    for split, split_data in raw_data.items():
        parsed_data[split] = parse_abcd_dataset_for_workflow_discovery(split_data, action_mappings, value_mappings)

    return parsed_data, list(action_mappings.values())


def write_split_data(processed_data_path: Path, dataset_name: str, split: str, data, task_name="workflow_discovery"):
    output_file = processed_data_path / f"{split}_{task_name}_{dataset_name}.json"
    with jsonlines.open(output_file.resolve(), mode="w") as writer:
        # We use jsonline to use the hugggingface dataset library
        for sample in data:
            writer.write(sample)


def create_workflow_discovery_dataset(
    processed_data_path: Path, parsed_data: Dict, all_actions: List, dataset_name: str
):
    for split, split_data in parsed_data.items():
        wd_split_data = create_workflow_discovery_split_dataset(split_data, all_actions)

        write_split_data(processed_data_path, dataset_name, split, wd_split_data)


def read_multiwoz_raw_data(raw_data_path: Path, split: str):
    file_path = raw_data_path / f"{split}_multiwoz_22.json"
    with jsonlines.open(file_path) as reader:
        samples = [s for s in reader]
    return samples


def is_empty_value(value_list):
    """Following MutliWOZ 2.4"""
    return any(
        [
            value in ["dont care", "dontcare", "don't care", "do not care", "not_mentioned", "none", "unknown"]
            for value in value_list
        ]
    )


def get_frame_slot_values(frame, replacements):
    parsed_slot_values = {}
    for slot_name, slot_value in zip(
        frame["slots_values"]["slots_values_name"], frame["slots_values"]["slots_values_list"]
    ):
        if is_empty_value(slot_value):
            continue
        parsed_slot_values[slot_name] = [multiwoz_24_nlp_utils.normalize(s, replacements) for s in slot_value]
    return parsed_slot_values


def get_slot_value_name(multiwoz_name, domain, act):
    match = re.match(f"^{domain}-{act}(.*)$", multiwoz_name)
    if not match:
        match = re.match(f"^{domain}-(.*)$", multiwoz_name)
        if not match:
            raise ValueError()

    return match.group(1)


def find_best_value(name, domain, act, values, dialogue_utterances_str):
    best_value = None
    if len(values) == 0:
        raise ValueError()

    for value in values:
        try:
            value = int(value)
            # Integer value

            exact_name = get_slot_value_name(name, domain, act)
            best_value = str(value) + " " + exact_name  # e.g., replace 2 to 2 people
            break
        except:
            pass

        if value in ["yes", "no"]:
            exact_name = get_slot_value_name(name, domain, act)
            # e.g., replace internet yes -> with internet
            best_value = "with" if value == "yes" else "without"
            best_value += " "
            best_value += exact_name
            break

        if value in dialogue_utterances_str:
            # In MultiWoz 2.2 some values have multiple candidate, we choose the one that exists in the dialogue
            best_value = value
            break

    if best_value is None:
        # Probably an annotation error
        best_value = values[0]

    return best_value


def convert_intents_to_workflow(dialogue_intents_w_values, original_dialogue: List, action_mappings: Dict):
    dialogue_str = " ".join([u[1] for u in original_dialogue])
    workflow = []
    intents = list(dialogue_intents_w_values.keys())
    for intent in intents:
        act, domain = intent.split("_")
        slot_values = dialogue_intents_w_values[intent]

        action_name = action_mappings[intent]
        action_values = []
        for name, values in slot_values.items():
            # Following Mutliwoz 2.4 where the "book" slot values are linked to the book intent
            if act == "book":
                match = re.match(f"^{domain}-book(.*)$", name)
                if not match:
                    continue
                value = find_best_value(name, domain, act, values, dialogue_str)
                action_values.append(value)
            else:
                if "book" in name:
                    continue
                value = find_best_value(name, domain, act, values, dialogue_str)
                action_values.append(value)

        workflow.append([action_name, action_values])

    return workflow


def parse_multiwoz_dataset(raw_data: List, action_mappings: Dict, replacements: List):
    parsed_samples = []

    for sample in raw_data:
        original_dialogue = []
        intents_w_slots = defaultdict(dict)  # Using python3.8, dictionaries keep insert order
        turns = sample["turns"]
        for speaker, utterance, active_frames in zip(turns["speaker"], turns["utterance"], turns["frames"]):
            utterance = multiwoz_24_nlp_utils.normalize(utterance, replacements)

            original_dialogue.append(["user" if speaker == 0 else "system", utterance])

            if speaker == 0:  # User
                for frame in active_frames["state"]:
                    active_intent = frame["active_intent"]
                    if active_intent.lower() == "none":
                        continue

                    parsed_slot_values = get_frame_slot_values(frame, replacements)
                    intents_w_slots[active_intent].update(parsed_slot_values)

        workflow = convert_intents_to_workflow(intents_w_slots, original_dialogue, action_mappings)

        parsed_samples.append({"original_dialogue": original_dialogue, "workflow": workflow})

    return parsed_samples


def get_workflow_discovery_data_from_multiwoz(raw_data_path: Path, dataset_name: str):
    action_mappings = read_action_mapping_file(raw_data_path, dataset_name)
    replacements = multiwoz_24_nlp_utils.get_replacements(raw_data_path)

    parsed_data = {}
    for split in ["train", "validation", "test"]:
        split_raw_data = read_multiwoz_raw_data(raw_data_path, split)
        parsed_data[split] = parse_multiwoz_dataset(split_raw_data, action_mappings, replacements)

    return parsed_data, list(action_mappings.values())


def get_ast_data_from_abcd(raw_data_path: Path):
    raw_data = abcd_utils.read_abcd_raw_data(raw_data_path)
    parsed_data = {}
    for split, split_data in raw_data.items():
        parsed_data[split] = abcd_utils.parse_abcd_dataset_for_ast(raw_data_path, split_data)

    return parsed_data


def get_cds_data_from_abcd(raw_data_path: Path):
    raw_data = abcd_utils.read_abcd_raw_data(raw_data_path)
    parsed_data = {}
    for split, split_data in raw_data.items():
        parsed_data[split] = abcd_utils.parse_abcd_dataset_for_cds(raw_data_path, split_data)

    return parsed_data


def convert_context_to_str(context):
    return "Context: " + " ".join(context)


def convert_cds_context_to_str(context, utterance_candidates):
    context_str = convert_context_to_str(context)
    if utterance_candidates:
        context_str += " Candidates: "
        context_str += " ".join(utterance_candidates)

    return context_str


def create_cds_split_dataset(parsed_data: List):
    cds_split_data = []
    for idx, sample in enumerate(parsed_data):
        context = sample["context"]
        candidates = sample["candidates"]
        next_step = sample["next_step"]

        context_str = convert_cds_context_to_str(context, candidates)

        target_parts = [sample["intent"], next_step]
        if next_step == "respond":
            target_parts.append(sample["target_utterance"])
        elif next_step == "action":
            target_parts.append(convert_workflow_to_str([sample["take_action_target"]]))

        target_str = "; ".join(target_parts)

        cds_sample = {
            "sample_id": len(cds_split_data),
            "convo_id": sample["convo_id"],
            "turn_id": sample["turn_id"],
            "target": target_str,
            "input": context_str,
            "target_data": json.dumps(sample),  # Used during metrics evaluation
        }

        cds_split_data.append(cds_sample)
    return cds_split_data


def create_ast_split_dataset(parsed_data: List):
    ast_split_data = []
    for idx, sample in enumerate(parsed_data):
        action = sample["action"]
        context = sample["context"]

        action_str = convert_workflow_to_str([action])  # Same format as worklow with a single action
        context_str = convert_context_to_str(context)

        ast_sample = {
            "sample_id": len(ast_split_data),
            "target": action_str,
            "input": context_str,
            "target_data": json.dumps(action),  # Used during metrics evaluation
        }

        ast_split_data.append(ast_sample)
    return ast_split_data


def create_ast_dataset(processed_data_path: Path, parsed_ast_data: Dict, dataset_name: str):
    for split, split_data in parsed_ast_data.items():
        wd_split_data = create_ast_split_dataset(split_data)
        write_split_data(processed_data_path, dataset_name, split, wd_split_data, task_name="AST")


def create_cds_dataset(processed_data_path: Path, parsed_cds_data: Dict, dataset_name: str):
    for split, split_data in parsed_cds_data.items():
        wd_split_data = create_cds_split_dataset(split_data)
        write_split_data(processed_data_path, dataset_name, split, wd_split_data, task_name="CDS")


def download_multiwoz_22_raw_data(raw_data_path: Path):
    dataset = load_dataset("multi_woz_v22")
    for split, data in dataset.items():
        file_path = raw_data_path / f"{split}_multiwoz_22.json"
        data.to_json(file_path)

    # Download replacement file
    replacement_file_path = raw_data_path / "mapping.pair"
    urlretrieve("https://raw.githubusercontent.com/smartyfh/MultiWOZ2.4/main/utils/mapping.pair", replacement_file_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--raw_data_folder", required=True, help="Raw datasets folder path")
    parser.add_argument("-o", "--processed_data_folder", required=True, help="Processed datasets folfer path (output)")
    return parser.parse_args()


def create_all_datasets(raw_data_path: Path, processed_data_path: Path):
    print("Creating datasets, this takes a while...")

    print("Creating datasets from ABCD ...")
    wd_data, all_actions = get_workflow_discovery_data_from_abcd(raw_data_path)
    create_workflow_discovery_dataset(processed_data_path, wd_data, all_actions, "abcd")

    ast_data = get_ast_data_from_abcd(raw_data_path)
    create_ast_dataset(processed_data_path, ast_data, "abcd")

    cds_data = get_cds_data_from_abcd(raw_data_path)
    create_cds_dataset(processed_data_path, cds_data, "abcd")

    print("Creating datasets from MultiWOZ ...")
    download_multiwoz_22_raw_data(raw_data_path)
    wd_data, all_actions = get_workflow_discovery_data_from_multiwoz(raw_data_path, "multiwoz")
    create_workflow_discovery_dataset(processed_data_path, wd_data, all_actions, "multiwoz")

    print("Done! Happy discovery")


if __name__ == "__main__":
    args = parse_args()
    create_all_datasets(Path(args.raw_data_folder), Path(args.processed_data_folder))
