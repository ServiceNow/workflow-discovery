"""All processing functions are adapted from https://github.com/asappresearch/abcd/blob/master/utils/process.py
for fair comparison for the AST and CDS tasks.
"""
import gzip
import shutil
import json
import logging
from urllib.request import urlretrieve
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_action_labels(ontology):
    action_list = []
    for section, buttons in ontology["actions"].items():
        actions = buttons.keys()
        action_list.extend(actions)
    return {action: idx for idx, action in enumerate(action_list)}


def prepare_value_labels(ontology):
    value_list = []
    for category, values in ontology["values"]["enumerable"].items():
        for val in values:
            if val not in value_list:
                value_list.append(val.lower())
    return {slotval: idx for idx, slotval in enumerate(value_list)}


def _read_json_file(raw_data_path: Path, file_name: str):
    file_path = raw_data_path / file_name
    with file_path.open() as f:
        data = json.load(f)

    return data

def read_abcd_raw_data(raw_data_path: Path):
    return _read_json_file(raw_data_path, "abcd_v1.1.json")


def read_abcd_guidelines(raw_data_path: Path):
    return _read_json_file(raw_data_path, "guidelines.json")


def read_abcd_ontology(raw_data_path: Path):
    return _read_json_file(raw_data_path, "ontology.json")


def read_utterances_file(raw_data_path: Path):
    return _read_json_file(raw_data_path, "utterances.json")


def prepare_labels_for_ast(raw_data_path: Path):
    ontology = read_abcd_ontology(raw_data_path)

    non_enumerable = ontology["values"]["non_enumerable"]
    enumerable = {}
    for category, values in ontology["values"]["enumerable"].items():
        enumerable[category] = [val.lower() for val in values]

    mappers = {"value": prepare_value_labels(ontology), "action": prepare_action_labels(ontology)}

    # Break down the slot values by action
    value_by_action = {}
    for section, actions in ontology["actions"].items():
        for action, targets in actions.items():
            value_by_action[action] = targets

    return non_enumerable, enumerable, mappers, value_by_action


def ast_value_to_id(_context, value, potential_vals, enumerable):
    for option in potential_vals:
        if option in enumerable:  # just look it up
            if value in enumerable[option]:
                # We need to return the exact value
                # potential_vals.pop(potential_vals.index(option))
                return value
        else:
            entity = f"<{option}>"  # calculate location in the context
            if entity in _context:
                # We need to return the entity
                return entity

    return value


def prepare_intent_labels(ontology):
    intent_list = []
    for flow, subflows in ontology["intents"]["subflows"].items():
        intent_list.extend(subflows)
    return {intent: idx for idx, intent in enumerate(intent_list)}


def prepare_nextstep_labels(ontology):
    nextstep_list = ontology["next_steps"]
    return {nextstep: idx for idx, nextstep in enumerate(nextstep_list)}


def prepare_labels_for_cds(raw_data_path: Path):
    ontology = read_abcd_ontology(raw_data_path)

    non_enumerable = ontology["values"]["non_enumerable"]
    enumerable = {}
    for category, values in ontology["values"]["enumerable"].items():
        enumerable[category] = [val.lower() for val in values]

    mappers = {
        "value": prepare_value_labels(ontology),
        "action": prepare_action_labels(ontology),
        "intent": prepare_intent_labels(ontology),
        "nextstep": prepare_nextstep_labels(ontology),
    }  # utterance is ranking, so not needed

    # Break down the slot values by action
    value_by_action = {}
    for section, actions in ontology["actions"].items():
        for action, targets in actions.items():
            value_by_action[action] = targets

    return non_enumerable, enumerable, mappers, value_by_action


def collect_one_example(dialog_history, targets, support_items, enumerable, mappers, utterances):
    def value_to_id(_context, value, potential_vals):
        for option in potential_vals:
            if option in enumerable:  # just look it up
                if value in enumerable[option]:
                    return value
            else:
                entity = f"<{option}>"  # calculate location in the context
                if entity in _context:
                    return entity

        return value

    def action_to_id(action):
        return mappers["action"][action]

    intent, nextstep, action, _, utt_id = targets
    take_action_target = ["none", ["none"]]
    utt_candidates = ""
    target_utterance = "none"

    if nextstep == "take_action":
        value, potential_vals, convo_id, turn_id = support_items
        if value != "not applicable":
            parsed_values = []
            for v in value:
                value_id = value_to_id(dialog_history, v, potential_vals)
                parsed_values.append(value_id)
        else:
            parsed_values = ["none"]
        take_action_target = [action, parsed_values]
        nextstep_target = "action"

    elif nextstep == "retrieve_utterance":
        candidates, convo_id, turn_id = support_items
        target_utt_id = candidates[utt_id]
        target_utterance = utterances[target_utt_id]
        real_candidates = [utterances[u] for u in candidates]
        utt_candidates = real_candidates
        nextstep_target = "respond"

    elif nextstep == "end_conversation":
        convo_id, turn_id = support_items
        nextstep_target = "end"
    else:
        raise ValueError()

    return {
        "context": [t.split("|")[1] for t in dialog_history],
        "intent": intent,
        "next_step": nextstep_target,
        "take_action_target": take_action_target,
        "target_utterance": target_utterance,
        "candidates": utt_candidates,
        "convo_id": convo_id,
        "turn_id": turn_id,
    }


def collect_examples(context, targets, convo_id, turn_id, value_by_action, enumerable, mappers, utterances):
    _, _, action, values, _ = targets
    potential_vals = value_by_action[action]

    if len(potential_vals) > 0:  # just skip if action does not require inputs
        return collect_one_example(
            context, targets, (values, potential_vals, convo_id, turn_id), enumerable, mappers, utterances
        )
    else:
        return collect_one_example(
            context, targets, ("not applicable", potential_vals, convo_id, turn_id), enumerable, mappers, utterances
        )


def parse_abcd_dataset_for_cds(raw_dat_path: Path, data: List):
    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_cds(raw_dat_path)
    utterances = read_utterances_file(raw_dat_path)

    parsed_samples = []

    for sample in tqdm(data, total=len(data)):
        so_far = []
        for turn in sample["delexed"]:
            speaker, text = turn["speaker"], turn["text"]
            utterance = f"{speaker}|{text}"

            if speaker == "agent":
                context = so_far.copy()
                support_items = turn["candidates"], sample["convo_id"], turn["turn_count"]
                parsed_samples.append(
                    collect_one_example(context, turn["targets"], support_items, enumerable, mappers, utterances)
                )
                so_far.append(utterance)
            elif speaker == "action":
                context = so_far.copy()
                parsed_samples.append(
                    collect_examples(
                        context,
                        turn["targets"],
                        sample["convo_id"],
                        turn["turn_count"],
                        value_by_action,
                        enumerable,
                        mappers,
                        utterances,
                    )
                )
                so_far.append(utterance)
            else:
                so_far.append(utterance)

        context = so_far.copy()  # the entire conversation
        end_targets = turn["targets"].copy()  # last turn after then end of the loop
        end_targets[1] = "end_conversation"
        end_targets[4] = -1
        support_items = sample["convo_id"], turn["turn_count"]
        parsed_samples.append(
            collect_one_example(context, end_targets, support_items, enumerable, mappers, utterances)
        )  # end

    return parsed_samples


def parse_abcd_dataset_for_ast(raw_data_path: Path, data: List):
    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
    parsed_samples = []
    for sample in data:
        context = []

        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]

                potential_values = value_by_action[target_action]
                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    corrected_target_values = []
                    for value in target_values:
                        context_str = " ".join(context)
                        corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
                        corrected_target_values.append(corrected_value)
                    target_values = corrected_target_values

                parsed_samples.append({"context": context, "action": [target_action, target_values]})

            context.append(turn["text"])

    return parsed_samples

def get_value_mappings(data_path: Path):
    """
    Create the mapping from values like "shirt_how_1" to "remove a stain from the shirt" per agent guidelines
    Mapping validated from the original ABCD guidelines
    Reference: https://docs.google.com/document/d/1_SZit-iUAzNCICJ6qahULoMhqVOJCspQF37QiEJzHLc
    """

    def _get_value_mappings(subflows_data, value_prefixes, expected_value_count, get_value_type_fct=None):
        _value_mappings = {}
        for value in value_prefixes:
            added_values_count = 0

            # Skipping the first value since it holds an instruction to the agent not the value
            nl_values = subflows_data[f"{value} FAQ"]["instructions"][1].split(",")

            for idx, nl_value in enumerate(nl_values):
                value_type = get_value_type_fct(idx + 1) if get_value_type_fct else "_{idx + 1}"
                _value_mappings[f"{value.lower()}{value_type}"] = nl_value.strip()
                added_values_count += 1

            assert added_values_count == expected_value_count
        return _value_mappings

    guidelines = read_abcd_guidelines(data_path)

    value_mappings = {}

    single_item_queries_subflows = guidelines["Single-Item Query"]["subflows"]
    # There are 4 "how" values (e.g., shirt_how_4) followed by 4 "other" values (e.g., shirt_other_4)
    get_value_type = lambda i: f"_how_{i if i <= 4 else i - 4}" if i <= 4 else f"_other_{i if i <= 4 else i - 4}"
    value_mappings.update(
        _get_value_mappings(
            single_item_queries_subflows,
            ["Shirt", "Jacket", "Jeans", "Boots"],
            8,  # there 8 values for each
            get_value_type_fct=get_value_type,
        )
    )

    store_wide_queries_subflows = guidelines["Storewide Query"]["subflows"]
    value_mappings.update(
        _get_value_mappings(
            store_wide_queries_subflows,
            ["Policy", "Timing", "Pricing", "Membership"],
            4,  # there only value (i.e., timing_1, ..., timing_4)
            get_value_type_fct=get_value_type,
        )
    )

    return value_mappings

def unzip_file(zip_file: Path):
    output_file_path = zip_file.parent / "abcd_v1.1.json"

    with gzip.open(str(zip_file.resolve()), 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        
def download_abcd_dataset(raw_data_path: Path):
    """Download the ABCD dataset from the official website"""
    logger.info(f"Downloading the ABCD dataset to {raw_data_path}")
    
    files_to_download = [
        "https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz",
        "https://raw.githubusercontent.com/asappresearch/abcd/master/data/guidelines.json",
        "https://raw.githubusercontent.com/asappresearch/abcd/master/data/ontology.json",
        "https://raw.githubusercontent.com/asappresearch/abcd/master/data/utterances.json",
        "https://raw.githubusercontent.com/asappresearch/abcd/master/data/utterances.json",
    ]
    
    for file_to_download in tqdm(files_to_download):
        output_file_path = raw_data_path / file_to_download.split("/")[-1]
        urlretrieve(file_to_download, output_file_path) 
        
        if file_to_download.endswith(".gz"):
            unzip_file(output_file_path)
        
    logger.info(f"Downloading the ABCD dataset to {raw_data_path}")
        