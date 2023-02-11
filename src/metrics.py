import re
import sys
from filelock import FileLock
from pathlib import Path
from copy import deepcopy

import jsonlines
from tqdm import tqdm
import numpy as np
import nltk  # Here to have a nice missing dependency error message early on
from nltk.corpus import stopwords
from transformers.file_utils import is_offline_mode
from bert_score import score


def verify_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("stopwords")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def remove_stop_words(string):
    filtered_words = [word for word in string.split(" ") if word not in stopwords.words("english")]
    return " ".join(filtered_words)


def is_action_name_match(target_action_name, predicted_action_name, use_bert_score=False):
    if use_bert_score:
        _, _, F1 = score([target_action_name], [predicted_action_name], lang="en")
        return F1 >= 95.0
    else:
        # We assume stop word does not change the action for our case (e.g., offer promo code is the same as offer a promo code)
        target_action_name = remove_stop_words(target_action_name)
        predicted_action_name = remove_stop_words(predicted_action_name)

        match = target_action_name == predicted_action_name

    return match


def is_slot_values_match(target_slots, predicted_slots):
    # we assume that predicting "the museum" is the same as "museum"
    target_slots = [remove_stop_words(s) for s in target_slots]
    predicted_slots = [remove_stop_words(s) for s in predicted_slots]

    not_found_count = len(target_slots)
    for target_slot in target_slots:
        if isinstance(target_slot, str):
            target_slot = [target_slot]
        for t in target_slot:
            if t in predicted_slots:
                not_found_count -= 1
                break

    match = not_found_count == 0
    return match


def is_flow_action_match(target, prediction, action_only=False, use_bert_score=False):
    target_action_name, target_action_slots = target
    predicted_action_name, predicted_action_slots = prediction

    if not is_action_name_match(target_action_name, predicted_action_name, use_bert_score=use_bert_score):
        return False

    if not action_only and not is_slot_values_match(target_action_slots, predicted_action_slots):
        return False

    return True


def is_exact_match_flow(target_flow, predicted_flow, action_only=False, use_bert_score=False):
    if len(target_flow) != len(predicted_flow):
        # If the length is not the same no need to go further, this will be covered by the CE metric.
        return False

    for target_action, predicted_action in zip(target_flow, predicted_flow):
        if not is_flow_action_match(
            target_action, predicted_action, action_only=action_only, use_bert_score=use_bert_score
        ):
            return False

    return True


def compute_flow_cascading_evaluation(targets, predictions, action_only=False, use_bert_score=False):
    targets = deepcopy(targets)
    predictions = deepcopy(predictions)

    scores = []
    for prediction, target in tqdm(zip(predictions, targets), total=len(targets)):
        if len(prediction) > len(target):
            prediction = [v for v in target if v in prediction]
        if len(prediction) < len(target):
            prediction.extend([["Missing", []]] * (len(target) - len(prediction)))

        current_score = 0
        length = len(target)
        for turn_id in range(length):
            num_remaining = length - turn_id

            num_correct = 0
            # count up how many were predicted correctly
            while turn_id < length and is_flow_action_match(
                target[turn_id], prediction[turn_id], action_only=action_only, use_bert_score=use_bert_score
            ):
                num_correct += 1
                turn_id += 1

            current_score += num_correct / num_remaining

        scores.append(current_score / length)

    return sum(scores) / len(scores)


def compute_flow_cascading_evaluation_w_aga(targets, predictions, use_bert_score=False):
    targets = deepcopy(targets)
    predictions = deepcopy(predictions)

    scores = []
    for prediction, target in tqdm(zip(predictions, targets), total=len(targets)):
        if len(prediction) > len(target):
            prediction = [v for v in target if v in prediction]
        if len(prediction) < len(target):
            prediction.extend([["Missing", []]] * (len(target) - len(prediction)))

        current_score = 0
        length = len(target)
        for turn_id in range(length):
            num_remaining = length - turn_id

            num_correct = 0
            # count up how many were predicted correctly
            while turn_id < length and is_flow_action_match(
                target[turn_id], prediction[turn_id], use_bert_score=use_bert_score
            ):
                num_correct += 1
                turn_id += 1

            current_score += num_correct / num_remaining

        scores.append(current_score / length)

    return sum(scores) / len(scores)


def compute_exact_match(targets, predictions, action_only=False, use_bert_score=False):
    targets = deepcopy(targets)
    predictions = deepcopy(predictions)

    exact_match_count = 0
    for target, prediction in tqdm(zip(targets, predictions), total=(len(targets))):
        if is_exact_match_flow(target, prediction, action_only=action_only, use_bert_score=use_bert_score):
            exact_match_count += 1
    return exact_match_count / float(len(targets))


def compute_metrics(targets, predictions, use_bert_score=False):
    metrics = {
        "EM": compute_exact_match(targets, predictions, use_bert_score=use_bert_score),
        "CE": compute_flow_cascading_evaluation(targets, predictions, use_bert_score=use_bert_score),
        "EM_action_only": compute_exact_match(targets, predictions, action_only=True, use_bert_score=use_bert_score),
        "CE_action_only": compute_flow_cascading_evaluation(
            targets, predictions, action_only=True, use_bert_score=use_bert_score
        ),
    }

    return metrics


def parse_cds_prediction(prediction_str):
    parts = prediction_str.split(";")
    intent = parts[0]
    next_step = "MISSING"
    action = ["MISSING", ["MISSING"]]
    next_utterance = "MISSING"
    if len(parts) > 1:
        next_step = parts[1].strip()
    if len(parts) > 2:
        match = re.match(r"(.*)\[(.*)]", parts[2])
        if match:
            # action w/ value
            action_name = match.group(1).strip()
            slot_str = match.group(2)
            slot_str = slot_str.replace(";", ",")
            slots = [s.strip() for s in slot_str.split(",")]
            action = [action_name, slots]
        else:
            # utterance
            next_utterance = parts[2].strip()

    return intent, next_step, action, next_utterance


def compute_cds_em_and_ce(predictions, labels, convo_ids, turn_ids):
    """Adapted from ABCD. """
    expected, predicted = [], []
    intent_preds = []
    intent_labels = []

    next_step_preds = []
    next_step_labels = []

    action_preds = []
    action_labels = []

    value_preds = []
    value_labels = []

    utterance_preds = []
    utterance_labels = []

    num_of_action_turn = 0

    num_utt_turns = 0
    for pred, label in zip(predictions, labels):
        expected.append(label.strip())
        predicted.append(pred.strip())

        intent_label, next_step_label, action_value_label, utterance_label = parse_cds_prediction(label)
        intent_pred, next_step_pred, action_value_pred, utterance_pred = parse_cds_prediction(pred)

        intent_preds.append(intent_pred)
        intent_labels.append(intent_label)

        next_step_preds.append(next_step_pred)
        next_step_labels.append(next_step_label)

        if next_step_label == "action":
            num_of_action_turn += 1

            action_label, values_label = action_value_label
            values_label.sort()

            action_pred, values_pred = action_value_pred
            values_pred.sort()

            action_labels.append(action_label)
            value_labels.append(values_label)

            if len(values_pred) > len(values_label):
                values_pred = [v for v in values_label if v in values_pred]
            if len(values_pred) < len(values_label):
                values_pred.extend(["MISSING"] * (len(values_label) - len(values_pred)))


            action_preds.append(action_pred)
            value_preds.append(values_pred)
        else:
            # Mimic abcd
            action_preds.append(-1)
            value_preds.append(-1)
            action_labels.append(-1)
            value_labels.append(-1)

        if next_step_label == "respond":
            num_utt_turns += 1
            utterance_preds.append(utterance_pred)
            utterance_labels.append(utterance_label)
        else:
            # Needed for CE calculation
            utterance_labels.append(-1)
            utterance_preds.append(-1)

    num_turns = len(expected)

    # Intent
    intent_preds_array = np.array(intent_preds)
    intent_labels_array = np.array(intent_labels)
    intent_match = intent_labels_array == intent_preds_array
    intent_acc = sum(intent_match) / float(num_turns)

    # Next Step
    next_step_preds_array = np.array(next_step_preds)
    next_step_labels_array = np.array(next_step_labels)
    next_step_match = next_step_labels_array == next_step_preds_array
    next_step_acc = sum(next_step_match) / float(num_turns)

    # action

    action_labels_arrary = np.array(action_labels)
    action_preds_arrary = np.array(action_preds)
    action_match = action_labels_arrary == action_preds_arrary
    selector = action_labels_arrary != "-1"
    action_match = action_match * selector
    action_acc = sum(action_match) / float(num_of_action_turn)

    value_labels_arrary = np.array(value_labels)
    value_preds_arrary = np.array(value_preds)
    value_match = value_labels_arrary == value_preds_arrary
    value_match = value_match * selector
    value_acc = sum(value_match) / float(num_of_action_turn)

    joint_match = action_match & value_match
    joint_acc = sum(joint_match) / float(num_of_action_turn)

    utterance_labels_array = np.array(utterance_labels)
    utterance_preds_array = np.array(utterance_preds)
    utterance_match = utterance_labels_array == utterance_preds_array
    utt_selector = utterance_labels_array != "-1"
    utterance_match = utterance_match * utt_selector
    utterance_recall_1 = sum(utterance_match) / num_utt_turns

    # group by convo_ids
    unique_convo_ids = list(set(convo_ids))
    conversations = {}
    for uci in unique_convo_ids:
        turns, correctness = [], []
        row_id = 0
        for convo_id, turn_count in zip(convo_ids, turn_ids):
            if convo_id == uci:
                turns.append(turn_count)
                correct = False
                intent_right = intent_match[row_id]
                nextstep_right = next_step_match[row_id]

                if next_step_labels[row_id] == "respond":
                    if intent_right and nextstep_right and utterance_match[row_id]:
                        correct = True
                    else:
                        correct = False
                elif next_step_labels[row_id] == "action":
                    if intent_right and nextstep_right and joint_match[row_id]:
                        correct = True
                    else:
                        correct = False
                elif next_step_labels[row_id] == "end":
                    if intent_right and nextstep_right:
                        correct = True
                    else:
                        correct = False
                else:
                    raise ValueError()

                correctness.append(correct)
            row_id += 1

        # sort by turn_counts
        ordered = [cor for _, cor in sorted(zip(turns, correctness), key=lambda tc: tc[0])]
        conversations[uci] = ordered

    # count how many correct
    turn_score, turn_correct = 0, 0
    my_scores = []
    for convo_id, convo_correctness in conversations.items():
        current_score = 0
        convo_length = len(convo_correctness)
        # we use turn_id rather than the true turn_count since turn counts will skip numbers
        # when looping through the conversation due to skipping over customer utterances
        for turn_id in range(convo_length):
            num_remaining = convo_length - turn_id

            num_correct = 0
            # count up how many were predicted correctly
            while turn_id < convo_length and convo_correctness[turn_id]:
                num_correct += 1
                turn_id += 1

            if num_correct > 0:
                turn_correct += 1
            # normalize by the number of turns remaining
            turn_score += num_correct / num_remaining
            current_score += num_correct / num_remaining

        my_scores.append(current_score / convo_length)

    # normalize by total number of turns possible
    turn_acc = turn_correct / float(num_turns)
    final_score = turn_score / float(num_turns)

    full_result = {
        "Intent_Accuracy": round(intent_acc, 4),
        "Nextstep_Accuracy": round(next_step_acc, 4),
        "Action_Accuracy": round(action_acc, 4),
        "Value_Accuracy": round(value_acc, 4),
        "Joint_Accuracy": round(joint_acc, 4),
        "Recall_at_1": round(utterance_recall_1, 4),
        "Recall_at_5": "N/A",
        "Recall_at_10": "N/A",
        "Turn_Accuracy": round(turn_acc, 4),
        "Cascading_Score": round(final_score, 4),
    }

    return full_result

def parse_ast_prediction(prediction_str):
    match = re.match(r"(.*)\[(.*)]", prediction_str)
    if match:
        # action w/ value
        action_name = match.group(1).strip()
        slot_str = match.group(2)
        slot_str = slot_str.replace(";", ",")
        slots = [s.strip() for s in slot_str.split(",")]
    else:
        action_name = "MISSING"
        slots = ["MISSING"]

    return action_name, slots


def compute_ast_acc_metrics(predictions, labels):
    """Adapted from ABCD. """
    action_preds = []
    action_labels = []

    value_preds = []
    value_labels = []

    for pred, label in zip(predictions, labels):

        action_label, values_label = parse_ast_prediction(label)
        values_label.sort()
        for value in values_label:
            action_labels.append(action_label)
            value_labels.append(value)

        action_pred, values_pred = parse_ast_prediction(pred)
        values_pred.sort()

        if len(values_pred) > len(values_label):
            values_pred = [v for v in values_label if v in values_pred]
        if len(values_pred) < len(values_label):
            values_pred.extend(["MISSING"] * (len(values_label) - len(values_pred)))

        for value in values_pred:
            action_preds.append(action_pred)
            value_preds.append(value)

    action_labels_arrary = np.array(action_labels)
    action_preds_arrary = np.array(action_preds)
    action_match = action_labels_arrary == action_preds_arrary
    action_acc = sum(action_match) / float(len(action_labels))

    value_labels_arrary = np.array(value_labels)
    value_preds_arrary = np.array(value_preds)
    value_match = value_labels_arrary == value_preds_arrary
    value_acc = sum(value_match) / float(len(action_labels))

    joint_match = action_match & value_match
    joint_acc = sum(joint_match) / float(len(action_labels))

    return {
        "action": action_acc,
        "value": value_acc,
        "joint": joint_acc
    }


def parse_workflow_string(workflow_str: str):
    workflow = []
    actions = workflow_str.split("; ")
    for action in actions:
        match = re.match(r"(.*)\[(.*)]", action)
        if match:
            # Has slots
            step_name = match.group(1).strip()
            slot_str = match.group(2)
            slot_str = slot_str.replace(";", ",")
            slots = [s.strip() for s in slot_str.split(",")]
        else:
            step_name = action.strip()
            slots = []

        workflow.append((step_name, slots))

    return workflow


def load_raw_test_dataset(file_path: Path, max_samples):
    convo_ids = []
    turn_counts = []
    with jsonlines.open(file_path) as reader:
        for sample in reader:
            convo_ids.append(sample["convo_id"])
            turn_counts.append(sample["turn_id"])
            if len(convo_ids) == max_samples:
                break
    return convo_ids, turn_counts


def create_compute_metric_fct(tokenizer, data_args, training_args, model_args):
    def decode(preds, labels):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        model_path = Path(model_args.model_name_or_path)
        file_name = "pred_mwoz.txt" if training_args.is_mwoz else "preds_test_set.txt"
        if not model_path.exists():
            # model name
            preds_file_path = Path(training_args.output_dir) / file_name
        else:
            preds_file_path = model_path / file_name

        with preds_file_path.open("w") as f:
            for pred, label in zip(decoded_preds, decoded_labels):
                label = label.replace("\n", " ")
                pred = pred.replace("\n", " ")
                f.write(f"{pred}\t{label}" + "\n")

        return decoded_preds, decoded_labels

    def parse_predictions(eval_preds):
        preds, labels = eval_preds
        decoded_predictions, decoded_labels = decode(preds, labels)
        return decoded_predictions, decoded_labels

    def compute_em_and_ce(eval_preds):
        predictions, labels = parse_predictions(eval_preds)
        predictions = [parse_workflow_string(w) for w in predictions]
        labels = [parse_workflow_string(w) for w in labels]
        return compute_metrics(labels, predictions, use_bert_score=training_args.use_bert_score)

    def compute_cds_metrics(eval_preds):
        predictions, labels = parse_predictions(eval_preds)
        convo_ids, turn_ids = load_raw_test_dataset(data_args.test_file, data_args.max_predict_samples)
        return compute_cds_em_and_ce(predictions, labels, convo_ids, turn_ids)

    def compute_ast_metrics(eval_preds):
        predictions, labels = parse_predictions(eval_preds)
        return compute_ast_acc_metrics(predictions, labels)

    def no_metrics(eval_preds):
        # Evaluation will be done during post hf_training
        preds, labels = eval_preds
        decode(preds, labels)
        return {}

    if training_args.no_metrics:
        return no_metrics
    elif training_args.use_cds_metrics:
        return compute_cds_metrics
    elif training_args.use_ast_metrics:
        return compute_ast_metrics
    else:
        return compute_em_and_ce