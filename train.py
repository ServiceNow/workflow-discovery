"""
Reference: https://github.com/huggingface/transformers/tree/main/examples/pytorch

Adapted from huggingface Transformers
"""

import logging
import os
import sys
from pathlib import Path

import datasets
import transformers
import transformers.trainer_utils as hf_trainer_utils

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    set_seed,
    MBartTokenizer,
    MBartTokenizerFast,
)

from src.data.data_args import DataArguments
from src.data.dataset_loader import DatasetLoader
from src.data.utils import group_col_name
from src.metrics import create_compute_metric_fct, verify_nltk
from src.model.hf_model_args import HfModelArguments
from src.hf_training.hf_training_args import HfSeq2SeqTrainingArgs

logger = logging.getLogger(__name__)


def hf_run():
    data_args, model_args, training_args = get_args()

    setup_wandb(training_args)

    setup_logging(training_args)

    verify_nltk()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: % distributed hf_training: %s 16-bits hf_training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    datasets_loader = DatasetLoader(data_args, training_args, tokenizer)
    train_dataset, validation_dataset, test_dataset = datasets_loader.load_datasets()

    model = load_model(model_args, data_args, tokenizer)

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            "`%s`. This will lead to loss being calculated twice and will take up more memory",
            model.__class__.__name__,
        )
    metric_fct = create_compute_metric_fct(tokenizer, data_args, training_args, model_args)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=create_data_collector(model, tokenizer, training_args, data_args),
        compute_metrics=metric_fct if training_args.predict_with_generate else None,
    )

    if training_args.do_train:
        train(trainer, train_dataset, training_args)

    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    if training_args.do_eval:
        do_eval(trainer, validation_dataset, max_length, num_beams)

    if training_args.do_predict:
        do_predict(trainer, test_dataset, tokenizer, training_args, data_args, model_args, max_length, num_beams)


def train(trainer, train_dataset, training_args):
    logger.info("*** train ***")

    check_point = get_resume_checkpoint(training_args)
    train_result = trainer.train(resume_from_checkpoint=check_point)

    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def do_eval(trainer, validation_dataset, max_length, num_beams):
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

    metrics["eval_samples"] = len(validation_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def do_predict(trainer, test_dataset, tokenizer, training_args, data_args, model_args, max_length, num_beams):
    logger.info("*** Predict ***")

    metrics = {}
    predictions = []

    if group_col_name in test_dataset.column_names:
        group_idx = 0

        while True:
            group_dataset = test_dataset.filter(lambda x: x[group_col_name] == group_idx)
            if group_dataset.num_rows == 0:
                # no groups left
                break
            logger.info("Predicting on test group %d", group_idx)

            predict_results = trainer.predict(
                group_dataset,
                metric_key_prefix=f"predict_group_{group_idx}",
                max_length=max_length,
                num_beams=num_beams,
            )
            metrics.update(predict_results.metrics)
            metrics[f"predict_samples_group_{group_idx}_size"] = len(group_dataset)

            group_idx += 1

            predictions.append(predict_results.predictions)

        for key in ["loss", "rouge1", "rouge2", "rougeL"]:
            metrics[f"overall_predict_{key}"] = round(
                sum([metrics[f"predict_group_{idx}_{key}"] for idx in range(group_idx)]) / group_idx, 4
            )
    else:
        predict_results = trainer.predict(
            test_dataset, metric_key_prefix="test", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        metrics["predict_samples_size"] = len(test_dataset)

    trainer.log(metrics)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


def get_args():
    parser = HfArgumentParser((HfModelArguments, DataArguments, HfSeq2SeqTrainingArgs))

    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    name_parts = [training_args.experiment_name]
    name_parts.extend([data_args.text_column, data_args.summary_column])

    name_parts.append(model_args.model_name_or_path)

    training_args.experiment_name = "_".join(name_parts)

    training_args.output_dir = str(Path(training_args.output_dir).joinpath(training_args.experiment_name))

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    return data_args, model_args, training_args


def load_model(model_args, data_args, tokenizer):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Forcing the generation min lenght, to avoid models preset for summarization tasks that are usually high
    config.min_length = 5

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))

    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get("summarization_cnn", {}))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id["en_XX"]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("en_XX")

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    return model


def get_resume_checkpoint(training_args):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    last_checkpoint = get_last_checkpoint(training_args)
    if last_checkpoint is not None:
        checkpoint = last_checkpoint

    return checkpoint


def get_last_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = hf_trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming hf_training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def create_data_collector(model, tokenizer, training_args, data_args):
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    return DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )


def setup_wandb(training_args):
    if training_args.use_wandb:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project_name
        training_args.run_name = training_args.experiment_name


if __name__ == "__main__":
    hf_run()
