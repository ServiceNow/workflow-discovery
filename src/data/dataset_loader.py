import sys
import logging
from pathlib import Path

from datasets import load_dataset

from src.data.utils import (
    summarization_name_mapping,
    train_dataset_name,
    validation_dataset_name,
    test_dataset_name,
    group_col_name,
    conversation_id_col_name,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class DatasetLoader:
    def __init__(self, data_args, training_args, tokenizer):
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer

    def load_datasets(self):
        if not any([self.training_args.do_train, self.training_args.do_eval, self.training_args.do_predict]):
            return

        raw_datasets = self._load_raw_datasets()

        train_dataset, validation_dataset, test_dataset = None, None, None
        if self.training_args.do_train:
            train_dataset = self._load_dataset(
                raw_datasets, train_dataset_name, self.data_args.max_target_length, self.data_args.max_train_samples
            )

        if self.training_args.do_eval:
            validation_dataset = self._load_dataset(
                raw_datasets,
                validation_dataset_name,
                self.data_args.val_max_target_length,
                self.data_args.max_eval_samples,
            )

        if self.training_args.do_predict:
            test_dataset = self._load_dataset(
                raw_datasets,
                test_dataset_name,
                self.data_args.val_max_target_length,
                self.data_args.max_predict_samples,
            )

        return train_dataset, validation_dataset, test_dataset

    def _load_raw_datasets(self):
        if self.data_args.dataset_name is not None:
            raw_datasets = load_dataset(
                self.data_args.dataset_name, self.data_args.dataset_config_name, cache_dir=self.data_args.data_cache_dir
            )
        else:
            raw_datasets = self._load_dataset_from_files()

        return raw_datasets

    def _load_dataset_from_files(self, train_file=None, validation_file=None, test_file=None):
        data_files = {}
        extensions = []

        train_file = train_file or self.data_args.train_file
        if train_file:
            train_file = Path(train_file)
            data_files[train_dataset_name] = str(train_file.absolute())
            extensions.append(train_file.suffix[1:])

        validation_file = validation_file or self.data_args.validation_file
        if validation_file:
            validation_file = Path(validation_file)
            data_files[validation_dataset_name] = str(validation_file.absolute())
            extensions.append(validation_file.suffix[1:])

        test_file = test_file or self.data_args.test_file
        if test_file:
            test_file = Path(test_file)
            extensions.append(test_file.suffix[1:])

        if len(set(extensions)) != 1:
            raise ValueError("Dataset file should have the same extension")

        supported_extensions = ["csv", "json"]
        if extensions[0] not in supported_extensions:
            raise ValueError(
                f"Invalid dataset extension ({extensions[0]}). Should be one of: {[','.join(supported_extensions)]}"
            )

        datasets = load_dataset(extensions[0], data_files=data_files, cache_dir=self.data_args.data_cache_dir)

        if test_file:
            # The test dataset contains column not in the train dataset like `group` and the dataset
            # library deletes the columns that does not exits in the train dataset.
            test_dataset = load_dataset(
                extensions[0], data_files={test_dataset_name: str(test_file)}, cache_dir=self.data_args.data_cache_dir
            )
            datasets.update(test_dataset)

        return datasets

    def _load_dataset(self, raw_datasets, dataset_name, max_target_length, max_samples):
        dataset = raw_datasets[dataset_name]

        if max_samples is not None:
            dataset = dataset.select(range(max_samples))

        preprocess_fct = self._create_preprocess_fct(dataset.column_names, max_target_length, dataset_name)

        columns_to_remove = dataset.column_names
        if dataset_name == test_dataset_name:
            for col_to_keep in [group_col_name, conversation_id_col_name]:
                if col_to_keep in columns_to_remove:
                    columns_to_remove.remove(col_to_keep)

        with self.training_args.main_process_first(desc=f"{dataset_name} dataset map pre-processing"):
            dataset = dataset.map(
                preprocess_fct,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=columns_to_remove,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc=f"Running tokenizer on {dataset_name} dataset",
            )

        return dataset

    def _create_preprocess_fct(self, column_names, max_target_length, dataset_name):
        prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""

        dataset_columns = summarization_name_mapping.get(self.data_args.dataset_name, None)
        if self.data_args.text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = self.data_args.text_column
            if text_column not in column_names:
                raise ValueError(
                    f"--text_column' value '{self.data_args.text_column}' needs to be one of: {', '.join(column_names)}"
                )

        if self.data_args.summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = self.data_args.summary_column
            if summary_column not in column_names:
                raise ValueError(
                    f"--summary_column' value '{self.data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
                )

        padding = "max_length" if self.data_args.pad_to_max_length else False

        # Setting the data_args and tokenizer as local variable is need for the hashing required for caching
        data_args = self.data_args
        tokenizer = self.tokenizer

        def preprocess_function(examples):
            inputs = examples[text_column]
            targets = examples[summary_column]
            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return preprocess_function
