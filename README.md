# Workflow Discovery from Dialogues in the Low Data Regime

Code base for ["Workflow Discovery from Dialogues in the Low Data Regime"](https://openreview.net/forum?id=L9othQvPks) accepted at TMLR.

We have created a **leaderboard** for the Workflow Disovery task on Paper with Code [here](https://paperswithcode.com/task/workflow-discovery).

## Data

In this work we use two dataset ABCD (Chen et al., 2021) and MultiWOZ 2.2 (Zang et al., 2020).

### Create folder Structure

Create a following folder structure to contain all the data

```txt
<Project Directory>/
└── data/
    ├── raw 
    └── processed 
```

```shell
mkdir -p data/raw
mkdir -p data/processed
```

### Download ABCD Dataset

Since ABCD is not on huggingface datasets, we need to download it manually:

```shell
cd data/raw
wget https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz
wget https://raw.githubusercontent.com/asappresearch/abcd/master/data/guidelines.json
wget https://raw.githubusercontent.com/asappresearch/abcd/master/data/ontology.json
wget https://raw.githubusercontent.com/asappresearch/abcd/master/data/utterances.json
wget https://raw.githubusercontent.com/asappresearch/abcd/master/data/utterances.json
gunzip abcd_v1.1.json.gz
```

### Create Workflow Discovery Datasets for both ABCD and MultiWOZ

```shell
# Enable you virtual env
pip install -r requirements.txt

python generate_datasets.py --raw_data_folder ./data/raw --processed_data_folder ./data/processed 
```

Once the script above runs successfully, you should see the following files in the processed data folder

```txt
<Project Directory>/
└── data/
    └── processed 
           └── workflow_discovery 
               └── abcd
                   ├── train.json 
                   ├── dev.json 
                   └── test.json 
               └── multiwoz 
                   ├── train.json 
                   ├── dev.json 
                   └── test.json 
           ├── action_state_tracking
               ├── train.json 
               ├── dev.json 
               └── test.json 
           └── cascading_dialogue_sucess 
               ├── train.json 
               ├── dev.json 
               └── test.json 
```

## Training

The training code is based on the Hunggingface Transformers [examples](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py).

### Set up you environment

```shell
# Enable you virtual env
pip install -r requirements.txt

# Or use docker
docker build .
docker run <IMG name> <arguments>
# same args as bellow
```

### Run

```shell
python train.py --experiment_name my_wd_experiment \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_workflow_discovery_abcd.json \
  --validation_file ./data/processed/dev_workflow_discovery_abcd.json \
  --test_file ./data/processed/dev_workflow_discovery_abcd.json \
  --text_column input \
  --summary_column target \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --predict_with_generate
  --output_dir /tmp/results/
  --save_strategy epoch \
  --source_prefix "Extract workflow : " \
  --max_source_length 1024 \
  --max_target_length 256 \
  --val_max_target_length 256 \
  --learning_rate 5e-5 \
  --warmup_steps 500
```

In other to fine-tune the model a different task, one only need to change the dataset path and the ``--source_prefix`` parameter as follows:

### Workflow Discovery Task on ABCD

```sh
--train_file ./data/processed/train_workflow_discovery_abcd.json \
--validation_file ./data/processed/dev_workflow_discovery_abcd.json \
--test_file ./data/processed/dev_workflow_discovery_abcd.json \
--source_prefix 'Extract workflow : ' 
```

### Workflow Discovery Task on MultiWoz

```sh
--train_file ./data/processed/train_workflow_discovery_multiwoz.json \
--validation_file ./data/processed/dev_workflow_discovery_multiwoz.json \
--test_file ./data/processed/test_workflow_discovery_multiwoz.json \
--source_prefix 'Extract workflow : ' 
```

### AST

```sh
--train_file ./data/processed/train_AST_abcd.json \
--validation_file ./data/processed/dev_AST_abcd.json \
--test_file ./data/processed/test_workflow_AST_abcd.json \
--source_prefix 'Predict AST: ' 
--use_ast_metrics
```

### CDS

```sh
--train_file ./data/processed/train_CDS_abcd.json \
--validation_file ./data/processed/dev_CDS_abcd.json \
--test_file ./data/processed/test_CDS_abcd.json \
--source_prefix 'Predict CDS: ' 
--use_cds_metrics
```

### Notes

- When using BART models, you will need to add `--label_smoothing_factor 0.1`
- You can use the `--no_metrics` flag to disable metric calculation.
- You can use the `--use_bert_score` flag to enable bert score during evaluation.
- To use our conditioning mechanism, use `--text_column input_w_possible_actions` or `--text_column input_w_possible_actions_plus`
- We use the huggingface trainer, so this code support any standard command line that you can check with `python train.py -h`
