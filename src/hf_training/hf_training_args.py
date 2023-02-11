from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class HfSeq2SeqTrainingArgs(Seq2SeqTrainingArguments):
    experiment_name: Optional[str] = field(default="workflow_discovery_exp", metadata={"help": "Experiment name"})

    use_wandb: bool = field(default=False, metadata={"help": "Use wand db for logging"})
    wandb_project_name: str = field(default="text2flow", metadata={"help": "Wandb project name"})
    no_metrics: bool = field(default=False, metadata={"help": "Do not calculate metrics"})
    use_bert_score: bool = field(default=False, metadata={"help": "Use BERT score during metric calculation"})
    use_cds_metrics: bool = field(default=False,  metadata={"help": "Use CDS metrics"})
    use_ast_metrics: bool = field(default=False,  metadata={"help": "Use CDS metrics"})

    is_mwoz: bool = field(default=False)
    ood_step: Optional[str] = field(default=None)
