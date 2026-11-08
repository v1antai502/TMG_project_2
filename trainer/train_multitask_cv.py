"""K-fold cross-validation of PhoBERT trained on clickbait + topic, same folds as the others.

    python -m trainer.train_multitask_cv --topic-weight 0.5

The topic is the dataset's `category` column (13 classes, including "Other"). It is used
only as a training signal: prediction uses the clickbait head alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from architecture.phobert import load_tokenizer
from architecture.phobert_multitask import PhoBERTMultiTask
from trainer import train_phobert_cv
from trainer.train_phobert import ROOT, parse_config
from trainer.train_phobert_cv import CVConfig
from utils.loss_function import ClassificationLoss, MultiTaskLoss


@dataclass
class MultiTaskConfig(CVConfig):
    out_dir: str = str(ROOT / "outputs" / "multitask_cv")
    topic_weight: float = 0.5  # loss = clickbait loss + topic_weight * topic loss
    epochs: int = 5
    layer_decay: float = 0.9
    monitor: str = "loss"


def setup(cfg: MultiTaskConfig, pool, test):
    """Adds `topic_id` to the article tables and returns what `train_phobert_cv.run` needs."""
    topics = sorted(set(pool["category"]) | set(test["category"]))
    index = {t: i for i, t in enumerate(topics)}
    # in place: run() builds the fold splits from these same tables after setup
    pool["topic_id"] = pool["category"].map(index)
    test["topic_id"] = test["category"].map(index)
    print(f"topic task: {len(topics)} topics, weight {cfg.topic_weight}")

    def make_model():
        return PhoBERTMultiTask(len(topics), cfg.model_name, dropout=cfg.dropout)

    def make_criterion(weights):
        # validation/test loss is the clickbait loss alone, comparable with the text model
        return MultiTaskLoss(ClassificationLoss(weights), cfg.topic_weight), ClassificationLoss(weights)

    trainer_kwargs = {"make_criterion": make_criterion}
    return load_tokenizer(cfg.model_name), make_model, trainer_kwargs, {"class_names": {"topic": topics}}


def run(cfg: MultiTaskConfig) -> dict:
    return train_phobert_cv.run(cfg, setup=setup)


if __name__ == "__main__":
    run(parse_config(MultiTaskConfig, __doc__.strip().splitlines()[0]))
