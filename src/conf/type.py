from dataclasses import dataclass
from typing import Any


@dataclass
class DirConfig:
    data_dir: str


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any]


@dataclass
class TrainConfig:
    dir: DirConfig
    model: ModelConfig
    n_folds: int
    fold: int
    n_epochs: int
    train_batch_size: int
    valid_batch_size: int
    scheduler: str
    lr: float


@dataclass
class InferConfig:
    dir: DirConfig
    model: ModelConfig
    n_folds: int
    model_dir: str
    valid_batch_size: int
