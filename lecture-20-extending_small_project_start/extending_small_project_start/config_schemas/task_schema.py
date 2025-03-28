from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING

from config_schemas.task import optimizer_schema, model_schema, loss_function_schema


@dataclass
class TaskConfig:
    _target_: str = MISSING
    optimizer: optimizer_schema.OptimizerConfig = MISSING


@dataclass
class MNISTClassifciationTaskConfig(TaskConfig):
    _target_: str = "tasks.MNISTClassificationTrainingTask"
    model: model_schema.ModelConfig = MISSING
    loss_function: loss_function_schema.LossFunctionConfig = MISSING


def setup_config() -> None:
    optimizer_schema.setup_config()
    model_schema.setup_config()
    loss_function_schema.setup_config()

    cs = ConfigStore.instance()
    cs.store(group="task", name="mnist_classification_training_task_schema", node=MNISTClassifciationTaskConfig)
