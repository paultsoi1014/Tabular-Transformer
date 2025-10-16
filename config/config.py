import json
import os

from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Any, Dict, List

if os.getenv("ENVIRONMENT", "development") == "development":
    load_dotenv(dotenv_path=".env", override=True)


@dataclass
class DataConfig:
    """Configuration for data module"""

    data_dir: str
    data_files: str
    val_size: float
    test_size: float
    batch_size: int
    target_name: str
    categorical_features: List[str]
    continuous_features: List[str]

    @classmethod
    def from_env(cls) -> "DataConfig":
        # parse categorical features as comma-separated string
        categorical_features_str = os.getenv("CATEGORICAL_FEATURES", "")
        categorical_features = (
            [
                feat.strip()
                for feat in categorical_features_str.split(",")
                if feat.strip()
            ]
            if categorical_features_str
            else []
        )

        # parse continuous features as comma-separated string
        continuous_features_str = os.getenv("CONTINUOUS_FEATURES", "")
        continuous_features = (
            [
                feat.strip()
                for feat in continuous_features_str.split(",")
                if feat.strip()
            ]
            if continuous_features_str
            else []
        )

        return cls(
            data_dir=os.getenv("DATA_DIR", "./horse_race/data/source"),
            data_files=os.getenv("DATA_FILES", "data.csv"),
            val_size=float(os.getenv("VAL_SIZE", 0.2)),
            test_size=float(os.getenv("TEST_SIZE", 0.1)),
            batch_size=int(os.getenv("BATCH_SIZE", 64)),
            target_name=os.getenv("TARGET_NAME", "None"),
            categorical_features=categorical_features,
            continuous_features=continuous_features,
        )


@dataclass
class ModelConfig:
    """Configuration for model module"""

    output_dim: int
    embedding_dim: int
    n_heads: int
    n_layers: int
    dim_feedforward: int
    dropout: float
    mlp_hidden_dims: List[int]
    activation: str

    @classmethod
    def from_env(cls) -> "ModelConfig":
        mlp_hidden_dims_str = os.getenv("MLP_HIDDEN_DIM", "32,16")
        mlp_hidden_dims = [int(dim.strip()) for dim in mlp_hidden_dims_str.split(",")]

        return cls(
            output_dim=int(os.getenv("OUTPUT_DIM", 3)),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", 64)),
            n_heads=int(os.getenv("N_HEADS", 8)),
            n_layers=int(os.getenv("N_LAYERS", 3)),
            dim_feedforward=int(os.getenv("DIM_FEEDFORWARD", 3)),
            dropout=float(os.getenv("DROPOUT", 0.01)),
            mlp_hidden_dims=mlp_hidden_dims,
            activation=os.getenv("ACTIVATION", "gelu"),
        )


@dataclass
class TrainingConfig:
    optimizer: str
    epochs: int
    loss_function: str
    scheduler: str
    early_stopping: bool
    save_model: bool
    optimizer_kwargs: Dict[str, Any] = None
    loss_kwargs: Dict[str, Any] = None
    scheduler_kwargs: Dict[str, Any] = None
    early_stopping_config: Dict[str, Any] = None
    save_model_config: Dict[str, Any] = None

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        # parse kwargs for optimizer
        optimizer_kwargs_str = os.getenv("OPTIMIZER_KWARGS", "{}")
        try:
            optimizer_kwargs = json.loads(optimizer_kwargs_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in OPTIMIZER_KWARGS: {e}")

        # parse kwargs for loss function
        loss_kwargs_str = os.getenv("LOSS_FUNCTION_KWARGS", "{}")
        try:
            loss_kwargs = json.loads(loss_kwargs_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LOSS_FUNCTION_KWARGS: {e}")

        # parse kwargs for scheduler
        scheduler_kwargs_str = os.getenv("SCHEDULER_KWARGS", "{}")
        try:
            scheduler_kwargs = json.loads(scheduler_kwargs_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in SCHEDULER_KWARGS: {e}")

        # parse config for earlystopping
        early_stopping_config_str = os.getenv("EARLY_STOPPING_CONFIG", "{}")
        try:
            early_stopping_config = json.loads(early_stopping_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in EARLY_STOPPING_CONFIG: {e}")

        # parse config for model save
        save_model_config_str = os.getenv("MODEL_SAVE_CONFIG", "{}")
        try:
            save_model_config = json.loads(save_model_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in MODEL_SAVE_CONFIG: {e}")

        return cls(
            optimizer=os.getenv("OPTIMIZER", None),
            optimizer_kwargs=optimizer_kwargs,
            epochs=int(os.getenv("EPOCHS", 200)),
            loss_function=os.getenv("LOSS_FUNCTION", None),
            loss_kwargs=loss_kwargs,
            scheduler=os.getenv("SCHEDULER", None),
            scheduler_kwargs=scheduler_kwargs,
            early_stopping=os.getenv("EARLY_STOPPING", "true").lower() == "true",
            early_stopping_config=early_stopping_config,
            save_model=os.getenv("SAVE_MODEL", "true").lower() == "true",
            save_model_config=save_model_config,
        )


@dataclass
class WandbConfig:
    """Configuration for WandB logger"""

    api_key: str
    base_url: str
    project_name: str
    run_name: str
    use_wandb: bool

    @classmethod
    def from_env(cls) -> "WandbConfig":
        return cls(
            api_key=os.getenv("WANDB_API_KEY", None),
            base_url=os.getenv("WANDB_BASE_URL", "http://192.168.2.25:8080"),
            project_name=os.getenv("WANDB_PROJECT_NAME", None),
            run_name=os.getenv("WANDB_RUN_NAME", None),
            use_wandb=os.getenv("USE_WANDB", "false").lower() == "true",
        )


@dataclass
class Config:
    """Main project configuration"""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    wandb: WandbConfig

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables"""
        return cls(
            data=DataConfig.from_env(),
            model=ModelConfig.from_env(),
            training=TrainingConfig.from_env(),
            wandb=WandbConfig.from_env(),
        )
