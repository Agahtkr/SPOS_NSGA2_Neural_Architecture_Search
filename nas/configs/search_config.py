# configs/search_config.py
import os
import yaml
from dataclasses import dataclass, fields


@dataclass
class NASConfig:
    num_layers: int
    base_channels: int
    in_channels: int
    num_classes: int
    class_mapping: dict
    input_resolution: int
    batch_size: int
    supernet_epochs: int
    learning_rate: float
    weight_decay: float
    save_dir: str
    bn_calib_batches: int
    train_img_dir: str
    train_lbl_dir: str
    val_image_dir: str
    val_lbl_dir: str
    pop_size: int
    num_generations: int
    elitism_ratio: float
    crossover_rate: float
    mutation_rate: float
    # Optional keys (must come after the required ones in a dataclass)
    paths_per_batch: int = 2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "config.yaml")


def load_config(yaml_path: str = DEFAULT_CONFIG_PATH) -> NASConfig:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Ignore unknown keys instead of crashing, but tell the user.
    known = {f.name for f in fields(NASConfig)}
    unknown = set(data) - known
    if unknown:
        print(f"[config] Ignoring unknown keys: {sorted(unknown)}")

    return NASConfig(**{k: v for k, v in data.items() if k in known})