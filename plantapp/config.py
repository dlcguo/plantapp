import json, pathlib
from dataclasses import dataclass
from typing import List

# This file contains the configs for the plantapp project

@dataclass
class Config:
    img_size:        int
    imagenet_mean:   List[float]
    imagenet_std:    List[float]
    data_root:       str
    train_split_dir: str
    val_split_dir:   str
    batch_size:      int
    num_workers:     int
    epochs:          int
    initial_lr:      float
    weight_decay:    float
    label_smoothing: float
    checkpoint_path: str
    gradcam_layer:   str
    plant_names:     List[str]
    disease_names:   List[str]
    gemini_api_key:  str

CONFIG_PATH = pathlib.Path(__file__).with_suffix(".json")

with CONFIG_PATH.open() as f:
    cfg_dict = json.load(f)

cfg = Config(**cfg_dict) 