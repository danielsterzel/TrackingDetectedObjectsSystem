from dataclasses import dataclass
from pathlib import Path


class Config:
    dataset_root: Path
    sequence_name: str

config = Config(
    dataset_root=Path("./evs_mot_public_dataset/evs_mot-train"),
    sequence_name="MOT_02"
)