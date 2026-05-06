import configparser
from dataclasses import dataclass
from pathlib import Path

from constants import SEQUENCE_CONFIG_FILE, TEST_PATH, TRAIN_PATH

@dataclass
class SequenceConfig:
    folder_path: Path
    name: str
    imDir: str
    frameRate: int
    seqLength: int
    imWidth: int
    imHeight: int
    imExt: str


def load_sequence_config(sequence_path: Path) -> SequenceConfig:

    config = configparser.ConfigParser()

    config.read(sequence_path / SEQUENCE_CONFIG_FILE)
    sequence = config["Sequence"]

    return SequenceConfig(
        folder_path=sequence_path,
        name=sequence["name"],
        imDir=sequence["imDir"],
        frameRate=int(sequence["frameRate"]),
        seqLength=int(sequence["seqLength"]),
        imWidth=int(sequence["imWidth"]),
        imHeight=int(sequence["imHeight"]),
        imExt=sequence["imExt"],
    )


def load_sequences(dataset_path: Path) -> dict[str, SequenceConfig]:

    sequences = {}
    for sequence_path in dataset_path.iterdir():
        if not sequence_path.is_dir():
            continue
        sequence_config = sequence_path
        config = load_sequence_config(sequence_path)

        sequences[config.name] = config

    return sequences


# name : config
TRAIN_SEQUENCES = load_sequences(TRAIN_PATH)
TEST_SEQUENCES = load_sequences(TEST_PATH)


if __name__ == "__main__":
    print(TEST_SEQUENCES)
