from pathlib import Path
import pandas as pd

from sequence_config import TRAIN_SEQUENCES, TEST_SEQUENCES, SequenceConfig

DETECTIONS_SUFFIX = "det/det.txt"


columns = ["frame", "id", "x", "y", "w", "h", "confidence"]

# name: Dataframe
def load_detections(
    sequence_dict: dict[str, SequenceConfig],
) -> dict[str, pd.DataFrame]:

    dataframes = {}
    for name, config in sequence_dict.items():

        det_path = config.folder_path / DETECTIONS_SUFFIX
        det_df = pd.read_csv(det_path, header=None, names=columns)
        dataframes[config.name] = det_df

    return dataframes


train_detections = load_detections(TRAIN_SEQUENCES)
test_detections = load_detections(TEST_SEQUENCES)


if __name__ == "__main__":
    for name, df in train_detections.items():
        print(name)
        print(df.head())
