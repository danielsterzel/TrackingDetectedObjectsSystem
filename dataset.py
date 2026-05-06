from pathlib import Path
import pandas as pd

from sequence_config import TRAIN_SEQUENCES, TEST_SEQUENCES, SequenceConfig

DETECTIONS_SUFFIX = "det/det.txt"


columns = ["frame", "id", "x", "y", "w", "h", "confidence"]

# name: Dataframe

def load_detections(sequence_dict : dict[str, SequenceConfig]) -> dict[str, pd.DataFrame]:

    dataframes = {}
    for name, config in sequence_dict.items():

        print(name)
        det_path = config.folder_path / DETECTIONS_SUFFIX
        print(det_path)

        det_df = pd.read_csv(det_path, header=None, names=columns)
        print(det_df.head())

        dataframes[config.name] = det_df

    return dataframes

train_detections = load_detections(TRAIN_SEQUENCES)
test_detections = load_detections(TEST_SEQUENCES)


for name, df in train_detections.items():
    print(name)
    print(df.head())