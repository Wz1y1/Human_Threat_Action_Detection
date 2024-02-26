import numpy as np

from PathUtils.load_save import load_pkl_file
from pathlib import Path
from project_info import PROJECT_PATH
import os
from DataCollection.data_collection import MediapipeAns
from sklearn.preprocessing import OneHotEncoder

"""
#######################################################################################################
Hand recognition mapping
#######################################################################################################
"""
HAND_CAT_MAPPING = {
    "NoPunch": 0,
    "RightPunch": 1,
    "LeftPunch": 2,
    "Boxing": 3
}
HAND_THREAT_MAPPING = {
    "NoPunch": 0,
    "RightPunch": 1,
    "LeftPunch": 1,
    "Boxing": 2
}
HAND_CAT_MAPPING_INV = {_val: _key for _key, _val in HAND_CAT_MAPPING.items()}
HAND_ONE_HOT_ENCODER = OneHotEncoder(sparse=False)
HAND_ONE_HOT_ENCODER.fit(np.array([val for val in HAND_CAT_MAPPING.values()]).reshape((-1, 1)))

"""
#######################################################################################################
Face recognition mapping
#######################################################################################################
"""
FACE_CAT_MAPPING = {
    "NormalFace": 0,
    "AngryFace": 1
}
FACE_THREAT_MAPPING = {
    "NormalFace": 0,
    "AngryFace": 2
}
FACE_CAT_MAPPING_INV = {_val: _key for _key, _val in FACE_CAT_MAPPING.items()}
FACE_ONE_HOT_ENCODER = OneHotEncoder(sparse=False)
FACE_ONE_HOT_ENCODER.fit(np.array([val for val in FACE_CAT_MAPPING.values()]).reshape((-1, 1)))


def __load_all():
    data_base_folder = PROJECT_PATH / "MP_Data"
    data = {}
    for task_name, mapping in zip(["hand", "face"], [HAND_CAT_MAPPING, FACE_CAT_MAPPING]):
        data[task_name] = dict()
        for cat in mapping:
            data[task_name][cat] = []

            # Get the list of all files and folders in the specified directory
            all_files_and_folders = os.listdir(data_base_folder / cat)

            # Filter out the folders
            folders = [data_base_folder / cat / name / "mediapipe_ans"
                       for name in all_files_and_folders
                       if os.path.isdir(data_base_folder / cat / name / "mediapipe_ans")]

            for folder in folders:
                all_files = os.listdir(folder)
                data[task_name][cat].append([])
                assert len(all_files) == 30
                for file in all_files:
                    data[task_name][cat][-1].append(load_pkl_file(Path(folder / file)))

    return data


def load_nn_train_val(stage_name: str):
    this_cat_mapping = HAND_CAT_MAPPING
    this_one_hot_encoder = HAND_ONE_HOT_ENCODER
    dims = {"pose", "lh", "rh"}
    if stage_name == "hand":
        pass
    elif stage_name == "face":
        this_cat_mapping = FACE_CAT_MAPPING
        this_one_hot_encoder = FACE_ONE_HOT_ENCODER
        dims = {"face"}
    else:
        raise "Unknown stage name"

    data = __load_all()[stage_name]
    train_data_x = {name: [] for name in MediapipeAns.input_feature_name}
    train_data_y = []
    val_data_x = {name: [] for name in MediapipeAns.input_feature_name}
    val_data_y = []

    for key, samples in data.items():
        # Go through all collected samples
        for i in range(len(samples)):
            this_data_x = train_data_x
            this_data_y = train_data_y
            if i % 5 == 0:
                this_data_x = val_data_x
                this_data_y = val_data_y

            # One sample for NN
            tmp_x = MediapipeAns.dict_to_nn_input(samples[i])

            # Append to all samples
            for name in MediapipeAns.input_feature_name:
                this_data_x[name].append(np.array(tmp_x[name]))

            this_data_y.append(this_cat_mapping[key])

    # All to numpy array
    for x in [train_data_x, val_data_x]:
        for name in MediapipeAns.input_feature_name:
            x[name] = np.array(x[name])

    train_data_y = np.array(train_data_y).reshape((-1, 1))
    val_data_y = np.array(val_data_y).reshape((-1, 1))

    return {"train": ({key: val for key, val in train_data_x.items() if key in dims},
                      this_one_hot_encoder.transform(train_data_y)),
            "val": ({key: val for key, val in val_data_x.items() if key in dims},
                    this_one_hot_encoder.transform(val_data_y))}


if __name__ == "__main__":
    load_nn_train_val("face")
