import numpy as np
from sklearn.metrics import precision_recall_curve

from Testing.full_model_real_time_testing import nn_predication, get_trained_full_model
from pathlib import Path
from PathUtils.load_save import load_pkl_file
from DataCollection.data_collection import MediapipeAns
import tensorflow as tf
from typing import List

import matplotlib.pyplot as plt

# noinspection PyTypeChecker
def test_one(mediapipe_result_folder: Path, model: tf.keras.Model):
    reads = []
    for i in range(30):
        reads.append(load_pkl_file(mediapipe_result_folder / f"{i}.pkl"))

    nn_input = MediapipeAns.dict_to_nn_input(reads)
    for key in nn_input.keys():
        nn_input[key] = np.array(nn_input[key])[np.newaxis, :, :]

    ans = nn_predication(model, nn_input)

    # Return the probability for 'NormalFace'
    return ans['face_prob'] if ans['face_class'] == 'NormalFace' else 1 - ans['face_prob']

    # return ans


def test_several(folders: List[Path]):
    model = get_trained_full_model()
    # ans = []
    # for folder in folders:
    #     ans.append(test_one(folder, model))
    #
    # return ans
    results = []
    for folder in folders:
        # Assuming folder name contains the true class
        true_label = 1 if 'NormalFace' in str(folder) else 0
        predicted_prob = test_one(folder, model)
        results.append((true_label, predicted_prob))
    return results



if __name__ == "__main__":
    to_test = []
    for i in range(200):
        to_test.append(Path(fr"C:\Users\Wz1y1\Desktop\PhaseTwoDelivery\MP_Data\Face_Validation\{i}\mediapipe_ans"))

    test_results = test_several(to_test)

    true_labels = [result[0] for result in test_results]
    predicted_probs = [result[1] for result in test_results]

    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    print(test_several(to_test))
