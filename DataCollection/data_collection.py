import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Dict
from numpy import ndarray
from mediapipe_utils import mediapipe_detection, draw_styled_landmarks, mp_holistic, extract_keypoints
from abc import abstractmethod, ABCMeta
from PathUtils.load_save import save_pkl_file
from PathUtils.path_and_file import try_to_find_file, try_to_find_file_if_exist_then_delete, \
    try_to_find_folder_path_otherwise_make_one, try_to_find_folder_if_exist_then_delete

# Path for exported data, numpy arrays
DATA_PATH = Path('../MP_Data')

NO_SEQ = 30
SEQ_LEN = 30
CHECK_RATIO = 0.8

HOLISTIC = None
CAP = None


def read_one_sample() -> List[ndarray]:
    collect_seq_len = SEQ_LEN * 2  # We collect double frames but only use some of them
    i = 0
    frames = []

    print(f"Start reading...")

    while i < collect_seq_len:
        success, frame = CAP.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        frames.append(frame)

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        i += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Finish!")

    return frames


def wait_to_read(num_sample: int) -> None:
    print(f"Get ready to read {num_sample}-th sample")
    for i in range(4):
        time.sleep(0.25)
        print(f"Count down: {3 - i}")


class AbstractCheckOneSample(metaclass=ABCMeta):
    def __init__(self, ratio: float, sample: List[ndarray]):
        """
        Constructor
        :param ratio: A threshold determining whether the current sample is OK or not.
        Because a sample has multiple frames, and it is possible that not all of them contain the same action.
        """
        assert 0 < ratio < 1, "Ratio shall be within 0 and 1, exclusive"
        self.ratio: float = ratio
        self.sample: List[ndarray] = sample

        self.left_index: int = -1
        self.right_index: int = -1

        self.detect_results: List[tuple] = []

    def run_check(self) -> bool:
        sample_len = len(self.sample)
        min_ok_len = int(self.ratio * SEQ_LEN)
        self.__detect()

        # Use sliding window algorithm
        left = 0
        right = 0
        ok_count = 0
        ok = False
        max_ratio = 0
        while right < sample_len:
            if self._check_one_frame(right):
                ok_count += 1

            if right - left + 1 == SEQ_LEN:  # Fixed window
                if ok_count >= min_ok_len:
                    ok = True
                    break
                else:
                    max_ratio = max(max_ratio, ok_count / SEQ_LEN)
                    right += 1
                    if self._check_one_frame(left):
                        ok_count -= 1
                    left += 1
            else:  # Increasing window
                right += 1

        if not ok:
            print(f"Invalid sample, try to read again, expected ratio >= {self.ratio}, actual = {max_ratio}")
            return False

        self.left_index = left
        self.right_index = right
        return True

    @abstractmethod
    def _check_one_frame(self, i: int) -> bool:
        pass

    def __detect(self) -> None:
        for frame in self.sample:
            image, result = mediapipe_detection(frame, HOLISTIC)
            self.detect_results.append((image, result))


class RightPunchCheck(AbstractCheckOneSample):
    def _check_one_frame(self, i: int) -> bool:
        return self.detect_results[i][1].right_hand_landmarks is not None


class LeftPunchCheck(AbstractCheckOneSample):
    def _check_one_frame(self, i: int) -> bool:
        return self.detect_results[i][1].left_hand_landmarks is not None


class BoxingCheck(AbstractCheckOneSample):
    def _check_one_frame(self, i: int) -> bool:
        return (self.detect_results[i][1].left_hand_landmarks is not None and
                self.detect_results[i][1].right_hand_landmarks is not None)


class AngryFaceCheck(AbstractCheckOneSample):
    def _check_one_frame(self, i: int) -> bool:
        return self.detect_results[i][1].face_landmarks is not None


class NormalFaceCheck(AbstractCheckOneSample):
    def _check_one_frame(self, i: int) -> bool:
        return self.detect_results[i][1].face_landmarks is not None


class NoPunchCheck(AbstractCheckOneSample):
    def _check_one_frame(self, i: int) -> bool:
        return True


class MediapipeAns:
    input_feature_name = ["pose", "face", "lh", "rh"]

    # https://developers.google.com/mediapipe/solutions/vision/face_landmarker
    shape = {
        "pose": (33, 4),
        "face": (478, 3),
        "lh": (21, 3),
        "rh": (21, 3)
    }

    def __init__(self, detect_result):
        ans = self.detect_result_to_ndarray(detect_result)
        self.pose = ans["pose"]
        self.face = ans["face"]
        self.lh = ans["lh"]
        self.rh = ans["rh"]

    @staticmethod
    def detect_result_to_ndarray(detect_result) -> Dict[str, ndarray]:
        ans = {
            "pose": np.zeros(MediapipeAns.shape["pose"]),
            "face": np.zeros(MediapipeAns.shape["face"]),
            "lh": np.zeros(MediapipeAns.shape["lh"]),
            "rh": np.zeros(MediapipeAns.shape["rh"])
        }

        if detect_result is not None:
            if detect_result.pose_landmarks:
                ans["pose"] = np.array([[res.x, res.y, res.z, res.visibility]
                                        for res in detect_result.pose_landmarks.landmark])

            if detect_result.face_landmarks:
                ans["face"] = np.array([[res.x, res.y, res.z]
                                        for res in detect_result.face_landmarks.landmark])

            if detect_result.left_hand_landmarks:
                ans["lh"] = np.array([[res.x, res.y, res.z]
                                      for res in detect_result.left_hand_landmarks.landmark])

            if detect_result.right_hand_landmarks:
                ans["rh"] = np.array([[res.x, res.y, res.z]
                                      for res in detect_result.right_hand_landmarks.landmark])

        return ans

    def to_dict(self):
        return {
            "pose": self.pose,
            "face": self.face,
            "lh": self.lh,
            "rh": self.rh
        }

    @staticmethod
    def dict_to_nn_input(one_sample: List[Dict[str, ndarray]]) -> Dict[str, List[ndarray]]:
        assert len(one_sample) == SEQ_LEN, f"One sample shall has a length of {SEQ_LEN}"

        nn_input = {key: [] for key in MediapipeAns.input_feature_name}
        for frame in one_sample:
            for key in nn_input:
                if frame[key].shape == (468, 3) and np.min(frame[key]) == np.max(frame[key]) == 0:
                    frame[key] = np.zeros((478, 3))

                nn_input[key].append(frame[key].flatten())

        return nn_input


def check_one_sample_factory(action: str, ratio: float, sample: List[ndarray]) -> AbstractCheckOneSample:
    if action == "RightPunch":
        return RightPunchCheck(ratio, sample)
    elif action == "LeftPunch":
        return LeftPunchCheck(ratio, sample)
    elif action == "Boxing":
        return BoxingCheck(ratio, sample)
    elif action == "AngryFace":
        return AngryFaceCheck(ratio, sample)
    elif action == "NormalFace":
        return NormalFaceCheck(ratio, sample)
    elif action == "NoPunch":
        return NoPunchCheck(ratio, sample)
    # else:
    #     raise "Unknown action"


def collect_to_images(action: str, num_samples: int = 300) -> None:
    num_sample = 0
    while num_sample < num_samples:
        base = Path(DATA_PATH / action / str(num_sample))

        if try_to_find_file(base / "good_sample.pkl"):
            num_sample += 1
            continue

        try_to_find_folder_if_exist_then_delete(base)  # Not good sample or there is no sample

        # Get ready to read
        wait_to_read(num_sample)

        # Read one sample
        one_sample: List[ndarray] = read_one_sample()

        # Process (discard any invalid sample)
        check_one_sample_obj: AbstractCheckOneSample = check_one_sample_factory(action, CHECK_RATIO, one_sample)
        if not check_one_sample_obj.run_check():
            continue

        # Save
        os.makedirs(base / "raw")
        os.makedirs(base / "detected")
        os.makedirs(base / "mediapipe_ans")

        for i in range(check_one_sample_obj.left_index, check_one_sample_obj.right_index + 1):
            j = i - check_one_sample_obj.left_index  # Offset

            cv2.imwrite((base / "raw" / f'{j}.png').__str__(), one_sample[i])

            draw_styled_landmarks(*check_one_sample_obj.detect_results[i])
            cv2.imwrite((base / "detected" / f'{j}.png').__str__(), check_one_sample_obj.detect_results[i][0])

            save_pkl_file(base / "mediapipe_ans" / f'{j}.pkl',
                          MediapipeAns(check_one_sample_obj.detect_results[i][1]).to_dict())

        # Mark this sample is good
        save_pkl_file(base / "good_sample.pkl", None)

        num_sample += 1


if __name__ == "__main__":
    # Initialize the webcam
    CAP = cv2.VideoCapture(0)

    current_action = "AngryFace"

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              model_complexity=2,
                              refine_face_landmarks=True,
                              min_tracking_confidence=0.5) as holistic:
        HOLISTIC = holistic
        collect_to_images(current_action)

    # Release the webcam and destroy all OpenCV windows
    CAP.release()
    cv2.destroyAllWindows()
