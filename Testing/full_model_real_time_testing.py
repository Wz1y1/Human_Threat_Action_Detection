import sys;

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\Wz1y1\\Desktop\\PhaseTwoDelivery', 'C:/Users/Wz1y1/Desktop/PhaseTwoDelivery'])
 
import time

from Model.full_model import load_full_model
from project_info import PROJECT_PATH
import cv2
from mediapipe_utils import mediapipe_detection, draw_styled_landmarks, mp_holistic, prob_viz
import tensorflow as tf
import numpy as np
from numpy import ndarray
from NNDataManager.nn_data_manager import HAND_CAT_MAPPING_INV, FACE_CAT_MAPPING_INV, FACE_THREAT_MAPPING, \
    HAND_THREAT_MAPPING
from DataCollection.data_collection import SEQ_LEN, MediapipeAns
from typing import Dict, Union
from threading import Thread, Condition, Lock
from queue import Queue
import copy
import psutil
import os

"""
"""
NN_HAND_RESULT_FRIENDLY_STR = ""
NN_FACE_RESULT_FRIENDLY_STR = ""

SHOW_LATENCY = True
MP_LATENCY = 0
NN_LATENCY = 0

SHOW_MP_RESULT = False
MP_RESULT = None

"""
"""
NN_PREDICT_MAX_ALLOWED_LATENCY = 0.0  # second
GIVE_UP_FRAME_ALLOWED_MAX_LATENCY = 0.05  # second
USE_CPU = True

if USE_CPU:
    tf.config.experimental.set_visible_devices([], 'GPU')  # Use CPU for test, in case the GPU is busy


def get_trained_full_model():
    return load_full_model()


class NNInputQueue:
    def __init__(self):
        self.queue = {key: [] for key in MediapipeAns.input_feature_name}
        self.use_ndarray = False
        self.lock = Lock()  # Create a lock object

    def get(self) -> Union[None, Dict[str, ndarray]]:
        with self.lock:  # Acquire the lock
            # No enough sample
            if not self.use_ndarray:
                return None

            # Enough sample
            return copy.deepcopy(self.queue)  # Return a copy to avoid external modification

    def enqueue(self, detect_result: Union[None, ndarray]):
        with self.lock:  # Acquire the lock
            detect_result_copy = copy.deepcopy(detect_result)

        detect_dict = MediapipeAns.detect_result_to_ndarray(detect_result_copy)

        with self.lock:  # Acquire the lock
            for key in self.queue:
                if not self.use_ndarray:
                    self.queue[key].append(detect_dict[key].flatten())
                else:
                    # Left shift
                    self.queue[key] = np.roll(self.queue[key], -1, axis=1)
                    self.queue[key][0, -1, :] = detect_dict[key].flatten()

            if len(self.queue[MediapipeAns.input_feature_name[0]]) == SEQ_LEN:
                if not self.use_ndarray:
                    for key in self.queue:
                        self.queue[key] = (np.array(self.queue[key]))
                        self.queue[key] = np.expand_dims(self.queue[key], axis=0)  # Expand batch dim
                    self.use_ndarray = True


def webcam_thread_func(frame_queue: Queue, frame_cv: Condition):
    PROCESSING_THRESHOLD = 0.1  # 100 milliseconds

    # Set thread priority
    p = psutil.Process(os.getpid())
    p.nice(psutil.NORMAL_PRIORITY_CLASS)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Adjust here to let the system dealing with different hardware
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Adjust width to 640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Adjust height to 480

    # Set frame rate
    cap.set(cv2.CAP_PROP_FPS, 30)  # Adjust frame rate

    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    # Initialize time and frame count
    prev_time = 0
    frame_count = 0
    fps = 0
    nn_latency = 0
    mp_latency = 0

    # Initialize prev_frame_time
    prev_frame_time = time.time()  # This line is added to initialize the variable
    while cap.isOpened():
        # Read feed
        success, frame = cap.read()
        current_frame_time = time.time()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Skip frame if processing is falling behind       The Frame Skipping Logic
        # if current_frame_time - prev_frame_time < PROCESSING_THRESHOLD:
        #     continue
        #
        # prev_frame_time = current_frame_time

        with frame_cv:
            frame_queue.put((time.time(), copy.deepcopy(frame)))
            frame_cv.notify()

        with frame_cv:
            # ! We have to do copy for the enqueued frame, as it will be changed in this thread later.
            frame_queue.put((time.time(), copy.deepcopy(frame)))
            frame_cv.notify()  # Notify that new data is available

        # Get the dimensions of the frame
        height, width, _ = frame.shape

        # Calculate FPS
        cur_time = time.time()
        frame_count += 1
        if cur_time - prev_time >= 1:  # Every second
            fps = frame_count / (cur_time - prev_time)
            frame_count = 0
            prev_time = cur_time
            if SHOW_LATENCY:
                mp_latency = MP_LATENCY
                nn_latency = NN_LATENCY

        # Show the latency if needed
        if SHOW_LATENCY:
            cv2.putText(frame, f"MP: {mp_latency:.1f} ms", (width - 128, height - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"NN: {nn_latency:.1f} ms", (width - 128, height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show MP result is needed
        if SHOW_MP_RESULT and MP_RESULT is not None:
            draw_styled_landmarks(frame, MP_RESULT)

        # Draw FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.1f}", (width - 100, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # noinspection DuplicatedCode
        # Simply calculation of threat
        threat = HAND_THREAT_MAPPING.get(NN_HAND_RESULT_FRIENDLY_STR, 0) + \
                 FACE_THREAT_MAPPING.get(NN_FACE_RESULT_FRIENDLY_STR, 0)
        cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, ', '.join(
            [NN_HAND_RESULT_FRIENDLY_STR,
             NN_FACE_RESULT_FRIENDLY_STR,
             f"threat={threat}"]
        ),
                    (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('WebCam', frame)

        # Break gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def mediapipe_thread_func(frame_queue: Queue, frame_cv: Condition, nn_queue: NNInputQueue):
    global MP_LATENCY
    global MP_RESULT

    # Set thread priority
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              model_complexity=1,
                              refine_face_landmarks=True,
                              min_tracking_confidence=0.5) as holistic:
        while True:
            with frame_cv:
                # ! Avoid spurious wake-ups
                # 等效于C++的 cv.wait(lock, [] { return !data_queue.empty(); });
                while frame_queue.empty():
                    frame_cv.wait()

            frame = frame_queue.get()

            # !? 强制同步, basically give up this frame
            t1 = time.time()
            if t1 - frame[0] > GIVE_UP_FRAME_ALLOWED_MAX_LATENCY:
                continue

            _, results = mediapipe_detection(frame[1], holistic)
            if SHOW_LATENCY:
                MP_LATENCY = (time.time() - t1) * 1000

            nn_queue.enqueue(results)

            # No concurrency protection here but it's OK
            if SHOW_MP_RESULT:
                MP_RESULT = results


# def nn_predication(model: tf.keras.Model, nn_input: Dict[str, ndarray]) -> Dict[str, str]:
#     nn_result = model.predict(nn_input, verbose=0)
#
#     return {
#         "hand": HAND_CAT_MAPPING_INV[np.argmax(nn_result["hand"])],
#         "face": FACE_CAT_MAPPING_INV[np.argmax(nn_result["face"])]
#     }

def nn_predication(model: tf.keras.Model, nn_input: Dict[str, ndarray]) -> Dict[str, Union[str, float]]:
    nn_result = model.predict(nn_input, verbose=0)

    hand_prob = np.max(nn_result["hand"])
    face_prob = np.max(nn_result["face"])

    return {
        "hand_class": HAND_CAT_MAPPING_INV[np.argmax(nn_result["hand"])],
        "face_class": FACE_CAT_MAPPING_INV[np.argmax(nn_result["face"])],
        "hand_prob": hand_prob,
        "face_prob": face_prob
    }


def nn_predication_thread_func(nn_queue: NNInputQueue):
    global NN_HAND_RESULT_FRIENDLY_STR
    global NN_FACE_RESULT_FRIENDLY_STR
    global NN_LATENCY

    # Define confidence threshold
    CONFIDENCE_THRESHOLD = 0.7

    # Set thread priority
    p = psutil.Process(os.getpid())
    p.nice(psutil.REALTIME_PRIORITY_CLASS)

    model = get_trained_full_model()
    # Warm up, because the first call of NN prediction is slow, this is a dummy call basically
    for i in range(31):
        nn_queue.enqueue(None)
    nn_predication(model, nn_queue.get())

    pre_nn_predict_ts = time.time()

    while True:
        # We do not need to wait here. There are always samples since it is a circular buffer.
        nn_input = nn_queue.get()

        # !? 强制同步, basically give up this frame
        t1 = time.time()
        if t1 - pre_nn_predict_ts > NN_PREDICT_MAX_ALLOWED_LATENCY:
            if nn_input is not None:
                nn_result = nn_predication(model, nn_input)
                NN_HAND_RESULT_FRIENDLY_STR = nn_result["hand_class"] if nn_result[
                                                                             "hand_prob"] >= CONFIDENCE_THRESHOLD else "Uncertain"
                NN_FACE_RESULT_FRIENDLY_STR = nn_result["face_class"] if nn_result[
                                                                             "face_prob"] >= CONFIDENCE_THRESHOLD else "Uncertain"

            pre_nn_predict_ts = time.time()

            if SHOW_LATENCY:
                NN_LATENCY = (pre_nn_predict_ts - t1) * 1000


def main():
    print("Main function started")
    frame_queue = Queue()
    frame_cv = Condition()

    nn_queue = NNInputQueue()

    # Create threads
    nn_predication_thread = Thread(target=nn_predication_thread_func, args=(nn_queue,))
    nn_predication_thread.start()
    time.sleep(10)

    mediapipe_thread = Thread(target=mediapipe_thread_func, args=(frame_queue, frame_cv, nn_queue))
    mediapipe_thread.start()

    time.sleep(2)
    webcam_thread = Thread(target=webcam_thread_func, args=(frame_queue, frame_cv))
    webcam_thread.start()

    # Wait for both threads to finish
    nn_predication_thread.join()
    mediapipe_thread.join()
    webcam_thread.join()


if __name__ == "__main__":
    main()
