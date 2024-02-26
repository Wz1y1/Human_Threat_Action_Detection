import tensorflow as tf
import json
from pathlib import Path
from PathUtils.load_save import try_to_find_folder_path_otherwise_make_one, try_to_find_file
from PathUtils.path_and_file import try_to_find_file_if_exist_then_delete
import os
from typing import Tuple, Callable, List, Any, Dict, Union
import re


class Heap:
    """
    Priority queue.
    """

    def __init__(self, comp: Callable[[Any, Any], bool]):

        self.heap: List[Any] = []
        self.comp: Callable[[Any, Any], bool] = comp

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def __len__(self):
        return self.heap.__len__()

    def shift_up(self, idx):
        idx_parent = (idx - 1) // 2
        while idx > 0 and self.comp(self.heap[idx], self.heap[idx_parent]):
            self.swap(idx, idx_parent)
            idx = idx_parent
            idx_parent = (idx - 1) // 2

    def insert(self, value):
        self.heap.append(value)
        self.shift_up(self.__len__() - 1)

    def shift_down(self, idx):
        idx_c1 = 2 * idx + 1
        while idx_c1 < self.__len__():
            idx_c2 = 2 * idx + 2
            idx_swap = idx_c1
            if idx_c2 < self.__len__() and self.comp(self.heap[idx_c2], self.heap[idx_c1]):
                idx_swap = idx_c2

            if self.comp(self.heap[idx_swap], self.heap[idx]):
                self.swap(idx_swap, idx)
                idx = idx_swap
                idx_c1 = 2 * idx + 1
            else:
                break

    def remove(self):
        self.swap(0, self.__len__() - 1)
        back = self.heap[-1]
        self.heap.pop()
        self.shift_down(0)

        return back

    def peek(self):
        return self.heap[0]


def heap_track_func(_epoch: int, logs, total_epoch: int, heap: Heap) -> Tuple[bool, int, int]:
    """
    :return: Tuple[Whether to execute the callback, The removed epoch number, The inserted epoch number]
    """
    if _epoch is None:
        return True, -1, -1

    auc: float = logs.get('val_auc')
    # There are no enough elements
    if len(heap) < max(int(total_epoch * 0.001), 10):
        heap.insert([_epoch, auc])
        return True, -1, _epoch

    # auc越大越好
    if heap.peek()[1] > auc:
        if _epoch != 0 and (_epoch % (max(int(total_epoch * 0.1), 1)) == 0):
            return True, -1, -1
        else:
            return False, -1, -1
    else:
        removed_epoch = heap.remove()
        heap.insert([_epoch, auc])
        return True, removed_epoch[0], _epoch


def find_new_working_folder(parent_folder):
    # Find proper folder
    tmp = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]

    used_folder = set()
    for ele in tmp:
        if try_to_find_file(parent_folder / ele / "hand_model_final.h5") and try_to_find_file(
                parent_folder / ele / "face_model_final.h5"):
            used_folder.add(ele)
            continue

        re_find = re.findall(r".*_(\d+)", ele)
        if len(re_find) > 0:
            return ele

    i = 1
    while f"Run_{i}" in tmp or f"Run_{i}" in used_folder:
        i += 1

    return f"Run_{i}"


class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epoch, save_folder_path: Path, std_out: bool = True,
                 *, which_model: str):
        """
        Constructor.
        :param total_epoch:
        :param save_folder_path: The folder path to record the callback execution.
        :param std_out: Whether to print to standard output.
        :param which_model: Shall either be "face_model" or "hand_model"
        """
        super().__init__()
        self.total_epoch = total_epoch
        self.heap = Heap(lambda a, b: a[1] < b[1])  # Min-heap，把最小的元素放到最前面

        self.save_folder_path = save_folder_path  # type: Path
        self.sub_f = find_new_working_folder(self.save_folder_path)
        try_to_find_folder_path_otherwise_make_one(self.save_folder_path / self.sub_f)
        self.history = dict()  # type: Dict[int, Dict[str, float]]
        self.std_out = std_out  # type: bool

        assert which_model in {"face_model", "hand_model"}
        self.which_model: str = which_model

    def call(self, epoch: Union[None, int] = None, logs=None, on_train_end: bool = False,
             *, model_save_name: str = "") -> None:
        """
        Wrapper for on_epoch_end and on_train_end
        """
        if (epoch is None) ^ on_train_end:
            raise "not ((epoch is None) ^ on_train_end)"

        if model_save_name == "":
            model_save_name = f'{self.which_model}_epoch_{epoch}.h5' \
                if not on_train_end \
                else f"{self.which_model}_final.h5"

        self.model.save_weights(self.save_folder_path / self.sub_f / model_save_name)

        record = {
            'loss': logs.get('loss'),
            'auc': logs.get('auc'),
            'val_loss': logs.get('val_loss'),
            'val_auc': logs.get('val_auc'),
        }
        self.history[epoch if epoch is not None else "final"] = record

        with open(self.save_folder_path / self.sub_f / f"{self.which_model}_history.json", 'w') as _fp:
            json.dump(self.history, _fp)

        if self.std_out:
            print(record)

    def on_epoch_end(self, epoch, logs=None):
        exe_ans = heap_track_func(epoch, logs, self.total_epoch, self.heap)
        if not exe_ans[0]:
            return

        if exe_ans[1] != -1:
            print(f"\nremove.... {self.which_model}_top_n_epoch_{exe_ans[1]}.h")
            try_to_find_file_if_exist_then_delete(
                self.save_folder_path / self.sub_f / f"{self.which_model}_top_n_epoch_{exe_ans[1]}.h5"
            )

        model_save_name = ""
        if exe_ans[2] != -1:
            model_save_name = f"{self.which_model}_top_n_epoch_{exe_ans[2]}.h5"

        self.call(epoch, logs, model_save_name=model_save_name)

    def on_train_end(self, logs=None):
        self.call(logs=logs, on_train_end=True)
