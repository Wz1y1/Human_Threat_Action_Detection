from NNDataManager.nn_data_manager import load_nn_train_val
from Model.full_model import make_hand_model, make_face_model
import tensorflow as tf
from project_info import PROJECT_PATH
from PathUtils.path_and_file import try_to_find_folder_path_otherwise_make_one
from Trainer.callback import SaveCallback
from tensorflow import keras
import psutil
import matplotlib.pyplot as plt


class CPUMonitorCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.cpu_usage = []
        self.losses = []  # List to store loss values
        self.accuracies = [] # List to store accuracy values
        self.val_accuracies = []  # List to store validation accuracy values
        self.auc = []  # For AUC
        self.val_auc = []  # For Validation AUC


    def on_epoch_end(self, epoch, logs=None):
        # CPU Usage
        # cpu_usage = psutil.cpu_percent()
        # self.cpu_usage.append(cpu_usage)
        # print(f'CPU Usage after epoch {epoch + 1}: {cpu_usage}%')

        # Recording loss value
        if logs is not None:
            loss = logs.get('loss')
            self.accuracies.append(logs.get('accuracy'))  # Assuming 'accuracy' is the metric name
            self.val_accuracies.append(logs.get('val_accuracy'))  # Validation accuracy
            self.auc.append(logs.get('auc'))  # AUC
            self.val_auc.append(logs.get('val_auc'))  # Validation AUC

            self.losses.append(loss)  # Store the loss
            # print(f'Epoch {epoch + 1}: Loss = {logs.get("loss")}, Accuracy = {logs.get("accuracy")}, Val Accuracy = {logs.get("val_accuracy")}')


def train(epochs: int = 200, batch_size: int = 4, *, train_hand: bool = False, train_face: bool = False):
    cpu_monitor = CPUMonitorCallback()

    save_folder = PROJECT_PATH / "Model/Saved/Final"
    try_to_find_folder_path_otherwise_make_one(save_folder)

    hand_data = load_nn_train_val("hand")
    face_data = load_nn_train_val("face")

    stages = {"hand": 0, "face": 1}

    for i, (make_model, data, which_model) in enumerate(zip([make_hand_model, make_face_model],
                                                            [hand_data, face_data],
                                                            ["hand_model", "face_model"])):
        if not train_hand and i == stages["hand"]:
            continue

        if not train_face and i == stages["face"]:
            continue

        model = make_model()
        save_callback = SaveCallback(total_epoch=epochs,
                                     save_folder_path=save_folder,
                                     which_model=which_model)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=80)
        model.fit(
            x=data["train"][0],
            y=data["train"][1],
            validation_data=data["val"],
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[save_callback, cpu_monitor]
        )

    # Test the CPU usage
    # cpu_usage = cpu_monitor.cpu_usage
    # plt.plot(cpu_usage, color='green')
    # plt.title('RMSprop CPU Usage Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('CPU Usage (%)')
    # plt.show()
    # return cpu_usage

    # plt.plot(cpu_monitor.losses, color='green')
    # plt.title('RMSprop Training Loss Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()
    # return cpu_monitor.losses

# Plotting training and validation accuracy
    plt.plot(cpu_monitor.auc, color='blue', label='Training Accuracy')
    plt.plot(cpu_monitor.accuracies, color='orange', label='Validation Accuracy')
    plt.title('Split Bi-LSTM')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return cpu_monitor.auc, cpu_monitor.accuracies

    # plt.plot(cpu_monitor.losses, color='green')
    # plt.title('ReLU Training Loss Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()
    # return cpu_monitor.losses

if __name__ == "__main__":
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    train(train_face=False, train_hand=True)
