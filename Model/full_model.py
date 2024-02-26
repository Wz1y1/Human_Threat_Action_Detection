import tensorflow as tf
from DataCollection.data_collection import MediapipeAns, SEQ_LEN
from NNDataManager.nn_data_manager import HAND_CAT_MAPPING, FACE_CAT_MAPPING
from project_info import PROJECT_PATH


# noinspection DuplicatedCode
def make_hand_model() -> tf.keras.Model:
    # %% Define hand model
    pose_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["pose"][0] * MediapipeAns.shape["pose"][1]),
                                       name="pose")
    pose_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(pose_input)
    pose_lstm = tf.keras.layers.Dropout(0.1)(pose_lstm)
    pose_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(pose_lstm)
    pose_lstm = tf.keras.layers.Dropout(0.1)(pose_lstm)

    left_hand_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["lh"][0] * MediapipeAns.shape["lh"][1]),
                                            name="lh")
    left_hand_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(left_hand_input)
    left_hand_lstm = tf.keras.layers.Dropout(0.1)(left_hand_lstm)
    left_hand_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(left_hand_lstm)
    left_hand_lstm = tf.keras.layers.Dropout(0.1)(left_hand_lstm)

    right_hand_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["rh"][0] * MediapipeAns.shape["rh"][1]),
                                             name="rh")
    right_hand_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(right_hand_input)
    right_hand_lstm = tf.keras.layers.Dropout(0.1)(right_hand_lstm)
    right_hand_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(right_hand_lstm)
    right_hand_lstm = tf.keras.layers.Dropout(0.1)(right_hand_lstm)

    # Concatenate the outputs of the sub-models
    hand_embedded = tf.keras.layers.Concatenate(axis=2)([pose_lstm, left_hand_lstm, right_hand_lstm])
    hand_embedded = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(hand_embedded)
    hand_embedded = tf.keras.layers.Dropout(0.2)(hand_embedded)
    hand_embedded = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(hand_embedded)

    hand_output = tf.keras.layers.Dense(len(HAND_CAT_MAPPING), activation='softmax',
                                        name="hand_output")(hand_embedded)
    hand_model = tf.keras.Model(inputs=[pose_input, left_hand_input, right_hand_input],
                                outputs=hand_output,
                                name="hand_model")
    hand_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['AUC'])
    hand_model.summary()

    return hand_model


# 01 LSTM
# def make_face_model() -> tf.keras.Model:
#     # %% Define face model
#     face_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["face"][0] * MediapipeAns.shape["face"][1]),
#                                        name="face")
#
#     #     # Adding more convolutional layers
#     # face_cnn = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="same")(face_input)
#     # face_cnn = tf.keras.layers.BatchNormalization()(face_cnn)
#     # face_cnn = tf.keras.layers.MaxPool1D()(face_cnn)
#
#     face_cnn = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same")(face_input)
#     face_cnn = tf.keras.layers.PReLU()(face_cnn)
#
#     face_cnn = tf.keras.layers.BatchNormalization()(face_cnn)
#     face_cnn = tf.keras.layers.MaxPool1D()(face_cnn)
#
#     face_lstm = tf.keras.layers.LSTM(64, return_sequences=True)(face_cnn)
#     face_lstm = tf.keras.layers.Dropout(0.4)(face_lstm)
#     face_lstm = tf.keras.layers.LSTM(48, return_sequences=True)(face_lstm)
#     face_lstm = tf.keras.layers.Dropout(0.3)(face_lstm)
#     face_lstm = tf.keras.layers.LSTM(32)(face_lstm)
#
#     face_output = tf.keras.layers.Dense(len(FACE_CAT_MAPPING), activation='softmax',
#                                         name="face_output")(face_lstm)
#
#     face_model = tf.keras.Model(inputs=face_input,
#                                 outputs=face_output,
#                                 name="face_model")
#     face_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['AUC', 'accuracy'])
#     face_model.summary()
#
#     return face_model


# #02 Bi-LSTM
# def make_face_model() -> tf.keras.Model:
#     # %% Define face model
#     face_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["face"][0] * MediapipeAns.shape["face"][1]),
#                                        name="face")
#
#     face_cnn = tf.keras.layers.Conv1D(32,
#                                       kernel_size=6,
#                                       activation='relu',
#                                       padding="valid")(face_input)
#     face_cnn = tf.keras.layers.BatchNormalization()(face_cnn)
#     face_cnn = tf.keras.layers.MaxPool1D()(face_cnn)
#
#     face_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(face_cnn)
#     face_lstm = tf.keras.layers.Dropout(0.4)(face_lstm)
#     face_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48, return_sequences=True))(face_lstm)
#     face_lstm = tf.keras.layers.Dropout(0.3)(face_lstm)
#     face_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(face_lstm)
#
#     face_output = tf.keras.layers.Dense(len(FACE_CAT_MAPPING), activation='softmax',
#                                         name="face_output")(face_lstm)
#
#     face_model = tf.keras.Model(inputs=face_input,
#                                 outputs=face_output,
#                                 name="face_model")
#     face_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['AUC', 'accuracy'])
#     face_model.summary()
#
#     return face_model


# #03 GRU
def make_face_model() -> tf.keras.Model:
    face_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["face"][0] * MediapipeAns.shape["face"][1]), name="face")

    # Adding more convolutional layers
    face_cnn = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="same")(face_input)
    face_cnn = tf.keras.layers.BatchNormalization()(face_cnn)
    face_cnn = tf.keras.layers.MaxPool1D()(face_cnn)
    # face_cnn = tf.keras.layers.AveragePooling1D()(face_cnn)

    face_cnn = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding="same")(face_cnn)
    face_cnn = tf.keras.layers.BatchNormalization()(face_cnn)
    face_cnn = tf.keras.layers.MaxPool1D()(face_cnn)
    # face_cnn = tf.keras.layers.AveragePooling1D()(face_cnn)

    # Using GRU instead of LSTM
    face_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(face_cnn)
    face_rnn = tf.keras.layers.Dropout(0.4)(face_rnn)
    face_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(48))(face_rnn)

    # Attention Layer can be added here if needed

    face_output = tf.keras.layers.Dense(len(FACE_CAT_MAPPING), activation='softmax', name="face_output")(face_rnn)

    face_model = tf.keras.Model(inputs=face_input, outputs=face_output, name="improved_face_model")
    face_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['AUC', 'accuracy'])
    face_model.summary()

    return face_model

# PReLU
# def make_face_model() -> tf.keras.Model:
#     face_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["face"][0] * MediapipeAns.shape["face"][1]), name="face")
#
#     # Adding more convolutional layers with PReLU activation
#     face_cnn = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same")(face_input)
#     face_cnn = tf.keras.layers.PReLU()(face_cnn)
#     face_cnn = tf.keras.layers.BatchNormalization()(face_cnn)
#     face_cnn = tf.keras.layers.MaxPool1D()(face_cnn)
#
#     face_cnn = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same")(face_cnn)
#     face_cnn = tf.keras.layers.PReLU()(face_cnn)
#     face_cnn = tf.keras.layers.BatchNormalization()(face_cnn)
#     face_cnn = tf.keras.layers.MaxPool1D()(face_cnn)
#
#     # Using GRU instead of LSTM
#     face_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(face_cnn)
#     face_rnn = tf.keras.layers.Dropout(0.4)(face_rnn)
#     face_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(48))(face_rnn)
#
#     # Attention Layer can be added here if needed
#
#     face_output = tf.keras.layers.Dense(len(FACE_CAT_MAPPING), activation='softmax', name="face_output")(face_rnn)
#
#     face_model = tf.keras.Model(inputs=face_input, outputs=face_output, name="improved_face_model")
#     face_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['AUC', 'accuracy'])
#     face_model.summary()
#
#     return face_model



def load_full_model():
    save_folder = PROJECT_PATH / "Model/Saved/Final"

    hand_model = make_hand_model()
    hand_model.load_weights(save_folder / f"hand_model_final.h5")

    face_model = make_face_model()
    face_model.load_weights(save_folder / f"face_model_final.h5")

    # %% Define the full model
    pose_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["pose"][0] * MediapipeAns.shape["pose"][1]),
                                       name="pose")
    left_hand_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["lh"][0] * MediapipeAns.shape["lh"][1]),
                                            name="lh")
    right_hand_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["rh"][0] * MediapipeAns.shape["rh"][1]),
                                             name="rh")
    face_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["face"][0] * MediapipeAns.shape["face"][1]),
                                       name="face")

    model = tf.keras.Model(inputs=[pose_input, left_hand_input, right_hand_input, face_input],
                           outputs={
                               "hand": hand_model([pose_input, left_hand_input, right_hand_input]),
                               "face": face_model(face_input)
                           })

    return model
