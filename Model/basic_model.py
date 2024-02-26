import tensorflow as tf
from DataCollection.data_collection import MediapipeAns, SEQ_LEN
from NNDataManager.nn_data_manager import HAND_CAT_MAPPING


# noinspection DuplicatedCode
def make_basic_model():
    # Define LSTM sub-models for each type of landmark
    pose_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["pose"][0] * MediapipeAns.shape["pose"][1]),
                                       name="pose")
    pose_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(pose_input)
    pose_lstm = tf.keras.layers.Dropout(0.1)(pose_lstm)
    pose_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(pose_lstm)

    face_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["face"][0] * MediapipeAns.shape["face"][1]),
                                       name="face")
    face_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(face_input)
    face_lstm = tf.keras.layers.Dropout(0.2)(face_lstm)
    face_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(face_lstm)

    left_hand_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["lh"][0] * MediapipeAns.shape["lh"][1]),
                                            name="lh")
    left_hand_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(left_hand_input)
    left_hand_lstm = tf.keras.layers.Dropout(0.1)(left_hand_lstm)
    left_hand_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(left_hand_lstm)

    right_hand_input = tf.keras.layers.Input(shape=(SEQ_LEN, MediapipeAns.shape["rh"][0] * MediapipeAns.shape["rh"][1]),
                                             name="rh")
    right_hand_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(right_hand_input)
    right_hand_lstm = tf.keras.layers.Dropout(0.1)(right_hand_lstm)
    right_hand_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))(right_hand_lstm)

    # Concatenate the outputs of the sub-models
    concat = tf.keras.layers.Concatenate(axis=2)([pose_lstm, face_lstm, left_hand_lstm, right_hand_lstm])

    # Main LSTM layers
    main_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(concat)
    main_lstm = tf.keras.layers.Dropout(0.2)(main_lstm)
    main_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(main_lstm)

    main_lstm = tf.keras.layers.Dense(32, activation='relu')(main_lstm)

    # Output layer
    output = tf.keras.layers.Dense(len(HAND_CAT_MAPPING), activation='softmax')(main_lstm)

    model = tf.keras.Model(inputs=[pose_input, face_input, left_hand_input, right_hand_input], outputs=output)

    model.summary()

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['AUC'])

    return model

