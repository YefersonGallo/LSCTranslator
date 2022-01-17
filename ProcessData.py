from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

import numpy as np
import os

actions = np.array(['a', 'b', 'c', 'd', 'h', 'j', 'l', 's', 'z'])

DATA_PATH = os.path.join('MP_Data')

no_sequences = 300

sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)

y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.08)

log_dir = os.path.join('Logs_Multiple_dataset_5')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
# model.add(Dropout(0.2))
# model.add(LSTM(256, return_sequences=True, activation='relu'))
# model.add(Dropout(0.2))
# model.add(LSTM(256, return_sequences=False, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

model.summary()

res = model.predict(X_test)

print(actions[np.argmax(res[3])])

print(actions[np.argmax(y_test[3])])

model.save('action_new_dataset_5.h5')

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(accuracy_score(ytrue, yhat))