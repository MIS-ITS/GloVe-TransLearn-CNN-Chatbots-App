# import tflearn
import pickle
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import metrics

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

ftr = np.load("../output/data-target-schematics/ftr-data-target.npy")
cls = np.load("../output/data-target-schematics/cls-data-target.npy")

test_size = 0.85

# X, X_test, Y, Y_test = train_test_split(ftr, cls, test_size=0.30)
# X = np.expand_dims(X, axis=2)
# X_test = np.expand_dims(X_test, axis=2)
sss = StratifiedShuffleSplit(n_splits=3, test_size=test_size, random_state=0)
for train_index, test_index in sss.split(ftr, cls):
    X, X_test = ftr[train_index], ftr[test_index]
    Y, Y_test = cls[train_index], cls[test_index]
    break
X = np.expand_dims(X, axis=2)
X_test = np.expand_dims(X_test, axis=2)

np.save("../output/test-case-" + str(test_size) + "/X.npy", X)
np.save("../output/test-case-" + str(test_size) + "/X_test.npy", X_test)
np.save("../output/test-case-" + str(test_size) + "/Y.npy", Y)
np.save("../output/test-case-" + str(test_size) + "/Y_test.npy", Y_test)

checkpoint_path = '../keras-model/checkpoints/intervene.data-target-'+str(test_size)+'.h5'
early_stopper = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)

model = Sequential()

model.add(Conv1D(64, kernel_size=3, activation='relu',input_shape=(50, 1)))
model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(180, activation='sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(len(Y[0]), activation='softmax'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy'])

# Fit the model
model.fit(X, Y,
          batch_size=32,
          shuffle=True,
          epochs=2000,
          validation_data=(X_test, Y_test),
          callbacks=[checkpointer, early_stopper])

# last model
model.save('../keras-model/cnn-data-target-' + str(test_size) + '.h5')
# load best model
model = load_model(checkpoint_path)

predictions = model.predict(X_test)
predictions = [np.argmax(predictions[i]) for i in range(len(predictions))]
predictions = np.array(predictions)
labels = [np.argmax(Y_test[i]) for i in range(len(Y_test))]
labels = np.array(labels)

print predictions
print labels

print "Accuracy: " + str(100*metrics.accuracy_score(labels, predictions))
print "Precision: " + str(100*metrics.precision_score(labels, predictions, average="weighted"))
print "Recall: " + str(100*metrics.recall_score(labels, predictions, average="weighted"))
print "f1_score: " + str(100*metrics.f1_score(labels, predictions, average="weighted"))
