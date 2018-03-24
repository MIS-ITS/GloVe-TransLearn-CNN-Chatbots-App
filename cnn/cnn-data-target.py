# import tflearn
import pickle
import numpy as np
import tensorflow as tf
from tflearn.data_utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras.models import load_model
from keras.initializers import glorot_normal
from keras.callbacks import ModelCheckpoint, EarlyStopping

# ftr = np.load("ftr-data-target.npy")
# cls = np.load("cls-data-target.npy")

# shuffle(ftr, cls)
# X, X_test, Y, Y_test = train_test_split(ftr, cls, test_size=0.5, shuffle=True)
# X = np.expand_dims(X, axis=2)
# X_test = np.expand_dims(X_test, axis=2)

test_size = 0.10
model_load_path = '../keras-model/checkpoints/intervene.data-source-0.1.h5'
output_load = "../output/test-case-" + str(test_size)

X = np.load(output_load + "/X.npy")
X_test = np.load(output_load + "/X_test.npy")
Y = np.load(output_load + "/Y.npy")
Y_test = np.load(output_load + "/Y_test.npy")

checkpoint_path = '../keras-model/checkpoints/intervene.data-target-tf-'+str(test_size)+'.h5'
early_stopper = EarlyStopping(monitor='loss', patience=40, verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)

trained_model = load_model(model_load_path)
print trained_model.summary()

model = Sequential()
for layer in trained_model.layers[:1]:
    model.add(layer)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180, activation='sigmoid', name='hidden_dense'))
model.add(Dropout(0.2, name='last_dropout'))
model.add(Dense(22, activation='softmax', name="output_dense"))

# for layer in model.layers[:1]:
#     layer.trainable = False
# model.layers[5].trainable = False

print model.summary()

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

print model.summary()
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
