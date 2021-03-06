# Filename: CNNLSTM_2018.py
# Description: This file implements a CNN-LSTM model for RADIOML 2018.10A dataset.

import tensorflow as tf
print(tf.keras.__version__)

from keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from keras import Input, Model
from keras.layers import Dropout, Dense, BatchNormalization, LSTM, Conv1D, MaxPool1D, Flatten

import h5py
import os, random
import numpy as np
from tensorflow.keras.layers import Input, Reshape, ZeroPadding2D, Conv2D, Dropout, Flatten, Dense, Activation, \
    MaxPool2D, AlphaDropout
#import tensorflow.keras.models as Model
import matplotlib.pyplot as plt


### Data Preprocessing ###

data_path = 'data'
for i in range(0, 24):
    # Load the data
    filename = os.path.join(data_path,'ExtractDataset','part') + str(i) + '.h5'
    print('Filename:',filename)
    f = h5py.File(filename, 'r')

    X_data = f['X'][:]
    Y_data = f['Y'][:]
    Z_data = f['Z'][:]
    f.close()

    # Read the data
    n_examples = X_data.shape[0]
    n_train = int(n_examples * 0.7)  # 70 percent of training data
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    if i == 0:
        X_train = X_data[train_idx]
        Y_train = Y_data[train_idx]
        Z_train = Z_data[train_idx]
        X_test = X_data[test_idx]
        Y_test = Y_data[test_idx]
        Z_test = Z_data[test_idx]
    else:
        X_train = np.vstack((X_train, X_data[train_idx]))
        Y_train = np.vstack((Y_train, Y_data[train_idx]))
        Z_train = np.vstack((Z_train, Z_data[train_idx]))
        X_test = np.vstack((X_test, X_data[test_idx]))
        Y_test = np.vstack((Y_test, Y_data[test_idx]))
        Z_test = np.vstack((Z_test, Z_data[test_idx]))

print('Training set X dimention:', X_train.shape)
print('Training set Y dimention:', Y_train.shape)
print('Training set Z dimention:', Z_train.shape)
print('Test set X dimention:', X_test.shape)
print('Test set Y dimention:', Y_test.shape)
print('Test set Z dimention:', Z_test.shape)


os.environ["KERAS_BACKEND"] = "tensorflow"
print(tf.test.gpu_device_name())
classes = ['32PSK',
           '16APSK',
           '32QAM',
           'FM',
           'GMSK',
           '32APSK',
           'OQPSK',
           '8ASK',
           'BPSK',
           '8PSK',
           'AM-SSB-SC',
           '4ASK',
           '16PSK',
           '64APSK',
           '128QAM',
           '128APSK',
           'AM-DSB-SC',
           'AM-SSB-WC',
           '64QAM',
           'QPSK',
           '256QAM',
           'AM-DSB-WC',
           'OOK',
           '16QAM']

# X_test = X_test.reshape(-1, 2, 1024, 1)
# X_train = X_train.reshape(-1, 2, 1024, 1)
data_format = 'channels_last'


### Build a CNN-LSTM model ###

in_shp = X_train.shape[1:]  # [1024,2]
print('Input Shape:',in_shp)

Xm_input = Input(in_shp, name='input')
# input layer
def CNN_LSTM():
    inputs = Input((1024, 2,))
    l = BatchNormalization()(inputs)
    l = Conv1D(filters=128, kernel_size=5, activation='relu')(l)
    l = MaxPool1D(3)(l)
    l = Conv1D(filters=128, kernel_size=5, activation='relu')(l)
    l = LSTM(128, return_sequences=True, activation='tanh', unroll=True)(l)
    l = LSTM(128, return_sequences=True, activation='tanh', unroll=True)(l)
    l = Dropout(0.8)(l)
    l = Flatten()(l)
    outputs = Dense(24, activation='softmax', kernel_constraint=max_norm(2.))(l)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.summary()
    return model


### Training ###

model = CNN_LSTM()

filepath = 'models/cnn_lstm_2018.h5'
history = model.fit(X_train,
                    Y_train,
                    # batch_size=1000,
                    batch_size=1000,  # already changed to 10, original one is 1000
                    epochs=100,  # changed to 10, original one is 100
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    # validation_split = 0.3,
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                                           mode='auto'),
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
                    ])

'''
### Loss ###

print('train finish')
val_loss_list = history.history['val_loss']
loss_list = history.history['loss']
plt.plot(range(len(loss_list)), val_loss_list)
plt.plot(range(len(loss_list)), loss_list)
plt.show()
'''

model.load_weights(filepath)

'''
### Confusion Matrix ###

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Plot confusion matrix
batch_size = 1024
test_Y_hat = model.predict(X_test, batch_size=1024)
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] = conf[j, k] + 1
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
plot_confusion_matrix(confnorm, labels=classes)
'''


### Accuracy for each SNR ###

'''
for i in range(len(confnorm)):
    print(classes[i], confnorm[i, i])
'''

acc = {}
Z_test = Z_test.reshape((len(Z_test)))
SNRs = np.unique(Z_test)
for snr in SNRs:
    X_test_snr = X_test[Z_test == snr]
    Y_test_snr = Y_test[Z_test == snr]

    pre_Y_test = model.predict(X_test_snr)
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, X_test_snr.shape[0]):
        j = list(Y_test_snr[i, :]).index(1)
        j = classes.index(classes[j])
        k = int(np.argmax(pre_Y_test[i, :]))
        k = classes.index(classes[k])
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

    '''
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)" % (snr))
    '''

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy %s: " % snr, cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)


### Save the results ###

snrs = list(acc.keys())
accs = list(acc.values())
results = np.concatenate((snrs,accs),axis=1)
np.savez('CNNLSTM_2018_', results)

'''
plt.plot(list(acc.keys()), list(acc.values()))
plt.ylabel('ACC')
plt.xlabel('SNR')
plt.show()
'''
