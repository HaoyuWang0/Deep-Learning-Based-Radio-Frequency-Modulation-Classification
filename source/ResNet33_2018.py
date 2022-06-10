# Filename: CNN_2018.py
# Description: This file implements a CNN model for RADIOML 2018.10A dataset.

import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
import DataGenerator
print(tf.keras.__version__)
import numpy as np
import h5py
import os, random
import numpy as np
from tensorflow.keras.layers import Input, Reshape, ZeroPadding2D, Conv2D, Dropout, Flatten, Dense, Activation, \
    MaxPool2D, AlphaDropout
from tensorflow.keras import layers
import tensorflow.keras.models as Model
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
    n_train = int(n_examples * 0.7)  # 70 percent training data
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

print('Training set X Dimension:', X_train.shape)
print('Training set Y Dimension:', Y_train.shape)
print('Training set Z Dimension:', Z_train.shape)
print('Test set X Dimension:', X_test.shape)
print('Test set Y Dimension:', Y_test.shape)
print('Test set Z Dimension:', Z_test.shape)

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

X_test = X_test.reshape(-1, 2, 1024, 1)
X_train = X_train.reshape(-1, 2, 1024, 1)
data_format = 'channels_last'


### Build a ResNet33 Model ###

def residual_stack(Xm, kennel_size, Seq, pool_size, if_max):
    # 1*1 Conv Linear original filtersize 32
    Xm = Conv2D(32, (1, 1), padding='same', name=Seq + "_conv1", kernel_initializer='glorot_normal',
                data_format=data_format)(Xm)
    # Residual Unit 1
    Xm_shortcut = Xm
    Xm = Conv2D(32, kennel_size, padding='same', activation="relu", name=Seq + "_conv2",
                kernel_initializer='glorot_normal', data_format=data_format)(Xm)
    Xm = Conv2D(32, kennel_size, padding='same', name=Seq + "_conv3", kernel_initializer='glorot_normal',
                data_format=data_format)(Xm)
    Xm = layers.add([Xm, Xm_shortcut])
    Xm = Activation("relu")(Xm)
    # Residual Unit 2
    Xm_shortcut = Xm
    Xm = Conv2D(32, kennel_size, padding='same', activation="relu", name=Seq + "_conv4",
                kernel_initializer='glorot_normal', data_format=data_format)(Xm)
    Xm = Conv2D(32, kennel_size, padding='same', name=Seq + "_conv5", kernel_initializer='glorot_normal',
                data_format=data_format)(Xm)
    Xm = layers.add([Xm, Xm_shortcut])
    Xm = Activation("relu")(Xm)
    # MaxPooling
    if (if_max):
        Xm = MaxPool2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format=data_format)(Xm)
    return Xm


in_shp = X_train.shape[1:]  # [1024,2]
print('Input Shape:',in_shp)
# input layer
Xm_input = Input(in_shp, name='input')
# Xm = Reshape([1,512,4], input_shape=in_shp)(Xm_input)


# Residual Srack

Xm = residual_stack(Xm_input, kennel_size=(3, 2), Seq="ReStk0", pool_size=(2, 2),
                    if_max=False)
X = MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid', data_format=data_format)(Xm)
Xm = residual_stack(Xm, kennel_size=(3, 2), Seq="ReStk1", pool_size=(1, 2), if_max=True)  # shape:(256,1,32)
Xm = residual_stack(Xm, kennel_size=(3, 2), Seq="ReStk2", pool_size=(1, 2), if_max=True)  # shape:(128,1,32)
Xm = residual_stack(Xm, kennel_size=(3, 2), Seq="ReStk3", pool_size=(1, 2), if_max=True)  # shape:(64,1,32)
Xm = residual_stack(Xm, kennel_size=(3, 2), Seq="ReStk4", pool_size=(1, 2), if_max=True)  # shape:(32,1,32)
Xm = residual_stack(Xm, kennel_size=(3, 2), Seq="ReStk5", pool_size=(1, 2), if_max=True)  # shape:(16,1,32)

Xm = Flatten(data_format=data_format, name='flat')(Xm)
Xm = Dense(128, activation='relu', kernel_initializer='glorot_normal', name="dense1")(Xm)
Xm = AlphaDropout(0.3)(Xm)
# Full Con 2
Xm = Dense(len(classes), kernel_initializer='glorot_normal', name="dense2")(Xm)
Xm = AlphaDropout(0.3)(Xm)
# SoftMax
Xm = Activation('softmax', name='activate')(Xm)
# Create Model
model = Model.Model(inputs=Xm_input, outputs=Xm)
adam = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam)
model.summary()

filepath = 'ResNet2018.h5'
history = model.fit(X_train,
                    Y_train,
                    # batch_size=1000,
                    batch_size=1000,  # already changed to 10, original one is 1000
                    epochs=10,  # changed to 10, original one is 100
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    # validation_split = 0.3,
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                                           mode='auto'),
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
                    ])

'''
### Plot the loss curve ###

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
np.savez('ResNet33_2018', results)


'''
plt.plot(list(acc.keys()), list(acc.values()))
plt.ylabel('ACC')
plt.xlabel('SNR')
plt.show()
'''
