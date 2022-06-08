import tensorflow as tf
from keras import Input, Model
from keras.layers import Dropout, Dense, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam

print(tf.keras.__version__)
import h5py
import os, random
import numpy as np

from tensorflow.keras.layers import Input, Reshape, ZeroPadding2D, Conv2D, Dropout, Flatten, Dense, Activation, \
    MaxPool2D, AlphaDropout
from tensorflow.keras import layers
import tensorflow.keras.models as Model
import matplotlib.pyplot as plt

"""
read files
"""
for i in range(0, 24):  # 24个数据集文件
    ########打开文件#######
    filename = 'full dataset/full_part' + str(i) + '.h5'
    print(filename)
    f = h5py.File(filename, 'r')
    ########读取数据#######
    X_data = f['X'][:]
    Y_data = f['Y'][:]
    Z_data = f['Z'][:]
    f.close()
    #########分割训练集和测试集#########
    # 每读取到一个数据文件就直接分割为训练集和测试集，防止爆内存
    n_examples = X_data.shape[0]
    n_train = int(n_examples * 0.7)  # 70%训练样本
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)  # 随机选取训练样本下标
    test_idx = list(set(range(0, n_examples)) - set(train_idx))  # 测试样本下标
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

print('训练集X维度：', X_train.shape)
print('训练集Y维度：', Y_train.shape)
print('训练集Z维度：', Z_train.shape)
print('测试集X维度：', X_test.shape)
print('测试集Y维度：', Y_test.shape)
print('测试集Z维度：', Z_test.shape)

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

in_shp = X_train.shape[1:]  # [1024,2]
print(in_shp)


# input layer


def baseline_lstm():
    inputs = Input((1024, 2,))
    l = BatchNormalization()(inputs)
    l = LSTM(1024, return_sequences=True, activation='tanh', unroll=True)(l)
    l = LSTM(1024, return_sequences=False, activation='tanh', unroll=True)(l)
    l = Dropout(0.2)(l)
    outputs = Dense(24, activation='softmax')(l)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.summary()
    return model


model = baseline_lstm()
filepath = 'lstm2018.h5'
history = model.fit(X_train,
                    Y_train,
                    # batch_size=1000,
                    batch_size=100,  # already changed to 10, original one is 1000
                    epochs=100,  # changed to 10, original one is 100
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    # validation_split = 0.3,
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                                           mode='auto'),
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
                    ])
"""
plot loss curve

"""
print('train finish')
val_loss_list = history.history['val_loss']
loss_list = history.history['loss']
plt.plot(range(len(loss_list)), val_loss_list)
plt.plot(range(len(loss_list)), loss_list)
plt.show()

model.load_weights(filepath)
"""
plot confusion matrix

"""


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

"""
analysis vs 24 SNRs
"""
for i in range(len(confnorm)):
    print(classes[i], confnorm[i, i])

acc = {}
Z_test = Z_test.reshape((len(Z_test)))
SNRs = np.unique(Z_test)
for snr in SNRs:
    X_test_snr = X_test[Z_test == snr]
    Y_test_snr = Y_test[Z_test == snr]

    pre_Y_test = model.predict(X_test_snr)
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, X_test_snr.shape[0]):  # 该信噪比下测试数据量
        j = list(Y_test_snr[i, :]).index(1)  # 正确类别下标
        j = classes.index(classes[j])
        k = int(np.argmax(pre_Y_test[i, :]))  # 预测类别下标
        k = classes.index(classes[k])
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)" % (snr))

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy %s: " % snr, cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)

"""
accuracy vs 24 snrs
"""
plt.plot(list(acc.keys()), list(acc.values()))
plt.ylabel('ACC')
plt.xlabel('SNR')
plt.show()
