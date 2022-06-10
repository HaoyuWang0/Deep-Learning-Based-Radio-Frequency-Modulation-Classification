# Filename: CNN_2016.py
# Description: This file implements a CNN model for RADIOML 2016.10A dataset.

import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from keras import Sequential
from keras.layers import ZeroPadding2D, Conv2D, Dropout, Flatten, Dense
from mlxtend.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


### Data Preprocessing ###

f = open("RML2016.10a_dict.pkl", 'rb')
Xd = pickle.load(f, encoding='latin1')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
X = np.vstack(X)
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0, int(n_examples)), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))
X_train = X[train_idx][:,:,:,np.newaxis]
X_test = X[test_idx][:,:,:,np.newaxis]
N, H, W,_ = X_train.shape


def to_onehot(yy):
    yy = list(yy)
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods  # modulations(11 classes)


### Build a CNN model ###

dr = 0.4  # dropout rate (%)
model = Sequential(name='CNN_Architecture')
model.add(ZeroPadding2D((0, 2), data_format='channels_last'))
model.add(Conv2D(256, (1, 3), activation='relu', data_format='channels_last', input_shape=(H, W, 1), name='conv1'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2), data_format='channels_last'))
model.add(Conv2D(80, (2, 3), activation='relu', data_format='channels_last'))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(dr))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(None, H, W, 1))
model.summary()


### Training ###

nb_epoch = 50
batch_size = 1024
filepath = 'baseline_simpleCNN_dr=0.5.h5'
history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                                        mode='auto'),
                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print('Score:',score)

'''
### Accuracy and Loss

plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['accuracy'], label='train_acc')
plt.plot(history.epoch, history.history['val_accuracy'], label='val_acc')
plt.legend()

plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train_loss')
plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
plt.legend()
'''

'''
### Confusion Matrix

model.load_weights("baseline_simpleCNN_dr=0.5.h5")
batch = 1024
y_pred = model.predict(X_test, batch)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch)
print(score)
predict_x = model.predict(X_test, batch)
y_pred_label = np.argmax(predict_x, axis=1)
y_test_label = np.argmax(Y_test, axis=1)
cm = confusion_matrix(y_test_label, y_pred_label, normalize='true')
cm = confusion_matrix(y_test_label, y_pred_label)
fg, ax = plot_confusion_matrix(cm, colorbar=True,
                               show_absolute=False,
                               show_normed=True)
ax.set_xticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=45)
ax.set_title("CNN Confusion Matrix")
ax.set_yticks(np.arange(len(classes)))
ax.set_yticklabels(classes)
fg.set_size_inches(16.5, 8.5, forward=True)
plt.show()
'''


### Accuracy for each SNR ###

acc = {}
for snr in snrs:
    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i,batch_size=128)
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    '''
    plt.figure()
    plot_confusion_matrix(confnorm, class_names=classes)
    plt.title("ConvNet Confusion Matrix(SNR=%d)" % (snr))
    '''
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)


### Save the results ###

accs = list(map(lambda x: acc[x], snrs))
results = np.concatenate((snrs,accs),axis=1)
np.savez('CNN_2016', results)

'''
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
'''
