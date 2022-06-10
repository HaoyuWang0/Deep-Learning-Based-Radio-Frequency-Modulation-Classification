# Filename: RNN_2016.py
# Description: This file implements an RNN model for RADIOML 2016.10A dataset.

import numpy as np
import pickle

from keras.constraints import max_norm
from sklearn.metrics import confusion_matrix
from keras import Input, Model
from keras.layers import Dropout, Dense, BatchNormalization, LSTM
from matplotlib import pyplot as plt
from tensorflow import keras


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
X_train = X[train_idx]
X_test = X[test_idx]
N, H, W = X_train.shape


def to_onehot(yy):
    yy = list(yy)
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
in_shp = list(X_train.shape[1:])
# print(X_train.shape, in_shp)
classes = mods  # modulations(11 classes)


### LSTM Model ###

def baseline_lstm():
    inputs = Input((2, 128,))
    l = BatchNormalization()(inputs)
    l = LSTM(128, return_sequences=True, activation='tanh', unroll=True)(l)
    l = LSTM(128, return_sequences=False, activation='tanh', unroll=True)(l)
    l = Dropout(0.5)(l)
    outputs = Dense(11, activation='softmax', kernel_constraint=max_norm(2.))(l)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


### Training ###

model = baseline_lstm()
checkpoint = keras.callbacks.ModelCheckpoint("lstm2016.h5",
                                             monitor='val_accuracy', verbose=2, save_best_only=False,
                                             save_weights_only=False, mode='auto', period=1)
history = model.fit(x=X_train,
                    y=Y_train,
                    validation_data=[X_test, Y_test],
                    batch_size=400,
                    epochs=20,
                    shuffle='batch',
                    callbacks=[checkpoint],
                    verbose=2)

acc_base_lstm = history.history['accuracy']
val_acc_base_lstm = history.history['val_accuracy']
loss_base_lstm = history.history['loss']
val_loss_base_lstm = history.history['val_loss']
epochs = range(1, len(acc_base_lstm) + 1)

'''
### Confusion Matrix ###

model.load_weights("lstm2016.h5")
batch = 512
y_pred = model.predict(X_test, batch)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch)
#print(score)
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
ax.set_title("LSTM Confusion Matrix")
ax.set_yticks(np.arange(len(classes)))
ax.set_yticklabels(classes)
fg.set_size_inches(16.5, 8.5, forward=True)
plt.show()
'''


### Accuracy for Each SNR ###

acc = {}

for snr in snrs:
    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    n = list(test_SNRs)
    test_X_i = X_test[np.where(np.array(n) == snr)]
    test_Y_i = Y_test[np.where(np.array(n) == snr)]
    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
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
    plot_confusion_matrix(confnorm, labels=classes, title="LSTM Confusion Matrix (SNR=%d)" % snr)
    plt.show()
    '''
    # cm.ax_.set_title()
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    # print("Overall Accuracy: ", cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)


### Save the results ###

accs = list(map(lambda x: acc[x], snrs))
results = np.concatenate((snrs,accs),axis=1)
np.savez('LSTM_2016', results)
