import numpy as np
import pickle

from keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras import Input, Model
from keras.layers import Dropout, Dense, BatchNormalization, LSTM, Conv1D, MaxPool1D, Flatten
from matplotlib import pyplot as plt
from tensorflow import keras


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def CNN_LSTM():
    inputs = Input((128, 2,))
    l = BatchNormalization()(inputs)
    l = Conv1D(filters=128, kernel_size=5, activation='relu')(l)
    l = MaxPool1D(3)(l)
    l = Conv1D(filters=128, kernel_size=5, activation='relu')(l)
    l = LSTM(128, return_sequences=True, activation='relu', unroll=True)(l)
    l = LSTM(128, return_sequences=True, activation='relu', unroll=True)(l)
    l = Dropout(0.8)(l)
    l = Flatten()(l)
    outputs = Dense(11, activation='softmax', kernel_constraint=max_norm(2.))(l)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.summary()
    return model


"""
pre-process data
"""

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
print(X_train.shape, in_shp)
classes = mods  # modulations(11 classes)

nb_epoch = 1
batch_size = 1024
filepath = 'CNN_LSTM_2016.h5'
model = CNN_LSTM()
X_train = np.reshape(X_train, (-1, 128, 2))
X_test = np.reshape(X_test, (-1, 128, 2))

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

# re-load the best weights once training is finished
model.load_weights(filepath)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print(score)
"""
show accuract & loss curve
"""

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
plt.show()

"""
plot confusion matrix
"""
model.load_weights(filepath)
batch = 1024
y_pred = model.predict(X_test, batch)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch)
print(score)
predict_x = model.predict(X_test, batch)
y_pred_label = np.argmax(predict_x, axis=1)
y_test_label = np.argmax(Y_test, axis=1)
cm = confusion_matrix(y_test_label, y_pred_label, normalize='true')
cm = confusion_matrix(y_test_label, y_pred_label)
plot_confusion_matrix(cm, labels=classes)
plt.show()

"""
plot confusion matrix vs SNRs
"""
acc = {}
for snr in snrs:

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
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="CNN_LSTM Confusion Matrix (SNR=%d)" % (snr))
    plt.show()
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor + ncor), "for snr: ", snr)
    acc[snr] = 1.0 * cor / (cor + ncor)

"""
print accuracy curve vs SNRs
"""

# Plot accuracy curve
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN_LSTM Classification Accuracy on RadioML 2016.10 Alpha")
plt.show()
