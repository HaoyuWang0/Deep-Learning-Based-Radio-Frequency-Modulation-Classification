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
#print(X_train.shape, in_shp)
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


### Save the results ###

outfile = 'RNN2016Results'
np.savez(outfile,acc_base_lstm=acc_base_lstm,val_acc_base_lstm=val_acc_base_lstm,
            loss_base_lstm=loss_base_lstm,val_loss_base_lstm=val_loss_base_lstm,
            epochs=epochs,cm=cm)
