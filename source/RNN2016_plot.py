import numpy as np
from matplotlib import pyplot as plt

### Plot Accuracy, Loss and Confusion Matrix
def RNN2016_visualize():

    # Load the result data
    outfile = 'RNN2016Results'
    npzfile = np.load(outfile+'.npz')
    acc_base_lstm = npzfile['acc_base_lstm']
    val_acc_base_lstm = npzfile['val_acc_base_lstm']
    loss_base_lstm = npzfile['loss_base_lstm']
    val_loss_base_lstm = npzfile['val_loss_base_lstm']
    epochs = npzfile['epochs']
    cm = npzfile['cm']

    # Plot acc and loss
    plt.plot(epochs, acc_base_lstm, label='Training acc')
    plt.plot(epochs, val_acc_base_lstm, label='Validation acc')
    plt.title('Training and validation accuracy for Baseline LSTM')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss_base_lstm, label='Training loss')
    plt.plot(epochs, val_loss_base_lstm, label='Validation loss')
    plt.title('Training and validation loss for Baseline LSTM')
    plt.legend()
    plt.show()

    # Plot confusion matrix
    plot_confusion_matrix(cm)


# Helper function for plotting cm
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



'''
"""
plot confusion matrix vs SNRs
"""
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
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="LSTM Confusion Matrix (SNR=%d)" % snr)
    plt.show()
    # cm.ax_.set_title()
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)

"""
print accuracy curve vs SNRs
"""

# Plot accuracy curve
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("LSTM Classification Accuracy on RadioML 2016.10 Alpha")
plt.show()
'''
