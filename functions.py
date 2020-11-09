import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

# Source: https://github.com/keras-team/keras/blob/1c630c3e3c8969b40a47d07b9f2edda50ec69720/keras/metrics.py
def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def plot_history(data_list, label_list, title, ylabel):

    epochs = range(1, len(data_list[0]) + 1)

    for data, label in zip(data_list, label_list):
        plt.plot(epochs, data, label=label)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()

    plt.show()
    
def arr_replacevalue(array, d_class_weights, start=0):
    labels = d_class_weights.keys()
    labels = list(labels)
    if start > 6:
        return array
    array = np.where(array ==  labels[start], d_class_weights.get(labels[start]), array)

    return arr_replacevalue(array, d_class_weights, start + 1)