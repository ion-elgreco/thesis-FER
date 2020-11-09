import scipy.sparse
import numpy as np

train_features = scipy.sparse.load_npz("data/train_features.npz") #CSR Matrix
val_features = scipy.sparse.load_npz("data/val_features.npz") #CSR Matrix
train_labels = np.load("data/train_labels.npy") #Numpy array
val_labels = np.load("data/val_labels.npy") #Numpy array
test_features = scipy.sparse.load_npz("data/test_features.npz") #CSR Matrix

# Function which reshapes the features to [sequences, sequence_length, features], 
# and labels to [sequences, sequence_length, labels]
def labels_reshaper(labels, sequence_length):
    # Find the amount of possible sequences with given sequence_length. Some data will be discarded this way.
    amount = labels.shape[0] // sequence_length
    
    # Reshapes the labels
    seq_labels = np.reshape(labels[:(amount * sequence_length)], (amount, sequence_length, labels.shape[1]))

    return seq_labels

# Function which reshapes the features to [sequences, sequence_length, features],
def features_reshaper(features, sequence_length):
    amount = features.shape[0] // sequence_length
    arr_features = features[:(amount * sequence_length)].toarray()
    
    # Reshapes the features
    seq_features = np.reshape(arr_features, (amount, sequence_length, arr_features.shape[1]))
    
    return seq_features