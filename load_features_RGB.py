import scipy.sparse
import numpy as np

# load Aff-Wild2 Features
train_features_AW2 = scipy.sparse.load_npz("data/features/train_features_RGB_AW2.npz") #CSR Matrix
val_features_AW2 = scipy.sparse.load_npz("data/features/val_features_RGB_AW2.npz") #CSR Matrix
train_labels_AW2 = np.load("data/labels/train_labels_RGB_AW2.npy") #Numpy array
val_labels_AW2 = np.load("data/labels/val_labels_RGB_AW2.npy") #Numpy array
test_features_AW2 = scipy.sparse.load_npz("data/features/test_features_RGB_AW2.npz") #CSR Matrix

# Load AFEW7.0 features
test_features_AF7 = scipy.sparse.load_npz("data/features/features_RGB_AF7.npz") #CSR Matrix
test_labels_AF7 = np.load("data/labels/labels_RGB_AF7.npy") #Numpy array

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