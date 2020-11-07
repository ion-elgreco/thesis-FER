import scipy.sparse
import numpy as np

train_features = scipy.sparse.load_npz("data/train_features.npz") #CSR Matrix
val_features = scipy.sparse.load_npz("data/val_features.npz") #CSR Matrix
train_labels = np.load("data/train_labels.npy") #Numpy array
val_labels = np.load("data/val_labels.npy") #Numpy array

# Function which reshapes the features to [sequences, sequence_length, features], 
# and labels to [sequences, sequence_length, labels]
def reshaper(features, labels, sequence_length):
    """
    :param features: extracted image features
    :type features: scipy CSR matrix (2-Dim)
    :param labels: one-hot encoded labels of each image
    :type labels: numpy array (2-Dim)
    :param sequence_length: the length of the sequence you want. E.g. You want 25 images in a seqeunce so, value is 25
    :type sequence_length: int
    
    """
    # Find the amount of possible sequences with given sequence_length. Some data will be discarded this way.
    amount = features.shape[0] // sequence_length
    
    # Convert Scipy matrix back to numpy array because we need to convert it to 3 Dimensional array
    arr_features = features[:(amount * sequence_length)].toarray()
    
    # Reshapes the labels and features
    seq_features = np.reshape(arr_features, (amount, sequence_length, arr_features.shape[1]))
    seq_labels = np.reshape(labels[:(amount * sequence_length)], (amount, sequence_length, labels.shape[1]))
    
    
    return seq_features, seq_labels