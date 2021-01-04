#!/usr/bin/env python
# coding: utf-8

# ## Load/import packages

# In[1]:


import time
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tqdm.notebook import tqdm
from os import listdir
from os.path import join

from tensorflow.keras import Sequential, layers
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# import fwRNN cell
from fwrnn_cell import FW_RNNCell

# Limit GPU memory usage
for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)


# ## Functions

# In[2]:


def shuffler(feature_fn, labels_fn):
    indices = np.arange(feature_fn.shape[0])
    np.random.shuffle(indices)
    return feature_fn[indices], labels_fn[indices]

def load_batch(batchsize, features_fnshuff, labels_fnshuff, train):
    if train:
        feature_dir = r"D:\block4\train_AW2\features"
        label_dir = r"D:\block4\train_AW2\labels"
    else:
        feature_dir = r"D:\block4\val_AW2\features"
        label_dir = r"D:\block4\val_AW2\labels"
        

    features = []
    labels = []
    
    global index
    
    for i in range(index, index+batchsize):
        features.append(np.expand_dims(np.load(join(feature_dir, features_fnshuff[i])),axis=0))
        labels.append(np.expand_dims(np.load(join(label_dir, labels_fnshuff[i])), axis=0))
    index += batchsize

    return np.vstack(features), np.vstack(labels)

def arr_replacevalue(array, d_class_weights, start=0):
    labels = d_class_weights.keys()
    labels = list(labels)
    if start > 6:
        return array
    array = np.where(array ==  labels[start], d_class_weights.get(labels[start]), array)

    return arr_replacevalue(array, d_class_weights, start + 1)

def comp_sampleweights(labels):
    # Convert one-hot encoded labels back to label integers
    train_label_ints = np.argmax(labels, axis=2)

    # Pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample
    return arr_replacevalue(train_label_ints, class_weights)


# In[3]:


# Build model with model subclassing and sequential API
def build_FWRNN(batch, units, activation_function):
    # Define model
    model = Sequential(name="FW-RNN")
    model.add(layers.InputLayer(input_shape=(60, 25088)))
    model.add(
        layers.RNN(
            FW_RNNCell(
                units=units,
                use_bias=True,
                activation=activation_function,
                step=1,
                decay_rate=0.95,
                learning_rate=0.5,
                batch_size=batch,
            ),
            return_sequences=True,
            name="FW-RNN",
        )
    )
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(7, activation="softmax", name="Dense_Output"))
    return model


# In[4]:


# Build baseline model (RNN or LSTM) with sequential API
def build_base(model_name, units):
    model = Sequential(name=model_name)
    model.add(layers.InputLayer(input_shape=(60, 25088)))
    if model_name == "RNN":
        model.add(layers.SimpleRNN(units, return_sequences=True,))
        model.add(layers.Dropout(0.4))
    else:
        model.add(layers.LSTM(units, return_sequences=True,))
        model.add(layers.Dropout(0.4))
    model.add(layers.LayerNormalization())
    model.add(layers.Dense(7, activation="softmax", name="Dense_Output"))
    return model


# # Training with custom loop 

# In[5]:


# List all files
train_feature_fn = np.array(listdir(r'D:/block4/train_AW2/features'))
train_labels_fn = np.array(listdir(r'D:/block4/train_AW2/labels'))

# Load class weights training set
with open('data/weights.json', 'r') as fp:
    class_weights = json.load(fp)


# In[6]:


#### RUN THIS ONCE ####

# val_feature_fn = np.array(listdir(r'D:/block4/val_AW2/features'))
# val_labels_fn = np.array(listdir(r'D:/block4/val_AW2/labels'))

# # Split validation in half to get validation- and test-set
# # Shuffle first
# for i in range(5):
#     val_feature_fn, val_labels_fn = shuffler(val_feature_fn, val_labels_fn)

# # Split features and labels in half
# val_feature_fn, test_feature_fn = np.split(val_feature_fn, 2)
# val_labels_fn, test_val_labels_fn = np.split(val_labels_fn, 2)

# np.save(r'D:\block4\val_feature_fn.npy', val_feature_fn)
# np.save(r'D:\block4\test_feature_fn.npy', test_feature_fn)
# np.save(r'D:\block4\val_labels_fn.npy', val_labels_fn)
# np.save(r'D:\block4\test_val_labels_fn.npy', test_val_labels_fn)


# In[7]:


val_feature_fn = np.load(r'D:\block4\val_feature_fn.npy')
val_labels_fn = np.load(r'D:\block4\val_labels_fn.npy')


# In[9]:


num_units = 50
epochs = 25
batchsize = 32

for model_name in ["FWRNN", "LSTM", "RNN"]:
    # Define optimizer
    optimizer = tf.keras.optimizers.RMSprop()

    # Define loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Prepare the metrics
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    
    if model_name == "RNN":
        model = build_base(model_name, num_units)
    elif model_name == "LSTM":
        model = build_base(model_name, num_units)
    elif model_name == "FWRNN":
        model = build_FWRNN(batchsize, num_units, "relu")

    # Training loop for 25 epochs
    for epoch in range(1, epochs+1):
        print(f"epoch: {epoch}")
        start = time.time()

        # Set index to 0 for training
        index = 0

        train_features_fnshuff, train_labels_fnshuff = shuffler(train_feature_fn, train_labels_fn)
        val_features_fnshuff, val_labels_fnshuff = shuffler(val_feature_fn, val_labels_fn)

        pbar = tqdm(range((len(train_feature_fn)//batchsize)))
        for i in pbar:
            X_train, y_train = load_batch(batchsize, train_features_fnshuff, train_labels_fnshuff, train=True)
            train_samples_weights = comp_sampleweights(y_train)

            with tf.GradientTape() as tape:
                logits = model(X_train, training = True)
                loss_value = loss_fn(y_train, logits, sample_weight=train_samples_weights)


            # Compute gradients
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Update weights
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_train, logits)

            pbar.set_postfix({'loss': loss_value.numpy(), 'acc': train_acc_metric.result().numpy()})

        train_acc = train_acc_metric.result()
        print(f"Training acc over epoch: {train_acc}")

        train_acc_metric.reset_states()


        # Set index to 0 for validation
        index = 0

        for i in range((len(val_feature_fn)//batchsize)):
            X_val, y_val = load_batch(batchsize, val_features_fnshuff, val_labels_fnshuff, train=False)
            val_logits = model(X_val, training=False)

            val_acc_metric.update_state(y_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()

        print(f"Validation acc over epoch: {val_acc}")
        print(f"Time taken: {time.time() - start}")

    # Save model
    tf.keras.Model.save(
        model,
        filepath=f"data/models/models_with_extractedfeatures_vgg19block4/{model_name}_{num_units}units.h5",
    )


# In[19]:


# Load AW2 test files
test_feature_fn = np.load(r'D:\block4\test_feature_fn.npy')
test_labels_fn = np.load(r'D:\block4\test_val_labels_fn.npy')

# Load AF7 test files
testAF7_feature_fn = np.array(listdir(r'D:/block4/test_AF7/features'))
testAF7_labels_fn = np.array(listdir(r'D:/block4/test_AF7/labels'))


# In[20]:


# test_features_fnshuff, test_labels_fnshuff = shuffler(test_feature_fn, test_labels_fn)
testAF7_features_fnshuff, testAF7_labels_fnshuff = shuffler(testAF7_feature_fn, testAF7_labels_fn)


# In[36]:


num_units = 50
epochs = 25
batchsize = 32

for model_name in ["FWRNN", "LSTM", "RNN"]:
    # Prepare the metrics
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    testAF7_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    if model_name == "FWRNN":
        model = tf.keras.models.load_model(
            filepath=f"data/models/models_with_extractedfeatures_vgg19block4/{model_name}_{num_units}units.h5",
            custom_objects={"FW_RNNCell": FW_RNNCell},
            compile=False,
        )
    else:
        model = tf.keras.models.load_model(
            filepath=f"data/models/models_with_extractedfeatures_vgg19block4/{model_name}_{num_units}units.h5",
            compile=False,
        )

    # Set index to 0 for evaluation
    index = 0

    for i in tqdm(range((len(test_feature_fn) // batchsize))):
        X_test, y_test = load_batch(batchsize, test_features_fnshuff, test_labels_fnshuff, train=False)
        test_logits = model(X_test, training=False)

        test_acc_metric.update_state(y_test, test_logits)
    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()

    print(f"{model_name} {num_units} units, AW2 test acc: {test_acc:.4f}")  

