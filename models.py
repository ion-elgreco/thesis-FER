#!/usr/bin/env python
# coding: utf-8

# ## Load/import packages

# In[ ]:


import json
import scipy
import numpy as np
import scipy.sparse
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from tensorflow.keras import Sequential, layers
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Import modules to run custom FW-RNN cell
from tensorflow.python.keras.layers.recurrent import (
    _generate_zero_filled_state_for_cell,
    _generate_zero_filled_state,
    activations,
    initializers,
    regularizers,
    nest,
)

get_ipython().run_line_magic('matplotlib', 'inline')

# Limit GPU memory usage
for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)


# In[ ]:


# load Aff-Wild2 Features
train_features_AW2 = scipy.sparse.load_npz("data/features/train_features_RGB_AW2.npz") #CSR Matrix
val_features_AW2 = scipy.sparse.load_npz("data/features/val_features_RGB_AW2.npz") #CSR Matrix
train_labels_AW2 = np.load("data/labels/train_labels_RGB_AW2.npy") #Numpy array
val_labels_AW2 = np.load("data/labels/val_labels_RGB_AW2.npy") #Numpy array

# test_features_AW2 = scipy.sparse.load_npz("data/features/test_features_RGB_AW2.npz") #CSR Matrix


# In[ ]:


# Load AFEW7.0 features
test_features_AF7 = scipy.sparse.load_npz("data/features/features_RGB_AF7.npz") #CSR Matrix
test_labels_AF7 = np.load("data/labels/labels_RGB_AF7.npy") #Numpy array


# ## Functions

# In[ ]:


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

def arr_replacevalue(array, d_class_weights, start=0):
    labels = d_class_weights.keys()
    labels = list(labels)
    if start > 6:
        return array
    array = np.where(array ==  labels[start], d_class_weights.get(labels[start]), array)

    return arr_replacevalue(array, d_class_weights, start + 1)


# # Prepare data

# In[ ]:


# Reshape data to specified sequence length
length = 60

X_train = features_reshaper(train_features_AW2, length)
del train_features_AW2  # wipe out of memory to free up space

X_val = features_reshaper(val_features_AW2, length)
del val_features_AW2  # wipe out of memory to free up space

y_train = labels_reshaper(train_labels_AW2, length)
del train_labels_AW2  # wipe out of memory to free up space

y_val = labels_reshaper(val_labels_AW2, length)
del val_labels_AW2  # wipe out of memory to free up space


# In[ ]:


X_test_AF7 = features_reshaper(test_features_AF7, length)
del test_features_AF7  # wipe out of memory to free up space

y_test_AF7 = labels_reshaper(test_labels_AF7, length)
del test_labels_AF7  # wipe out of memory to free up space


# In[ ]:


# Split the validation set into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5, random_state = 1234, shuffle=True)


# In[ ]:


def comp_sampleweights(labels):
    # Convert one-hot encoded labels back to label integers
    train_label_ints = np.argmax(labels, axis=2)

    # Compute class weights with sklearn
    class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(train_label_ints), train_label_ints.flatten()
    )
    d_class_weights = dict(enumerate(class_weights))

    # Pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample
    return arr_replacevalue(train_label_ints, d_class_weights)


train_samples_weights = comp_sampleweights(y_train)


# # Build FW-RNN model
# -  Build custom FW_RNN cell and wrap it in RNN layer (https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN), like this: RNN(FW_RNN)
#     -  "The cell abstraction, together with the generic keras.layers.RNN class, make it very easy to implement custom RNN architectures for your research."
# 
# Created by using this guide: https://www.tensorflow.org/guide/keras/custom_layers_and_models

# In[ ]:


# Build model with model subclassing and sequential API
def build_FWRNN(batch, units, activation_function):
    class FW_RNNCell(layers.Layer):
        def __init__(
            self,
            units,
            use_bias,
            batch_size,
            decay_rate,
            learning_rate,
            activation,
            step,
            LN=layers.LayerNormalization(),
            **kwargs
        ):
            super(FW_RNNCell, self).__init__(**kwargs)
            self.units = units
            self.step = step
            self.use_bias = use_bias
            self.activation = activations.get(activation)
            self.l = decay_rate
            self.e = learning_rate
            self.LN = LN
            self.batch = batch_size
            self.state_size = self.units

            # Initializer for the slow input-to-hidden weights matrix
            self.C_initializer = initializers.get("glorot_uniform")

            # Initializer for the slow hidden weights matrix
            self.W_h_initializer = initializers.get("identity")

            # Initializer for the fast weights matrix
            self.A_initializer = initializers.get("zeros")

            # Initializer for the bias vector.
            self.b_x_initializer = initializers.get("zeros")

        def build(self, input_shape):
            # Build is only called at the start, to initialize all the weights and biases

            # C = Slow input-to-hidden weights [shape (features_vector, units)]
            self.C = self.add_weight(
                shape=(input_shape[-1], self.units),
                name="inputweights",
                initializer=self.C_initializer,
            )

            # W_h The previous hidden state via the slow transition weights [shape (units, units)]
            
            self.W_h = self.add_weight(
                shape=(self.units, self.units),
                name="hiddenweights",
                initializer=self.W_h_initializer,
            )
            self.W_h = tf.scalar_mul(0.05, self.W_h)

            # A (fast weights) [shape (batch_size, units, units)]
            self.A = self.add_weight(
                shape=(self.batch, self.units, self.units),
                name="fastweights",
                initializer=self.A_initializer,
            )

            if self.use_bias:
                self.bias = self.add_weight(
                    shape=(self.units,), name="bias", initializer=self.b_x_initializer,
                )
            else:
                self.bias = None
            self.built = True

        def call(self, inputs, states, training=None):
            prev_output = states[0] if nest.is_sequence(states) else states

            # Next hidden state h(t+1) is computed in two steps:
            # Step 1 calculate preliminary vector: h_0(t+1) = f(W_h ⋅ h(t) + C ⋅ x(t))
            h = K.dot(prev_output, self.W_h) + K.dot(inputs, self.C)
            if self.bias is not None:
                h = h + self.bias
            if self.activation is not None:
                h = self.activation(h)

            # Reshape h to use with a
            h_s = tf.reshape(h, [self.batch, 1, self.units])

            # Define preliminary vector in variable
            prelim = tf.reshape(K.dot(prev_output, self.W_h), (h_s.shape)) + tf.reshape(
                K.dot(inputs, self.C), (h_s.shape)
            )

            # Fast weights update rule: A(t) = λ*A(t-1) + η*h(t) ⋅ h(t)^T
            self.A.assign(
                tf.math.add(
                    tf.scalar_mul(self.l, self.A),
                    tf.scalar_mul(
                        self.e, tf.linalg.matmul(tf.transpose(h_s, [0, 2, 1]), h_s)
                    ),
                )
            )

            # Step 2: Initiate inner loop with preliminary vector, which runs for S steps
            # to progressively change the hidden state into h(t+1) = h_s(t+1)
            # h_s+1(t+1) f([W_h ⋅ h(t) + C ⋅ x(t)]) + A(t)h_s(t+1)
            for _ in range(self.step):
                h_s = tf.math.add(prelim, tf.linalg.matmul(h_s, self.A))
                if self.activation is not None:
                    h_s = self.activation(h_s)

                # Apply layer normalization on hidden state
                h_s = self.LN(h_s)

            h = tf.reshape(h_s, [self.batch, self.units])

            output = h
            new_state = [output] if nest.is_sequence(states) else output
            return output, new_state

        def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
            return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

        def get_config(self):
            config = {
                "units": self.units,
                "step": self.step,
                "batch_size": self.batch,
                "use_bias": self.use_bias,
                "activation": activations.serialize(self.activation),
                "decay_rate": self.l,
                "learning_rate": self.e,
                "LN": self.LN,
            }
            base_config = super(FW_RNNCell, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    # Define model
    model = Sequential(name="FW-RNN")
    model.add(tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])))
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
    model.compile(
        optimizer="rmsprop",
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
        run_eagerly=False,
    )
    return model


# In[ ]:


# Build baseline model (RNN or LSTM) with sequential API
def build_base(model_name, units):
    model = Sequential(name=model_name)
    model.add(tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])))
    if model_name == "RNN":
        model.add(layers.SimpleRNN(units, return_sequences=True))
        model.add(layers.Dropout(0.4))
    else:
        model.add(layers.LSTM(units, return_sequences=True))
        model.add(layers.Dropout(0.4))
    model.add(layers.LayerNormalization())
    model.add(layers.Dense(7, activation="softmax", name="Dense_Output"))
    model.compile(
        optimizer="rmsprop", loss=CategoricalCrossentropy(), metrics=["accuracy"],
    )
    return model


# In[ ]:


# Set early stopping
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0025,
    patience=6,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)


# In[ ]:


# Define batch size for models
batchsize = 32

# Disivible length of train and val features by batchsize, necessary because fwRNN batch size is fixed
train_div = (X_train.shape[0] // batchsize) * batchsize
val_div = (X_val.shape[0] // batchsize) * batchsize

# Slice the sets to fit the batch size
X_train = X_train[:train_div]
y_train = y_train[:train_div]

X_val = X_val[:val_div]
y_val = y_val[:val_div]


for num_units in [5, 20, 50, 100]:
    for model in ["RNN", "LSTM", "FWRNN"]:
        #     for model in ["RNN", "LSTM", "FWRNN"]:
        # Access tensorboard in cmd of the main repo folder with following code:
        # tensorboard --logdir='logs/'
        name = f"final_{model}_{num_units}units_rmsprop_dropout0.4"
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=f"logs/models_with_extractedfeatures_vgg19block5/{name}"
        )

        if model == "RNN":
            NN = build_base(model, num_units)
        elif model == "LSTM":
            NN = build_base(model, num_units)
        elif model == "FWRNN":
            NN = build_FWRNN(batchsize, num_units, "relu")
        
        # Check model summary
        NN.summary()
        
        # Fit model to training set and evaluate
        history = NN.fit(
            X_train,
            y_train,
            batch_size=batchsize,
            sample_weight=train_samples_weights[:train_div],
            validation_data=(X_val, y_val),
            callbacks=[es, tb_callback],
            epochs=50,
            verbose=2,
            shuffle=True,
        )

        # Plot model
        tf.keras.utils.plot_model(
            NN,
            to_file=f"data/model_architectures/models_with_extractedfeatures_vgg19block5/{model}_{num_units}units_architecture.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="LR",
            expand_nested=False,
            dpi=96,
        )

        # Save model
        tf.keras.Model.save(
            NN,
            filepath=f"data/models/models_with_extractedfeatures_vgg19block5/{model}_{num_units}units.h5",
        )


# ## Evaluate on test sets

# In[ ]:


# Disivible length of test_AW2 and test_AF7 features by batchsize, necessary because fwRNN batch size is fixed
test_div_AW2 = (X_test.shape[0] // batchsize) * batchsize
test_div_AF7 = (X_test_AF7.shape[0] // batchsize) * batchsize

# Slice the sets to fit the batch size
X_test = X_test[:test_div_AW2]
y_test = y_test[:test_div_AW2]

X_test_AF7 = X_test_AF7[:test_div_AF7]
y_test_AF7 = y_test_AF7[:test_div_AF7]


# In[ ]:


for num_units in [5, 20, 50, 100]:
    for model in ["RNN", "LSTM", "FWRNN"]:

        class FW_RNNCell(layers.Layer):
            def __init__(
                self,
                units,
                use_bias,
                batch_size,
                decay_rate,
                learning_rate,
                activation,
                step,
                LN=layers.LayerNormalization(),
                **kwargs,
            ):
                super(FW_RNNCell, self).__init__(**kwargs)
                self.units = units
                self.step = step
                self.use_bias = use_bias
                self.activation = activations.get(activation)
                self.l = decay_rate
                self.e = learning_rate
                self.LN = LN
                self.batch = batch_size
                self.state_size = self.units

                # Initializer for the slow input-to-hidden weights matrix
                self.C_initializer = initializers.get("glorot_uniform")

                # Initializer for the slow hidden weights matrix
                self.W_h_initializer = initializers.get("identity")

                # Initializer for the fast weights matrix
                self.A_initializer = initializers.get("zeros")

                # Initializer for the bias vector.
                self.b_x_initializer = initializers.get("zeros")

            def build(self, input_shape):
                # Build is only called at the start, to initialize all the weights and biases

                # C = Slow input-to-hidden weights [shape (features_vector, units)]
                self.C = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    name="inputweights",
                    initializer=self.C_initializer,
                )

                # W_h The previous hidden state via the slow transition weights [shape (units, units)]

                self.W_h = self.add_weight(
                    shape=(self.units, self.units),
                    name="hiddenweights",
                    initializer=self.W_h_initializer,
                )
                self.W_h = tf.scalar_mul(0.05, self.W_h)

                # A (fast weights) [shape (batch_size, units, units)]
                self.A = self.add_weight(
                    shape=(self.batch, self.units, self.units),
                    name="fastweights",
                    initializer=self.A_initializer,
                )

                if self.use_bias:
                    self.bias = self.add_weight(
                        shape=(self.units,),
                        name="bias",
                        initializer=self.b_x_initializer,
                    )
                else:
                    self.bias = None
                self.built = True

            def call(self, inputs, states, training=None):
                prev_output = states[0] if nest.is_sequence(states) else states

                # Next hidden state h(t+1) is computed in two steps:
                # Step 1 calculate preliminary vector: h_0(t+1) = f(W_h ⋅ h(t) + C ⋅ x(t))
                h = K.dot(prev_output, self.W_h) + K.dot(inputs, self.C)
                if self.bias is not None:
                    h = h + self.bias
                if self.activation is not None:
                    h = self.activation(h)

                # Reshape h to use with a
                h_s = tf.reshape(h, [self.batch, 1, self.units])

                # Define preliminary vector in variable
                prelim = tf.reshape(
                    K.dot(prev_output, self.W_h), (h_s.shape)
                ) + tf.reshape(K.dot(inputs, self.C), (h_s.shape))

                # Fast weights update rule: A(t) = λ*A(t-1) + η*h(t) ⋅ h(t)^T
                self.A.assign(
                    tf.math.add(
                        tf.scalar_mul(self.l, self.A),
                        tf.scalar_mul(
                            self.e, tf.linalg.matmul(tf.transpose(h_s, [0, 2, 1]), h_s)
                        ),
                    )
                )

                # Step 2: Initiate inner loop with preliminary vector, which runs for S steps
                # to progressively change the hidden state into h(t+1) = h_s(t+1)
                # h_s+1(t+1) f([W_h ⋅ h(t) + C ⋅ x(t)]) + A(t)h_s(t+1)
                for _ in range(self.step):
                    h_s = tf.math.add(prelim, tf.linalg.matmul(h_s, self.A))
                    if self.activation is not None:
                        h_s = self.activation(h_s)

                    # Apply layer normalization on hidden state
                    h_s = self.LN(h_s)

                h = tf.reshape(h_s, [self.batch, self.units])

                output = h
                new_state = [output] if nest.is_sequence(states) else output
                return output, new_state

            def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
                return _generate_zero_filled_state_for_cell(
                    self, inputs, batch_size, dtype
                )

            def get_config(self):
                config = {
                    "units": self.units,
                    "step": self.step,
                    "batch_size": self.batch,
                    "use_bias": self.use_bias,
                    "activation": activations.serialize(self.activation),
                    "decay_rate": self.l,
                    "learning_rate": self.e,
                    "LN": self.LN,
                }
                base_config = super(FW_RNNCell, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
        
        # Load models (if FWRNN load it with custom objects fw RNNcel)
        if model == "FWRNN":
            NN = tf.keras.models.load_model(
                filepath=f"data/models/models_with_extractedfeatures_vgg19block5/{model}_{num_units}units.h5",
                custom_objects={"FW_RNNCell": FW_RNNCell},
                compile=True,
            )
        else:
            NN = tf.keras.models.load_model(
                filepath=f"data/models/models_with_extractedfeatures_vgg19block5/{model}_{num_units}units.h5",
                compile=True,
            )
        
        # Check model summary
        NN.summary()

        #####################################################################################
        #         # Evaluate on validation set to get validation scores
        #         csvlog_AW2_val = tf.keras.callbacks.CSVLogger(
        #             f"data/models/val_scores/{model}_{num_units}units_AW2_validation_scores.csv",
        #             separator=",",
        #             append=False,
        #         )
        #         NN.evaluate(
        #             X_val, y_val, batch_size=batchsize, callbacks=[csvlog_AW2_val],
        #         )

        #         # Get F1 scores for validation set
        #         val_pred = NN.predict(X_val, verbose=0)
        #         val_pred = np.reshape(
        #             val_pred, (val_pred.shape[0] * val_pred.shape[1], val_pred.shape[2])
        #         )
        #         # Convert one hot encoding to integers
        #         val_pred = np.argmax(val_pred, axis=1)

        #         # Reshape back to (frame, label)
        #         val_true = np.reshape(
        #             y_val, (y_val.shape[0] * y_val.shape[1], y_val.shape[2],),
        #         )
        #         val_true = np.argmax(val_true, axis=1)
        #         f1scores_val = {
        #             avg: f1_score(val_true, val_pred, average=avg) for avg in [None, "macro"]
        #         }
        #         f1scores_val[None] = f1scores_val.get(None).tolist()
        #         print(f1scores_val)

        #         with open(
        #             f"data/models/val_scores/{model}_{num_units}units_AW2_validation_F1scores.json",
        #             "w",
        #         ) as fp:
        #             json.dump(f1scores_val, fp)

        #####################################################################################
        # Evaluate on test set of AW2 to get test scores
        csvlog_AW2_test = tf.keras.callbacks.CSVLogger(
            f"data/models/test_scores/{model}_{num_units}units_AW2_test_scores.csv",
            separator=",",
            append=False,
        )
        NN.evaluate(
            X_test, y_test, batch_size=batchsize, callbacks=[csvlog_AW2_test],
        )

        # Get F1 scores for AW2 test set
        test_pred = NN.predict(X_test, verbose=0)
        test_pred = np.reshape(
            test_pred, (test_pred.shape[0] * test_pred.shape[1], test_pred.shape[2])
        )
        # Convert one hot encoding to integers
        test_pred = np.argmax(test_pred, axis=1)

        # Reshape back to (frame, label)
        test_true = np.reshape(
            y_test, (y_test.shape[0] * y_test.shape[1], y_test.shape[2],),
        )
        test_true = np.argmax(test_true, axis=1)

        f1scores_test = {
            avg: f1_score(test_pred, test_true, average=avg) for avg in [None, "macro"]
        }
        f1scores_test[None] = f1scores_test.get(None).tolist()
        print(f1scores_test)

        with open(
            f"data/models/test_scores/{model}_{num_units}units_AW2_test_F1scores.json",
            "w",
        ) as fp:
            json.dump(f1scores_test, fp)

        ##################################################################################

        # Evaluate on test set of AF7 to get test scores for cross-dataset performance
        csvlog_AF7 = tf.keras.callbacks.CSVLogger(
            f"data/models/test_scores/{model}_{num_units}units_AF7_test_scores.csv",
            separator=",",
            append=False,
        )
        NN.evaluate(
            X_test_AF7, y_test_AF7, batch_size=batchsize, callbacks=[csvlog_AF7],
        )

        # Get F1 scores for AF7 test set
        test_pred = NN.predict(X_test_AF7, verbose=0)
        test_pred = np.reshape(
            test_pred, (test_pred.shape[0] * test_pred.shape[1], test_pred.shape[2])
        )
        # Convert one hot encoding to integers
        test_pred = np.argmax(test_pred, axis=1)

        # Reshape back to (frame, label)
        test_true = np.reshape(
            y_test_AF7,
            (y_test_AF7.shape[0] * y_test_AF7.shape[1], y_test_AF7.shape[2],),
        )
        test_true = np.argmax(test_true, axis=1)

        f1scores_test = {
            avg: f1_score(test_pred, test_true, average=avg) for avg in [None, "macro"]
        }
        f1scores_test[None] = f1scores_test.get(None).tolist()
        print(f1scores_test)

        with open(
            f"data/models/test_scores/{model}_{num_units}units_AF7_test_F1scores.json",
            "w",
        ) as fp:
            json.dump(f1scores_test, fp)

