### Load/import packages
import json
import scipy
import numpy as np
import scipy.sparse
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Sequential, layers
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Import fwRNN cell
from fwrnn_cell import FW_RNNCell

# Limit GPU memory usage
for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

# load Aff-Wild2 Features
train_features_AW2 = scipy.sparse.load_npz("data/features/train_features_RGB_AW2.npz") #CSR Matrix
val_features_AW2 = scipy.sparse.load_npz("data/features/val_features_RGB_AW2.npz") #CSR Matrix
train_labels_AW2 = np.load("data/labels/train_labels_RGB_AW2.npy") #Numpy array
val_labels_AW2 = np.load("data/labels/val_labels_RGB_AW2.npy") #Numpy array

# Load AFEW7.0 features
test_features_AF7 = scipy.sparse.load_npz("data/features/features_RGB_AF7.npz") #CSR Matrix
test_labels_AF7 = np.load("data/labels/labels_RGB_AF7.npy") #Numpy array


### Functions
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


## Prepare data
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

X_test_AF7 = features_reshaper(test_features_AF7, length)
del test_features_AF7  # wipe out of memory to free up space

y_test_AF7 = labels_reshaper(test_labels_AF7, length)
del test_labels_AF7  # wipe out of memory to free up space


# Split the validation set into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5, random_state = 1234, shuffle=True)

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

# Build model with model subclassing and sequential API
def build_FWRNN(batch, units, activation_function):
    # Define model
    model = Sequential(name="FW-RNN")
    model.add(layers.InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])))
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

# Build baseline model (RNN or LSTM) with sequential API
def build_base(model_name, units):
    model = Sequential(name=model_name)
    model.add(layers.InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])))
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

# Set early stopping
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0025,
    patience=6,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

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
        print(model, num_units)
        
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


### Evaluate on test sets
# Disivible length of test_AW2 and test_AF7 features by batchsize, necessary because fwRNN batch size is fixed
test_div_AW2 = (X_test.shape[0] // batchsize) * batchsize
test_div_AF7 = (X_test_AF7.shape[0] // batchsize) * batchsize

# Slice the sets to fit the batch size
X_test = X_test[:test_div_AW2]
y_test = y_test[:test_div_AW2]

X_test_AF7 = X_test_AF7[:test_div_AF7]
y_test_AF7 = y_test_AF7[:test_div_AF7]


for num_units in [5, 20, 50, 100]:
    for model in ["RNN", "LSTM", "FWRNN"]:
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

