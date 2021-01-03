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

# Limit GPU memory usage
for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

# load Aff-Wild2 Features
X_train = scipy.sparse.load_npz("data/features/train_features_RGB_AW2.npz") #CSR Matrix
X_val = scipy.sparse.load_npz("data/features/val_features_RGB_AW2.npz") #CSR Matrix
y_train = np.load("data/labels/train_labels_RGB_AW2.npy") #Numpy array
y_val = np.load("data/labels/val_labels_RGB_AW2.npy") #Numpy array

# Load AFEW7.0 features
X_test_AF7 = scipy.sparse.load_npz("data/features/features_RGB_AF7.npz") #CSR Matrix
y_test_AF7 = np.load("data/labels/labels_RGB_AF7.npy") #Numpy array

# Split the validation set into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5, random_state = 1234, shuffle=True)


## Prepare data
train_label_ints = np.argmax(y_train, axis=1) 
class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(train_label_ints), train_label_ints.flatten()
    )
class_weights = {i : class_weights[i] for i in range(len(class_weights))}


## Define and train model
model = Sequential()
model.add(layers.InputLayer(input_shape=4608,))
model.add(layers.Dense(32))
model.add(layers.Dense(7, activation="softmax", name="Dense_Output"))
model.compile(optimizer="adam", loss=CategoricalCrossentropy(), metrics=["accuracy"])

model.summary()

# Fit model to training set and evaluate
batchsize = 512
history = model.fit(
    train_features_AW2,
    train_labels_AW2,
    batch_size=batchsize,
    class_weight=class_weights,
    validation_data=(val_features_AW2, val_labels_AW2),
    epochs=5,
    verbose=1,
    shuffle=True,
)

# Save model
tf.keras.Model.save(
    model,
    filepath=f"data/models/models_with_extractedfeatures_vgg19block5/FCN.h5",
)


### Evaluate on test sets
NN = tf.keras.models.load_model(
    filepath=f"data/models/models_with_extractedfeatures_vgg19block5/FCN.h5",
    compile=True,
)

# Check model summary
NN.summary()

# Evaluate on test set of AW2 to get test scores
csvlog_AW2_test = tf.keras.callbacks.CSVLogger(
    f"data/models/test_scores/FCN_AW2_test_scores.csv",
    separator=",",
    append=False,
)
NN.evaluate(
    X_test, y_test, batch_size=batchsize, callbacks=[csvlog_AW2_test],
)

# Get F1 scores for AW2 test set
test_pred = NN.predict(X_test, verbose=0)

# Convert one hot encoding to integers
test_pred = np.argmax(test_pred, axis=1)

# Reshape back to (frame, label)
test_true = np.argmax(y_test, axis=1)

f1scores_test = {
    avg: f1_score(test_pred, test_true, average=avg) for avg in [None, "macro"]
}
f1scores_test[None] = f1scores_test.get(None).tolist()
print(f1scores_test)

with open(
    f"data/models/test_scores/FCN_AW2_test_F1scores.json", "w",
) as fp:
    json.dump(f1scores_test, fp)

##################################################################################

# Evaluate on test set of AF7 to get test scores for cross-dataset performance
csvlog_AF7 = tf.keras.callbacks.CSVLogger(
    f"data/models/test_scores/FCN_AF7_test_scores.csv",
    separator=",",
    append=False,
)
NN.evaluate(
    X_test_AF7, y_test_AF7, batch_size=batchsize, callbacks=[csvlog_AF7],
)

# Get F1 scores for AF7 test set
test_pred = NN.predict(X_test_AF7, verbose=0)

# Convert one hot encoding to integers
test_pred = np.argmax(test_pred, axis=1)

# Reshape back to (frame, label)
test_true = np.argmax(y_test_AF7, axis=1)

f1scores_test = {
    avg: f1_score(test_pred, test_true, average=avg) for avg in [None, "macro"]
}
f1scores_test[None] = f1scores_test.get(None).tolist()
print(f1scores_test)

with open(
    f"data/models/test_scores/FCN_AF7_test_F1scores.json", "w",
) as fp:
    json.dump(f1scores_test, fp)

