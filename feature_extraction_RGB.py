### Load/import packages
import time
import scipy.sparse
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential, layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg19 import preprocess_input

# Limit GPU memory usage
for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)


### Functions
def feature_extractor(model, data_dir, batch_size=128, test=False):

    # Define ImageDataGenerator with precoessing function set to preprocess_input for vgg19 model
    # Also using class_model categorical which automatically one-hot encodes the labels
    if test == False:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        generator = datagen.flow_from_directory(
            directory=data_dir,
            target_size=(112, 112),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        generator = datagen.flow_from_directory(
            directory=data_dir,
            target_size=(112, 112),
            color_mode="rgb",
            class_mode=None,
            batch_size=batch_size,
            shuffle=False,
        )
    
    # Grab all filenames in the generator
    filenames = generator.filenames

    # Reset batch index to 0 for the train_generator
    generator.reset()

    # Onehot encoding of labels
    if test == False:
        labels_arr = to_categorical(generator.classes)

    # Create list for the predictions, using list is more memory efficient then concatenating an array each time
    # when adding the new batch into it.
    features = []

    start = time.time()

    if test == False:
        for batch, label in generator:
            pred = model.predict(batch, verbose=0)
            features.append(scipy.sparse.csr_matrix(pred))

            print(
                f"Progress: {round(((generator.total_batches_seen/len(generator))*100),2)}%"
            )
            # Allow loop to break when it has looped over all the batches in the generator, else it will keep looping over them.
            if len(generator) == generator.total_batches_seen:
                break
    else:
        for batch in generator:
            pred = model.predict(batch, verbose=0)
            features.append(scipy.sparse.csr_matrix(pred))

            print(
                f"Progress: {round(((generator.total_batches_seen/len(generator))*100),2)}%"
            )
            # Allow loop to break when it has looped over all the batches in the generator, else it will keep looping over them.
            if len(generator) == generator.total_batches_seen:
                break

    # Stack features in scipy CSR matrix
    features_arr = scipy.sparse.vstack(features)

    seconds = round(time.time() - start, 2)
    print(f"Finished with extraction, Execution duration: {seconds//60}m:{seconds%60}s")

    if test == False:
        return features_arr, labels_arr, filenames
    else:
        return features_arr, filenames


## Initiate Base CNN
# For the feature extraction the pre-trained **VGG19** network will be used with the imagenet weights. Input shape is set to 112,112,3. The top is not included because we only want to extract features, so we remove the classification layers.
base_VGG19 = tf.keras.applications.VGG19(
    include_top=False, weights="imagenet", input_shape=(112, 112, 3)
)
base_VGG19.summary()

## Initiate Feature Extraction model
# We add a flatten layer to the base VGG19 layer to just get a simple 1-Dimensional feature vector as output for our RNN/LSTM as input
def build_FE_model():
    model = Sequential()
    for layer in base_VGG19.layers:
        model.add(layer)
    model.add(layers.Flatten(name="Flatten"))
    return model

FE_model = build_FE_model()
FE_model.summary()

# Plot model
tf.keras.utils.plot_model(
    FE_model,
    to_file=f"data/model_architectures/FE_model.jpg",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)

## Extract all Features with FE model1 - AFF-Wild2
AW2_train_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_per_class_RGB\train"
AW2_val_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_per_class_RGB\val"

# In this directory each video folder contains all its frames
AW2_test_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_RGB\test"

### Extract Training set Features

# Extract training features and labels
train_features, train_labels, train_filenames = feature_extractor(
    FE_model, AW2_train_dir
)

# Save features as NPZ Numpy’s compressed array format and labels as numpy
scipy.sparse.save_npz("data/features/train_features_RGB_AW2.npz", train_features)
np.save("data/labels/train_labels_RGB_AW2.npy", train_labels)
with open("data/filenames/train_filenames_RGB_AW2.txt", "w") as fp:
    fp.write("\n".join(train_filenames))

### Extract Validation set Features
# Extract validation features and labels
val_features, val_labels, val_filenames = feature_extractor(FE_model, AW2_val_dir)

# Save features as NPZ Numpy’s compressed array format and labels as numpy
scipy.sparse.save_npz("data/features/val_features_RGB_AW2.npz", val_features)
np.save("data/labels/val_labels_RGB_AW2.npy", val_labels)
with open("data/filenames/val_filenames_RGB_AW2.txt", "w") as fp:
    fp.write("\n".join(val_filenames))

### Extract Test set Features
test_features, test_filenames = feature_extractor(FE_model, AW2_test_dir, test=True)

# Save features as NPZ Numpy’s compressed array format
scipy.sparse.save_npz("data/features/test_features_RGB_AW2.npz", test_features)
with open("data/filenames/test_filenames_RGB_AW2.txt", "w") as fp:
    fp.write("\n".join(test_filenames))

## Extract all Features with FE model1 - AFEW 7.0
AF7_dir = r"D:\AFEW 7.0 Dataset\Val+train_faces"

# Extract train+val features and labels
AF7_features, AF7_labels, AF7_filenames = feature_extractor(FE_model, AF7_dir)

# Save features as NPZ Numpy’s compressed array format and labels as numpy
scipy.sparse.save_npz("data/features/features_RGB_AF7.npz", AF7_features)
np.save("data/labels/labels_RGB_AF7.npy", AF7_labels)
with open("data/filenames/filenames_RGB_AF7.txt", "w") as fp:
    fp.write("\n".join(AF7_filenames))

