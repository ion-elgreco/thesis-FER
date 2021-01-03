#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# # Initiate Base CNN
# For the feature extraction the pre-trained **VGG19** network will be used with the imagenet weights. Input shape is set to 112,112,3. The top is not included because we only want to extract features, so we remove the classification layers.

# ## Load/import packages

# In[3]:


base_VGG19 = tf.keras.applications.VGG19(
    include_top=False, weights="imagenet", input_shape=(112, 112, 3)
)


# In[4]:


base_VGG19.summary()


# # Initiate Feature Extraction model

# In[5]:


# Remove block 5 from FE model 
# (Source: Ravi, A. (2018). Pre-Trained Convolutional Neural Network Features for Facial Expression Recognition. ArXiv:1812.06387 [Cs]. 
#  Retrieved from http://arxiv.org/abs/1812.06387)

# We add a flatten layer to the base VGG19 layer to just get a simple 1-Dimensional feature vector as output for our RNN/LSTM as input
def build_FE_model():
    model = Sequential()
    for layer in base_VGG19.layers[:-5]:
        model.add(layer)
    model.add(layers.Flatten(name="Flatten"))
    return model

FE_model = build_FE_model()
FE_model.summary()


# # Extract all Features with FE model1 - AFF-Wild2

# In[6]:


AW2_train_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_per_class_RGB\train"
AW2_val_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_per_class_RGB\val"

# In this directory each video folder contains all its frames
AW2_test_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_RGB\test"


# In[7]:


def feature_extractor(model):
    # Define index to extract in parts
    for batch, labels in generator:
        pred = model.predict(batch, verbose=0)
        return pred, labels


# ## Extract Training set Features

# In[25]:


datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(
    directory=AW2_train_dir,
    target_size=(112, 112),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=60,
    shuffle=False,
)


# In[29]:


while len(generator) != generator.total_batches_seen:
    print(
        f"Progress: {round(((generator.total_batches_seen/len(generator))*100),2)}%, batch index: {generator.batch_index}"
    )
    features, labels = feature_extractor(FE_model)
    
    np.save(f"D:/block4/train_AW2/features/features_part{generator.total_batches_seen}.npy", features)
    np.save(f"D:/block4/train_AW2/labels/labels_part{generator.total_batches_seen}.npy", labels)


# ## Extract Validation set Features

# In[11]:


datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(
    directory=AW2_val_dir,
    target_size=(112, 112),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=60,
    shuffle=False,
)


# In[12]:


while len(generator) != generator.total_batches_seen:
    print(
        f"Progress: {round(((generator.total_batches_seen/len(generator))*100),2)}%, batch index: {generator.batch_index}"
    )
    features, labels = feature_extractor(FE_model)
    
    np.save(f"D:/block4/val_AW2/features/features_part{generator.total_batches_seen}.npy", features)
    np.save(f"D:/block4/val_AW2/labels/labels_part{generator.total_batches_seen}.npy", labels)


# ## Extract Test set Features
# 

# # Extract all Features with FE model1 - AFEW 7.0

# In[17]:


AF7_dir = r"D:\AFEW 7.0 Dataset\Val+train_faces"


# In[18]:


datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(
    directory=AF7_dir,
    target_size=(112, 112),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=60,
    shuffle=False,
)


# In[20]:


while len(generator) != generator.total_batches_seen:
    print(
        f"Progress: {round(((generator.total_batches_seen/len(generator))*100),2)}%, batch index: {generator.batch_index}"
    )
    features, labels = feature_extractor(FE_model)
    
    np.save(f"D:/block4/test_AF7/features/features_part{generator.total_batches_seen}.npy", features)
    np.save(f"D:/block4/test_AF7/labels/labels_part{generator.total_batches_seen}.npy", labels)

