#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Load/import packages
import json
import time
import shutil
import numpy as np

from os import listdir, mkdir, rename
from os.path import join, splitext
from skimage import io, color, exposure

# Import directories and filenames from own function
from load_filenames import (
    AW2_cropped_aligned_dir,  # 'D:\\Aff-Wild2 Dataset\\Aff-wild2\\Images\\cropped_aligned'
    AW2_cropped_aligned_folders,  # Returns foldernames
    AW2_train_FN_split,  # Returns videonames with
    AW2_val_FN_split,
)


# In[2]:


# Steps in this script
# 1. Apply illumination normalization (histogram equalization)
# 2. Store the frames (images) in a named folder same name as video. 
# 3. Move all files to the corresponding train, validation, test directory
# 4. Put each corresponding frame in the correct class folder for training and validiation set.
#    This is required for keras to read the image data from the directory in batches and assign the classes
#    inferred from the directory structure.


# In[15]:


# Pre-processing function

# Each image has the shape (112, 112, 3)
# The images are adaptive histogram equalized, in RGB or grayscale. 
# If its grayscale it afterwards gets converted back to RGB but remains the grayscale color. 
# It's also converted back to values ranging from 0-255

def pre_processing(RGB=True):
    # Set timer
    start = time.time()
    if RGB:
        AW2_normalized_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Images\cropped_aligned_normalized_RGB"
    else:
        AW2_normalized_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Images\cropped_aligned_normalized"
    
    foldercount = 0
    for folder in AW2_cropped_aligned_folders:
        print(f'Processing this folder: {folder}, Progress: {foldercount/len(AW2_cropped_aligned_folders)*100:.2f}%')
        foldercount+=1
        # Define the folder path to each video to grab the frames (images)
        folder_dir = join(AW2_cropped_aligned_dir, folder)

        # Create list of all the filenames of the corresponding folder
        frames_FN = listdir(folder_dir)

        
        # Try to create a folder with the name of the video to save all the normalized frames
        try:
            mkdir(join(AW2_normalized_dir, folder))
        except FileExistsError:
            print("Directory already exists!")

        for frame_name in frames_FN:
            # Added if statement, to skip this step if it sees this .ds_store file
            if frame_name == ".DS_Store":
                continue
            # Create path to save the normalized file
            file = join(join(AW2_normalized_dir, folder), frame_name)
            
            # Read img, convert it to grayscale else you cant adaptive histogram equalize it.
            frame = io.imread(join(folder_dir, frame_name))
            if RGB:
                frame_hist = exposure.equalize_adapthist(frame)
            else:
                frame_g = color.rgb2gray(frame)
                frame_hist = exposure.equalize_adapthist(frame_g)
                # Convert it back to 3 channels, because VGG19 model requirs 3channel images as input
                frame_hist = color.gray2rgb(frame_hist)
            
            # Save the file
            io.imsave(file, (frame_hist * 255).astype(np.uint8))
        print(
            f"Done! Time ran since start: {round(time.time()-start)//60 }:{round(time.time()-start)%60}"
        )
    print('Finished pre-processing')
    
    print('Start moving files')
    ### Move images in their correct train, val and test set directory
    # Set directories to move the images to
    if RGB:
        train_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_RGB\train"
        val_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_RGB\val"
        test_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_RGB\test"
    else:
        train_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets\train"
        val_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets\val"
        test_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets\test"

    # Load Aff-Wild2 AW2_train_classes.json
    with open("data/AW2_train_classes.json", "r") as fp:
        AW2_train_classes = json.load(fp)

    # Load Aff-Wild2 AW2_val_classes.json
    with open("data/AW2_val_classes.json", "r") as fp:
        AW2_val_classes = json.load(fp)

    # Move training files
    for folder, ext in AW2_train_FN_split:
        shutil.move(join(AW2_normalized_dir, folder), train_dir)

    for folder, ext in AW2_val_FN_split:
        shutil.move(join(AW2_normalized_dir, folder), val_dir)

    for folder in listdir(AW2_normalized_dir):
        shutil.move(join(AW2_normalized_dir, folder), test_dir)


    ### Put each corresponding frame in the correct class folder for training and validiation set. 
    ### This is necessary to create a dataset in keras
    
    if RGB:
        train_perclass_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_per_class_RGB\train"
    else:
        # Extract image from each videofolder and move it to the corresponding class folder
        train_perclass_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_per_class\train"
    label_folders = listdir(train_perclass_dir)

    for folder in listdir(train_dir):
        ##e.g. returns folder: 51-30-1280x720

        # Gets labels from the corresponding video from the AW2_train_classes dict
        labels = AW2_train_classes.get(folder + ".txt")

        # Loop through each frame, and take the corresponding label for this frame from labels
        for file in listdir(join(train_dir, folder)):
            frame_n, ext = splitext(file)
            frame_n = int(frame_n)
            label = labels[(frame_n - 1)]

            # If the label is between 0-6, create corresponding new directory
            if label in [0, 1, 2, 3, 4, 5, 6]:
                new_dir = join(train_perclass_dir, label_folders[label])
            else:
                continue
            shutil.copy(join(train_dir, join(folder, file)), new_dir)
            rename(join(new_dir, file), join(new_dir, folder + "_" + file))


    # Extract image from each videofolder and move it to the corresponding class folder
    if RGB:
        val_perclass_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_per_class_RGB\val"
    else:
        val_perclass_dir = r"D:\Aff-Wild2 Dataset\Aff-wild2\Sets_per_class\val"
    label_folders = listdir(val_perclass_dir)

    for folder in listdir(val_dir):
        ##e.g. returns folder: 51-30-1280x720
        # Gets labels from the corresponding video from the AW2_train_classes dict
        labels = AW2_val_classes.get(folder + ".txt")

        # Loop through each frame, and take the corresponding label for this frame from labels
        for file in listdir(join(val_dir, folder)):
            frame_n, ext = splitext(file)
            frame_n = int(frame_n)
            label = labels[(frame_n - 1)]

            # If the label is between 0-6, create corresponding new directory
            if label in [0, 1, 2, 3, 4, 5, 6]:
                new_dir = join(val_perclass_dir, label_folders[label])
            else:
                continue
            shutil.copy(join(val_dir, join(folder, file)), new_dir)
            rename(join(new_dir, file), join(new_dir, folder + "_" + file))
    print('Finished moving files')


# In[ ]:


pre_processing()

