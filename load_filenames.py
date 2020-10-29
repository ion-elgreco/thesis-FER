from os import listdir
from os.path import splitext
from os.path import join

### Acronyms
# Aff-Wild2 = AW2
# AFEW 7.0  = AF7
# filenames = FN

### Aff-wild2 ###
# Directories for labels/annotations for each set
AW2_dir_train_labels = (r"D:\Aff-Wild2 Dataset\Aff-wild2\Videos\annotations\EXPR_Set\Training_Set")
AW2_dir_val_labels = (r"D:\Aff-Wild2 Dataset\Aff-wild2\Videos\annotations\EXPR_Set\Validation_Set")

# Directories for videos
AW2_dir_allvideos = r"D:\Aff-Wild2 Dataset\Aff-wild2\Videos\all_videos"

# Save the filenames in a list for each set, and split the filename from the extension
AW2_train_FN = listdir(AW2_dir_train_labels)
AW2_train_FN_split = [splitext(file) for file in AW2_train_FN]

AW2_val_FN = listdir(AW2_dir_val_labels)
AW2_val_FN_split = [splitext(file) for file in AW2_val_FN]

AW2_videos_FN = listdir(AW2_dir_allvideos)
AW2_videos_FN_split = [splitext(file) for file in AW2_videos_FN]


### AFEW 7.0 ###
# Directories for labels/annotations for each set
AF7_dir_videos = r"D:\AFEW 7.0 Dataset\Val+train_AFEW"

# Save the foldernames (class) in a list 
AF7_dir_labels = listdir(AF7_dir_videos) 

# Create dictionary with class as key, and videonames as values
AF7_labeled_videos_FN = {}
for label in AF7_dir_labels:
    videos = listdir(join(AF7_dir_videos, label))
    AF7_labeled_videos_FN[label] = videos
