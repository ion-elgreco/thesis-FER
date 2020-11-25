### Load/import packages
import ffmpeg
import cv2
import numpy as np
import tensorflow as tf
# tf.__version__ 1.1.4

from os import mkdir
from os.path import join, splitext
from skimage import io, exposure, color
from skimage.transform import resize

# Import MTCNN for face detection (method 1)
from mtcnn.mtcnn import MTCNN



# ## Directories
# ### Aff-Wild2
from load_filenames import (AF7_dir_videos, 
                            AF7_dir_labels, 
                            AF7_labeled_videos_FN)

### Start Pre-processing

#assign the MTCNN detector
detector = MTCNN()

# Define extract face function
def extract_face(frame):              
    f = detector.detect_faces(frame)
    # If the face is not found return the empty list
    if not f:
        return f
    #fetching the (x,y)co-ordinate and (width-->w, height-->h) of the image
    x1,y1,w,h = f[0]['box']             
    x1, y1 = abs(x1), abs(y1)
    x2 = abs(x1+w)
    y2 = abs(y1+h)
    
    #locate the co-ordinates of face in the image
    face = frame[y1:y2,x1:x2]
    face = resize(face, (112,112,3))
    return face


# # Only face detection with MTCNN
# Define directory to save the frames
frame_dir = r'D:\AFEW 7.0 Dataset\Val+train_faces'

for label in AF7_dir_labels:
    try:
        mkdir(join(frame_dir, label))
    except FileExistsError:
        print(f"{label} Directory already exists!")
    
    videos = AF7_labeled_videos_FN.get(label)
    for video in videos:
        fn, ext = splitext(video)
        
        video_dir = join(join(frame_dir, label) , video)
        probe = ffmpeg.probe(join(AF7_dir_videos, join(label, video)))
        video_info = next(x for x in probe["streams"] if x["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])

        out, _ = (
            ffmpeg.input(join(AF7_dir_videos, join(label, video)))
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True)
        )
        frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        
        frame_n = 0
        for frame in frames:
            frame_n +=1
            face = extract_face(frame)
            # If the face is not found, continue in the loop
            if face == []:
                continue
            io.imsave(join(join(frame_dir, label), f'{fn}_{frame_n:05n}'+'.jpg'), (face*255).astype('uint8'), check_contrast=False)
        