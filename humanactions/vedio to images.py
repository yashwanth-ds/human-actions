import cv2     # for capturing videos
import math
import numpy
# for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
#matplotlib inline
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
#from skimage.transform import resize   # for resizing images
count = 0
for i in range(1,100):
    videoFile = r"C:\\Users\\Admin\\Desktop\\spec\\KTH dataset\\handclapping\2i_%dv.avi"%i
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename =r"frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
            a=numpy.asarray(frame)
            a.tofile('foo.csv',sep=",",format='%10.5f')
            print(frame)
            print('----------------------------------------------')
    print('done')
    cap.release()

#cap.release()

