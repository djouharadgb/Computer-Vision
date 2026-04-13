import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
imgL = cv.imread(os.path.join(script_dir, 'scene1.png'), 0)
imgR = cv.imread(os.path.join(script_dir, 'scene2.png'), 0)

if imgL is None or imgR is None:
    raise FileNotFoundError("Could not load scene1.png or scene2.png")

stereo = cv.StereoBM_create(numDisparities=16,
                             blockSize=15)
#Parameters
#numDisparities  the disparity search range. 
#For each pixel algorithm will find the best disparity from 0 (default minimum 
#disparity) to numDisparities. The search range can then be shifted by changing 
#the minimum disparity.
#blockSize   the linear size of the blocks compared by the algorithm. 
#The size should be odd (as the block is centered at the current pixel). 
#Larger block size implies smoother, though less accurate disparity map. 
#Smaller block size gives more detailed disparity map, but there is higher chance 
#for algorithm to find a wrong correspondence.

disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity)
#plt.imshow(disparity,'gray')
plt.show()