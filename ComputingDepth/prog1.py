import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
imgL = cv2.imread(os.path.join(script_dir, 'scene2.png'), 0)
imgR = cv2.imread(os.path.join(script_dir, 'scene3.png'), 0)

if imgL is None or imgR is None:
    raise FileNotFoundError("Could not load scene2.png or scene3.png")

window_size = 5
stereo = cv2.StereoSGBM_create()

# StereoSGBM is the implementation of Hirschmüller’s original SGM [2] algorithm. 
# SGBM stands for Semi-Global Block Matching. It also implements the sub-pixel 
#estimation proposed by Brichfield et al. [3]
#[2] Hirschmüller, Heiko (2005). “Accurate and efficient stereo processing 
#by semi-global matching and mutual information”. 
#IEEE Conference on Computer Vision and Pattern Recognition. pp. 807–814.

#[3] . Birchfield and C. Tomasi, “Depth discontinuities by pixel-to-pixelstereo,
#”International Journal of Computer Vision, vol. 35, no. 3,pp. 269–293, 1999.

disparity = stereo.compute(imgL, imgR)

plt.imshow(disparity, 'gray')
plt.show()