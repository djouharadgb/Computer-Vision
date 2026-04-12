# import computer vision library(cv2) in this code
import cv2

img_path = "im1.jpg"
image_c = cv2.imread(img_path)
image = cv2.cvtColor(image_c,cv2.COLOR_BGR2GRAY)
n = 10
#image=image_c

# mentioning absolute path of the image
#cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
#borderType: This specify boundaries of an image while kernel is applied 
#on borders of an image.
# cv2.BORDER_DEFAULT: gfedcb|abcdefgh|gfedcba4
blur_img1 = cv2.GaussianBlur(image, (3,3),2,0, cv2.BORDER_DEFAULT)

for i in range(1, n + 1):

  blur_img1 = cv2.GaussianBlur(blur_img1, (3,3),2,0, cv2.BORDER_DEFAULT)
  print(f"Iteration {i} : Convolution appliquée")
  cv2.imshow('Blur image1',blur_img1)
  cv2.waitKey(0)


cv2.imshow('Blur image1',blur_img1)
cv2.waitKey(0)
cv2.imwrite("im1b.png", blur_img1)
cv2.waitKey(0)
