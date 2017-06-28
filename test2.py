import cv2
import numpy as numpy
import os
from matplotlib import pyplot as plt


os.chdir("/home/musigma/Desktop/")
img = cv2.imread('img2.jpg',0)
# ret_bin_gauss,thresh_img_bin_gauss = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# ##Edge detection
# edges = cv2.Canny(thresh_img_bin_gauss,100,200)
# edges_orig = cv2.Canny(img,100,200)


# plt.subplot(131),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(edges,cmap = 'gray')
# plt.title('Gaussian edge'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(edges_orig,cmap = 'gray')
# plt.title('Regular edge'), plt.xticks([]), plt.yticks([])
# plt.show()
#############-------------------GLOBAL VALUES-----------------#################
##cv2.threshold
##img is the source image (in grayscale)
###Second argument 127 is used to classify the pixel values
####Third argument 255 is the maxVal which represents the value to be given if pixel value > (or sometimes less than)
#####the threshold value
##### there are 4 different types of thresholding..thresh_binary,binary_inv,
######thresh_trunc, thresh_tozero,thresh_tozero_inv 
# ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)


# cv2.imshow('img',thresh_img)
# cv2.imwrite("thresh_binary_inv.jpg", thresh_img)
# cv2.destroyAllWindows()

# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# #ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# #ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# #ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# titles = ['Original Image','BINARY','TRUNC']
# images = [img, thresh1, thresh2]
# for i in xrange(3):
#     plt.subplot(1,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


#############-------------------adaptive threshold-----------------#################
# img = cv2.medianBlur(img,5)
# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# for i in xrange(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()



###########Gaussian filtering and then binary#################
# global thresholding
# ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # Otsu's thresholding after Gaussian filtering
# blur = cv2.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # plot all the images and their histograms
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#           'Original Noisy Image','Histogram',"Otsu's Thresholding",
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# for i in xrange(3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()


############BACKGROUND SUBTRACTION###########
import numpy as np
import cv2
 
cap = cv2.VideoCapture('vtest.avi')
 
fgbg = cv2.createBackgroundSubtractorMOG()
 
while(1):
 ret, frame = cap.read()

 fgmask = fgbg.apply(frame)

 cv2.imshow('frame',fgmask)
 k = cv2.waitKey(30) & 0xff
 if k == 27:
 	break

cap.release()
cv2.destroyAllWindows()
