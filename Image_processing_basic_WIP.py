import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

##Change directory to desktop
os.chdir("/home/user_name/Desktop/")
img = cv2.imread('img2.jpg',0)
img2 = cv2.imread('img2.jpg',1)
#img1 = cv2.imread('img1.jpg',0)
##Global threshold value
##ret_bin,thresh_img_bin = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
##ret_trunc,thresh_img_trunc = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)


##Gaussian filtering to get threshold value for converting to binary (0 is default)
##2 versions for binary truncated and regular

# ret_bin_gauss,thresh_img_bin_gauss = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# ret_trunc_gauss,thresh_img_tunc_gauss = cv2.threshold(img,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)

##Edge detection using canny
##First argument is the image . 2nd and 3rd are minVal and maxVal respectively
##Can use trial and error to get the optimal values

# edges_orig = cv2.Canny(img,100,200)


##Plotting all 4 images on the same window 

# titles = ['Original Image','GAUSS BI','GAUSS TR', 'EDGES']
# images = [img,thresh_img_bin_gauss,thresh_img_tunc_gauss,edges_orig]
# for i in xrange(4):
#     plt.subplot(1,4,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.savefig('/home/user_name/Desktop/image_process_basics.png')

##Background subtraction  

#####BackgroundSubtractorMOG#####
# _,cap = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# _1,cap1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# _,contours,hierarchy = cv2.findContours(cap,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# _1,contours1,hierarchy1 = cv2.findContours(cap1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# i=0
# cnt = contours[i]
# cnt1 = contours1[i]
# area = cv2.contourArea(cnt)
# area1 = cv2.contourArea(cnt1)

# if area<=1.0:
# 	cnt = contours[i+1]
# if area1<=1.0:
# 	cnt1 = contours1[i+1]

# area = cv2.contourArea(cnt)
# area1 = cv2.contourArea(cnt1)
	
# x,y,w,h= cv2.boundingRect(cnt)
# x1,y1,w1,h1= cv2.boundingRect(cnt1)

# crop = img[y:y+h,x:x+w]
# crop1 = img1[y1:y1+h1,x1:x1+w1]

# cv2.imwrite('image2.png',crop)
# cv2.imwrite('image1.png',crop1)



###CONTOURS FOR IMAGE SEGMENTAITON####

ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

###First argument img is the source of image
###Second is the countours which should be passed as python list
###Third is index of contours (to draw all contours pass -1)
####remaining are color and thickness

mask2 = cv2.drawContours(thresh, contours, 0, (255,0,0), -1)

masked_data = cv2.bitwise_and(img,img, mask = mask2)

b,g,r = cv2.split(img2)
rgba = [b,g,r, thresh]
dst = cv2.merge(rgba,4)
#idx=1
#print(contours)
#print(type(contours))
# x,y,w,h = cv2.boundingRect(dst)
# roi=dst[y:y+h,x:x+w]
#cv2.imwrite(str(idx) + '.jpg', dst)
cv2.imwrite('image2sfdsf.png',dst)
