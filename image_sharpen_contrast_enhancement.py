
import cv2
##Assertion failed is usually when there is no input image
##Sharpening an image (image contrast improvement)

def sharpen():
	###Make sure that the input images data is in unsigned 8 bit format
	my_image = cv2.cvtColor(my_image, cv2.CV_8U)
	##my_image is the raw input image and the cv2 thing is to convert it to 8 bit stuff
	height, width, n_channels = my_image.shape
	result = np.zeros(my_image.shape,my_image.dtype)
	#Return a new array of given shape and type, filled with zeros
	#shape is my_image.shape and type is like float etc.


	#We need to access multiple rows and columns which can be done by adding or subtracting 1 to the current center (i,j).
	#Then we apply the sum and put the new value in the Result matrix.
    for j in range  (1, height-1):
    	for i in range  (1, width-1):
        	for k in range  (0, n_channels):
            	sum = 5 * my_image[j, i, k] - my_image[j + 1, i, k] - my_image[j - 1, i, k]\
                 - my_image[j, i + 1, k] - my_image[j, i - 1, k];
            	if sum > 255:
                	sum = 255
            	if sum < 0:
                	sum = 0
            	result[j, i, k] = sum


##FIlter 2D function so that we don't have to use that loop stuff

#1. Define object that holds the mask(kernel)
#Type should be float
#This mask is for contrast enhancement / image sharpening
kernel = np.array([ [0,-1,0],
                    [-1,5,-1],
                    [0,-1,0] ],np.float32)

##Function call
K = cv2.filter2D(I, -1, kernel)
# ddepth = -1, means destination image has depth same as input image.
##I is the input image
##Read an image in grayscale with parameter 0