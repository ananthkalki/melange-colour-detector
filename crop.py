from PIL import Image
import cv2
import imutils

image = cv2.imread(r".jpg")
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = imutils.grab_contours(cnts)


image2=Image.open('cloth.jpg')
peri = cv2.arcLength(cnts, True)
approx = cv2.approxPolyDP(c, 0.04 * peri, True)
(x, y, w, h) = cv2.boundingRect(approx)

croppedIm = image2.crop((x,y,x+w,y+h))

croppedIm.save('cropped.jpg')