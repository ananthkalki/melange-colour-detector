from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76

def rotateImage90R(src):
        img = cv2.imread(src)
        #print(type(img))
        #print(img.shape)

        img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(src, img_rotate_90_clockwise)

def loadImages(src):
        print('')
        print("[INFO] loading images...")
        imagePaths = sorted(list(paths.list_images(src)))
        images = []

        for imagePath in imagePaths:
                image = cv2.imread(imagePath)
                images.append(image)

        return images

def stitchImages(images, dest):
        print("[INFO] stitching images...")
        stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
        (status, stitched) = stitcher.stitch(images)

        if status == 0:
                print("[INFO] stitching complete...")

                cv2.imwrite(dest, stitched)
                
                '''
                cv2.imshow("Stitched", stitched)
                cv2.waitKey(0)
                #'''

                rotateImage90R(dest)
                return True
        else:
                print("[INFO] image stitching failed ({})".format(status))
                return False


# Driver Code
args = {}
args['images'] = 'input'# input('Source folder:')
args['output'] = 'output'# input('Dest/file:')
#print(args)

print('Source     : ', args['images'])
print('Destination: ', args['output'])
count = 2#int(input('Enter no. of rows: '))

images = loadImages(args['images']+'/raw')
#print(images)

if(count == 3):
        print('')
        print('Row 1:')
        stitchImages(images[:3], args['images']+'/row/1.jpg')
        print('')
        print('Row 2:')
        stitchImages(images[3:6], args['images']+'/row/2.jpg')
        print('')
        print('Row 3:')
        stitchImages(images[6:], args['images']+'/row/3.jpg')
elif(count == 2):
        print('')
        print('Row 1:')
        stitchImages(images[:2], args['images']+'/row/1.jpg')
        print('')
        print('Row 2:')
        stitchImages(images[2:], args['images']+'/row/2.jpg')

images = loadImages(args['images']+'/row')
#print(images)

print('')
print('Final Image:')
stitchImages(images, args['output']+'/output.png')

rotateImage90R(args['output']+'/output.png')
rotateImage90R(args['output']+'/output.png')

image = cv2.imread('output/output.png')
cv2.imshow("Composite Image",image)
cv2.waitKey(0)

#-----------------------------------------------------------------------------
#
#       Square Detection - Cropping
#
#-----------------------------------------------------------------------------

print('')
print('Cropping Image...')
image = cv2.imread('output/output.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
image_number=0				

min_area = 160000			
for c in cnts:
    area = cv2.contourArea(c)
    if area > min_area :
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+h]
        cv2.imwrite('output/detect{}.jpg'.format(image_number), ROI)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        image_number += 1

print('Cropping complete...')
cv2.imshow("Cloth", ROI)

#-----------------------------------------------------------------------------
#
#       Color Detection
#
#-----------------------------------------------------------------------------

print('')
print('Detecting Color...')
image = cv2.imread('output/detect0.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


modified_image = cv2.resize(image, (1200, 900), interpolation = cv2.INTER_AREA)
modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

clf = KMeans(n_clusters = 4)
labels = clf.fit_predict(modified_image)

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
counts = Counter(labels)

center_colors = clf.cluster_centers_

ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]

plt.figure(figsize = (8, 6))
plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

plt.savefig('output/color.png')
print('Process Complete...')

image = cv2.imread('output/color.png')
cv2.imshow("Color composition",image)
cv2.waitKey(0)
