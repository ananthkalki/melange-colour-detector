from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

def rotateImage90R(src):
        img = cv2.imread(src)
        #print(type(img))
        #print(img.shape)

        img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(src, img_rotate_90_clockwise)

def loadImages(src):
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
        print('Row 1:')
        stitchImages(images[:3], args['images']+'/row/1.jpg')
        print('Row 2:')
        stitchImages(images[3:6], args['images']+'/row/2.jpg')
        print('Row 3:')
        stitchImages(images[6:], args['images']+'/row/3.jpg')
elif(count == 2):
        print('Row 1:')
        stitchImages(images[:2], args['images']+'/row/1.jpg')
        print('Row 2:')
        stitchImages(images[2:], args['images']+'/row/2.jpg')

images = loadImages(args['images']+'/row')
#print(images)

print('Final Image:')
stitchImages(images, args['output']+'/output.png')

rotateImage90R(args['output']+'/output.png')
rotateImage90R(args['output']+'/output.png')

image = cv2.imread('output/output.png')
cv2.imshow('image',image)
