# Process image,

#Find all contours
#Filter to contours of certain size

# Convert images to greyscale, and compare


from PIL import Image
import cv2
import sys, os
import imutils
import numpy
import time
image = cv2.imread("All Cards.png",cv2.IMREAD_GRAYSCALE)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150, 255)
cnts = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

croppedImgs =[]

a = 3
b = 4
height = 26
width = 20
for y in (a,a+98,a+196,a+294):
    for x in range(13):
        c = b+70*x
        #cv2.rectangle(image, (c, y), (c + width, y + height), (255, 255, 0), 1)
        croppedImgs.append(image[y:y+height,c:c+width])
numpy.set_printoptions(linewidth=500)
#cv2.imshow('gdgd', image)


bitArray = []

for i in range (len(croppedImgs)):
    th3 = cv2.adaptiveThreshold(croppedImgs[i],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    bitArray.append(th3)
    numpy.ndarray
    #print (bitArray[i])
#print (bitArray[5])
#print (bitArray[6])
print (len(bitArray[2][0]))
for k in range(13):
    print (bitArray[k])

megaArray = numpy.zeros((26,20))
totalValue = 0
#for q in range (13):
for r in range(height):
    for p in range(width):
        for q in range(13):
            if (q ==0):
                totalValue = 0
            totalValue = bitArray[q][r][p] + totalValue
            if (q ==12):
                megaArray[r][p] = totalValue

print (megaArray)
"""
## Ttry masking stuff
x = numpy.array(bitArray[5],dtype='int64')
y = numpy.array(bitArray[6],dtype='int64')
m = numpy.zeros(shape = (len(x), len(x[0])))
for i in range (len(x)):
    for j in range (len(x[0])):
        value = x[i][j] + y[i][j]
        m[i][j] = value
print (m)
"""
## small distinction between 8 and 9
