import numpy
from PIL import ImageGrab
import pyautogui as mouseCtrl
import time
import copy


# USAGE
# python recognize_digits.py

# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

### SCreenshot EXAMPLE and IMAGE SAVE
"""
snapshot = ImageGrab.grab()
save_path = "C:\\Users\\dabes\\Desktop\\Vision Recognition Card Poker\\MySnapshot.jpg"
snapshot.save(save_path)
"""

### Mouse Click on Screen ###  ### THIS WORKS RELATIVE TO SCREEN WHICH IS GOOOD!!!!
### UNITS ARE IN PIXELS####
"""
mouseCtrl.click(100,100)
time.sleep(3)
mouseCtrl.rightClick(200,300)
"""
### TRANSLATE CODE BELOW TO work with PokerStarz#####
##############################################

# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

### Top segment
##CARD LOOKUP
CARD_LOOKUP = {
        

}

#This function returns a standardized array in clockwise form where the top left point is Point 0
def boxTransformConsistent(inP):
        outP = copy.copy(inP)
        #outP = numpy.zeros(shape=(4,2), dtype=int)
        # THIS WILL STOP YOU FROM GOING CRAZY. PYTHON ACTUALLY REFERECES the original array when you make a copy unless you use the copy library
        inPDummy = copy.copy(inP)
        if len(inP) != 4 or len(inP[0]) != 2:
                return ("The input must be a 4 by 2 array")
        #This gets point 0 on box, as well as point 1 (which is below point 0)
        for j in range(4):
                if (inP[j,0] <= inP[0,0] and inP[j,0] <= inP[1,0] and inP[j,0] <= inP[2,0] and inP[j,0] <= inP[3,0]):
                        
                        #print("inp j 1: ", inP[j,1])
                        
                        #initialize flags
                        aPointIsToLeft = False
                        aPointIsBelow = False
                        secondCounter = 0
                        count = 0
                        for k in range(4):
                                if inP[j,1] >= inP[k,1]:
                                        #Coutns up if our Y coord point is lower or equal to other points
                                        count = count + 1
                        for k in range(4):
                                if inP[j,0] >= inP[k,0]:
                                        #Counts up if it is to the left or equal to other point (X-coor)
                                        secondCounter = 1 + secondCounter
                        if (secondCounter ==2):
                                aPointIsToLeft = True
                        elif (secondCounter ==1):
                                pass
                        else:
                                print ("FUCK I MISSED SOMETHING")
                        for k in range(4):
                                if inP[j,1] > inP[k,1]:
                                        #makes it true if the j point is on top of any other point
                                        aPointIsBelow = True
                        #If leftmost point is NOT at the top, then we apply algorithim below
                        #If point chosen is lower/same level as all points except for 1, then apply alg below
                        if count >=3 and aPointIsToLeft and aPointIsBelow:
                                #print ("j:",j)
                                #Point 2 is certain then
                                outP[1] = inPDummy[j]
                                if j==0:
                                        outP[0] =inPDummy[3] if (inP[1,0] >inP[3,0]) else inPDummy[1] #case 1
                                if j ==1:
                                        outP[0] =inPDummy[0] if (inP[2,0] >inP[0,0]) else inPDummy[2] #case 2
                                if j ==2:
                                        outP[0] =inPDummy[1] if (inP[3,0] >inP[1,0]) else inPDummy[3] #case 3
                                if j==3:
                                        outP[0] =inPDummy[2] if (inP[0,0] >inP[2,0]) else inPDummy[0] #case 3

                        else:
                                outP[0] = inPDummy[j]
                                #Make point 2 be closest point to the right of it AND below
                                tempArr =[]
                                #print ("inP: ", inP,"j: ",j ,"inpDummy" , inPDummy[j])
                                #print ("outP ", outP)
                                for q in range(4):
                                        #print ("q: ", q)
                                        #print ("inP", inP[q,1])
                                        #outP has to be higher than inPq
                                        if (outP[0,1] < inP[q,1]):
                                                tempArr.append(inP[q])
                                if (len(tempArr) ==1):
                                        outP[1] = tempArr[0]
                                elif (len(tempArr) ==2):
                                        outP[1] = tempArr[0] if (tempArr[0][0] < tempArr[1][0]) else tempArr[1]
                                else:
                                        print(" ERROR:  SHAPE DOES NOT MATCH VALID GEOMETRY, number of points below top left point: ", len(tempArr))
                        #print ("count: ", count, " outP0,outP1: ", outP[0], outP[1])
        #Reset Flags
        aPointIsToLeft = False
        aPointIsBelow = False
        counter = 0
        secondCounter = 0
        mark1 = 99
        mark2 = 99

        #Now we find the remaining two points that are uncharted and map them Clockwise as the rest of the points
        #tempInP =copy.copy(inP)
        #print ("outP :",outP)
        tempInP = copy.copy(inP)
        for t in range(len(inP)):
                if (inP[t,0] ==outP[0,0] and inP[t,1] ==outP[0,1]):
                        mark1 = t
                        #print ("mark1" , mark1)
                if (inP[t,0] ==outP[1,0] and inP[t,1] ==outP[1,1]):
                        mark2 = t
                        #print ("mark2" , mark2)
        #print("begoretemppin" , tempInP)
        if (mark1 ==99 or mark2 ==99):
                print ("you fucked up. Fix your code")
        #Delete the greater mark first since it will lay deeper in the array thus messing things up
        if (mark1>mark2):
                tempInP = numpy.delete(tempInP,mark1,0)
                tempInP = numpy.delete(tempInP,mark2,0)
        else:
                #for e in [mark2,mark1]:
                tempInP = numpy.delete(tempInP,mark2,0)
                tempInP = numpy.delete(tempInP,mark1,0)
                        
        #print (len(tempInP))
        if (len(tempInP) != 2):
                print ("ERROR: 2 values were not left over in the algorithim. Fix algorithim")
        else:
                if (tempInP[0,1] > tempInP[1,1]):
                        outP[2] = tempInP[0]
                        outP[3] = tempInP[1]
                else:
                        outP[3] = tempInP[0]
                        outP[2] = tempInP[1]
        return outP
        
        

### CROP IMAGE TO ONLY CONTAIN POKER STARS GAME SCREEN##############################################################
# load the example image
image = cv2.imread("PokerStarzScreenshot5.jpg")
#image = cv2.imread("All Cards.png")

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
#image = imutils.resize(image, height=500)                 # DONT RESIZE
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150, 255)

#cv2.imshow("edged",edged)
#cv2.waitKey(0)


# find contours in the edge map, then sort them by their size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse=True)
displayCnt = None
croppedImg = None

# loop over the contours
for c in cnts:        
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if the contour has four vertices, then we have found
    # the thermostat display
    if len(approx) == 4:
                #blablabl
                (x, y, w, h) = cv2.boundingRect(c)
                if (w>400 and h>500):
                        print (x, y, w, h)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        croppedImg = image[y:y+h,x:x+w]
                        #displayCnt = approx

try:
        if (len(croppedImg) >0):
                print ("Valid length")
except:
        croppedImg = image
        
#cv2.imshow("marked",croppedImg)
#cv2.waitKey(0)

############FIND CIRCLES inside poker game window######################
circleArrayCardsOnRightSide = []
circleArrayCardsOnLeftSide = []
circleArray = []
circleX = []

croppedGray = cv2.cvtColor(croppedImg,cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(croppedGray, cv2.HOUGH_GRADIENT, 0.956, 70, param1=30,param2=30,maxRadius = 30, minRadius=26)
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = numpy.round(circles[0, :]).astype("int")
 
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(croppedImg, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(croppedImg, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
        circleArray.append([x,y])
        circleX.append(x)
    # Sorts circles out to left and right side respectfully
    circleXSorted = sorted(circleX)
    for i in range(len(circleX)):
                if (circleX[i] == circleXSorted[0] or circleX[i] == circleXSorted[1] or circleX[i] == circleXSorted[3]):
                        circleArrayCardsOnLeftSide.append(circleArray[i])
                else:
                        circleArrayCardsOnRightSide.append(circleArray[i])

    # show the output image
    #cv2.imshow("croppedImg", croppedImg)
    #cv2.waitKey(0)

####################################################################################################################
### Find second set of contours for inside the poker game window###

#modify cropped Image
blurredCr = cv2.GaussianBlur(croppedImg, (5, 5), 0)
croppedEdges = cv2.Canny(blurredCr, 50, 150, 255)

#cv2.imshow("edged",croppedEdges)
#cv2.waitKey(0)

# find contours in the cropped image, and sorts them in descending order (by size)
cnts = cv2.findContours(croppedEdges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


# loop over the contours
for c in cnts:
    valid1 = False
    valid2 = False
    peri = cv2.arcLength(c, True)
    if (peri <35):
            continue
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
   
    box = numpy.int0(box)

    #goes to next contour if contour is a fucking line....
    if box[0,0] == box[1,0] == box[2,0] == box[3,0]:
            continue
    if box[0,1] == box[1,1] == box[2,1] == box[3,1]:
            continue
    #This narrows down the search to box size , eliminates large AND small
    if ((numpy.absolute(box[0,0]-box[3,0]) <40) and (numpy.absolute(box[1,1]-box[2,1]) <40)):
            # this narrows it down to straight boxes
            if (numpy.absolute(box[0,1] - box[3,1]<5) or numpy.absolute(box[1,1] - box[2,1])<5):
                    boxStorage = copy.copy(box)
                    #print ("peri",peri)
                    box = copy.copy(boxTransformConsistent(boxStorage))
                    #print("box after: ", box)
                    #print(box)
                    
                    #This one checks if it is anywhere above ANY of the circles
                    for (pointX, pointY) in circleArray:
                            if (pointY - box[1,1] ) > 35 and (pointY - box[1,1]) < 150:
                                    valid1 = True
                                    print ("pointY - box[1,1]: " , pointY - box[1,1])
                                    print("pointY",pointY)
                                    print("pointXx",pointX)
                                    print("box11",box[1,1])
                                    
                    #checks left side circle relations
                    for (pointX, pointY) in circleArrayCardsOnLeftSide:
                            print ("pointX -box[3,0]: " , pointX -box[3,0])
                            print("pointY",pointY)
                            print("pointXx",pointX)
                            print("box30",box[3,0])
                            if ((pointX -box[3,0] >0) and (pointX -box[3,0] < 120)):
                                   valid2 = True
                                   
                                   break
                    #checks right side circle relations               
                    for (pointX, pointY) in circleArrayCardsOnRightSide:
                            if ((box[3,0] - pointX  >0) and (box[2,0] - pointX < 150)):
                                   valid2 = True
                                   print ("box[3,0] - pointX: " , box[3,0] - pointX)
                                   break
                                   
                    if (valid1 and valid2):
                        cv2.drawContours(croppedImg,[box],0,(0,0,255),1)
                        print (box)
for (pointX, pointY) in circleArray:
    print ("x: y: ", pointX, pointY)
            
    """ THIS IS GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOD /option A
    if (len(approx) >=5):
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(croppedImg,ellipse,(0,255,0),2)
    """
    # approximate the contour

    #print (len(approx))
    #print (approx)
    #if (len(approx)) >10):
    #        cv2.bounding

    """
    # if the contour has four vertices, then we have found
    # the thermostat display
    if len(approx) == 4:
                #blablabl
                (x, y, w, h) = cv2.boundingRect(c)
                #if (w>400 and h>500):
                        #print (x, y, w, h)
                cv2.rectangle(croppedImg, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        #croppedImg = image[y:y+h,x:x+w]
                        #displayCnt = approx
    """

cv2.imshow("marked2",croppedImg)
cv2.waitKey(0)

####################################################################################################################

# extract the thermostat display, apply a perspective transform
# to it
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))

# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh = cv2.threshold(warped, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []

# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    # if the contour is sufficiently large, it must be a digit
    if w >= 15 and (h >= 30 and h <= 40):
        digitCnts.append(c)

# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts,
    method="left-to-right")[0]
digits = []

# loop over each of the digits
for c in digitCnts:
    # extract the digit ROI
    (x, y, w, h) = cv2.boundingRect(c)
    roi = thresh[y:y + h, x:x + w]

    # compute the width and height of each of the 7 segments
    # we are going to examine
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)

    # define the set of 7 segments
    segments = [
        ((0, 0), (w, dH)),  # top
        ((0, 0), (dW, h // 2)), # top-left
        ((w - dW, 0), (w, h // 2)), # top-right
        ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
        ((0, h // 2), (dW, h)), # bottom-left
        ((w - dW, h // 2), (w, h)), # bottom-right
        ((0, h - dH), (w, h))   # bottom
    ]
    on = [0] * len(segments)

    # loop over the segments
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)

        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if total / float(area) > 0.5:
            on[i]= 1

    # lookup the digit and draw it on the image
    digit = DIGITS_LOOKUP[tuple(on)]
    digits.append(digit)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(output, str(digit), (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# display the digits
print(u"{}{}.{} \u00b0C".format(*digits))
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)

