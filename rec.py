import numpy as np
import cv2 as cv
import sys

userInput = sys.argv
print(userInput)

def empty():
    pass

def videoCapture():
    if len(userInput) > 1:
        filePath = userInput[1]
        frame = cv.imread(filePath)
        return frame
    capture = cv.VideoCapture(1)
    isTrue, frame = capture.read()
    return  frame

def getContours(img, imgContour):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for x in contours:
        area = cv.contourArea(x)
        if area > 2000:
            cv.drawContours(imgContour, x, -1, (255, 0, 255), 7)    
            peri = cv.arcLength(x, True)
            approx = cv.approxPolyDP(x, 0.02 * peri, True)
            x1, y1, x2, y2 = cv.boundingRect(approx)   # x, y, w, h

            cv.rectangle(imgContour, (x1, y1), (x1+x2, y1+y2), (0, 255, 0), 5)
            cv.putText(imgContour, "Points: " + str(len(approx)), (x1 + x2 + 20, y1 + y2 + 20), cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            cv.putText(imgContour, "Area: " + str(int(area)), (x1 + x2 + 45, y1 + y2 + 45), cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            if len(approx) <= 7 and area <= 6200:
                cv.putText(imgContour, "Nutt",(x1 + x2 + 70, y1 + y2 + 70), cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            if (len(approx) <= 7 and area >=17000) or (area > 7000 and len(approx) >= 9):
                cv.putText(imgContour, "Bolt",(x1 + x2 + 70, y1 + y2 + 70), cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            if len(approx) == 8 and (area >= 3150 and area <= 14000):
                cv.putText(imgContour, "Washer",(x1 + x2 + 70, y1 + y2 + 70), cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

cv.namedWindow("Main")
cv.resizeWindow("Main", 420, 80)
cv.createTrackbar("Threshold1", "Main", 255, 255, empty)
cv.createTrackbar("Threshold2", "Main", 52, 255, empty)

while True:
    
    img = videoCapture()
    imgContour = img.copy()

    cv.waitKey(100)

    blurred = cv.GaussianBlur(img, (7,7), 0)
    imgGray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    threshold1 = cv.getTrackbarPos("Threshold1",  "Main")
    threshold2 = cv.getTrackbarPos("Threshold2", "Main")

    imgCanny = cv.Canny(imgGray, threshold1, threshold2)

    kernel = np.ones((5,5))
    imgDil = cv.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDil, imgContour)

    imgStack = stackImages(0.8, ([img, imgDil, imgContour]))
    cv.imshow("Result", imgStack)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break