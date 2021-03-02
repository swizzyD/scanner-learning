import numpy as np
import cv2
import pytesseract

# Global Variables and webcam capture
frameWidth = 1280
frameHeight = 720
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe '
globalFlag = 0


img = cv2.imread('sample.jpg')


# img = cv2.resize(img,(frameWidth,frameHeight))

cv2.imshow('doc',img)

###########################################################
def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text
###########################################################

def initProcess(img):
    # Pre-processing image for finding document
    dIteration = 2  # dilation iterations
    eIteration = 1  # erosion iterations

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(gray,100,200)
    kernel = np.ones((5,5))
    dilate = cv2.dilate(canny,kernel,iterations=dIteration)
    imgResult = cv2.erode(dilate,kernel,iterations=eIteration)

    return imgResult


def printContours(img):
    areaThreshold = 500
    maxArea =0
    finalPoints = np.array([])
    cntrs, hierarchy = cv2.findContours(imgResult, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in cntrs:
        area = cv2.contourArea(cnt)
        if area > areaThreshold: #locating large rectangle and returning only the largest
            # cv2.drawContours(imgcnt, cnt, -1, (255, 0, 255), 5)
            peri = cv2.arcLength(cnt,True)
            sides = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(sides)==4: #returns only the largest parallelogram
                finalPoints = sides
                maxArea = area

    cv2.drawContours(imgcnt, finalPoints, -1, (255, 0, 0), 3)

    return finalPoints



def arrangePoints(rectPoints):
    # rectPoints.shape is (4,1,2), so reshaping to 4-by-2 and returning a 4-by-1-by-2

    rectPoints = rectPoints.reshape((4,2))
    orderedPts = np.zeros((4,1,2),np.int32)
    summed = rectPoints.sum(1)
    orderedPts[0] = rectPoints[np.argmin(summed)] #(0,0) top left point
    orderedPts[3] = rectPoints[np.argmax(summed)] #(max,max) bottom right point
    diff = np.diff(rectPoints,axis=1)
    orderedPts[1] = rectPoints[np.argmin(diff)] #(maxX,0) top right point
    orderedPts[2] = rectPoints[np.argmax(diff)] #(0,maxY) bottom left point

    # print(rectPoints)


    return orderedPts


def getHeightAndWidth(orderedPts):
    # Getting the appropriate height and width for the document (ratio)
    # calculating heights and widths using pythagoras and finding averages of both for final height and width

    # Heights (Left then Right)
    hA = np.sqrt((orderedPts[2,0,0] - orderedPts[0,0,0])**2 + (orderedPts[2,0,1] - orderedPts[0,0,1])**2)
    hB = np.sqrt((orderedPts[3,0,0] - orderedPts[1,0,0])**2 + (orderedPts[3,0,1] - orderedPts[1,0,1])**2)

    # Widths (Top then Bottom)
    wA = np.sqrt((orderedPts[1, 0, 0] - orderedPts[0, 0, 0]) ** 2 + (orderedPts[1, 0, 1] - orderedPts[0, 0, 1]) ** 2)
    wB = np.sqrt((orderedPts[3, 0, 0] - orderedPts[2, 0, 0]) ** 2 + (orderedPts[3, 0, 1] - orderedPts[2, 0, 1]) ** 2)

    height = int((hA+hB)/2)
    width = int((wA+wB)/2)
    return height, width


def warp(img,orderedPoints):
    # warps the image according to the appropriate height and width
    height, width = getHeightAndWidth(orderedPts)
    pts1 = np.float32(orderedPoints)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]]) #pts2 must be in the same orientation as pts1 so arrangePoints() is needed
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarped = cv2.warpPerspective(img,matrix,(width,height))
    return imgWarped

def imgProcess(img):
    # applying adaptive thresholding to the final warped image and also crops out the excess
    imgFinal = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    imgFinal = cv2.adaptiveThreshold(imgFinal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 11)
    cv2.imshow('final',imgFinal)
    return imgFinal

while True:
    imgResult = initProcess(img) # Processing initial image
    imgcnt = img.copy()
    rectPoints = printContours(imgResult) # gathering 4 points of document
    orderedPts = arrangePoints(rectPoints) # arranging points in order for warp algorithm
    imgWarped = warp(img,orderedPts) # warping

    cv2.imshow('imgWarped',imgWarped)
    imgFinal = imgProcess(imgWarped)

    ###################################################
    if globalFlag == 0:
        imgWarped = cv2.cvtColor(imgWarped,cv2.COLOR_BGR2RGB)
        text = ocr_core(imgWarped)
        print(text)

        # Writing to text file
        f = open("text.txt","w")
        f.write(text)
        f.close()

        globalFlag = globalFlag + 1
    #####################################################

    # cv2.imshow('imgcnt',imgcnt)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Press 'q' to exit program
        break

cv2.destroyAllWindows()



