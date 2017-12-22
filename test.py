import cv2
import numpy as np
import argparse


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True,help = "Path to the image to be scanned")
#args = vars(ap.parse_args())

image=cv2.imread("doc.jpg")

r = 500.0 / image.shape[1]
dim = (500, int(image.shape[0] * r))
orig = image.copy()
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 50, 200)
_,cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
screenCnt=[]
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break
print screenCnt
cv2.drawContours(image,screenCnt,-1,[0,255,0],7)

output=four_point_transform(image,screenCnt.reshape(4,2))
print screenCnt.reshape(4,2)
#output = cv2.adaptiveThreshold(cv2.cvtColor(output,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
output=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
ret,output = cv2.threshold(output,200,255,cv2.THRESH_TRUNC)


cv2.imshow("image",output)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
