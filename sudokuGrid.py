import cv2
import numpy as np
import joblib
import SUDOKU

clf = joblib.load('classifier.pkl')
font = cv2.FONT_HERSHEY_SIMPLEX

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

ratio2 = 3
kernel_size = 3
lowThreshold = 30
im=cv2.imread("blank.jpg")
l=0

im=cv2.imread("sudo.png")
sudoku1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
sudoku1 = cv2.blur(sudoku1, (3,3))

edges = cv2.Canny(sudoku1, lowThreshold, lowThreshold*ratio2, kernel_size)
lines = cv2.HoughLines(edges, 2, np.pi /180, 220, 0, 0)
line=[]
for i in lines:
    line.append(i[0])
line = sorted(line, key=lambda line:line[0])
pos_hori = 0
pos_vert = 0
c=0

New_lines = []
Points = []
for rho,theta in line:
         a = np.cos(theta)
         b = np.sin(theta)
         x0 = a*rho
         y0 = b*rho
         x1 = int(x0 + 1000*(-b))
         y1 = int(y0 + 1000*(a))
         x2 = int(x0 - 1000*(-b))
         y2 = int(y0 - 1000*(a))
         if (b>0.5):
          if(rho-pos_hori>20):
           pos_hori=rho
           c=c+1
           New_lines.append([rho, theta, 0])
         else:
          if(rho-pos_vert>20):
           pos_vert=rho
           c=c+1
           New_lines.append([rho, theta, 1])
print c
for i in range(len(New_lines)):
            if(New_lines[i][2] == 0):
                for j in range(len(New_lines)):
                    if (New_lines[j][2]==1):
                        theta1=New_lines[i][1]
                        theta2=New_lines[j][1]
                        p1=New_lines[i][0]
                        p2=New_lines[j][0]
                        xy = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                        p = np.array([p1,p2])
                        res = np.linalg.solve(xy, p)
                        Points.append(res)
im1=im.copy()
im2=im.copy()
result=[]
board=[]
if(len(Points)==100):
            sudoku1=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            sudoku1 = cv2.adaptiveThreshold(sudoku1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 101, 1)
            for i in range(0,9):
                for j in range(0,9):
                    y1=int(Points[j+i*10][1]+4)
                    y2=int(Points[j+i*10+11][1]-1)
                    x1=int(Points[j+i*10][0]+4)
                    x2=int(Points[j+i*10+11][0]-1)
                    cv2.rectangle(im1,(x1+(x2-x1)/10,y1+(y2-y1)/10),(x2-(x2-x1)/10,y2-(y2-y1)/10),(0,255,0),2)
                    #arr=np.array([[[x1,y1],[x2,y2],[x1,y2],[x2,y1]]])
                    X = sudoku1[y1:y2,x1:x2]
                    #X=four_point_transform(X,arr.reshape(4,2))
                    #X = X[(y2-y1)/10:9*(y2-y1)/10,(x2-x1)/10:9*(x2-x1)/10]
                    if(X.size!=0):
                        X = cv2.resize(X, (36,36))
                        num = clf.predict(np.reshape(X, (1,-1)))
                        result.append(num)
                        board.append(num)
                        if (num[0] != 0):
                            cv2.putText(im,str(num[0]),(int(Points[j+i*10+10][0]+10),int(Points[j+i*10+10][1]-30)),font,1,(225,0,0),2)
                        #else:
                         #   cv2.putText(im,str(num[0]),(int(Points[j+i*10+10][0]+10),
                          #                                       int(Points[j+i*10+10][1]-15)),font,1,(225,0,0),2)
            result = np.reshape(result, (9, 9))
            result1=result.copy()
            board = SUDOKU.SolveSudoku(result)
            print board
            if board is not None:
               for i in range(0, 9):
                   for j in range(0, 9):
                        if (result1[i][j] == 0):
                            cv2.putText(im2, str(result[i][j]), (int(Points[j + i * 10 + 10][0] + 15),
                                                    int(Points[j + i * 10 + 10][1] - 10)), font, 1,(0, 0, 255), 2)

cv2.imshow("Result", im2)
cv2.imshow("sudoku",im)
cv2.imshow("boxes",im1)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows
