import cv2
import numpy as np

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

cv2.namedWindow("SUDOKU Solver")
vc = cv2.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
while rval:
 # Preprocess image, convert from RGB to Gray
    sudoku1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sudoku1 = cv2.blur(sudoku1, (3,3))
    # Apply Canny edge detection
    edges = cv2.Canny(sudoku1, lowThreshold, lowThreshold*ratio2, kernel_size)
    # Apply Hough Line Transform, return a list of rho and theta
    lines = cv2.HoughLines(edges, 2, np.pi /180, 300, 0, 0)
    if (lines is not None):
        if l==0 and len(lines)>30:
            l=1
            im=frame

    cv2.imshow("SUDOKU Solver", frame)
    cv2.imshow("image", im)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27 or l==1:  # exit on ESC
                break
vc.release()
cv2.destroyAllWindows()



sudoku1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
sudoku1 = cv2.blur(sudoku1, (3,3))

edges = cv2.Canny(sudoku1, lowThreshold, lowThreshold*ratio2, kernel_size)
lines = cv2.HoughLines(edges, 2, np.pi /180, 180, 0, 0)
line=[]
for i in lines:
    line.append(i[0])
line = sorted(line, key=lambda line:line[0])
print line
pos_hori = 0
pos_vert = 0
c=0
# Create a list to store new bundle of lines
New_lines = []
        # Store intersection points
Points = []
        # Define the position of horizontal and vertical line
for rho,theta in line:
         a = np.cos(theta)
         b = np.sin(theta)
         x0 = a*rho
         y0 = b*rho
         x1 = int(x0 + 1000*(-b))
         y1 = int(y0 + 1000*(a))
         x2 = int(x0 - 1000*(-b))
         y2 = int(y0 - 1000*(a))
         # If b > 0.5, the angle must be greater than 45 degree
         # so we consider that line as a vertical line
         if (b>0.5):
          # Check the position
          if(rho-pos_hori>20):
           # Update the position
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
if len(Points)==100:
     for i in range(0,9):
                for j in range(0,9):
                    y1=int(Points[j+i*10][1]+5)
                    y2=int(Points[j+i*10+11][1]-5)
                    x1=int(Points[j+i*10][0]+5)
                    x2=int(Points[j+i*10+11][0]-5)
                    # Saving extracted block for training, uncomment for saving digit blocks
                    # cv2.imwrite(str((i+1)*(j+1))+".jpg", sudoku1[y1: y2,
                    #                                            x1: x2])
                    cv2.rectangle(im,(x1,y1),(x2, y2),(0,255,0),2)
print len(Points)
cv2.imshow("sudoku",im)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows
