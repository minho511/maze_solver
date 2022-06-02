from transform import transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2


imgPath = "./dataset/l/maze1.png"
img = cv2.imread(imgPath)

def sq_detect(image):
    origin = image.copy()
    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 75, 200)
    # dilate 하지 않으면 미로를 인식하지 못함
    kernel = np.ones((10,10), np.uint8)
    dilate = cv2.dilate(edge, kernel, iterations=1)
    cv2.imwrite('d.png', dilate)
    contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    for contour in contours:
        
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            document = approx
            break
    warp = transform(origin, document.reshape(4, 2))
    contour = cv2.drawContours(origin, [document], -1, (0, 255, 0), 2)
    
    return contour, warp  # contour한 결과를 출력

a4_contour, a4_warp = sq_detect(img)
maze_contour, maze_warp = sq_detect(a4_warp)

# cv2.imshow('hough line', maze_warp)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 허프변환으로는 미로 벽의 정보를 잃는다.

# def hough_line_detect(image):
#     img = image.copy()
#     h, w, _ = img.shape
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edge = cv2.Canny(gray, 50, 150)
#     kernel = np.ones((7, 7), np.uint8)
#     # dilate = cv2.dilate(edge, kernel, iterations=1)
#     # erode = cv2.erode(dilate, kernel, iterations=1)
#     cv2.imshow('hough line', edge)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#     lines = cv2.HoughLines(edge, 1, np.pi/180, 150)
#     for line in lines: # 검출된 모든 선 순회
#         r,theta = line[0] # 거리와 각도
#         tx, ty = np.cos(theta), np.sin(theta) # x, y축에 대한 삼각비
#         x0, y0 = tx*r, ty*r  #x, y 기준(절편) 좌표
#         # 기준 좌표에 빨강색 점 그리기
#         # cv2.circle(img, int(x0), int(y0), 3, (0,0,255), -1)
#         # 직선 방정식으로 그리기 위한 시작점, 끝점 계산
#         x1, y1 = int(x0 + w*(-ty)), int(y0 + h * tx)
#         x2, y2 = int(x0 - w*(-ty)), int(y0 - h * tx)
#         # 선그리기
#         cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 1)

#     #결과 출력    
#     merged = np.hstack((image, img))
#     cv2.imshow('hough line', merged)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
# hough_line_detect(maze_warp)

def find_point(image):
    '''
    이미지에서 출발점 (초록점), 도착점 (빨간점)의 위치를 출력
    BGR 이미지를 입력으로 받고 HSV로 변환
    H값(0-180)을 사용하여 초록과 빨간 점의 위치 출력
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rmap = hsv[:,:,0]<10
    rmap2 = hsv[:,:,1]>180
    gmap = (hsv[:,:,0]>50) * (hsv[:,:,0]<70)
    gmap2 = hsv[:,:,1]>180

    # print(map)
    plt.imshow(gmap*gmap2, cmap='gray')
    plt.show()
    # startX, startY = np.argmax(image[:,:,1])
    # endX, endY = np.argmax(image[:,:,2])
    # print(startX, startY, endX, endY)
find_point(maze_warp)
