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
from make_gif import mkgif
# mimi
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
# 에이포 용지의 윤곽선이 남기도 하여 잘라냄
a4_warp = a4_warp[20:-20,20:-20,:]
maze_contour, maze_warp = sq_detect(a4_warp)
cv2.imshow('a4_contour', a4_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('a4_warp', a4_warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('maze_contour', maze_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('maze_warp', maze_warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
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
    rmap2 = np.array(hsv[:,:,1]>180)
    rmap2 = rmap2.astype(np.uint8)
    kernel = np.ones((3,3))
    rmap3 = cv2.erode(rmap2, kernel, iterations=1)
    gmap = (hsv[:,:,0]>50) * (hsv[:,:,0]<70)
    gmap2 = hsv[:,:,1]>180
    # start = np.where(gmap2*gmap == 1)
    # end = np.where(rmap3*rmap == 1)
    plt.subplot(2, 6, 1)
    plt.imshow(rmap, cmap='gray')
    plt.subplot(2, 6, 2)
    plt.imshow(rmap2, cmap='gray')
    plt.subplot(2, 6, 3)
    plt.imshow(rmap*rmap2, cmap='gray')
    plt.subplot(2, 6, 4)
    plt.imshow(gmap, cmap='gray')
    plt.subplot(2, 6, 5)
    plt.imshow(gmap2, cmap='gray')
    plt.subplot(2, 6, 6)
    plt.imshow(gmap*gmap2, cmap='gray')
    plt.show()
    exit()
    # 영역의 중앙 점 좌표를 사용
    startX = np.median(start[0])
    startY = np.median(start[1])
    endX = np.median(end[0])
    endY = np.median(end[1])

    # 추가로 미로에서 출발 도착 지점에 해당하는 부분은 통로로 만들어줘야하기 때문에
    # end와 start에 담긴 좌표를 사용하여 image의 출발 도착 지점을 255로 만들어줌
    img_erased = image.copy()
    cv2.circle(img_erased, (int(startY),int(startX)), 2, (255,255,255), thickness = 10)
    cv2.circle(img_erased, (int(endY),int(endX)), 2, (255,255,255), thickness = 10)
    return (int(startX), int(startY)), (int(endX), int(endY)), img_erased

start, end, img_erased = find_point(maze_warp)

# 미로 사진을 이진화 한다.
maze_gray = cv2.cvtColor(img_erased, cv2.COLOR_BGR2GRAY)
T = threshold_local(maze_gray, 11, offset = 10, method = "gaussian")
maze_bin = (maze_gray<= T).astype("uint8")*255

# 미로 길찾기
points = []
cnt = 0


def maze_solver(dila, start_point, end_point, maze_warp):
    map = np.zeros_like(dila, np.float32)
    startx, starty = start_point
    endx, endy = end_point
    h, w = dila.shape
    labeled = maze_warp.copy()
    # 이동방향
    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]

    q = deque([(startx, starty)])
    branch = []
    
    cnt = 0
    while q:
        x, y = q.popleft()
        print(x, y)
        for i in range(4):
            nx = x+dx[i]
            ny = y+dy[i]
            if nx < 0 or nx >= h or ny < 0 or ny >=w:
                continue
            if nx == endx and ny == endy:
                print('finish!')
                branch.reverse()
                temx, temy, _, _ = branch[0]
                road = []
                for ix, iy, jx, jy in branch[1:]:
                    if temx == jx and temy == jy and map[temx][temy] - map[ix][iy] >0:
                        temx, temy = ix, iy
                        road.append((iy, ix))
                # draw
                road.reverse()
                for ix, iy in road:
                    cv2.circle(labeled, (ix, iy), 2, (0, 0, 255), thickness = 1)          
                    cnt += 1
                    if cnt%30 == 0:
                        cv2.imwrite(f'./gifs/img{cnt}.png', labeled)
                cv2.imshow('res', labeled)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                mkgif()
                return
            if dila[nx][ny] != 0:
                continue
            elif dila[nx][ny] == 0:
                q.append((nx,ny))
                dila[nx][ny] = 255
                map[nx][ny] = map[x][y] + 1
                branch.append((x, y, nx, ny))

    return
def erase_outside(image):
    '''
    길을 밖으로 찾지 않도록 최외각선 밖의 영역을 막아줘야됨
    '''
    img = image.copy()
    h, w = image.shape
    lu = [0,0]
    ld = [h-1,0]
    ru = [0,w-1]
    rd = [h-1,w-1]
    while img[lu[0],lu[1]] == 0:
        img[lu[0], lu[1]] = 255
        img[lu[0], lu[1]+1] = 255
        img[lu[0]+1, lu[1]] = 255
        lu[0] += 1
        lu[1] += 1
    while img[ld[0],ld[1]] == 0:
        img[ld[0], ld[1]] = 255
        img[ld[0], ld[1]+1] = 255
        img[ld[0]-1, ld[1]] = 255
        ld[0] -= 1
        ld[1] += 1
    while img[ru[0],ru[1]] == 0:
        img[ru[0], ru[1]] = 255
        img[ru[0], ru[1]-1] = 255
        img[ru[0]+1, ru[1]] = 255
        ru[0] += 1
        ru[1] -= 1
    while img[rd[0],rd[1]] == 0:
        img[rd[0], rd[1]] = 255
        img[rd[0], rd[1]-1] = 255
        img[rd[0]-1, rd[1]] = 255
        rd[0] -= 1
        rd[1] -= 1
    return img

maze_result = erase_outside(maze_bin)
maze_solver(maze_result, start, end, maze_warp)


