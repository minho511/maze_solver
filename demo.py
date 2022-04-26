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
import time
# from roi import select

def drawROI(img, corners):
    cpy = img.copy()

    c1 = (192, 192, 255)
    c2 = (128, 128, 255)

    for pt in corners:
        cv2.circle(cpy, tuple(pt.astype(int)), 50, c1, -1, cv2.LINE_AA)

    cv2.line(cpy, tuple(corners[0].astype(int)), tuple(corners[1].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1].astype(int)), tuple(corners[2].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2].astype(int)), tuple(corners[3].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3].astype(int)), tuple(corners[0].astype(int)), c2, 2, cv2.LINE_AA)

    disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0)

    return disp


def onMouse(event, x, y, flags, param):
    global srcQuad, dragSrc, ptOld, src

    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(srcQuad[i] - (x, y)) < 25:
                dragSrc[i] = True
                ptOld = (x, y)
                break

    if event == cv2.EVENT_LBUTTONUP:
        for i in range(4):
            dragSrc[i] = False

    if event == cv2.EVENT_MOUSEMOVE:
        for i in range(4):
            if dragSrc[i]:
                dx = x - ptOld[0]
                dy = y - ptOld[1]

                srcQuad[i] += (dx, dy)

                cpy = drawROI(src, srcQuad)
                # cv2.imshow('img', cpy)
                ptOld = (x, y)
                break

def select():
    global srcQuad, dragSrc, ptOld, src
    # 입력 영상 크기 및 출력 영상 크기
    h, w = src.shape[:2]
    dw = 500
    dh = round(dw * 297 / 210)  # A4 용지 크기: 210x297cm

    # 모서리 점들의 좌표, 드래그 상태 여부
    srcQuad = np.array([[70, 70], [70, h-70], [w-70, h-70], [w-70, 70]], np.float32)
    dstQuad = np.array([[0, 0], [0, dh-1], [dw-1, dh-1], [dw-1, 0]], np.float32)
    dragSrc = [False, False, False, False]

    # 모서리점, 사각형 그리기
    disp = drawROI(src, srcQuad)

    # cv2.imshow('img', disp)
    cv2.setMouseCallback('img', onMouse)

    while True:
        key = cv2.waitKey()
        if key == 13:  # ENTER 키
            break
        elif key == 27:  # ESC 키
            cv2.destroyWindow('img')
            sys.exit()

    # 투시 변환
    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers, (dw, dh), flags=cv2.INTER_CUBIC)
    return dst

imgPath = "pract_image.jpeg"
src = cv2.imread(imgPath)
if src is None:
    print('Image open failed!')
    sys.exit()
orig = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# detecting edges
edge = cv2.Canny(blur, 75, 200)

# cv2.imshow("Original Image", src)
# cv2.imshow("Edge Detected Image", edge)


# finding contours
contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

for contour in contours:
    
	perimeter = cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
	
	if len(approx) == 4:
		document = approx
		break

if len(approx) == 4:
    cv2.drawContours(src, [document], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    warped = transform(orig, document.reshape(4, 2))
else :
    warped = select()
    
warped_g = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# calculate a threshold mask
T = threshold_local(warped_g, 11, offset = 10, method = "gaussian")

result = (warped_g > T).astype("uint8") * 255
# cv2.imshow("Original", src)

# img = cv2.imread('./maze2.png')
img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray,150,255, cv2.THRESH_BINARY_INV)

kernel = np.ones((15,15), np.uint8)
dilate = cv2.dilate(255-result, kernel, iterations=1)
cv2.imwrite('dial.png', dilate)
# cv2.imshow('image', dilate)
points = []
cnt = 0

# 가장 바깥 테두리부터 살펴보며 
def erase_outside(thresh):
    output = thresh.copy()
    h, w = thresh.shape
    l, r, u, d = 0, w-1, 0, h-1
    while True:
        val = np.sum(thresh[:, l])+np.sum(thresh[:, r])+\
            np.sum(thresh[u,:])+np.sum(thresh[d,:])
        
        if val != 0:
            output[:,:l] = 255
            output[:,r:] = 255
            output[:u,:] = 255
            output[d:,:] = 255
            return output
        else:
            l, r, u, d = l+1, r-1, u+1, d-1

def maze_solver(dila, start_point, end_point):
    global img
    map = np.zeros_like(dila, np.float32)
    startx, starty = start_point
    endx, endy = end_point
    
    # 이동방향
    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]

    q = deque([(startx, starty)])
    branch = []
    while q:

        x, y = q.popleft()
        print(x, y)
        # img[x, y, 2] += 255
        for i in range(4):
            nx = x+dx[i]
            ny = y+dy[i]
            
            if nx == endx and ny == endy:
                print('finish!')
                branch.reverse()
                temx, temy, _, _ = branch[0]
                
                for ix, iy, jx, jy in branch[1:]:
                    if temx == jx and temy == jy and map[temx][temy] - map[ix][iy] >0:
                        temx, temy = ix, iy
                        cv2.circle(img, (iy, ix), 2, (0, 0, 255), thickness = 2)          
                
                cv2.imwrite('res.png', img)
                
                return
            if dila[nx][ny] != 0:
                continue
            elif dila[nx][ny] == 0:
                q.append((nx,ny))
                dila[nx][ny] = 255
                map[nx][ny] = map[x][y] + 1
                branch.append((x, y, nx, ny))
            
            
                
                # print((nx, ny))
        cv2.imshow("image", img)
    return


def MouseLeftClick(event, x, y, flags, param):
    global cnt, img, points, img_erased
    if event == cv2.EVENT_LBUTTONDOWN:
        cnt += 1
        
        print(event, x, y,)
        points.append((y, x))
        
		# 출발지점과 도착지점을 순서대로 클릭 받고 원을 그려 표시
        if len(points) <= 2:
            for point in points:
                # 두번째 입력까지만 그림으로 그려 표시
                cv2.circle(img, (point[1], point[0]), 2, (0, 255, 0), thickness = 5)
        # 세번째 아무데나 클릭하면 경로를 찾기 시작함
        if len(points) == 3:
            # points[0] 에는 start points[1]에는 end의 좌표가 담긴다.
            res = maze_solver(img_erased, points[0], points[1])
    cv2.imshow("image", img)

cv2.namedWindow('image')
img_erased = erase_outside(dilate)
cv2.imshow('image', img)
cv2.setMouseCallback('image', MouseLeftClick)




while True:
    if cv2.waitKey(0) == ord('q'):
        break
cv2.destroyAllWindows()

