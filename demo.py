# select 없는버전

# a4 용지가 전체적으로 나오게 찍었을때 contour 잡는 과정이 두번 필요함
# img1.jpeg는 아래 책상에 반사된 빛 때문에 에이포가 잘 안잡힘
# img2.jpeg는 에이포 꼭짓점 부분이 짤려서 안잡힘


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

imgPath = "./imgs/data/f/"
src = cv2.imread(imgPath)
h, w, c = src.shape
# if h > 1000 or w > 1000:
#     src = cv2.resize(src, (w//2, h//2), interpolation=cv2.INTER_AREA)
if src is None:
    print('Image open failed!')
    sys.exit()

def draw_square(input_img, d=1):
    orig = input_img.copy()
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # detecting edges
    edge = cv2.Canny(blur, 75, 200)
    if d == 1:
        cv2.imwrite('draw_square1.png', edge)
    kernel = np.ones((d,d), np.uint8)  # kernel값에 따라 범용 불가할 수도 있음 --- 해결필요
    dilate = cv2.dilate(edge, kernel, iterations=1)
    cv2.imwrite('check.png', edge)
    # finding contours
    contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    for contour in contours:
        
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            document = approx
            break
    warped = transform(orig, document.reshape(4, 2))
    outlined = cv2.drawContours(orig, [document], -1, (0, 255, 0), 2)
    
    # warped_gray = cv2.cvtColor(warped, cv2q.COLOR_BGR2GRAY)
    return outlined, warped  # contour한 결과를 출력

# a4용지 검출


a4_outlined, a4_warped = draw_square(src, 3)

cv2.imshow('a4_outlined', a4_outlined)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imshow('a4_warped', a4_warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# a4_warped = a4_warped[10:-10,10:-10,:]
# 그 안 미로 검출
maze_outlined, maze_warped = draw_square(a4_warped, 10)
cv2.imshow('maze_outlined', maze_outlined)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('maze_warped', maze_warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('warped_g.png', warped_g)
maze_warped_gray = cv2.cvtColor(maze_warped, cv2.COLOR_BGR2GRAY)
# calculate a threshold mask
T = threshold_local(maze_warped_gray, 11, offset = 10, method = "gaussian")
result = (maze_warped_gray<= T).astype("uint8") * 255

# ret, result = cv2.threshold(warped_g,150,255, cv2.THRESH_BINARY_INV)
img = cv2.cvtColor(255-result, cv2.COLOR_GRAY2BGR)
# img는 다른 색으로 draw하기 위해 3 채널 사진으로 바꿔준 결과, 여기에 draw를 할 것임
# result는 threshold값을 주어 2진변환한 결과
cv2.imwrite('result.png', result)

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(img_gray,150,255, cv2.THRESH_BINARY_INV)

kernel = np.ones((2,2), np.uint8)  # kernel값에 따라 범용 불가할 수도 있음 --- 해결필요
dilate = cv2.dilate(result, kernel, iterations=1)
# dilate는 이진화된 사진에 팽창을 적용하여 길을 좁혀 탐색할 통로를 좁히고자함
cv2.imwrite('dilate.png', dilate)

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
            cv2.imwrite('erased.png', output)
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

    cnt = 0
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
                        cv2.circle(img, (iy, ix), 2, (0, 0, 255), thickness = 1)          
                        cnt += 1
                        if cnt%50 == 0:
                            cv2.imwrite(f'./gifs/img{cnt}.png', img)
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
            maze_solver(img_erased, points[0], points[1])
    cv2.imshow("image", img)


cv2.namedWindow('image')
img_erased = erase_outside(dilate)
cv2.imshow('image', img)
cv2.setMouseCallback('image', MouseLeftClick)


while True:
    if cv2.waitKey(0) == ord('q'):
        break

