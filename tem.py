import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2
import time
# global img, points, cnt
img = cv2.imread('./maze2.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray,150,255, cv2.THRESH_BINARY_INV)
# cv2.imshow('image', img)
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
        
        



def maze_solver(thresh, start_point, end_point):
    global img
    startx, starty = start_point
    endx, endy = end_point
    
    # 이동방향
    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]

    q = deque([(startx, starty)])
    load = []
    print(q)
    while q:
        x, y = q.popleft()
        for i in range(4):
            nx = x+dx[i]
            ny = y+dy[i]
            if nx == endx and ny == endy:
                print('finish!')
                cv2.imwrite('res.png', img)
                return
            if thresh[nx][ny] != 0:
                continue
            elif thresh[nx][ny] == 0:
                q.append((nx,ny))
                thresh[nx][ny] = thresh[nx][ny] + 1
                img[nx, ny, 2] += 10
                print((nx, ny))
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
                cv2.circle(img, (point[1], point[0]), 2, (0, 0, 255), thickness = 5)
        # 세번째 아무데나 클릭하면 경로를 찾기 시작함
        if len(points) == 3:
            # points[0] 에는 start points[1]에는 end의 좌표가 담긴다.
            res = maze_solver(img_erased, points[0], points[1])


    
    cv2.imshow("image", img)

cv2.namedWindow('image')
img_erased = erase_outside(thresh)
cv2.imshow('image', img)
cv2.setMouseCallback('image', MouseLeftClick)




while True:
    if cv2.waitKey(0) == ord('q'):
        break
cv2.destroyAllWindows()

