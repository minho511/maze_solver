from skimage.filters import threshold_local
import numpy as np
import cv2
import os
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
import time 
# 이미지를 복사하여 터미널에서 붙여넣기하면 경로가 복사됨
imgPath = input("이미지의 경로를 입력 >> ").rstrip()
# imgPath = "./dataset/d/mazed1.png"
img = cv2.imread(imgPath)

def order_points(pts):

    rect = np.zeros((4, 2), dtype = "float32")

    s= np.sum(pts, axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    d = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    
    return rect
    
def transform(img, pts):
    
    rect = order_points(pts)
    (tl, tr, br, bl) = rect	
    
    width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))	
    width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(width_1), int(width_2))
    
    height_1 = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))	
    height_2 = np.sqrt(((bl[0] - tl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(height_1), int(height_2))
    
    dst = np.array([
             [0, 0],
             [maxWidth - 1, 0],
             [maxWidth - 1, maxHeight - 1],
             [0, maxHeight - 1]], dtype = "float32")
             
    M = cv2.getPerspectiveTransform(rect, dst) 
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def sq_detect(image):
    origin = image.copy()
    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY) # gray scale로 변환
    edge = cv2.Canny(gray, 75, 200) # Canny edge 적용하여 edge를 검출함
    kernel = np.ones((10,10), np.uint8)
    dilate = cv2.dilate(edge, kernel, iterations=1) # 없으면 미로검출 안됨
    # contour를 검출
    contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    # 미로의 입구와 출구로 인한 왜곡과 A4용지의 구부러짐또는 빛에 의한 정보손실을 고려하여 approxPolyDP 사용
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4: # 4개의 윤곽을 찾았다면 종료
            document = approx
            break
    # 윤곽선의 정보를 기준으로
    warped_image = transform(origin, document.reshape(4, 2))
    # 기존의 이미지에 검출한 윤곽을 그림
    contour_image = cv2.drawContours(origin, [document], -1, (0, 255, 0), 2) 
    
    return contour_image, warped_image  # contour한 결과를 출력

a4_contour, a4_warp = sq_detect(img)

# 에이포 용지의 윤곽선이 남기도 하여 잘라냄
def cut_edge(image):
    return image[20:-20,20:-20,:]

a4_warp = cut_edge(a4_warp)
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

def find_widest_region(mat):
    '''
    이진 영상에서 가장 큰 영역에 속하는 좌표값들을 반환
    이때 배경은 0 영역은 1
    '''
    map = mat.copy()
    map = map.astype(np.uint8)
    
    h, w = mat.shape
    dx = [-1, 1, 0, 0]
    dy = [0, 0, 1, -1]
    region = 2
    
    for x in range(h):
        for y in range(w):
            if map[x][y] == 1:
                
                q = deque([(x, y)])
                map[x][y] = region
                while q:
                    cx, cy = q.popleft()
                    map[cx][cy] = region
                    for k in range(4):
                        nx = cx + dx[k]
                        ny = cy + dy[k] 
                        if nx < 0 or ny < 0 or nx >=h or ny >=w:
                            continue
                        if map[nx][ny] == 0 or map[nx][ny] == region:
                            continue
                        if map[nx][ny] == 1:
                            q.append((nx, ny))
                            map[nx][ny] = region
                region += 1
    points = []
    size_of_region = []
    for r in range(2,region):
        p =np.where(map==r)
        points.append(p)
        size_of_region.append(len(p[0]))
    return points[np.argmax(size_of_region)]


def find_point(image):
    '''
    이미지에서 출발점 (초록점), 도착점 (빨간점)의 위치를 출력
    BGR 이미지를 입력으로 받고 HSV로 변환
    H값(0-180)을 사용하여 초록과 빨간 점의 위치 출력
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rmap = hsv[:,:,0]<10
    rmap2 = hsv[:,:,1]>120

    gmap = (hsv[:,:,0]>40) * (hsv[:,:,0]<80)
    gmap2 = hsv[:,:,1]>80

    start = find_widest_region(rmap2*rmap)
    end = find_widest_region(gmap2*gmap)

    # 가장 큰 영역
    # 영역의 중앙 점 좌표를 사용
    startX = np.median(start[0])
    startY = np.median(start[1])
    endX = np.median(end[0])
    endY = np.median(end[1])

    # 추가로 미로에서 출발 도착 지점에 해당하는 부분은 통로로 만들어줘야하기 때문에
    # end와 start에 담긴 좌표를 사용하여 image의 출발 도착 지점을 255로 만들어줌
    img_erased = image.copy()
    cv2.circle(img_erased, (int(startY),int(startX)), 2, (255,255,255), thickness = 15)
    cv2.circle(img_erased, (int(endY),int(endX)), 2, (255,255,255), thickness = 15)
    return (int(startX), int(startY)), (int(endX), int(endY)), img_erased

start, end, maze_erased = find_point(maze_warp)

# 미로 사진을 이진화 한다.
def binary_inv(maze_erased):
    maze_gray = cv2.cvtColor(maze_erased, cv2.COLOR_BGR2GRAY)
    T = threshold_local(maze_gray, 11, offset = 10, method = "gaussian")
    maze_bin = (maze_gray<= T).astype("uint8")*255
    return maze_bin

def mkgif():
    path = [f"./for_mkgif/{i}" for i in os.listdir("./for_mkgif")]
    for idx, p in enumerate(path):
        if p[-4:] != '.png':
            path.pop(idx)
    path.sort()
    paths = [ Image.open(i) for i in path]
    imageio.mimsave('./result.gif', paths, fps=20)

def maze_solver(dila, start_point, end_point, maze_warp):
    t = time.time()
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
    # gif 파일을 만들기위해 경로를 저장할 폴더를 만든다.
    if not os.path.isdir('./for_mkgif'):
        os.mkdir('./for_mkgif')
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
                print("time: ", time.time()-t)
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
                        cv2.imwrite(f'./for_mkgif/img{cnt}.png', labeled)
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

def block_outside(image):
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

# 미로 길찾기
maze_bin = binary_inv(maze_erased)
maze_in = block_outside(maze_bin)
maze_solver(maze_in, start, end, maze_warp)