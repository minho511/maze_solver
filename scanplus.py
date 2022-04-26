from transform import transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from roi import select

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
                cv2.imshow('img', cpy)
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

    cv2.imshow('img', disp)
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

imgPath = "wifi.jpeg"
src = cv2.imread(imgPath)
if src is None:
    print('Image open failed!')
    sys.exit()
orig = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# detecting edges
edge = cv2.Canny(blur, 75, 200)

cv2.imshow("Original Image", src)
cv2.imshow("Edge Detected Image", edge)

cv2.waitKey(0)
cv2.destroyAllWindows()

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
cv2.imshow("Original", src)
cv2.imshow("Scanned", result)

cv2.imwrite("output.jpg", result)
cv2.waitKey(0)

cv2.destroyAllWindows()