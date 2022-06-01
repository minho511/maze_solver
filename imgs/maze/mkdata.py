'''
미로에 출발점(초록) 도착점(빨강)을 찍어 저장하기 위해 만든 코드
'''
import cv2

cv2.namedWindow('maze')

for i in range(1, 11):
    maze = cv2.imread(f'./maze{i}.png')
    
    cv2.imshow(f'maze{i}', maze)
    cnt = 0
    def MouseLeftClick(event, x, y, flags, param):
        global cnt, maze, i
        
        if event == cv2.EVENT_LBUTTONDOWN:
            cnt += 1
            if cnt == 1:
                BGR = (0, 255, 0) # 출발점
            elif cnt == 2:
                BGR = (0, 0, 255) # 도착점
            cv2.circle(maze, (x, y), 2, BGR, thickness = 5)
        cv2.imshow(f"maze{i}", maze)
    cv2.setMouseCallback(f'maze{i}', MouseLeftClick)

    if cv2.waitKey() == ord('q'):
        cv2.imwrite(f'maze{i}_labeld.png', maze)
        cv2.destroyAllWindows()

