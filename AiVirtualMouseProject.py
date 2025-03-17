import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui  # 用于鼠标控制

# 摄像头参数
wCam, hCam = 640, 480
frameR = 100  # 限制活动区域
smoothening = 5  # 平滑系数

# 记录鼠标位置
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# 拖拽状态变量
dragging = False  # 是否正在拖拽

# 滚动状态变量
scrolling = False  # 是否在滚动模式
scroll_start_time = None  # 记录滚动的开始时间

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# 初始化手部跟踪
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    # 1. 读取摄像头图像
    success, img = cap.read()
    img = detector.findHands(img)  # 识别手部
    lmList, bbox = detector.findPosition(img)

    # 2. 获取手指坐标
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # 食指指尖
        x2, y2 = lmList[12][1:]  # 中指指尖

        # 3. 识别手指状态
        fingers = detector.fingersUp()

        # 4. 鼠标移动模式（仅食指抬起）
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 平滑鼠标移动
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(wScr - clocX, clocY)

            # 更新位置
            plocX, plocY = clocX, clocY

        # 5. **拖拽模式（食指和中指抬起）**
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
            # 计算两指间距离
            length, _, _ = detector.findDistance(8, 12, img, draw=False)

            # 确保手指间距足够小，防止误触
            if length < 40:
                if not dragging:
                    pyautogui.mouseDown()  # 按住鼠标左键
                    dragging = True
                    print("进入拖拽模式")

                # 拖拽时鼠标跟随移动
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                autopy.mouse.move(wScr - clocX, clocY)

                # 更新位置
                plocX, plocY = clocX, clocY
            else:
                print("保持拖拽模式")

        # 6. **释放拖拽（手势完全松开才释放）**
        elif dragging and (fingers[1] == 0 or fingers[2] == 0):
            pyautogui.mouseUp()  # 释放鼠标
            dragging = False
            print("退出拖拽模式")

        # 7. **右键点击（拇指和食指接触）**
        length_thumb_index, _, _ = detector.findDistance(4, 8, img, draw=False)
        if length_thumb_index < 30:
            pyautogui.rightClick()
            time.sleep(0.3)  # 防止多次触发

        # 8. **滚动模式（食指和中指都伸出超过3秒）**
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            if scroll_start_time is None:
                scroll_start_time = time.time()  # 记录开始时间
            elif time.time() - scroll_start_time > 3:  # 超过3秒
                scrolling = True
                print("进入滚动模式")
        else:
            scroll_start_time = None
            scrolling = False

        # 9. **执行滚动**
        if scrolling:
            if y1 < y2 - 20:  # 食指比中指高（手向上移动）
                pyautogui.scroll(5)  # 向上滚动
                print("向上滚动")
            elif y2 < y1 - 20:  # 中指比食指高（手向下移动）
                pyautogui.scroll(-5)  # 向下滚动
                print("向下滚动")

    # 计算 FPS
    cTime = time.time()
    fps = 1 / (cTime - (plocX or time.time()))
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # 显示窗口
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
