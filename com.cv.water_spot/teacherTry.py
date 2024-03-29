import numpy as np
import cv2 as cv
from myUtils import *

# 背景建模法
if __name__ == "__main__":

    # 测试视频
    cap = cv.VideoCapture("D:\develop\python3\jupyter\opencv\img\\all1.mp4")
    # cap = cv.VideoCapture(0)
    # 形态学操作需要内核
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    # 创建混合高斯模型用于背景建模
    # fgbg = cv.createBackgroundSubtractorMOG2()
    # 基于KNN的背景/前景分割算法
    fgbg = cv.createBackgroundSubtractorKNN()
    history = 10
    fgbg.setHistory(history)
    ret, orignal = cap.read()
    fps = cap.get(cv.CAP_PROP_FPS)
    while (True):
        ret, frame = cap.read()
        if ret == False:
            break
        # 算法比对
        fgmask = fgbg.apply(frame)
        # 建立一下历史
        if history + 10 > 0:
            history -= 1
            continue

        frame_copy = frame.copy()
        # 形态学开运算加腐蚀去噪点
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        fgmask = cv.dilate(fgmask, None, iterations=3)

        # 寻找轮廓
        binary, contours, hierarchy = cv.findContours(image=fgmask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        for c in contours:
            # 计算各轮廓面积
            area = cv.contourArea(c)
            # cv.line(frame, (x,y), (x,y), (255, 0, 0), 2, 2)
            if area > 200:
                # 轮廓切割为矩形
                x, y, w, h = cv.boundingRect(c)
                frame_roi = frame[y:y+h, x:x+w]
                roi = orignal[y:y+h, x:x+w]
                # 查看与原来模块roi区域的相似度
                similarity = hist_similarity(cv.cvtColor(frame_roi, cv.COLOR_BGR2GRAY),
                                             cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
                # similarity = classify_hist_with_split(frame_roi, roi)
                print(similarity)
                if similarity < 0.5:
                    # 画出这个矩形
                    cv.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv.putText(frame_copy, "water spot", (x, y - 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        cv2.namedWindow("frame", 0)
        cv2.resizeWindow("frame", 800, 600)
        cv2.namedWindow("fgmask", 0)
        cv2.resizeWindow("fgmask", 800, 600)
        cv.imshow("frame", frame_copy)
        cv.imshow("fgmask_down", fgmask)
        k = cv.waitKey(int(1000.0 / fps)) & 0xff
        # k = cv.waitKey(200) & 0xff
        if k == 27:
            break

    cv.destroyAllWindows()
    cap.release()
