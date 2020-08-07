import numpy as np
import cv2 as cv
import collections
from myUtils import *

# 背景建模法
if __name__ == "__main__":

    # 测试视频
    videoUrl = "D:\develop\python3\jupyter\water-spot-detection\\pink-red-drop.mp4"
    # videoUrl = "D:\develop\python3\jupyter\water-spot-detection\\pink-red-drop.mp4"
    # videoUrl = "D:\develop\python3\jupyter\water-spot-detection\\black-drop.mp4"
    # videoUrl = "D:\develop\python3\jupyter\water-spot-detection\\green-drop.mp4"
    # videoUrl = "D:\develop\python3\jupyter\water-spot-detection\\green-ground.mp4"
    cap = cv.VideoCapture(videoUrl)
    fps = cap.get(cv.CAP_PROP_FPS)

    # cap = cv.VideoCapture(0)
    # 形态学操作需要内核
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # 创建混合高斯模型用于背景建模
    # fgbg = cv.createBackgroundSubtractorMOG2()
    # 基于KNN的背景/前景分割算法
    history = 10
    fgbg = cv.createBackgroundSubtractorKNN()
    fgbg.setHistory(history)
    ret, orignal = cap.read()
    orignal = cv.resize(orignal, (0, 0), None, fx=0.5, fy=0.5)

    while (True):
        ret, frame = cap.read()
        if ret == False :
            break

        frame = cv.resize(frame,(0,0),None,fx=0.5,fy=0.5)
        frame_change = hsv_change(frame)
        # 算法比对
        fgmask = fgbg.apply(frame_change)

        if history+10 > 0 :
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
            if area > 200:
                # 轮廓切割为矩形
                x, y, w, h = cv.boundingRect(c)
                frame_roi = frame[y:y+h, x:x+w]
                roi = orignal[y:y+h, x:x+w]
                # 查看与原来模块roi区域的相似度
                similarity = hist_similarity(cv.cvtColor(frame_roi,cv.COLOR_BGR2GRAY), cv.cvtColor(roi,cv.COLOR_BGR2GRAY))
                # similarity = classify_hist_with_split(frame_roi, roi)
                print(similarity)
                if similarity < 0.48:
                    # 画出这个矩形
                    cv.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv.putText(frame_copy, "water spot", (x, y-50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)



        cv.imshow("frame", frame_copy)
        cv.imshow("frame_change", frame_change)
        cv.imshow("fgmask_down", fgmask)
        k = cv.waitKey(int(1000.0/fps)) & 0xff
        # k = cv.waitKey(200) & 0xff
        if k == 27:
            break


    cv.destroyAllWindows()
    cap.release()
