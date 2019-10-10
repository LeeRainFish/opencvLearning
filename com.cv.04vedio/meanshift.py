import numpy as np
import cv2 as cv



if __name__ == "__main__":

    # 测试视频
    cap = cv.VideoCapture("D:\develop\python3\jupyter\opencv\img\\all1.mp4")
    # 形态学操作需要
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # 创建混合高斯模型用于背景建模
    fgbg = cv.createBackgroundSubtractorMOG2()
    ret, orignal = cap.read()
    fps = cap.get(cv.CAP_PROP_FPS)
    while (True):
        ret, frame = cap.read()
        if ret == True :
            fgmask = fgbg.apply(frame)
            # 形态学开运算去噪点
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
            fgmask = cv.dilate(fgmask, None, iterations=3)
            binary, contours, hierarchy = cv.findContours(image=fgmask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

            for c in contours:
                # 计算各轮廓面积
                area = cv.contourArea(c)

                if area > 500:
                    # 找到一个直矩形
                    x, y, w, h = cv.boundingRect(c)
                    # print(x,y,w,h)
                    frame_roi = frame[y:y+h, x:x+w]
                    roi = orignal[y:y+h, x:x+w]

                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv.putText(frame, "water spot", (x, y-50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            frame_down = cv.pyrDown(frame)
            fgmask_down = cv.pyrDown(fgmask)
            cv.imshow("frame", frame_down)
            cv.imshow("fgmask_down", fgmask_down)
            k = cv.waitKey(int(1000.0/fps)) & 0xff
            if k == 27:
                break
        else:
            break

    # cv.polylines(frame)
    cv.destroyAllWindows()
    cap.release()
