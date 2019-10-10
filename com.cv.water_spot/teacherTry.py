import numpy as np
import cv2 as cv
import collections

def hist_similarity(img1, img2):
    # 计算灰度单通道的直方图的相似值
    hist1 = cv.calcHist([img1], [0], None, [256], [0.0, 255.0])
    hist2 = cv.calcHist([img2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = float (degree) / len(hist1)
    return degree


def getColorDicList():
    dict = collections.defaultdict(list)
    #灰色
    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['gray']=color_list
    return dict

def hsv_change(origin_img):
    # 转换色域
    hsv_img = cv.cvtColor(origin_img, cv.COLOR_BGR2HSV)
    color_dict = getColorDicList()
    mask_pre = origin_img.copy()
    for d in color_dict:
        mask_gray = cv.inRange(hsv_img, color_dict[d][0], color_dict[d][1])
        mask_pre[mask_gray==0] = [0, 0, 0]
    return mask_pre


# 背景建模法
if __name__ == "__main__":

    # 测试视频
    cap = cv.VideoCapture("D:\develop\python3\jupyter\opencv\img\\all1.mp4")
    # cap = cv.VideoCapture(0)
    # 形态学操作需要内核
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # 创建混合高斯模型用于背景建模
    # fgbg = cv.createBackgroundSubtractorMOG2()
    # 基于KNN的背景/前景分割算法
    fgbg = cv.createBackgroundSubtractorKNN()
    ret, orignal = cap.read()
    fps = cap.get(cv.CAP_PROP_FPS)
    while (True):
        ret, frame = cap.read()
        if ret == True :

            # frame = hsv_change(frame)
            # 算法比对
            fgmask = fgbg.apply(frame)
            # frame_copy = frame.copy()

            # 形态学开运算加腐蚀去噪点
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
            fgmask = cv.dilate(fgmask, None, iterations=3)
            # 寻找轮廓
            binary, contours, hierarchy = cv.findContours(image=fgmask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

            for c in contours:
                # 计算各轮廓面积
                area = cv.contourArea(c)
                # cv.line(frame, (x,y), (x,y), (255, 0, 0), 2, 2)
                if area > 100:
                    # 轮廓切割为矩形
                    x, y, w, h = cv.boundingRect(c)
                    frame_roi = frame[y:y+h, x:x+w]
                    roi = orignal[y:y+h, x:x+w]
                    # 查看与原来模块roi区域的相似度
                    similarity = hist_similarity(cv.cvtColor(frame_roi,cv.COLOR_BGR2GRAY), cv.cvtColor(roi,cv.COLOR_BGR2GRAY))
                    # similarity = classify_hist_with_split(frame_roi, roi)
                    print(similarity)
                    if similarity < 0.45:
                        # 画出这个矩形
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv.putText(frame, "water spot", (x, y-50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)


            # frame = cv.bitwise_and(frame, frame_copy)
            # 调整视频画面大小为原来的二分之一
            frame = cv.pyrDown(frame)
            fgmask = cv.pyrDown(fgmask)
            cv.imshow("frame", frame)
            cv.imshow("fgmask_down", fgmask)
            k = cv.waitKey(int(1000.0/fps)) & 0xff
            # k = cv.waitKey(200) & 0xff
            if k == 27:
                break
        else:
            break

    # cv.polylines(frame)
    cv.destroyAllWindows()
    cap.release()
