import numpy as np
import cv2 as cv
import collections
from myUtils import *
import ransac
import math

font = cv2.FONT_HERSHEY_SIMPLEX


def getBatchs(points, w, slice):
    array = [[] for i in range(slice)]
    for point in points:
        i = math.ceil(point[0] / w * slice) - 1
        array[i].append(point)

    max = 0
    ret = 0
    for i in range(len(array)):
        if len(array[i]) > max:
            max = len(array[i])
            ret = i
    print(array)
    return array[ret]


def left90degree(frame):
    frame = cv.transpose(frame)
    frame = cv.flip(frame, 0)
    return frame


# 背景建模法
if __name__ == "__main__":

    # 测试视频
    videoUrl = "D:\develop\python3\jupyter\water-spot-detection\\pink-red-drop.mp4"
    # videoUrl = "D:\develop\python3\jupyter\water-spot-detection\drop\drop3.mp4"
    # videoUrl = "http://192.168.1.155:8080/video"\    # videoUrl = 0
    cap = cv.VideoCapture(videoUrl)

    fps = cap.get(cv.CAP_PROP_FPS)
    fwidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    fheight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(fwidth)
    print(fheight)
    # 形态学操作需要内核
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    # 基于KNN的背景/前景分割算法
    history = 5
    fgbg = cv.createBackgroundSubtractorKNN()
    fgbg.setHistory(history)
    ret, orignal = cap.read()
    # cv2.imwrite("D:\develop\python3\jupyter\water-spot-detection\\drop.jpg",orignal)
    # 左转90度
    orignal = left90degree(orignal)

    # 记录滴液质心位置
    points_list = []

    # 拟合数据
    n_inputs = 1
    n_outputs = 1
    input_columns = range(n_inputs)  # the first columns of the array
    output_columns = [n_inputs + i for i in range(n_outputs)]  # the last columns of the array
    model = ransac.LinearLeastSquaresModel(input_columns, output_columns)
    cv2.namedWindow("frame", 0)
    cv2.resizeWindow("frame", 960, 720)
    cv2.namedWindow("fgmask", 0)
    cv2.resizeWindow("fgmask", 960, 720)

    like = np.zeros((int(fheight), int(fwidth)), dtype=np.uint8)
    spots = 0

    while (True):
        ret, frame = cap.read()
        # 左转90度
        frame = left90degree(frame)

        if ret == False:
            break

        # 算法比对
        fgmask = fgbg.apply(frame)
        # 建立一下历史
        # if history + 10 > 0:
        #     history -= 1
        #     continue

        frame_copy = frame.copy()

        # frame = cv.blur(frame, (5, 5))
        # 形态学开运算加腐蚀去噪点
        rettt, fgmask = cv.threshold(fgmask, 170, 255, cv.THRESH_BINARY)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        fgmask = cv.dilate(fgmask, None, iterations=3)
        # 寻找轮廓
        binary, contours, hierarchy = cv.findContours(image=fgmask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        if (len(contours) > 0):
            print("有%d个轮廓" % int(len(contours)))
        for c in contours:
            # 计算各轮廓面积
            area = cv.contourArea(c)

            x, y, w, h = cv.boundingRect(c)

            # if w<fwidth/20 and area > 500:
            if w < fwidth / 20 and h < fheight / 20 and area > 500:

                cv.imwrite("D:\develop\python3\jupyter\water-spot-detection\drop\img\drop3-" + str(spots) + ".jpg", frame_copy)
                spots += 1
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                gravity = center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])

                points_list.append((gravity[0], gravity[1]))
                # 轮廓切割roi
                frame_roi = frame[y:y + h, x:x + w]
                roi = orignal[y:y + h, x:x + w]
                # 查看与原来模块roi区域的相似度
                similarity = hist_similarity(cv.cvtColor(frame_roi, cv.COLOR_BGR2GRAY),
                                             cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
                # similarity = classify_hist_with_split(frame_roi, roi)
                # print(similarity)
                if similarity < 0.5:
                    # 画出这个矩形
                    cv.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv.putText(frame_copy, "water spot", (x, y - 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        #
        for i in range(len(points_list)):
            if i == 0:
                continue
            cv.line(frame_copy, points_list[i - 1], points_list[i], (255, 255, 0), 2)
        #
        np_array = []
        if len(points_list) > 40:
            points_list = getBatchs(points_list, fwidth, 20)
            print(points_list)
            for i in range(len(points_list)):
                # like = np.zeros_like((960, 720))
                # cv.imwrite("like2.jpg", like)
                np_array.append(list(points_list[i]))
            np_array = np.array(np_array, dtype=np.float64)
            try:
                #
                # ransac_fit, ransac_data = ransac.ransac(np_array, model,
                #                                         5, 500, 3e5, 20,  # misc. parameters
                #                                         debug=False, return_all=True)
                ransac_fit, ransac_data =  ransac.ransac(np_array, model,
                                                        math.ceil(len(points_list) / 20),
                                                        500,
                                                        3e5,
                                                        math.ceil(len(points_list) / 10 * 3),
                                                        # misc. parameters
                                                        debug=False, return_all=True)


            except ValueError as e:
                print("not find", e)
                pass
            else:
                print("ransac_fit:", ransac_fit)

                k = np.max(ransac_fit[1])
                if k < 0:
                    k = -k
                print("K:", k)
                angle = math.atan2(k, 1)
                theta = angle * (180 / math.pi)
                if theta > 90:
                    print(theta)
                    theta = 180 - theta
                if (theta >= 80 and theta <= 90):
                    cv2.putText(frame_copy, "water line appear", (20, 100)
                                , font, 1,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA)
                print("theta:", theta, "度")

                print(points_list)


                print(list)
                points_list.clear()

        # TODO 根据points_list的点拟合线性函数

        cv.imshow("frame", frame_copy)
        cv.imshow("fgmask", fgmask)
        k = cv.waitKey(int(1000.0 / 40)) & 0xff
        # k = cv.waitKey(70) & 0xff
        if k == 27:
            break

    cv.destroyAllWindows()
    cap.release()
