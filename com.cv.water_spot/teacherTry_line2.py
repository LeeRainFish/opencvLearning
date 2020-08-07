import numpy as np
import cv2 as cv
import collections
from myUtils import *
import ransac
import math

font = cv2.FONT_HERSHEY_SIMPLEX

def left90degree(frame):
    frame = cv.transpose(frame)
    frame = cv.flip(frame, 0)
    return frame

# 背景建模法
if __name__ == "__main__":

    # 测试视频
    videoUrl = "D:\develop\python3\jupyter\water-spot-detection\\pink-red-drop.mp4"
    cap = cv.VideoCapture(videoUrl)

    fps = cap.get(cv.CAP_PROP_FPS)
    # 形态学操作需要内核
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    # 基于KNN的背景/前景分割算法
    history = 5
    fgbg = cv.createBackgroundSubtractorKNN()
    fgbg.setHistory(history)
    ret, orignal = cap.read()
    # 左转90度
    orignal = left90degree(orignal)
    # orignal = cv.resize(orignal, (0, 0), None, fx=0.5, fy=0.5)
    # orignal = cv.blur(orignal, (5, 5))
    # 记录滴液质心位置
    points_list = []

    # 拟合数据
    n_inputs = 1
    n_outputs = 1
    input_columns = range(n_inputs)  # the first columns of the array
    output_columns = [n_inputs + i for i in range(n_outputs)]  # the last columns of the array
    model = ransac.LinearLeastSquaresModel(input_columns, output_columns)
    while (True):
        ret, frame = cap.read()
        # 左转90度
        frame = left90degree(frame)

        if ret == False:
            break

        # 算法比对
        fgmask = fgbg.apply(frame)
        # 建立一下历史
        if history + 10 > 0:
            history -= 1
            continue

        frame_copy = frame.copy()

        # frame = cv.blur(frame, (5, 5))
        # 形态学开运算加腐蚀去噪点
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        fgmask = cv.dilate(fgmask, None, iterations=3)
        # 寻找轮廓
        binary, contours, hierarchy = cv.findContours(image=fgmask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        for c in contours:
            # 计算各轮廓面积
            area = cv.contourArea(c)

            if area > 500:
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(frame, (x, y), (x + h, y + w), (255, 255, 0), 2)
                gravity = center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
                # gravity = center(x, y, w, h)
                # points_list.append([int(gravity[0]), int(gravity[1])])
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
        if len(points_list) > 50:
            for i in range(len(points_list)):
                np_array.append(list(points_list[i]))
            np_array = np.array(np_array, dtype=np.float64)
            lstsq_array = np_array[:, input_columns]
            lstsq_array = lstsq_array.reshape((1, -1))
            A = np.vstack([lstsq_array**0,lstsq_array**1])

            lstsq_array = np_array[:, output_columns]
            lstsq_array = lstsq_array.ravel()

            print("A:",A)
            print("B:",lstsq_array)
            np_array2  = np_array
            print(np_array)
            # print(np_array[:, input_columns])
            # print(np_array[:, output_columns])
            try:
                linear_fit, resids, rank, s = scipy.linalg.lstsq(A.T,
                                                                 lstsq_array )
                b = linear_fit[0]
                a = linear_fit[1]
                linear_fit = np.array(linear_fit)
                linear_fit.reshape((2,1))
                print("b:",b)
                print("a:",a)
                # run RANSAC algorithm
                ransac_fit, ransac_data= ransac.ransac(np_array, model,
                                           5, 500, 3e5, 30,  # misc. parameters
                                           debug=False, return_all=True)
                tmp = ransac_fit[0][0]
                ransac_fit[0][0] = ransac_fit[1][0]
                ransac_fit[1][0] = tmp

            except ValueError as e:
                print("not find", e)
                pass
            else :
                print("ransac_fit:", ransac_fit)
                # print(type(ransac_fit))
                np_array = np_array[:, 0]
                np_array = np_array.reshape(-1, 1)
                # print("np_array:",np_array)
                sort_idxs = np.argsort(np_array[:, 0])
                # print("sort_idxs:",sort_idxs)

                k = np.max(ransac_fit[0])
                if k<0:
                    k = -k
                print("K:", k)
                angle = math.atan2(k, 1)
                theta = angle * (180 / math.pi)
                if theta > 90:
                    print(theta)
                    theta = 180 - theta
                if (theta >= 0 and theta <= 22) or (theta >= 70 and theta <= 90):
                    cv2.putText(frame_copy, "water line appear", (20, 100)
                                , font, 1,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA)
                print("theta:", theta, "度")

                # angle = Math.atan2((p2.x - p1.x), (p2.y - p1.y)) // 弧度
                # 0.9272952180016122
                # theta = angle * (180 / Math.PI); // 角度
                # 53.13010235415598
                A_col0_sorted = np_array[sort_idxs]  # maintain as rank-2 array
                import pylab
                pylab.plot(A_col0_sorted[:, 0],
                           np.dot(A_col0_sorted, ransac_fit.T)[:, 0],
                           label='RANSAC fit')
                # pylab.plot(A_col0_sorted[:, 0],
                #            np.dot(A_col0_sorted, linear_fit.T)[:, 0],
                #            label='linear_fit')
                # pylab.scatter(ransac_data[:, 0], ransac_data[:, 1], color="b")
                pylab.scatter(np_array2[:, 0], np_array2[:, 1], color="r")
                # x y 互换
                # pylab.scatter(np_array2[:, 1], np_array2[:, 0], color="r")
                pylab.show()
                print(points_list)
                points_list.clear()

        # TODO 根据points_list的点拟合线性函数
        cv2.namedWindow("frame", 0)
        cv2.resizeWindow("frame", 600, 800)
        cv2.namedWindow("fgmask", 0)
        cv2.resizeWindow("fgmask", 600, 800)
        cv.imshow("frame", frame_copy)
        cv.imshow("fgmask", fgmask)
        k = cv.waitKey(int(1000.0 / fps)) & 0xff
        # k = cv.waitKey(70) & 0xff
        if k == 27:
            break

    cv.destroyAllWindows()
    cap.release()
