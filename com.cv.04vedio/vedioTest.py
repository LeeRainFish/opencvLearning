import numpy as np
import cv2 as cv


def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值
    hist1 = cv.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            # degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
            pass
        else:
            degree = degree + 1
    degree = float (degree) / len(hist1)
    return degree


def classify_hist_with_split(image1, image2):
    # RGB每个通道的直方图相似度
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    sub_image1 = cv.split(image1)
    sub_image2 = cv.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data

if __name__ == "__main__":

    # 测试视频
    cap= cv.VideoCapture("D:\develop\python3\jupyter\opencv\img\VID_20191002_105546.mp4")


    while (True):
        ret,frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret,binary = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)

        cv.imshow("frame",frame)
        cv.imshow("binary",binary)

        k = cv.waitKey(80) & 0xff
        if k==27:
            break

    cap.release()

