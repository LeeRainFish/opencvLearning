import collections
import threading
import time
import cv2
import numpy as np
import scipy

def center(points):
    """calculates centroid of a given matrix"""
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    x = int (x)
    y  = int(y)
    return x, y


def hist_similarity(img1, img2):
    # 计算灰度单通道的直方图的相似值
    hist1 = cv2.calcHist([img1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = float(degree) / len(hist1)
    return degree


def getColorDicList():
    dict = collections.defaultdict(list)
    # 红色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red'] = color_list

    # 红色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    # pink
    lower_pink = np.array([160, 0, 0])
    upper_pink = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_pink)
    color_list.append(upper_pink)
    dict['pink'] = color_list
    #
    # 绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    return dict


def hsv_change(origin_img):
    # 转换色域
    img_copy = origin_img.copy()
    hsv_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    color_dict = getColorDicList()
    ret = np.zeros_like(origin_img)
    for d in color_dict:
        mask_gray = cv2.inRange(hsv_img, color_dict[d][0], color_dict[d][1])
        img_copy = cv2.bitwise_and(origin_img, origin_img, mask=mask_gray)
        ret = cv2.bitwise_or(ret, img_copy)
    return ret

