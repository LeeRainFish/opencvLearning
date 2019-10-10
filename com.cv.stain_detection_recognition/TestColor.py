import cv2
import numpy as np
from carColors import getColorDicList

filename = 'img/sample4.jpg'


# 处理图片
def get_color(origin_img):
    # 转换色域
    hsv_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = getColorDicList()
    for d in color_dict:
        mask = cv2.inRange(hsv_img, color_dict[d][0], color_dict[d][1])
        ret,binary_img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        binary_img = cv2.dilate(binary_img, None, iterations=5)

        img, cnts, hiera = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 计算白色区域数量
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if sum > maxsum:
            maxsum = sum
            color = d

    return color# 处理图片
def hsv_change(origin_img):
    # 转换色域
    hsv_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)
    color_dict = getColorDicList()
    for d in color_dict:
        hsv_img = cv2.inRange(hsv_img, color_dict[d][0], color_dict[d][1])
    return hsv_img


if __name__ == '__main__':
    frame = cv2.imread(filename)
    print(get_color(frame))
