import cv2 as cv
import numpy as np
import argparse

# 原始-》灰度-》二值 -》轮廓
# 原始-》灰度-》二值 -》形态学-》轮廓 -》切分
# 模板匹配

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to input image")
ap.add_argument("-t","--template",required=True,help="path to input template OCR image")

args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER={
    "3":"American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card",
}

if __name__ =="__main__":

    print()
