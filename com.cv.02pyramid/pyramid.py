import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


def cv_show(name,src):

    cv.imshow(name,src)
    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":

    root_dir = os.path.dirname(os.path.abspath('.'))  # 获取当前文件(这里是指run_report.py)所在目录的父目录的绝对路径,也就是项目所在路径E:\DDT_Interface
    img_dir = root_dir+"/res/toma.jpg"
    img= cv.imread(img_dir)

    print(img.shape)
    cv_show("img",img)


#     高斯金字塔

    down = cv.pyrDown(img)
    print(down.shape)
    cv_show("down",down)

    up = cv.pyrUp(img)
    print(up.shape)
    cv_show("up",up)



# np.vstack():在竖直方向上堆叠
#
# np.hstack():在水平方向上平铺
# 经过两次操作后与原图比较

    up_down = cv.pyrDown(up)
    cv_show("up_down",np.hstack((img,up_down)))

    img_up_down = img - up_down

    cv_show("img_up_down",img_up_down)