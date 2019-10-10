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
    img_dir = root_dir+"/res/carcode.jpg"

    # 为了更高的准确率，使用二值图像
    img = cv.imread(img_dir)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh_img = cv.threshold(gray, 155, 255, type=cv.THRESH_BINARY)
    cv_show("thresh_img",thresh_img)
    print(ret)
    # 寻找轮廓
    binary,contours,hierarchy = cv.findContours(image=thresh_img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    draw_img= img.copy()

    # 传入绘制图像,轮廓,轮廓索引（-1 默认为所有轮廓）,颜色,线条厚度
    # cv.drawContours(draw_img,contours,-1,(0,0,255),2)
    cv.drawContours(draw_img,contours,-1,(0,0,255),2)


    #轮廓特征
    for index,c in enumerate(contours):
        # 轮廓面积
        if cv.contourArea(c) > 900:
            cnt = c
            cv.drawContours(draw_img, contours, index, (255, 0, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # break

    # 周长 Ture表示闭合
    # cv.arcLength(cnt,True)

    # 轮廓近似
    # epsilon
    epsilon = 0.1*cv.arcLength(cnt,True)
    approx_poly_dp = cv.approxPolyDP(cnt, epsilon, True)

    # cv.drawContours(draw_img,[approx_poly_dp],-1,(255,0,0),2)
    # cv.drawContours(draw_img,[cnt],-1,(255,0,0),2)
    cv_show("draw_img",draw_img)

    # 绑定为一个边界矩形
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(draw_img,(x,y),(x+w,y+h),(0,255,0),2)
    cv_show("draw_img", draw_img)

    rect_area = w*h
    area = cv.contourArea(cnt)
    extent = float(rect_area)/area
    print("轮廓面积与边界矩形比",extent)

    # 外接圆

    (x,y),radius = cv.minEnclosingCircle(cnt)

    center =(int(x),int(y))
    radius = int(radius)
    cv.circle(draw_img,center=center,radius=radius,color=(0,255,0),thickness=2)
    cv_show("draw_img", draw_img)


