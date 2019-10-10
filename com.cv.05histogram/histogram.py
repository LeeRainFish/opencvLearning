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
    img_toma_dir = root_dir + '/res/toma.jpg'  # 根据项目所在路径，找到用例所在的相对项目的路径

    img= cv.imread(img_toma_dir, cv.IMREAD_GRAYSCALE)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    hist.shape

    equalize_hist = cv.equalizeHist(img)
    plt.hist(equalize_hist.ravel(),256)
    # plt.show()

    cv_show("origin_hist",np.hstack((img,equalize_hist)))

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
    res_clach = clahe.apply(img)



    colors = ('b', 'g', 'r')
    img= cv.imread(img_toma_dir)
    for i,color in enumerate(colors):
        calc_hist = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(calc_hist,color=color)
        plt.xlim([0,256])


    # 创建mask
    mask= np.zeros(img.shape[0:2], dtype=np.uint8)
    mask[100:300,100:400] =255
    cv_show("mask",mask)

    masked_img = cv.bitwise_and(img, img, mask=mask) #与操作
    cv_show("maskimg",masked_img)

    hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])

    plt.subplot(221),plt.imshow(img,"gray")
    plt.subplot(222),plt.imshow(mask,"gray")
    plt.subplot(223),plt.imshow(masked_img,"gray")
    plt.subplot(224),plt.imshow(hist_full,"gray"),plt.plot(hist_mask)
    plt.xlim([0,256])
    plt.show()
