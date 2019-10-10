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



def cv_show(name,src):
    cv.imshow(name,src)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ =="__main__":

    img = cv.imread(args["image"])
    img = cv.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
    cv_show("original",img)
    # 灰度
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 中值滤波
    # gray = cv.medianBlur(gray, 5)
    # 直方图处理
    # equalize_hist = cv.equalizeHist(gray)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    equalize_hist = clahe.apply(gray_img)
    cv_show("equalize_hist",equalize_hist)
    
    # 模糊
    blur_img_blur = cv.blur(equalize_hist, (5,5))
    blur_img_medianBlur = cv.medianBlur(equalize_hist,5)
    res = np.hstack((blur_img_blur, blur_img_medianBlur))
    blur_img_GaussianBlur = cv.GaussianBlur(equalize_hist, (5,5),0)
    res = np.hstack((res, blur_img_GaussianBlur))
    cv_show("blur_img",res)


    # 形态学
    kernel = np.ones((3, 3), dtype=np.uint8)

    morphology_img= cv.morphologyEx(blur_img_medianBlur, cv.MORPH_OPEN, kernel)

    # 二值化
    # ret,binary = cv.threshold(gray, 127, 255, type=cv.THRESH_TRUNC)
    ret,binary = cv.threshold(morphology_img , 155, 255, type=cv.THRESH_BINARY_INV)
    print(ret)
    cv_show("binary",binary)

