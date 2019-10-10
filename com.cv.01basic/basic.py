import cv2
import os
import matplotlib.pyplot as plt
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    root_dir = os.path.dirname(os.path.abspath('.'))  # 获取当前文件(这里是指run_report.py)所在目录的父目录的绝对路径,也就是项目所在路径E:\DDT_Interface
    img_car_dir = root_dir + '/res/target.png'  # 根据项目所在路径，找到用例所在的相对项目的路径
    img_toma_dir = root_dir + '/res/toma.jpg'  # 根据项目所在路径，找到用例所在的相对项目的路径
    img_car = cv2.imread(img_car_dir);
    img_toma = cv2.imread(img_toma_dir);
    # cv_show("image", img_car)
    cv2.imshow("image",img_car)
    # print(root_dir)
    # print(case_dir)

    h,w,c = img_toma.shape
    print(h,w,c)

    img_car = cv2.resize(img_car,dsize=(w,h))  #先宽后高
    img_add = cv2.add(img_car, img_toma) #越界不处理
    # img_add = img_car+img_toma #越界%255
    cv2.imshow("after",img_add)

    weighted = cv2.addWeighted(src1=img_car, alpha=0.4, src2=img_toma, beta=0.6,gamma=0)

    cv2.imshow("weighter",weighted)

    img_car = cv2.resize(img_car, (0, 0), fx=3, fy=4)
    cv2.imshow("car_after",img_car)



    cv2.waitKey(0)
    cv2.destroyAllWindows()


