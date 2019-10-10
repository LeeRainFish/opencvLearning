import numpy as np
import cv2 as cv

# 背景建模法
if __name__ == "__main__":
    cap = cv.VideoCapture("D:\develop\python3\jupyter\opencv\img\\all1.mp4")

    # 角点检测需要的参数
    feature_params = dict(maxCorners=100,
                           qualityLevel=0.3,
                           minDistance=7)

    lk_params = dict(winSize=(15, 15), maxLevel=2)
    
    #随机线条颜色
    colors = np.random.randint(0, 255, (100, 3))

    # 获取第一帧图像

    ret,old_frame = cap.read()
    old_frame = cv.pyrDown(old_frame)

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    # p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    h,w = old_gray.shape[:2]
    print(h,w)
    p0 = np.array([[[h/2,w/2]]],dtype=np.float32)
    print("p0:")
    print(p0)
    # print(p0)
    # print(type(p0))
    # print(p0)
    mask = np.zeros_like(old_frame)
    img = np.zeros_like(old_frame)

    while (True):
        ret,frame = cap.read()
        frame = cv.pyrDown(frame)
        if ret ==False:
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts[, status[, err[, winSize[, maxLevel[, criteria[, flags[,
        #                      minEigThreshold]]]]]]]) -> nextPts, status, err

        p1,st,err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # print("err: "+err)
        # cv.calcOpticalFlowFarneback()
        print("p1: ")
        print(p1)
        # print(st)
        # print(err)
        if(np.all(p1!=None)):
            good_new=p1[st==1]
            good_old=p0[st==1]

            # 绘制轨迹
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
                frame = cv.circle(frame, (a,b), 30, colors[i].tolist(), -1)
                # print("a,b:")
                print(a, b)
            img = cv.add(frame, mask)

            cv.imshow("frame", img)
            k = cv.waitKey(100) & 0xff
            if k==27:
                break

            # 更新
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            cv.imshow("frame", img)
            k = cv.waitKey(100) & 0xff
            if k == 27:
                break
            old_gray = frame_gray.copy()
    cv.destroyAllWindows()
    cap.release()
        