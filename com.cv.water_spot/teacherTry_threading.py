
from myUtils import *

flag_spot = False
flag_color = False
flag_line = False

lock_spot = threading.Lock()
lock_line = threading.Lock()
lock_color = threading.Lock()


def task_spot(img):

    global flag_spot
    while True:



        pass


def task_color(img):

    global flag_color
    while True:

        pass


def task_line(img):

    global  flag_line
    while True:
        # 记录滴液质心位置
        points_list = []
    pass


def main():
    #
    # event_obj = threading.Event()  # 创建event事件对象
    # t_spot = threading.Thread(target=task_spot, args=(event_obj))
    # t_color = threading.Thread(target=task_color, args=(event_obj))
    # t_line = threading.Thread(target=task_line, args=(event_obj))
    # #
    # t_spot .setDaemon(True)
    # t_line.setDaemon(True)
    # t_color.setDaemon(True)
    #
    # t_spot.start()
    # t_line.start()
    # t_color.start()

    # 测试视频
    videoUrl = "D:\develop\python3\jupyter\water-spot-detection\\pink-red-drop.mp4"
    cap = cv2.VideoCapture(videoUrl)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 形态学操作需要内核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # 基于KNN的背景/前景分割算法
    history = 10
    fgbg = cv2.createBackgroundSubtractorKNN()
    fgbg.setHistory(history)
    ret, orignal = cap.read()
    orignal = cv2.resize(orignal, (0, 0), None, fx=0.5, fy=0.5)

    while (True):
        ret, frame = cap.read()
        if ret == False:
            break

            frame = cv.resize(frame, (0, 0), None, fx=0.5, fy=0.5)
            # 算法比对
            fgmask = fgbg.apply(frame)
            # 建立一下历史
            if history + 10 > 0:
                history -= 1
                continue
            frame_copy = frame.copy()
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
            fgmask = cv.dilate(fgmask, None, iterations=3)
            binary, contours, hierarchy = cv.findContours(image=fgmask, mode=cv.RETR_EXTERNAL,
                                                          method=cv.CHAIN_APPROX_NONE)
            for c in contours:
                area = cv.contourArea(c)
                if area > 200:
                    x, y, w, h = cv.boundingRect(c)
                    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    gravity = center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
                    points_list.append((int(gravity[0]), int(gravity[1])))
                    # 轮廓切割roi
                    frame_roi = frame[y:y + h, x:x + w]
                    roi = orignal[y:y + h, x:x + w]
                    # 查看与原来模块roi区域的相似度
                    similarity = hist_similarity(cv.cvtColor(frame_roi, cv.COLOR_BGR2GRAY),
                                                 cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
                    print(similarity)
                    if similarity < 0.5:
                        # 画出这个矩形
                        cv.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv.putText(frame_copy, "water spot", (x, y - 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            #
            if len(points_list) > 20:
                print(points_list)
                points_list.clear()
            #
            for i in range(len(points_list)):
                if i == 0:
                    continue
                cv.line(frame_copy, points_list[i - 1], points_list[i], (255, 255, 0), 2)

            # TODO 根据points_list的点拟合线性函数

            cv.imshow("frame", frame_copy)
            cv.imshow("fgmask", fgmask)
            k = cv.waitKey(int(1000.0 / fps)) & 0xff
            if k == 27:
                break

        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
