import cv2

if __name__ =="__main__":

    img = cv2.imread()

    size = img.shape[:2]

    # 1 filename 2 编码器 3帧率 4size
    writer = cv2.VideoWriter("filename", -1, 5, size)

    for i in range(1,11):
        filename = "image"+str(i)+".jpg"
        img = cv2.imread(filename)
        writer.write(img)


