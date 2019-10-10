import cv2

if __name__ =="__main__":

    cap = cv2.VideoCapture("")

    with cv2.VideoCapture() as capture:

        pass

    opened= cap.isOpened()
    fps= cap.get(cv2.CAP_PROP_FPS)
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not opened:

        pass

    i = 0
    while True:
        ret,frame = cap.read()
        i+=1
        if ret ==True:
            frame_copy = frame.copy()
            filename = "image"+str(i)+".jpg"
            cv2.imwrite(filename,frame,[cv2.IMWRITE_JPEG_QUALITY,100])
        else:
            break

    cv2.destroyAllWindows()
    cap.release()