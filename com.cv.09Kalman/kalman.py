import cv2
import numpy as np

def mousemove(event,x,y,s,p):
    global frame ,lass_measurement , current_measurement ,\
        lass_prediction ,current_prediction ,mesurements
    last_measurement = current_measurement
    last_prediction = current_prediction
    current_measurement = np.array((x,y),np.float32)
    kalman.correct(current_measurement)
    current_prediction = kalman.predict()
    lmx,lmy = last_measurement[:2]
    cmx,cmy = current_measurement[:2]
    lpx,lpy = last_prediction[:2]
    cpx,cpy = current_prediction[:2]

    cv2.line(frame,(lmx,lmy),(cmx,cmy),(0,100,0))
    cv2.line(frame,(lpx,lpy),(cpx,cpy),(0,0,255))





if __name__ =="__main__":
    frame = np.zeros((800, 800, 3), np.uint8)
    last_measurement = current_measurement = np.array((2, 1), np.float32)
    last_prediction = current_prediction = np.zeros((2, 1), np.float32)

    cv2.namedWindow("kalman_tracker")
    cv2.setMouseCallback("kalman_tracker",mousemove)

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)*0.03

    while True:
        cv2.imshow("kalman_tracker",frame)
        if(cv2.waitKey(30)&0xff)==27:
            break

    cv2.destroyAllWindows()