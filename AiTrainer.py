import cv2
import numpy as np
import time
import PoseModule as pm

# skończyłeś na 4:38  18
#cap = cv2.VideoCapture("AiTrainer/Stop Doing This Bicep Curl Mistake.mp4")
cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
count =0
dir = 0


while True:
    success, img = cap.read()
    # img = cv2.resize(img, (1280,720))
    #img = cv2.imread("AiTrainer/pullup.jpg")
    img = detector.findPose(img=img,draw=False)
    lmList = detector.findPosition(img, False)
    #print(lmList)

    if len(lmList) != 0:
        #right arm
        #detector.findAngle(img,11,13,15)
        #left arm
        angle = detector.findAngle(img,12,14,16)
        per = np.interp(angle,(210,310),(0,100))
        print(per)



    cv2.imshow("Image",img)
    cv2.waitKey(1)