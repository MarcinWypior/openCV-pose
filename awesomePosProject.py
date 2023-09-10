import cv2
import mediapipe as mp
import time
import PoseModule


cap = cv2.VideoCapture(0)
detector = PoseModule.poseDetector()
pTime = 0

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img,draw=False)
    # cv2.circle(img,(lmList[14][1],lmList[14][2]) , 25, (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # img = cv2.flip(img, 1)
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)

    img = cv2.flip(img,1)

    cv2.imshow("frame", img)
    cv2.waitKey(1)