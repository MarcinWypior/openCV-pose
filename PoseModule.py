import cv2
import mediapipe as mp
import time


class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackingCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,model_complexity=self.upBody,smooth_landmarks= self.smooth,
                                     min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackingCon)

        static_image_mode = False,
        model_complexity = 1,
        smooth_landmarks = True,
        enable_segmentation = False,
        smooth_segmentation = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5


    def findPose(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self,img, draw=True):
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList
def main():
    cap = cv2.VideoCapture("videos/1 Pajacyki.mp4")
    detector = poseDetector()
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

        cv2.imshow("frame", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
