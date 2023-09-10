import cv2
import numpy as np
import pyautogui

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set up canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Initialize variables
prev_x, prev_y = 0, 0
draw = False

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip horizontally for a mirrored view

    # Region of Interest (ROI) for hand gesture
    roi = frame[100:400, 300:600]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary image
    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (the hand)
        hand_contour = max(contours, key=cv2.contourArea)

        # Get the centroid of the hand
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if draw:
                # Draw a line on the canvas
                cv2.line(canvas, (prev_x, prev_y), (cX, cY), (0, 0, 255), 5)
                pyautogui.moveTo(cX * 2, cY * 2)

            prev_x, prev_y = cX, cY

    else:
        prev_x, prev_y = 0, 0

    cv2.imshow("Canvas", canvas)
    cv2.imshow("Hand", threshold)

    # Exit the program on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()