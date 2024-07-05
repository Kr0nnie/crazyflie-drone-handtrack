from cvzone.HandTrackingModule import HandDetector
import cv2
import math

def calculate_hand_angle(wrist, tip):
    """
    Calculate the angle of the line between the wrist and tip of the middle finger with respect to the horizontal axis.
    """
    deltaY = tip[1] - wrist[1]
    deltaX = tip[0] - wrist[0]
    angle_radians = - math.atan2(deltaY, deltaX)  # Calculate angle in radians
    angle_degrees = math.degrees(angle_radians)  # Convert to degrees
    return angle_degrees

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        for hand in hands:
            lmList = hand["lmList"]  # Get landmark list
            wrist = lmList[0]  # Wrist coordinate
            tip_of_middle_finger = lmList[12]  # Tip of the middle finger coordinate
            hand_angle = calculate_hand_angle(wrist, tip_of_middle_finger)

            hand_type = "Right" if hand["type"] == "Right" else "Left"
            text = f'{hand_type} Hand Angle: {hand_angle:.2f} degrees'
            
            # Positioning the text on the video feed
            org = (wrist[0] + 20, wrist[1] + 20)  # Just slightly offset from the wrist point
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.6
            color = (255, 8, 127)  # White color text
            thickness = 2
            img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

            fingers = detector.fingersUp(hand)
            fingers_text = f'{hand_type} Fingers Up: {fingers}'
            img = cv2.putText(img, fingers_text, (org[0], org[1] + 20), font, fontScale, color, thickness, cv2.LINE_AA)
            
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
