import cv2
import time
import math
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cvzone.HandTrackingModule import HandDetector

def calculate_hand_angle(wrist, tip):
    deltaY = tip[1] - wrist[1]
    deltaX = tip[0] - wrist[0]
    angle_radians = math.atan2(deltaY, deltaX)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

# Initialize the low-level drivers without debug info
cflib.crtp.init_drivers(enable_debug_driver=False)

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    uri = "radio://0/80/2M/E7E7E7E7E7"  # Your Crazyflie configuration URI

    with SyncCrazyflie(uri) as scf:
        with MotionCommander(scf) as mc:
            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)

                if hands:
                    hand = hands[0]  # We use the first hand detected
                    wrist = hand["lmList"][0]
                    tip_of_middle_finger = hand["lmList"][12]
                    hand_angle = calculate_hand_angle(wrist, tip_of_middle_finger)

                    # Control logic based on hand angle
                    if -160 <= hand_angle <= -110:
                        mc.right(0.01, velocity=2)  # Move drone to the right
                        print('Drone Moving Right')
                    elif -50 <= hand_angle <= 0:
                        mc.left(0.01, velocity=2)  # Move drone to the left
                        print('Drone Moving Left')
                    else:
                        action = 'Holding Position'

                    # Display action on the screen

                cv2.imshow("Image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print('Landing')
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
