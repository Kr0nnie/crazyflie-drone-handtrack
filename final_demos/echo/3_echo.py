import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

# Initialize Crazyflie URI for Drone A
URI_A = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E9')

cflib.crtp.init_drivers()

# Initialize hand tracking
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
hand_grabbed = False

# Drone A initial position (y-axis value is fixed)
position_A = [0, -1.5, 0.5]

def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

def position_control(scf, position):
    with scf.cf.high_level_commander as hlc:
        hlc.go_to(position[0], position[1], position[2], 0, 2.0)

def main():
    global hand_grabbed, position_A

    # Connect to Crazyflie
    with SyncCrazyflie(URI_A, cf=Crazyflie(rw_cache='./cache')) as scf_A:
        
        # Reset estimator
        reset_estimator(scf_A)

        # Take off
        position_control(scf_A, position_A)
        time.sleep(3)  # Wait for takeoff

        try:
            while True:
                success, img = cap.read()
                if not success:
                    print("Error: Could not read frame from video capture.")
                    break

                img = cv2.flip(img, 1)
                hands, img = detector.findHands(img)
                cv2.imshow("Image", img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if hands:
                    hand = hands[0]
                    lmList = hand['lmList']
                    fingers = detector.fingersUp(hand)

                    if fingers == [0, 0, 0, 0, 0]:
                        hand_grabbed = True
                        palm = lmList[0]
                        position_A[0] = (palm[0] - 320) / 150  # Convert screen coordinates to drone coordinates
                        position_A[2] = (palm[1] - 240) / 150

                        # Update Drone A position
                        position_control(scf_A, position_A)

        except KeyboardInterrupt:
            pass

        finally:
            # Land the drone
            scf_A.cf.high_level_commander.land(0.0, 2.0)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
