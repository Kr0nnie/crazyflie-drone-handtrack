import logging
import time
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

# Initialize Crazyflie URIs
URI_A = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E7E9')
URI_B = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E9')
URI_C = uri_helper.uri_from_env(default='radio://0/90/2M/E7E7E7E7E9')

cflib.crtp.init_drivers()
logging.basicConfig(level=logging.ERROR)

# Initialize hand tracking
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
hand_grabbed = False

# Convert screen width to initial point
center_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
center_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

# Initial positions
positions = {
    'A': [0, -1.0, 0.5],
    'B': [0, 0, 0.5],
    'C': [0, 1.0, 0.5]
}

# Delays for following
delay_B = 0.3  # seconds delay for Drone B
delay_C = 0.6  # seconds delay for Drone C

# Last update times
last_update_B = time.time()
last_update_C = time.time()

def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

def main():
    global hand_grabbed, positions, last_update_B, last_update_C

    # Connect to Crazyflies
    with SyncCrazyflie(URI_A, cf=Crazyflie(rw_cache='./cache')) as scf_A, \
         SyncCrazyflie(URI_B, cf=Crazyflie(rw_cache='./cache')) as scf_B, \
         SyncCrazyflie(URI_C, cf=Crazyflie(rw_cache='./cache')) as scf_C:
        
        # Reset estimator
        reset_estimator(scf_A)
        reset_estimator(scf_B)
        reset_estimator(scf_C)

        # Take off
        hlc_A = scf_A.cf.high_level_commander
        hlc_B = scf_B.cf.high_level_commander
        hlc_C = scf_C.cf.high_level_commander
        
        hlc_A.go_to(positions['A'][0], positions['A'][1], positions['A'][2], 0, 2.0)
        hlc_B.go_to(positions['B'][0], positions['B'][1], positions['B'][2], 0, 2.0)
        hlc_C.go_to(positions['C'][0], positions['C'][1], positions['C'][2], 0, 2.0)
        time.sleep(2)  # Wait for takeoff

        try:
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    break

                img = cv2.flip(img, 1)
                hands, img = detector.findHands(img)

                if hands:
                    hand = hands[0]
                    lmList = hand['lmList']
                    fingers = detector.fingersUp(hand)
                    palm = lmList[9]

                    if fingers == [0, 0, 0, 0, 0]:
                        hand_grabbed = True
                        positions['A'][0] = (palm[0] - center_x) / 150
                        positions['A'][2] = (480 - palm[1]) / 400
                    else:
                        if hand_grabbed:
                            hand_grabbed = False

                if hand_grabbed:
                    Pos_A = np.array([positions['A'][0], positions['A'][2]])
                    print("Pos_A:", Pos_A)

                    hlc_A.go_to(Pos_A[0], positions['A'][1], Pos_A[1], 0, 0.5, relative=False)

                    current_time = time.time()
                    if current_time - last_update_B >= delay_B:
                        hlc_B.go_to(Pos_A[0], positions['B'][1], Pos_A[1], 0, 0.5, relative=False)
                        last_update_B = current_time
                    
                    if current_time - last_update_C >= delay_C:
                        hlc_C.go_to(Pos_A[0], positions['C'][1], Pos_A[1], 0, 0.5, relative=False)
                        last_update_C = current_time

                cv2.imshow('Hand Tracking', img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass

        finally:
            hlc_A.land(0.0, 2.0)
            hlc_B.land(0.0, 2.0)
            hlc_C.land(0.0, 2.0)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
