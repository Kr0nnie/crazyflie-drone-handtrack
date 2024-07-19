import logging
import time
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

# Initialize Crazyflie URI
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E9')

cflib.crtp.init_drivers()
logging.basicConfig(level=logging.ERROR)

# Initialize hand 
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
hand_grabbed = False

# Convert screen width to initial point
center_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
center_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

def main():
    global hand_grabbed

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        
        reset_estimator(scf)

        # Take off
        hlc = scf.cf.high_level_commander
        hlc.go_to(0, 0, 0.5, 0, 0.5, relative=False)
        time.sleep(2)  

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
                        POINT_x = (palm[0] - center_x) / 160
                        POINT_y = 2.4 - ((palm[1]) / 200)
                    else:
                        if hand_grabbed:
                            hand_grabbed = False


                if hand_grabbed:
                    POINT = np.array([POINT_x,  POINT_y])
                    print("POINT:", POINT)
                    hlc.go_to(POINT[0], 0, POINT[1], 0, 0.5, relative=False)

                cv2.imshow('Hand Tracking', img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            

        except KeyboardInterrupt:
            pass

        finally:
            hlc.land(0.0, 2.0)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
