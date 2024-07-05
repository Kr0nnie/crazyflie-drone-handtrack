import logging
import time
import cv2
import math
from cvzone.HandTrackingModule import HandDetector
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

cflib.crtp.init_drivers()

logging.basicConfig(level=logging.ERROR)

def calculate_hand_angle(wrist, tip):
    deltaY = tip[1] - wrist[1]
    deltaX = tip[0] - wrist[0]
    angle_radians = math.atan2(deltaY, deltaX)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def position_callback(timestamp, data, logconf):
    x = data['kalman.stateX']
    y = data['kalman.stateY']
    z = data['kalman.stateZ']
    print(f'Position: x={x:.2f}, y={y:.2f}, z={z:.2f}')

def start_position_logging(scf):
    log_conf = LogConfig(name='Position', period_in_ms=500)
    log_conf.add_variable('kalman.stateX', 'float')
    log_conf.add_variable('kalman.stateY', 'float')
    log_conf.add_variable('kalman.stateZ', 'float')
    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(position_callback)
    log_conf.start()

def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        reset_estimator(scf)
        start_position_logging(scf)

        hlc = scf.cf.high_level_commander
        hlc.takeoff(0.5, 2.0)
        time.sleep(1)

        try:
            while True:
                success, img = cap.read()
                hands, img = detector.findHands(img)

                if hands:
                    hand = hands[0]
                    wrist = hand["lmList"][0]
                    tip_of_middle_finger = hand["lmList"][12]
                    hand_angle = calculate_hand_angle(wrist, tip_of_middle_finger)
                    fingers = detector.fingersUp(hand)
                    fingers_count = sum(fingers)

                    if fingers_count == 5:
                        hlc.up(0.1)
                        print('Moving up')
                    elif fingers_count == 4:
                        hlc.down(0.1)
                        print('Moving down')
                    elif -160 <= hand_angle <= -110:
                        hlc.right(0.1)
                        print('Moving right')
                    elif -50 <= hand_angle <= 0:
                        hlc.left(0.1)
                        print('Moving left')

                cv2.imshow("Image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            hlc.land(0.5, 2.0)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
