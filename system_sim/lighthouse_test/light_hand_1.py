import cv2
import time
import math
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cvzone.HandTrackingModule import HandDetector

# Initialize the low-level drivers
cflib.crtp.init_drivers(enable_debug_driver=False)

# Calculate the angle of the hand
def calculate_hand_angle(wrist, tip):
    deltaY = tip[1] - wrist[1]
    deltaX = tip[0] - wrist[0]
    angle_radians = math.atan2(deltaY, deltaX)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

# Log the position callback
def position_callback(timestamp, data, logconf):
    x = data['kalman.stateX']
    y = data['kalman.stateY']
    z = data['kalman.stateZ']
    print(f'Position: x={x:.2f}, y={y:.2f}, z={z:.2f}')

# Start logging the position
def start_position_logging(scf):
    log_conf = LogConfig(name='Position', period_in_ms=500)
    log_conf.add_variable('kalman.stateX', 'float')
    log_conf.add_variable('kalman.stateY', 'float')
    log_conf.add_variable('kalman.stateZ', 'float')
    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(position_callback)
    log_conf.start()

# Wait for the position estimator to find the position
def wait_for_position_estimator(scf):
    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')
    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10
    threshold = 0.001
    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]
            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)
            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)
            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break

# Reset the position estimator
def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    wait_for_position_estimator(cf)

def main():
    uri = 'radio://0/80/2M/E7E7E7E7E7'
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    with SyncCrazyflie(uri) as scf:
        with MotionCommander(scf) as mc:
            reset_estimator(scf)
            start_position_logging(scf)

            print('Taking off!')
            mc.take_off()
            time.sleep(1)

            try:
                while True:
                    success, img = cap.read()
                    hands, img = detector.findHands(img)

                    if hands:
                        hand = hands[0]
                        fingers = detector.fingersUp(hand)
                        fingers_count = sum(fingers)
                        wrist = hand["lmList"][0]
                        tip_of_middle_finger = hand["lmList"][12]
                        hand_angle = calculate_hand_angle(wrist, tip_of_middle_finger)

                        if fingers_count == 5:
                            mc.up(0.1)
                            print('Moving up')
                        elif fingers_count == 4:
                            mc.down(0.1)
                            print('Moving down')
                        elif -160 <= hand_angle <= -110:
                            mc.right(0.1)
                            print('Moving right')
                        elif -50 <= hand_angle <= 0:
                            mc.left(0.1)
                            print('Moving left')

                    cv2.imshow("Image", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            finally:
                print('Landing!')
                mc.land()
                cap.release()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
