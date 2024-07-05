import cv2
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cvzone.HandTrackingModule import HandDetector

# Drone connection test function
def test_drone_connection(uri):
    cf = Crazyflie(rw_cache='./cache')

    def connected(link_uri):
        print(f"Connected successfully to {link_uri}")
        return True

    def connection_failed(link_uri, msg):
        print(f"Connection to {link_uri} failed: {msg}")
        return False

    def disconnected(link_uri):
        print(f"Disconnected from {link_uri}")

    cf.connected.add_callback(connected)
    cf.connection_failed.add_callback(connection_failed)
    cf.disconnected.add_callback(disconnected)

    cf.open_link(uri)
    time.sleep(2)  # Wait a bit to see if it connects
    cf.close_link()

URI1 = 'radio://0/60/2M/E7E7E7E706'  # Adjust this to your Crazyflie configuration
URI2 = 'radio://0/90/2M/E7E7E7E709'  # Adjust this to your second Crazyflie configuration

# Initialize the low-level drivers
cflib.crtp.init_drivers(enable_debug_driver=False)

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    test_drone_connection(URI1)
    test_drone_connection(URI2)

    with SyncCrazyflie(URI1) as scf1, SyncCrazyflie(URI2) as scf2:
        with MotionCommander(scf1) as mc1, MotionCommander(scf2) as mc2:
            print('Taking off!')
            time.sleep(1)

            drone_selection = None  # This will determine which drone(s) to control

            try:
                while True:
                    success, img = cap.read()
                    hands, img = detector.findHands(img)

                    left_hand = None
                    right_hand = None

                    for hand in hands:
                        if hand["type"] == "Left":
                            left_hand = hand
                        elif hand["type"] == "Right":
                            right_hand = hand

                    # Determine drone selection left hand
                    if left_hand:
                        left_fingers = detector.fingersUp(left_hand)
                        left_fingers_count = sum(left_fingers)

                        if left_fingers_count == 1:
                            drone_selection = 'drone1'
                        elif left_fingers_count == 2:
                            drone_selection = 'drone2'
                        elif left_fingers_count == 5:
                            drone_selection = 'all'
                        else:
                            drone_selection = None

                    # Control selected drone with right hand
                    if right_hand and drone_selection:
                        right_fingers = detector.fingersUp(right_hand)
                        right_fingers_count = sum(right_fingers)

                        if right_fingers_count == 3:
                            if drone_selection in ['drone1', 'all']:
                                mc1.down(0.01, velocity=2)
                                print('Drone 1: Moving down')
                            if drone_selection in ['drone2', 'all']:
                                mc2.down(0.01, velocity=2)
                                print('Drone 2: Moving down')

                        elif right_fingers_count == 5:
                            if drone_selection in ['drone1', 'all']:
                                mc1.up(0.01, velocity=2)
                                print('Drone 1: Moving up')
                            if drone_selection in ['drone2', 'all']:
                                mc2.up(0.01, velocity=2)
                                print('Drone 2: Moving up')

                    cv2.imshow("Image", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            finally:
                print('Landing!')
                cap.release()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
