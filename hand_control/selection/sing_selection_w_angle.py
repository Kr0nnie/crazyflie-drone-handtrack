import cv2
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cvzone.HandTrackingModule import HandDetector
import math


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


def calculate_hand_angle(wrist, tip):
    deltaY = tip[1] - wrist[1]
    deltaX = tip[0] - wrist[0]
    angle_radians = math.atan2(deltaY, deltaX)  # Calculate angle in radians
    angle_degrees = math.degrees(angle_radians)  # Convert to degrees
    return angle_degrees


URI1 = 'radio://0/80/2M/E7E7E7E7E6'  # Adjust this to your Crazyflie configuration

# Initialize the low-level drivers
cflib.crtp.init_drivers(enable_debug_driver=False)

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    test_drone_connection(URI1)

    with SyncCrazyflie(URI1) as scf1:
        with MotionCommander(scf1) as mc1:
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

                        wrist = hand["lmList"][0]  # Wrist coordinate
                        tip_of_middle_finger = hand["lmList"][12]  # Tip of the middle finger
                        hand_angle = calculate_hand_angle(wrist, tip_of_middle_finger)    

                        hand_type = "Right" 
                        text = f'{hand_type} Hand Angle: {hand_angle:.2f} degrees'

                        if hand["type"] == "Left":
                            left_hand = hand
                        elif hand["type"] == "Right":
                            right_hand = hand
                        

                        org = (wrist[0] + 20, wrist[1] + 20)  # Just slightly offset from the wrist point
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.6
                        color = (255, 8, 127)  # White color text
                        thickness = 2
                        img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)                            



                    # Control selected drone with right hand
                    if right_hand:
                        right_fingers = detector.fingersUp(right_hand)
                        right_fingers_count = sum(right_fingers)

                        #left right
                        if right_fingers_count == 1:

                                if -160 <= hand_angle <= -110:
                                    mc1.right(0.01, velocity=2)  # Move drone to the right
                                    print('Drone 1 Moving Right')
                                elif -50 <= hand_angle <= 0:
                                    mc1.left(0.01, velocity=2)  # Move drone to the left
                                    print('Drone 1 Moving Left')



                        #Down
                        elif right_fingers_count == 4:
                                mc1.down(0.01, velocity=2)
                                print('Drone 1: Moving down')


                        #UP
                        elif right_fingers_count == 5:
                                mc1.up(0.01, velocity=2)
                                print('Drone 1: Moving up')






                    cv2.imshow("Image", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            finally:
                print('Landing!')
                cap.release()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
