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
 
URI = 'radio://0/90/2M/E7E7E7E709'  # Adjust this to your Crazyflie configuration
 
# Initialize the low-level drivers
cflib.crtp.init_drivers(enable_debug_driver=False)
 
def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=1)  # Only track one hand (left hand)
 
    # Test drone connection first
    test_drone_connection(URI)
 
    with SyncCrazyflie(URI) as scf:
        # We take off when the commander is created
        with MotionCommander(scf) as mc:
            print('Taking off!')
            time.sleep(1)
 
            try:
                while True:
                    success, img = cap.read()
                    hands, img = detector.findHands(img)
 
                    if hands and hands[0]["type"] == "Left":  # Check if the left hand is detected
                        fingers = detector.fingersUp(hands[0])
                        fingers_count = sum(fingers)
 
                        if fingers_count == 5:  # If 4 or 5 fingers are up, move up
                            mc.up(0.05)
                            print('Moving up')
                        elif fingers_count == 3:  # If 0 or 1 finger is up, move down
                            mc.down(0.05)
                            print('Moving down')
 
                    cv2.imshow("Image", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                print('Landing!')
                cap.release()
                cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()