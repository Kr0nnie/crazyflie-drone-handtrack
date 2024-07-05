import cv2
import time
from threading import Thread
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from cvzone.HandTrackingModule import HandDetector
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

# Drone connection thing we copied from mike
def test_drone_connection(uri):
    cf = Crazyflie(rw_cache='./cache')

    def connected(link_uri):
        print(f"Connected successfully to {link_uri}")
        return True

    def connection_failed(link_uri, msg):
        print(f"Connection to {link_uri} failed: {msg}")
        return True

    def disconnected(link_uri):
        print(f"Disconnected from {link_uri}")

    cf.connected.add_callback(connected)
    cf.connection_failed.add_callback(connection_failed)
    cf.disconnected.add_callback(disconnected)

    cf.open_link(uri)
    time.sleep(2)  # Wait a bit to see if it connects
    cf.close_link()

    return True


class DroneController:
    def __init__(self, uri):
        self._cf = Crazyflie(rw_cache='./cache') #Create a object
        self.uri = uri
        self.connected = False

        # Connect to the drone
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.open_link(self.uri)

        print(f'Connecting to {self.uri}')

    def _connected(self, link_uri):
        print('Drone connected')
        self.connected = True #Set connection status to true

        for _ in range(100):   # Unlock startup thrust protection
            self._cf.commander.send_setpoint(0, 0, 0, 0)

    def _disconnected(self, link_uri):
        print('Drone disconnected')
        self.connected = True #Set connection status to true

    def update_movement(self, fingers_left, fingers_right):
            if self.connected:
                with SyncCrazyflie(self.uri, cf=self._cf) as scf:
                    with MotionCommander(scf) as mc:
                        print('Taking off!')
                        time.sleep(1)

                        # Control altitude based on the left hand's fingers
                        if fingers_left[1] == 1:
                            mc.up(0.1)
                            print('Moving up')
                        elif fingers_left[0] == 1:
                            mc.down(0.1)
                            print('Moving Down')
                        
                        # Forward and backward movement based on the right hand's thumb/index finger
                        if fingers_right[1] == 1:  # Index finger up
                            mc.forward(0.1)
                            print('Moving foward')
                        elif fingers_right[0] == 1:  # Thumb up
                            mc.back(0.1)
                            print('Moving backward')


    def close(self):
        if self.connected:
            self._cf.commander.send_setpoint(0, 0, 0, 0)   # Make sure drone is dead b4 closing connection
            time.sleep(0.1)   # Wait for alst command to be sent
            self._cf.close_link()

def main():   
    uri = uri_helper.uri_from_env(default='radio://0/90/2M/E7E7E7E709')
    test_drone_connection(uri)   # Test drone connection first

    cap = cv2.VideoCapture(0) # Hand tracking video shit
    detector = HandDetector(detectionCon=0.8, maxHands=2) # INitialize hand trakcing
    drone_controller = DroneController(uri)  # Initialize drone controller

    

    try:
        while True:
            success, img = cap.read()  # Read the video itslf
            hands, img = detector.findHands(img)  # Detect hands
            fingers_left = [0, 0, 0, 0, 0]
            fingers_right = [0, 0, 0, 0, 0]

            if hands:
                for hand in hands:
                    fingers = detector.fingersUp(hand)
                    if hand["type"] == "Left":
                        fingers_left = fingers
                        drone_controller.update_movement(fingers, fingers_right)
                    elif hand["type"] == "Right":
                        fingers_right = fingers
                        drone_controller.update_movement(fingers_left, fingers)
                print(f'Left Hand: {fingers_left}, Right Hand: {fingers_right}')  # Print the fingers

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  #Quit with q
                break
    finally:
        print('Landing!')
        drone_controller.close()  #make sure drone connection is closed
        cap.release()  
        cv2.destroyAllWindows()  

if __name__ == "__main__":
    # Initialize the low-level drivers for the drone
    cflib.crtp.init_drivers()
    main()   # Run the main function from the class
