import cv2
import time
from threading import Thread
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from cvzone.HandTrackingModule import HandDetector


# Drone connection thing we copied from mike
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


class DroneController:
    def __init__(self, uri):
        self._cf = Crazyflie(rw_cache='./cache') #Create a object
        self.uri = uri
        self.thrust_base = 20000  # Minimum thrust required to hover
        self.thrust_max = 50000   # Maximum thrust value
        self.thrust_increment = 5000  # Thrust increment for each finger
        self.current_thrust = self.thrust_base 
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
        self.connected = False #Set connection status to true

    def update_thrust(self, fingers):
        if self.connected:
            desired_thrust = self.thrust_base + self.thrust_increment * fingers # Calculate desired thrust based on fingers 
            self.current_thrust = min(max(self.thrust_base, desired_thrust), self.thrust_max) # Ensure thrust stains within limit values
            self._cf.commander.send_setpoint(0, 0, 0, int(self.current_thrust))   # Send the thrust to drone
            print(f'Thrust: {self.current_thrust}')
        else:
            print("Drone is not connected.")

    def close(self):
        if self.connected:
            self._cf.commander.send_setpoint(0, 0, 0, 0)   # Make sure drone is dead b4 closing connection
            time.sleep(0.1)   # Wait for alst command to be sent
            self._cf.close_link()

def main():   
    uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E6')
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
                        drone_controller.update_thrust(sum(fingers_left))
                    elif hand["type"] == "Right":
                        fingers_right = fingers

                print(f'Left Hand: {fingers_left}, Right Hand: {fingers_right}')  # Print the fingers

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  #Quit with q
                break
    finally:
        drone_controller.close()  #make sure drone connection is closed
        cap.release()  
        cv2.destroyAllWindows()  

if __name__ == "__main__":
    # Initialize the low-level drivers for the drone
    cflib.crtp.init_drivers()
    main()   # Run the main function from the class
