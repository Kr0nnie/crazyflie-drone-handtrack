#simplified ver. just left fingers # and thrust. much quicker response time

import cv2
import time
from threading import Thread
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
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

# Drone control class
class DroneController:
    def __init__(self, uri):
        self._cf = Crazyflie(rw_cache='./cache')
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
        self.connected = True
        # Unlock startup thrust protection
        for _ in range(100):
            self._cf.commander.send_setpoint(0, 0, 0, 0)

    def _disconnected(self, link_uri):
        print('Drone disconnected')
        self.connected = False

    def update_thrust(self, fingers):
        if self.connected:
            desired_thrust = self.thrust_base + self.thrust_increment * fingers
            self.current_thrust = min(max(self.thrust_base, desired_thrust), self.thrust_max)
            self._cf.commander.send_setpoint(0, 0, 0, int(self.current_thrust))
            print(f'Left Fingers: {fingers}, Thrust: {self.current_thrust}')
        else:
            print("Drone is not connected.")

    def close(self):
        if self.connected:
            self._cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.1)
            self._cf.close_link()

# Main function to run the drone control with hand tracking
def main():
    uri = uri_helper.uri_from_env(default='radio://0/90/2M/E7E7E7E709')
    # Test drone connection first
    test_drone_connection(uri)

    # Initialize hand tracking and drone controller
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    drone_controller = DroneController(uri)

    try:
        while True:
            success, img = cap.read()
            hands, img = detector.findHands(img)
            fingers_left = 0

            if hands:
                for hand in hands:
                    if hand["type"] == "Left":
                        fingers = detector.fingersUp(hand)
                        fingers_left = sum(fingers)
                        drone_controller.update_thrust(fingers_left)
                        break  # Assuming only one left hand to control the drone

                print(f'Left Fingers: {fingers_left}')

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        drone_controller.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize the low-level drivers for the drone
    cflib.crtp.init_drivers()
    main()
