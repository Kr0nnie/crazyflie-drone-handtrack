import cv2
import time
from threading import Thread
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from cvzone.HandTrackingModule import HandDetector
from simple_pid import PID  # Ensure you have a PID library or implement PID logic

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
        self._cf = Crazyflie(rw_cache='./cache')
        self.uri = uri
        self.thrust_base = 20000
        self.thrust_max = 50000
        self.thrust_increment = 5000
        self.current_thrust = self.thrust_base
        self.connected = False
        self.desired_altitude = 0.5  # Desired altitude in meters
        self.pid = PID(1.0, 0.1, 0.05, setpoint=self.desired_altitude)
        self.pid.output_limits = (0, 10000)  # Limit the thrust adjustment

        # Connect to the drone
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.open_link(self.uri)
    
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

    def adjust_for_altitude(self):
        current_altitude = self.read_current_altitude()  # Implement this method
        thrust_adjustment = self.pid(current_altitude)
        self.current_thrust = max(min(self.thrust_base + thrust_adjustment, self.thrust_max), self.thrust_base)
        if self.connected:
            self._cf.commander.send_setpoint(0, 0, 0, int(self.current_thrust))

    def read_current_altitude(self):
        # Implement this method based on your drone's altitude sensor
        return altitude  # Placeholder: Return actual altitude here

    def close(self):
        if self.connected:
            self._cf.commander.send_setpoint(0, 0, 0, 0)   # Make sure drone is dead b4 closing connection
            time.sleep(0.1)   # Wait for alst command to be sent
            self._cf.close_link()

def main():
    uri = uri_helper.uri_from_env(default='radio://0/90/2M/E7E7E7E709')
    test_drone_connection(uri)

    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    drone_controller = DroneController(uri)

    try:
        while True:
            success, img = cap.read()
            hands, img = detector.findHands(img)
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

                drone_controller.adjust_for_altitude()  # Adjust for altitude

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        drone_controller.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cflib.crtp.init_drivers()
    main()
