import logging
import time
from threading import Thread

import cv2
import cflib
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from cvzone.HandTrackingModule import HandDetector

# Initialize the low-level drivers
cflib.crtp.init_drivers()

uri = uri_helper.uri_from_env(default='radio://0/90/2M/E7E7E7E709')
logging.basicConfig(level=logging.ERROR)


class MotorRampExample:
    """Example that connects to a Crazyflie and ramps the motors up/down and
    then disconnects"""

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """

        self._cf = Crazyflie(rw_cache='./cache')

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        self._cf.open_link(link_uri)

        print('Connecting to %s' % link_uri)

    def _connected(self, link_uri):
        """ This callback is called from the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        # Start a separate thread to do the motor test.
        # Do not hijack the calling thread!
        Thread(target=self._control_with_hand).start()

    def _connection_failed(self, link_uri, msg):
        """Callback when initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)

    def _control_with_hand(self):
        cap = cv2.VideoCapture(0)
        detector = HandDetector(detectionCon=0.8, maxHands=2)

        thrust_step = 1000
        min_thrust = 20000
        max_thrust = 65000
        thrust = min_thrust
        thrust_direction = 1

        while True:
            success, img = cap.read()
            hands, img = detector.findHands(img)
            if hands:
                left_hand, right_hand = hands[:2] if len(hands) >= 2 else (None, None)

                if left_hand:
                    left_fingers = detector.fingersUp(left_hand)
                    print("Left Hand Fingers:", left_fingers)

                if right_hand:
                    right_fingers = detector.fingersUp(right_hand)
                    print("Right Hand Fingers:", right_fingers)

                hand = left_hand if left_hand else right_hand
                fingers = detector.fingersUp(hand)

                # Calculate thrust based on the number of fingers
                num_fingers = fingers.count(1)
                thrust = min_thrust + num_fingers * thrust_step

                if num_fingers == 0:
                    # Drone starts descending slowly until it reaches the floor
                    thrust_direction = -1

                thrust = max(min(thrust, max_thrust), min_thrust)
                thrust += thrust_step * thrust_direction

                # Send the thrust to the drone
                self._cf.commander.send_setpoint(0, 0, 0, thrust)
                
                # Print thrust level
                print("Thrust Level:", thrust)

            else:
                # No hands detected, stop the drone
                self._cf.commander.send_setpoint(0, 0, 0, min_thrust)

            cv2.imshow("Image", img)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()
        self._cf.close_link()


if __name__ == '__main__':
    le = MotorRampExample(uri)
