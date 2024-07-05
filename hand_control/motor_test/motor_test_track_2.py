#Works ok, kind of. Does not have cvimshow, so not able to see camera nor the tracking

import logging
import time
from threading import Thread

import cv2
import cflib
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from cvzone.HandTrackingModule import HandDetector

# Initialize Crazyflie URI
uri = uri_helper.uri_from_env(default='radio://0/90/2M/E7E7E7E709')

# Initialize logging
logging.basicConfig(level=logging.ERROR)


class MotorRampExample:
    """Example that connects to a Crazyflie and ramps the motors up/down and
    the disconnects"""

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """

        self._cf = Crazyflie(rw_cache='./cache')

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        self._cf.open_link(link_uri)

        print('Connecting to %s' % link_uri)

        # Initialize hand detector
        self.detector = HandDetector(detectionCon=0.8, maxHands=2)

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        # Start a separate thread to do the motor test.
        # Do not hijack the calling thread!
        Thread(target=self._control_drone).start()

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)

    def _control_drone(self):
        thrust_mult = 1
        thrust_step = 500
        thrust = 25000
        pitch = 0
        roll = 0
        yawrate = 0

        # Unlock startup thrust protection
        for i in range(100):
            self._cf.commander.send_setpoint(0, 0, 0, 0)

        while True:
            # Get image from camera
            success, img = cap.read()
            # Find hands
            hands, img = self.detector.findHands(img)

            if hands:
                left_hand = None
                right_hand = None

                for hand in hands:
                    if hand['type'] == 'Left':
                        left_hand = hand
                    else:
                        right_hand = hand

                if left_hand:
                    fingers = self.detector.fingersUp(left_hand)
                    num_fingers = sum(fingers)
                    thrust = 25000 + num_fingers * 500
                    print("Left Hand Fingers:", fingers)
                else:
                    thrust -= 100  # Decrease thrust slowly if no fingers detected

                if right_hand:
                    print("Right Hand Fingers:", self.detector.fingersUp(right_hand))

            self._cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
            time.sleep(0.1)

    def stop(self):
        self._cf.commander.send_setpoint(0, 0, 0, 0)
        # Make sure that the last packet leaves before the link is closed
        # since the message queue is not flushed before closing
        time.sleep(0.1)
        self._cf.close_link()


if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Create MotorRampExample instance
    le = MotorRampExample(uri)

    # Keep the program running until interrupted
    try:
        while True:
            pass
    except KeyboardInterrupt:
        le.stop()
        cap.release()
        cv2.destroyAllWindows()
