import logging
import time

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

URI = 'radio://0/60/2M/E7E7E7E706'

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)

    with SyncCrazyflie(URI) as scf:
        with MotionCommander(scf) as mc:
            print('Taking off!')
            time.sleep(1)

            print('Moving forward 0.5m')
            mc.forward(0.5)
            time.sleep(1)

            print('Moving up 0.2m')
            mc.up(0.2)
            # Wait a bit
            time.sleep(1)

            print('Doing a 270deg circle')
            mc.circle_right(0.5, velocity=0.5, angle_degrees=270)

            print('Moving down 0.2m')
            mc.down(0.2)
            # Wait a bit
            time.sleep(1)

            print('Rolling left 0.2m at 0.6m/s')
            mc.left(0.2, velocity=0.6)
            # Wait a bit
            time.sleep(1)

            print('Moving forward 0.5m')
            mc.forward(0.5)

            print('Landing!')
