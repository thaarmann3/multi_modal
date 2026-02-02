from UR5Controller import UR5Controller as ur5
import numpy as np
import time
import logging

# Configure logging to show messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

max_force = 50 # Newtons
max_velocity = 0.5 # meters per second
max_acceleration = 0.3 # meters per second squared

controller = ur5(default_acceleration=max_acceleration)

kx = 0.05
ky = 0.05
kz = 0.005

dt = 0.002

def main():
    while not controller.check_termination():
        # time.sleep(0.1)
        
        # Get force reading with error checking
        force_data = controller.get_force()
        if force_data is None:
            print("Failed to get force data - robot not connected?")
            continue
            
        fx, fy, fz, tx, ty, tz = -force_data
        fmag = np.sqrt(fx**2 + fy**2 + fz**2)
        
        print(f"Force magnitude: {fmag:.3f} N")
        
        if fmag > max_force:
            print("Force too high - stopping robot")
            # Emergency stop instead of speedL
            controller.rtde_c.servoStop()
        elif fmag < 10:  # Very low force threshold
            print("No force detected - going to home position")
            # controller.home_position()
            controller.move_joints([0, -np.pi/2, -np.pi/2, -np.pi/2, 0, 0], 1)
        else:
            dx = fx*kx*dt
            dy = fy*ky*dt
            dz = fz*kz*dt

            print(f"Moving relative: [{dx:.4f}, {dy:.4f}, {dz:.4f}]")
            
            # Try the movement, if it fails, reduce step size
            success = controller.move_servo_relative([0, 0, dz, 0, 0, 0])

            if success:
                print("Moved relative successfully")
            else:
                print("Failed to move relative")
            


if __name__ == "__main__":
    main()