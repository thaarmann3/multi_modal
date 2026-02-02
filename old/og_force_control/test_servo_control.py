import time
import sys
from UR5Controller import UR5Controller as ur5
import numpy as np

robot = ur5(robot_ip="192.168.1.104")

rtde_c = robot.rtde_c
rtde_r = robot.rtde_r

def force_compliant_motion(rtde_c, rtde_r):
    # Set a low, safe speed for compliant motion
    v = 0.1  # Velocity [m/s]
    a = 0.5  # Acceleration [m/s^2]
    dt = 1.0/500  # 500 Hz sampling frequency

    # Desired force threshold in Newtons (e.g., 5N in the Z-direction)
    desired_force_z = 5.0
    k_p = 0.005  # Proportional gain for force control

    print("Starting compliant motion. Press Ctrl+C to stop.")

    try:
        rtde_c.zeroFtSensor()
        robot.move_joints([0, -np.pi/2, -np.pi/2, -np.pi/2, 0, 0], 0.1)
        time.sleep(0.7)
        home_points = rtde_r.getActualTCPPose()

        while not robot.check_termination():
            # time.sleep(0.1)
            
            # Read the current end-effector force (Fx, Fy, Fz, Tx, Ty, Tz)
            force = rtde_r.getActualTCPForce()
            position = rtde_r.getActualTCPPose()
            # print(f"position: {position}")
            fx, fy, fz, tx, ty, tz = force

            print(f"Force: {fz}")
            
            # Get current pose
            px, py, pz, rx, ry, rz = np.array(position) - home_points
            
            print(f"pz: {pz}")

            # Create the target pose with the calculated velocity offset
            target_pose = [px, py, pz, rx, ry, rz]

            # Use proportional control to calculate a velocity offset
            # Move in the Z-direction based on force feedback
            # Note: The sign of k_p depends on your frame definition.
            # You may need to experiment.
            # dz = -1 * k_p * fz * (1/pz) * dt
            dz = -1 * k_p * fz * dt
            print(f"dz: {dz}")

            # Move based on force input
            if abs(fz) > 5:  # Check if force error is significant
                target_pose[2] += dz  # Update Z position with velocity offset
                 # Send the updated pose command
                rtde_c.servoL(target_pose, v, a, dt, 0.1, 300)
                time.sleep(dt)

            # else:
            #     robot.move_joints([0, -np.pi/2, -np.pi/2, -np.pi/2, 0, 0], 0.1)
            #     time.sleep(0.7)

           

    except KeyboardInterrupt:
        print("Stopping compliant motion.")
    finally:
        rtde_c.servoStop()
        rtde_c.disconnect()

if __name__ == "__main__":
    force_compliant_motion(rtde_c, rtde_r)
