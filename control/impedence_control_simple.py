from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
import sympy as sp
from typing import Tuple
import math
import time

robot_ip = "192.168.1.104"

rtde_c = RTDEControlInterface(robot_ip)
rtde_r = RTDEReceiveInterface(robot_ip)

alpha = 10.0 # 10
k = 25.0 # 5.0
b = 100.0 # 100.0
dt = 1.0/500.0

x, y, z = sp.symbols('x y z')

# near axis
# q_goal = sp.Point(-0.00600044, -0.1427485, 0.73381762)

# outright
q_goal = sp.Point(-0.29139538, -0.13562899, 0.68669564)
# Convert SymPy Point to regular Python floats
goal_pos = np.array([float(q_goal.x), float(q_goal.y), float(q_goal.z)])

# Reset Joints
j_goal = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0]
rtde_c.moveJ(j_goal, 0.5)


# Uatt = alpha*((x - q_goal.x)**2 + (y - q_goal.y)**2 + (z - q_goal.z)**2)**2
Uatt_2d = alpha*((x - q_goal.x)**2 + (y - q_goal.y)**2)
# Urep = -0.1*alpha*((x - q_goal.x)**2 + (y - q_goal.y)**2 + (z - q_goal.z)**2)**2

# U = Uatt + Urep
U_2d = Uatt_2d
# grad_U = [sp.diff(U, var) for var in (x, y, z)]
grad_U_2d = [sp.diff(U_2d, var) for var in (x, y)]

# Lambdify for fast numerical computation
# U_func = sp.lambdify((x, y, z), U, 'numpy')
U_func_2d = sp.lambdify((x, y), U_2d, 'numpy')
# grad_func = sp.lambdify((x, y, z), grad_U, 'numpy')
grad_func_2d = sp.lambdify((x, y), grad_U_2d, 'numpy')

# Move to the goal position in Cartesian space
goal_pose = [q_goal.x, q_goal.y, q_goal.z, 0, 0, 0]  # [x, y, z, rx, ry, rz]
rtde_c.moveL(goal_pose, 0.1, 0.1)
time.sleep(2.0)  # Wait longer for movement to complete
print("Moved to goal position")
rtde_c.zeroFtSensor()
while True:
    try:
       
        qx, qy, qz, qrx, qry, qrz = rtde_r.getActualTCPPose()
        q_act = np.array([qx, qy, qz])
        # print(q_act)

        Fx, Fy, Fz, T, Ty, Tz = rtde_r.getActualTCPForce()
        F_inp = np.array([Fx, Fy, Fz])
        
        # F_pot = -2.0*np.array(grad_func(qx, qy, qz))
        F_pot = -2.0*np.array(grad_func_2d(qx, qy))
        
        qdot = np.zeros(6)
       
        if np.linalg.norm(F_inp) > 5:
            # qdot[:3] = (F_inp + F_pot - k*(q_act - goal_pos))/b
            qdot[:2] = (F_inp[:2] + F_pot[:2] - k*(q_act[:2] - goal_pos[:2]))/b
        else:
            # qdot[:3] = 3*(F_pot - k*(q_act - goal_pos))/b
            qdot[:2] = 3*(F_pot[:2] - k*(q_act[:2] - goal_pos[:2]))/b
            print("Under Force Threshold")

        # Limit velocity to maximum of 0.5 for any component
        qdot = np.clip(qdot, -0.75, 0.75)

        # print(f"Force: {F_inp}")
        # print(f"Potential Force: {F_pot}")
        # print(f"Velocity: {qdot}")
        rtde_c.speedL(qdot, 0.3, dt)
        # time.sleep(0.1)



    except KeyboardInterrupt:
        rtde_c.disconnect()
        print("Stopping...")
        break
    

