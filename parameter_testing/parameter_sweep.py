from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
import sympy as sp
import time
import csv
import itertools

robot_ip = "192.168.1.104"

rtde_c = RTDEControlInterface(robot_ip)
rtde_r = RTDEReceiveInterface(robot_ip)

alpha_values = [10.0]
k_values = [10.0, 25.0, 50.0, 100.0, 200.0, 350.0, 500.0]
b_values = [10.0, 50.0, 100.0, 200.0, 350.0, 500.0, 1000.0]

dt = 0.0005
displacement = 0.10  # 10cm
tolerance = displacement*0.02  # 0.5% of 10cm
settle_time = 1
trial_time = 10.0

x, y, z = sp.symbols('x y z')
q_goal = sp.Point(-0.29139538, -0.13562899, 0.68669564)
goal_pos = np.array([float(q_goal.x), float(q_goal.y), float(q_goal.z)])

j_goal = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0]
rtde_c.moveJ(j_goal, 0.5)
time.sleep(0.5)
rtde_c.zeroFtSensor()

results = []

def save_results():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'parameter_testing', 'parameter_sweep_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['alpha', 'k', 'b', 'settled', 'response_time', 'final_error', 'max_overshoot', 'max_speed', 'avg_speed'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} results to {csv_path}")

try:
    for alpha, k, b in itertools.product(alpha_values, k_values, b_values):
        if k == b:
            print(f"Skipping: alpha={alpha}, k={k}, b={b} (k == b)")
            continue
        if b <= k:
            print(f"Skipping: alpha={alpha}, k={k}, b={b} (b <= k)")
            continue
        print(f"Testing: alpha={alpha}, k={k}, b={b}")
        
        Uatt = 0.5*alpha*((x - q_goal.x)**2 + (y - q_goal.y)**2 + (z - q_goal.z)**2)
        # Urep = -0.1*alpha*((x - q_goal.x)**2 + (y - q_goal.y)**2 + (z - q_goal.z)**2)**2
        # U = Uatt + Urep
        U = Uatt
        grad_U = [sp.diff(U, var) for var in (x, y, z)]
        grad_func = sp.lambdify((x, y, z), grad_U, 'numpy')
        
        goal_pose = [q_goal.x, q_goal.y, q_goal.z, 0, 0, 0]
        rtde_c.moveL(goal_pose, 0.1, 0.1)
        time.sleep(2.0)
        rtde_c.zeroFtSensor()
        time.sleep(0.5)
        
        # Apply 10cm displacement
        displaced_pose = [q_goal.x, q_goal.y, q_goal.z + displacement, 0, 0, 0]
        rtde_c.moveL(displaced_pose, 0.1, 0.1)
        time.sleep(2.0)
        
        # Record response
        start_time = time.time()
        positions = []
        times = []
        speeds = []
        settled = False
        settle_start = None
        
        while time.time() - start_time < trial_time:
            try:
                qx, qy, qz, qrx, qry, qrz = rtde_r.getActualTCPPose()
                q_act = np.array([qx, qy, qz])
                
                Fx, Fy, Fz, T, Ty, Tz = rtde_r.getActualTCPForce()
                F_inp = np.array([Fx, Fy, Fz])
            except:
                print("Robot connection lost - stopping script")
                save_results()
                rtde_c.disconnect()
                exit()
            
            F_pot = -2.0*np.array(grad_func(qx, qy, qz))
            
            qdot = np.zeros(6)
            if np.linalg.norm(F_inp) > 5:
                qdot[:3] = (F_inp + F_pot - k*(q_act - goal_pos))/b
            else:
                qdot[:3] = 3*(F_pot - k*(q_act - goal_pos))/b
            
            qdot = np.clip(qdot, -1.0, 1.0)  # Cap at 1.0 m/s
            try:
                rtde_c.speedL(qdot, 0.3, dt)
                # time.sleep(dt)
            except:
                print("Robot control lost - stopping script")
                save_results()
                rtde_c.disconnect()
                exit()
            
            current_time = time.time() - start_time
            error = np.linalg.norm(q_act - goal_pos)
            current_speed = np.linalg.norm(qdot[:3])  # Record speed magnitude
            
            positions.append(q_act.copy())
            times.append(current_time)
            speeds.append(current_speed)
            
            if error < tolerance:
                if settle_start is None:
                    settle_start = current_time
                elif current_time - settle_start >= settle_time:
                    settled = True
                    break
            else:
                settle_start = None
        
        response_time = time.time() - start_time
        final_error = np.linalg.norm(positions[-1] - goal_pos) if positions else float('inf')
        
        results.append({
            'alpha': alpha,
            'k': k,
            'b': b,
            'settled': settled,
            'response_time': response_time,
            'final_error': final_error,
            'max_overshoot': max([np.linalg.norm(pos - goal_pos) for pos in positions]) if positions else 0,
            'max_speed': max(speeds) if speeds else 0,
            'avg_speed': np.mean(speeds) if speeds else 0
        })
    
        print(f"Response time: {response_time:.2f}s, Settled: {settled}, Final error: {final_error:.4f}m")
        
        # Stop any ongoing speedL commands and wait between tests
        rtde_c.speedStop()
        time.sleep(1.0)

except KeyboardInterrupt:
    print("\nInterrupted by user. Saving partial results...")
    save_results()
    rtde_c.disconnect()
    exit()

save_results()
rtde_c.disconnect()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'parameter_testing', 'parameter_sweep_results.csv')
print(f"Parameter sweep complete. Results saved to {csv_path}")
