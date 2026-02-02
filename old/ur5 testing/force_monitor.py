import rtde_receive
import time
import math

def monitor_force_sensor(robot_ip="192.168.1.104"):
    """Simple force sensor monitoring script"""
    print(f"Connecting to UR5 robot at {robot_ip}...")
    
    try:
        # Initialize RTDE receive interface
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print("Connected successfully!")
        print("Monitoring force sensor readings...")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        while True:
            try:
                # Get the actual TCP force (force at the tool center point)
                actual_tcp_force = rtde_r.getActualTCPForce()
                
                # Calculate the magnitude of the force vector
                force_magnitude = math.sqrt(
                    actual_tcp_force[0]**2 + 
                    actual_tcp_force[1]**2 + 
                    actual_tcp_force[2]**2
                )
                
                # Display force readings
                print(f"Force: {force_magnitude:6.2f} N | "
                      f"Fx: {actual_tcp_force[0]:6.2f} | "
                      f"Fy: {actual_tcp_force[1]:6.2f} | "
                      f"Fz: {actual_tcp_force[2]:6.2f} | "
                      f"Tx: {actual_tcp_force[3]:6.2f} | "
                      f"Ty: {actual_tcp_force[4]:6.2f} | "
                      f"Tz: {actual_tcp_force[5]:6.2f}")
                
                # Small delay to prevent excessive output
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error reading force data: {e}")
                time.sleep(1.0)
                
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        try:
            rtde_r.disconnect()
            print("Disconnected from robot")
        except:
            pass

if __name__ == "__main__":
    monitor_force_sensor()
