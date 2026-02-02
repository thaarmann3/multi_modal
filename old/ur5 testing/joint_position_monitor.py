#!/usr/bin/env python3
import rtde_receive
import time
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UR5JointMonitor:
    def __init__(self, robot_ip="192.168.1.104"):
        """Initialize UR5 joint position monitoring"""
        self.robot_ip = robot_ip
        self.rtde_r = None
        
    def connect(self):
        """Connect to the UR5 robot"""
        try:
            logger.info(f"Connecting to UR5 robot at {self.robot_ip}...")
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            
            if self.rtde_r.isConnected():
                logger.info("✅ Successfully connected to UR5 robot")
                return True
            else:
                logger.error("❌ Failed to connect to UR5 robot")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False
    
    def get_joint_positions(self):
        """Get current joint positions in radians"""
        try:
            if self.rtde_r and self.rtde_r.isConnected():
                joint_positions = self.rtde_r.getActualQ()
                return joint_positions
            else:
                logger.error("Robot not connected")
                return None
        except Exception as e:
            logger.error(f"Error getting joint positions: {e}")
            return None
    
    def radians_to_degrees(self, radians):
        """Convert radians to degrees"""
        return radians * 180.0 / math.pi
    
    def format_joint_output(self, joint_positions):
        """Format joint positions for display"""
        if joint_positions is None:
            return "No data available"
        
        # Joint names
        joint_names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3"]
        
        output = []
        output.append("=" * 80)
        output.append("UR5 JOINT POSITIONS")
        output.append("=" * 80)
        
        for i, (name, pos_rad) in enumerate(zip(joint_names, joint_positions)):
            pos_deg = self.radians_to_degrees(pos_rad)
            # pos_deg = pos_rad
            output.append(f"Joint {i}: {name:8} | {pos_rad:8.4f} rad | {pos_deg:7.2f}°")
        
        output.append("=" * 80)
        
        # Add copyable list format
        output.append("")
        output.append("COPYABLE LIST FORMAT:")
        output.append("-" * 40)
        output.append("joint_positions = [")
        for i, pos_rad in enumerate(joint_positions):
            comma = "," if i < len(joint_positions) - 1 else ""
            output.append(f"    {pos_rad:8.4f}{comma}  # {joint_names[i]}")
        output.append("]")
        output.append("")
        output.append("COPYABLE ARRAY FORMAT:")
        output.append("-" * 40)
        output.append("joint_positions = [")
        for i, pos_rad in enumerate(joint_positions):
            comma = "," if i < len(joint_positions) - 1 else ""
            output.append(f"    {pos_rad:8.4f}{comma}")
        output.append("]")
        
        return "\n".join(output)
    
    def monitor_joints(self, update_interval=0.5):
        """Continuously monitor and display joint positions"""
        if not self.connect():
            return
        
        logger.info("Starting joint position monitoring...")
        logger.info("Press Ctrl+C to stop")
        logger.info("-" * 80)
        
        try:
            while True:
                joint_positions = self.get_joint_positions()
                
                if joint_positions is not None:
                    # Clear screen and display current positions
                    print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
                    print(self.format_joint_output(joint_positions))
                    print(f"\nLast updated: {time.strftime('%H:%M:%S')}")
                    print("Press Ctrl+C to stop monitoring")
                else:
                    logger.warning("Failed to get joint positions")
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up connections"""
        if self.rtde_r and self.rtde_r.isConnected():
            self.rtde_r.disconnect()
            logger.info("✅ Disconnected from UR5 robot")

def main():
    """Main function"""
    print("UR5 Joint Position Monitor")
    print("=" * 50)
    
    # You can change the robot IP here if needed
    robot_ip = "192.168.1.104"
    
    monitor = UR5JointMonitor(robot_ip)
    monitor.monitor_joints(update_interval=0.5)  # Update every 0.5 seconds

if __name__ == "__main__":
    main()
