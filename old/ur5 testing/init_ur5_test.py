import rtde_control
import rtde_receive
import time
import math
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UR5ForceController:
    def __init__(self, robot_ip="192.168.1.104"):
        """Initialize UR5 robot control interfaces"""
        self.robot_ip = robot_ip
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        
        # Force threshold (50 Newtons)
        self.force_threshold = 50.0
        
        # Circle parameters
        self.circle_radius = 0.5  # meters
        self.circle_points = 50   # Number of points to define the circle
        
        # Safety parameters
        self.max_velocity = 0.1   # m/s
        self.max_acceleration = 0.1  # m/s²
        
        logger.info("UR5 Force Controller initialized")
    
    def get_force_magnitude(self):
        """Get the current force magnitude from the robot's force sensor"""
        try:
            # Get the actual TCP force (force at the tool center point)
            actual_tcp_force = self.rtde_r.getActualTCPForce()
            
            # Calculate the magnitude of the force vector
            force_magnitude = math.sqrt(
                actual_tcp_force[0]**2 + 
                actual_tcp_force[1]**2 + 
                actual_tcp_force[2]**2
            )
            
            return force_magnitude, actual_tcp_force
        except Exception as e:
            logger.error(f"Error reading force data: {e}")
            return 0.0, [0, 0, 0, 0, 0, 0]
    
    def get_current_pose(self):
        """Get the current robot pose"""
        try:
            return self.rtde_r.getActualTCPPose()
        except Exception as e:
            logger.error(f"Error reading robot pose: {e}")
            return None
    
    def generate_circle_trajectory(self, center_pose):
        """Generate a circle trajectory around the current position"""
        if center_pose is None:
            logger.error("Cannot generate trajectory: invalid center pose")
            return []
        
        trajectory = []
        
        # Extract position from pose (first 3 elements are x, y, z)
        center_x, center_y, center_z = center_pose[0], center_pose[1], center_pose[2]
        
        # Generate circle points in the XY plane (keeping Z constant)
        for i in range(self.circle_points + 1):  # +1 to close the circle
            angle = 2 * math.pi * i / self.circle_points
            
            # Calculate circle point
            x = center_x + self.circle_radius * math.cos(angle)
            y = center_y + self.circle_radius * math.sin(angle)
            z = center_z  # Keep Z constant
            
            # Create pose (position + orientation)
            pose = [x, y, z, center_pose[3], center_pose[4], center_pose[5]]
            trajectory.append(pose)
        
        logger.info(f"Generated circle trajectory with {len(trajectory)} points")
        return trajectory
    
    def execute_circle_motion(self, trajectory):
        """Execute the circle motion trajectory"""
        if not trajectory:
            logger.error("Cannot execute motion: empty trajectory")
            return False
        
        try:
            logger.info("Starting circle motion execution")
            
            # Move to the starting position first
            start_pose = trajectory[0]
            self.rtde_c.moveL(start_pose, self.max_velocity, self.max_acceleration)
            
            # Execute the circle trajectory
            for i, pose in enumerate(trajectory[1:], 1):
                # Check if we should stop (force still above threshold)
                current_force, _ = self.get_force_magnitude()
                if current_force < self.force_threshold:
                    logger.info(f"Force dropped below threshold during motion. Stopping at point {i}")
                    break
                
                # Move to next point
                self.rtde_c.moveL(pose, self.max_velocity, self.max_acceleration)
                
                # Small delay to ensure smooth motion
                time.sleep(0.01)
            
            logger.info("Circle motion completed")
            return True
            
        except Exception as e:
            logger.error(f"Error executing circle motion: {e}")
            return False
    
    def monitor_force_and_respond(self):
        """Main monitoring loop - detects force threshold and responds"""
        logger.info("Starting force monitoring...")
        logger.info(f"Force threshold: {self.force_threshold} N")
        logger.info("Waiting for force detection...")
        
        try:
            while True:
                # Get current force
                force_magnitude, force_vector = self.get_force_magnitude()
                
                # Log force data periodically (every 10 iterations to avoid spam)
                if hasattr(self, '_log_counter'):
                    self._log_counter += 1
                else:
                    self._log_counter = 0
                
                if self._log_counter % 10 == 0:
                    logger.info(f"Current force: {force_magnitude:.2f} N (threshold: {self.force_threshold} N)")
                
                # Check if force exceeds threshold
                if force_magnitude > self.force_threshold:
                    logger.warning(f"Force threshold exceeded! Force: {force_magnitude:.2f} N")
                    logger.info("Force vector: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]" % tuple(force_vector))
                    
                    # Get current pose to use as circle center
                    current_pose = self.get_current_pose()
                    if current_pose is None:
                        logger.error("Cannot get current pose. Aborting circle motion.")
                        continue
                    
                    # Generate and execute circle trajectory
                    trajectory = self.generate_circle_trajectory(current_pose)
                    if trajectory:
                        success = self.execute_circle_motion(trajectory)
                        if success:
                            logger.info("Circle motion completed successfully")
                        else:
                            logger.error("Circle motion failed")
                    else:
                        logger.error("Failed to generate circle trajectory")
                    
                    # Wait a bit before resuming monitoring
                    time.sleep(1.0)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up robot connections"""
        try:
            logger.info("Cleaning up robot connections...")
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main function to run the UR5 force controller"""
    try:
        # Initialize the controller
        controller = UR5ForceController("127.0.0.1")
        
        # Start monitoring
        controller.monitor_force_and_respond()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())