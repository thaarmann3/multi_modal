import rtde_control
import rtde_receive
import time
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UR5ForceSquareController:
    def __init__(self, robot_ip="192.168.1.104"):
        """Initialize UR5 robot control interfaces"""
        self.robot_ip = robot_ip
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        
        # Force threshold (75 Newtons)
        self.force_threshold = 50.0
        
        # Square parameters
        self.square_side_length = 0.5  # meters
        self.square_points = 20        # Number of points per side (for smooth movement)
        
        # Movement method: 'joint' only (avoid moveL to prevent safety stops)
        self.movement_method = 'joint'  # Use joint-based movement to avoid safety stops
        
        # Safety parameters (joint movements only)
        self.max_velocity = 0.3        # rad/s for joint movements
        self.max_acceleration = 0.1    # rad/s² for joint movements
        
        logger.info("UR5 Force Square Controller initialized")
        logger.info(f"Force threshold: {self.force_threshold} N")
        logger.info(f"Square side length: {self.square_side_length} m")
    
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

    def move_to_start_position(self):
        """Move the UR5 arm to a start position (90 degree bend)"""
        logger.info("Moving UR5 to start position...")
        
        # Joint positions in radians: [base, shoulder, elbow, wrist1, wrist2, wrist3]
        starting_joints = [
            0,  # Base: 0 degrees (point forward)
            -math.pi/2,         # Shoulder: -90 degrees (bend down)
            -math.pi/2,         # Elbow: -90 degrees (bend)
            -math.pi/2,  # Wrist1: -90 degrees (adjust for orientation)
            0.0,         # Wrist2: 0 degrees (neutral)
            0.0          # Wrist3: 0 degrees (neutral)
        ]
        
        try:
            # Move to start position using joint movements
            self.rtde_c.moveJ(starting_joints, self.max_velocity, self.max_acceleration)
            logger.info("✅ Successfully moved to start position")
            
            # Wait a moment for the movement to complete
            time.sleep(2.0)
            
            # Verify position
            current_joints = self.get_current_joint_positions()
            if current_joints:
                logger.info(f"Current joint positions: {[f'{j:.3f}' for j in current_joints]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error moving to start position: {e}")
            return False
    
    def get_current_pose(self):
        """Get the current robot pose"""
        try:
            return self.rtde_r.getActualTCPPose()
        except Exception as e:
            logger.error(f"Error reading robot pose: {e}")
            return None
    
    def get_current_joint_positions(self):
        """Get current joint positions"""
        try:
            return self.rtde_r.getActualQ()
        except Exception as e:
            logger.error(f"Error reading joint positions: {e}")
            return None
    
    def spin_wrist_joint(self):
        """Spin the wrist 3 joint (joint 6) for the specified duration"""
        logger.info(f"Starting wrist spin for {self.spin_duration} seconds...")
        
        try:
            # Get current joint positions
            current_joints = self.get_current_joint_positions()
            if current_joints is None:
                logger.error("Cannot get current joint positions")
                return False
            
            # Calculate total rotation for the spin duration
            total_rotation = self.spin_velocity * self.spin_duration
            
            # Create target position with wrist3 rotated
            target_joints = current_joints.copy()
            target_joints[5] += total_rotation  # Joint 6 (index 5) is wrist3
            
            logger.info(f"Spinning wrist3 joint by {total_rotation:.2f} radians ({math.degrees(total_rotation):.1f} degrees)")
            logger.info(f"Spin velocity: {self.spin_velocity} rad/s")
            
            # Start the spin movement
            start_time = time.time()
            self.rtde_c.moveJ(target_joints, self.spin_velocity, self.move_acceleration)
            
            # Monitor the spin
            while True:
                elapsed = time.time() - start_time
                remaining = self.spin_duration - elapsed
                
                # Get current joint positions to show progress
                current_pos = self.get_current_joint_positions()
                if current_pos:
                    wrist3_angle = math.degrees(current_pos[5])
                    logger.info(f"Spinning... {elapsed:.1f}s elapsed, {remaining:.1f}s remaining, Wrist3: {wrist3_angle:.1f}°")
                
                time.sleep(0.3)  # Update every 0.5 seconds
            
            logger.info("✅ Wrist spin completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during wrist spin: {e}")
            return False

    def move_sqaure(self):
        """Move the UR5 arm to in a square pattern based on previsouly set joint poisitions"""
        logger.info("Moving UR5 arm in a square pattern...")
        
        # Get current joint positions
        current_joints = self.get_current_joint_positions()
        if current_joints is None:
            logger.error("Cannot get current joint positions")
            return False
        
        points_in_sqaure = [
            [0, -math.pi/2, -math.pi/2, -math.pi/2, 0.0, 0.0],
            # finish
            ]
    
    def generate_square_joint_trajectory(self, center_pose):
        """Generate a square trajectory using joint movements around the current position"""
        if center_pose is None:
            logger.error("Cannot generate trajectory: invalid center pose")
            return []
        
        # Get current joint positions as starting point
        current_joints = self.get_current_joint_positions()
        if current_joints is None:
            logger.error("Cannot get current joint positions")
            return []
        
        trajectory = []
        
        # Extract position from pose (first 3 elements are x, y, z)
        center_x, center_y, center_z = center_pose[0], center_pose[1], center_pose[2]
        
        # Calculate half side length
        half_side = self.square_side_length / 2.0
        
        # Define square corners relative to center
        # Starting from bottom-left, going clockwise
        corners = [
            (center_x - half_side, center_y - half_side),  # Bottom-left
            (center_x + half_side, center_y - half_side),  # Bottom-right
            (center_x + half_side, center_y + half_side),  # Top-right
            (center_x - half_side, center_y + half_side),  # Top-left
            (center_x - half_side, center_y - half_side)   # Back to start (close the square)
        ]
        
        logger.info(f"Square center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
        logger.info(f"Square dimensions: {self.square_side_length}m x {self.square_side_length}m")
        logger.info("Generating square trajectory using base and shoulder joint movements...")
        
        # Create a more realistic square movement using multiple joints
        # This approach creates actual square-like movement by combining joint rotations
        
        # Calculate joint angle changes needed for the square
        # Use base, shoulder, and elbow joints to create square movement
        base_angle_per_meter = 0.4  # radians per meter (adjust as needed)
        shoulder_angle_per_meter = 0.3  # radians per meter (adjust as needed)
        elbow_angle_per_meter = 0.2  # radians per meter (adjust as needed)
        
        # Generate trajectory for each side of the square
        for side in range(4):
            logger.info(f"Generating trajectory for side {side + 1}/4")
            
            # Calculate the movement for this side
            if side == 0:  # Bottom side: move in +X direction
                base_change = half_side * 2 * base_angle_per_meter
                shoulder_change = 0
                elbow_change = 0
            elif side == 1:  # Right side: move in +Y direction
                base_change = 0
                shoulder_change = half_side * 2 * shoulder_angle_per_meter
                elbow_change = half_side * 2 * elbow_angle_per_meter
            elif side == 2:  # Top side: move in -X direction
                base_change = -half_side * 2 * base_angle_per_meter
                shoulder_change = 0
                elbow_change = 0
            elif side == 3:  # Left side: move in -Y direction
                base_change = 0
                shoulder_change = -half_side * 2 * shoulder_angle_per_meter
                elbow_change = -half_side * 2 * elbow_angle_per_meter
            
            # Generate points along this side
            for j in range(self.square_points):
                # Linear interpolation for this side
                t = j / (self.square_points - 1)
                
                # Create target joint positions
                target_joints = current_joints.copy()
                
                # Apply the joint changes for this side
                target_joints[0] += t * base_change  # Base joint
                target_joints[1] += t * shoulder_change  # Shoulder joint
                target_joints[2] += t * elbow_change  # Elbow joint
                
                # Add small variations to wrist joints to make movement more natural
                target_joints[3] += 0.03 * math.sin(t * math.pi * 2)  # Small wrist1 movement
                target_joints[4] += 0.02 * math.cos(t * math.pi * 2)  # Small wrist2 movement
                
                trajectory.append(target_joints)
        
        logger.info(f"Generated square joint trajectory with {len(trajectory)} points")
        return trajectory
    
    def generate_square_pose_trajectory(self, center_pose):
        """Generate a square trajectory using pose movements for more accurate squares"""
        if center_pose is None:
            logger.error("Cannot generate trajectory: invalid center pose")
            return []
        
        trajectory = []
        
        # Extract position from pose (first 3 elements are x, y, z)
        center_x, center_y, center_z = center_pose[0], center_pose[1], center_pose[2]
        
        # Calculate half side length
        half_side = self.square_side_length / 2.0
        
        # Define square corners relative to center
        # Starting from bottom-left, going clockwise
        corners = [
            (center_x - half_side, center_y - half_side),  # Bottom-left
            (center_x + half_side, center_y - half_side),  # Bottom-right
            (center_x + half_side, center_y + half_side),  # Top-right
            (center_x - half_side, center_y + half_side),  # Top-left
            (center_x - half_side, center_y - half_side)   # Back to start (close the square)
        ]
        
        logger.info(f"Square center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
        logger.info(f"Square dimensions: {self.square_side_length}m x {self.square_side_length}m")
        logger.info("Generating square trajectory using pose movements...")
        
        # Generate smooth trajectory between corners
        for i in range(len(corners) - 1):
            start_x, start_y = corners[i]
            end_x, end_y = corners[i + 1]
            
            logger.info(f"Generating trajectory for side {i + 1}/4: ({start_x:.3f}, {start_y:.3f}) to ({end_x:.3f}, {end_y:.3f})")
            
            # Generate points along this side
            for j in range(self.square_points):
                # Linear interpolation between start and end
                t = j / (self.square_points - 1)
                x = start_x + t * (end_x - start_x)
                y = start_y + t * (end_y - start_y)
                z = center_z  # Keep Z constant
                
                # Create pose (position + orientation)
                pose = [x, y, z, center_pose[3], center_pose[4], center_pose[5]]
                trajectory.append(pose)
        
        logger.info(f"Generated square pose trajectory with {len(trajectory)} points")
        return trajectory
    
    def execute_square_motion(self, trajectory):
        """Execute the square motion trajectory using joint movements only"""
        if not trajectory:
            logger.error("Cannot execute motion: empty trajectory")
            return False
        
        try:
            logger.info("Starting square motion execution using joint movements")
            logger.info("Moving in a square pattern...")
            
            # Move to the starting position first
            start_joints = trajectory[0]
            self.rtde_c.moveJ(start_joints, self.max_velocity, self.max_acceleration)
            logger.info("✅ Reached starting position")
            
            # Execute the square trajectory (complete the full square regardless of force)
            total_points = len(trajectory)
            logger.info(f"Executing complete square trajectory with {total_points} points")
            logger.info("Note: Will complete full square even if force drops below threshold")
            
            for i, joints in enumerate(trajectory[1:], 1):
                # Move to next joint position
                self.rtde_c.moveJ(joints, self.max_velocity, self.max_acceleration)
                
                # Progress logging
                if i % (self.square_points // 4) == 0:  # Log every quarter of the square
                    side = (i // (self.square_points // 4)) + 1
                    logger.info(f"Completed side {side}/4 of the square")
                
                # Small delay to ensure smooth motion
                time.sleep(0.01)
            
            logger.info("✅ Square motion completed")
            return True
            
        except Exception as e:
            logger.error(f"Error executing square motion: {e}")
            return False
    
    def monitor_force_and_execute_square(self):
        """Main monitoring loop - detects force threshold and executes square motion"""
        logger.info("Starting force monitoring...")
        logger.info(f"Force threshold: {self.force_threshold} N")
        logger.info("Waiting for force detection...")
        logger.info("Press Ctrl+C to stop")
        logger.info("-" * 60)
        
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
                    logger.warning(f"🚨 Force threshold exceeded! Force: {force_magnitude:.2f} N")
                    logger.info(f"Force vector: [Fx: {force_vector[0]:.2f}, Fy: {force_vector[1]:.2f}, Fz: {force_vector[2]:.2f}]")
                    
                    # Move to start position first
                    self.move_to_start_position()
                    
                    # Get current pose to use as square center
                    current_pose = self.get_current_pose()
                    if current_pose is None:
                        logger.error("Cannot get current pose. Aborting square motion.")
                        continue
                    
                    logger.info("🎯 Starting square motion sequence...")
                    
                    # Generate and execute square trajectory using joint movements only
                    trajectory = self.generate_square_joint_trajectory(current_pose)
                    
                    if trajectory:
                        success = self.spin_wrist_joint(self)
                    else:
                        logger.error("❌ Failed to generate square trajectory")
                    
                    # Wait a bit before resuming monitoring
                    logger.info("Resuming force monitoring in 3 seconds...")
                    time.sleep(3.0)
                
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
    """Main function to run the UR5 force square controller"""
    try:
        # Initialize the controller
        controller = UR5ForceSquareController("192.168.1.104")
        
        # Start monitoring and executing square motions
        controller.monitor_force_and_execute_square()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
