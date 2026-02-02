import rtde_control
import rtde_receive
import time
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UR5VerticalSpinner:
    def __init__(self, robot_ip="192.168.1.104"):
        """Initialize UR5 robot control interfaces"""
        self.robot_ip = robot_ip
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        
        # Motion parameters
        self.move_velocity = 0.5    # rad/s for joint movements
        self.move_acceleration = 0.3  # rad/s² for joint movements
        self.spin_velocity = 0.5   # rad/s for wrist spinning
        self.spin_duration = 5.0   # seconds to spin
        
        logger.info("UR5 Vertical Spinner initialized")
    
    def get_current_joint_positions(self):
        """Get current joint positions"""
        try:
            return self.rtde_r.getActualQ()
        except Exception as e:
            logger.error(f"Error reading joint positions: {e}")
            return None
    
    def move_to_vertical_position(self):
        """Move the UR5 arm to a vertical position (pointing straight up)"""
        logger.info("Moving UR5 to vertical position...")
        
        # Vertical position: all joints at 0 except joint 1 at -90 degrees (pointing up)
        # Joint positions in radians: [base, shoulder, elbow, wrist1, wrist2, wrist3]
        vertical_joints = [
            0,  # Base: -90 degrees (point forward)
            -math.pi/2,         # Shoulder: 0 degrees (straight up)
            0.0,         # Elbow: 0 degrees (straight)
            -math.pi/2,  # Wrist1: -90 degrees (adjust for vertical)
            0.0,         # Wrist2: 0 degrees (neutral)
            0.0          # Wrist3: 0 degrees (neutral)
        ]
        
        try:
            # Move to vertical position
            self.rtde_c.moveJ(vertical_joints, self.move_velocity, self.move_acceleration)
            logger.info("✅ Successfully moved to vertical position")
            
            # Wait a moment for the movement to complete
            time.sleep(2.0)
            
            # Verify position
            current_joints = self.get_current_joint_positions()
            if current_joints:
                logger.info(f"Current joint positions: {[f'{j:.3f}' for j in current_joints]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error moving to vertical position: {e}")
            return False
    
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
    
    def run_sequence(self):
        """Run the complete sequence: move to vertical, then spin wrist"""
        logger.info("🤖 Starting UR5 vertical position and wrist spin sequence")
        logger.info("=" * 60)
        
        try:
            # Step 1: Move to vertical position
            logger.info("Step 1: Moving to vertical position")
            if not self.move_to_vertical_position():
                logger.error("Failed to move to vertical position. Aborting sequence.")
                return False
            
            # Wait a moment between movements
            time.sleep(1.0)
            
            # Step 2: Spin the wrist joint
            logger.info("Step 2: Spinning wrist joint")
            if not self.spin_wrist_joint():
                logger.error("Failed to spin wrist joint.")
                return False
            
            logger.info("=" * 60)
            logger.info("🎉 Sequence completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in sequence: {e}")
            return False
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
    """Main function to run the UR5 vertical spinner"""
    try:
        # Initialize the controller
        spinner = UR5VerticalSpinner("192.168.1.104")
        
        # Run the sequence
        success = spinner.run_sequence()
        
        if success:
            logger.info("✅ All operations completed successfully!")
            return 0
        else:
            logger.error("❌ Some operations failed!")
            return 1
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
