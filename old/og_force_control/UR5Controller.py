"""
ur5_basic_funcs.py

A reusable module for basic UR5 robot control functions using ur-rtde.
Provides initialization, connection, and common movement operations.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, Tuple, List, Union
import logging
import math
import signal
import sys

# Import ur-rtde modules
import rtde_receive
import rtde_control


# Configure logging
logger = logging.getLogger(__name__)

__all__ = ["UR5Controller"]
__version__ = "0.1.0"


class UR5Controller:
    """
    Basic UR5 robot controller using ur-rtde for real-time data exchange.
    Provides initialization, connection, movement, and safety operations.
    """
    
    def __init__(self, robot_ip: str = "192.168.1.104", 
                default_speed: float = 0.1, 
                default_acceleration: float = 0.1, 
                default_joint_speed: float = 0.5):
        """
        Initialize UR5 controller.
        
        Args:
            robot_ip: IP address of the UR5 robot
            default_speed: Default movement speed (m/s)
            default_acceleration: Default movement acceleration (m/s²)
            default_joint_speed: Default joint movement speed (rad/s)
        """
        self.robot_ip = robot_ip
        self.connected = False
        self.rtde_r = None  # RTDE receive interface
        self.rtde_c = None  # RTDE control interface
        
        # Movement parameters
        self.default_speed = 0.1  # m/s
        self.default_acceleration = 0.1  # m/s²
        self.default_joint_speed = 0.5  # rad/s
        
        # Termination tracking
        self.termination_requested = False
        self._setup_signal_handlers()
        
        logger.info(f"UR5Controller initialized for IP: {robot_ip}")
        self.connect()
    
    def connect(self) -> bool:
        """
        Establish connection to the UR5 robot using ur-rtde.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            
            # Initialize RTDE receive interface
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            
            # Initialize RTDE control interface
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            
            # Test connection by getting current position
            if self.rtde_r.isConnected():
                self.connected = True
                logger.info("Successfully connected to UR5 robot via RTDE")
                return True
            else:
                logger.error("Failed to establish RTDE connection")
                return False
            
        except ImportError:
            logger.error("ur-rtde library not found. Install with: pip install ur-rtde")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to UR5: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the UR5 robot."""
        if self.connected:
            try:
                if self.rtde_r:
                    self.rtde_r.disconnect()
                if self.rtde_c:
                    self.rtde_c.disconnect()
                self.connected = False
                logger.info("Disconnected from UR5 robot")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
    
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.connected and self.rtde_r is not None and self.rtde_r.isConnected()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful termination."""
        def signal_handler(signum, frame):
            logger.info("Termination signal received (Ctrl+C)")
            self.termination_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def check_termination(self) -> bool:
        """
        Check if user has requested termination via Ctrl+C.
        
        Returns:
            bool: True if termination requested, False otherwise
        """
        return self.termination_requested
    
    def reset_termination_flag(self) -> None:
        """Reset the termination flag."""
        self.termination_requested = False
    
    def get_position(self) -> Optional[np.ndarray]:
        """
        Get current robot position (x, y, z, rx, ry, rz).
        
        Returns:
            np.ndarray: Current position [x, y, z, rx, ry, rz] or None if not connected
        """
        if not self.is_connected():
            logger.warning("Robot not connected")
            return None
        
        try:
            position = self.rtde_r.getActualTCPPose()
            return np.array(position)
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None
    
    def get_joints(self) -> Optional[np.ndarray]:
        """
        Get current joint angles.
        
        Returns:
            np.ndarray: Current joint angles [j1, j2, j3, j4, j5, j6] or None if not connected
        """
        if not self.is_connected():
            logger.warning("Robot not connected")
            return None
        
        try:
            joints = self.rtde_r.getActualQ()
            return np.array(joints)
        except Exception as e:
            logger.error(f"Failed to get joints: {e}")
            return None
    
    def move_to(self, target_position: Union[List[float], np.ndarray], 
                speed: Optional[float] = None,
                use_linear: bool = False) -> bool:
        """
        Move robot to target position in Cartesian space...
            Takes a pose and uses the inverse kinematics to move to the target joint position.
        
        Args:
            target_position: Target position [x, y, z, rx, ry, rz]
            speed: Movement speed (m/s for linear, rad/s for joint), uses default if None
            use_linear: If True, use linear movement (can cause singularities)
            
        Returns:
            bool: True if movement successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Robot not connected")
            return False
        
        try:
            speed = speed or self.default_speed
            if use_linear:
                # Linear movement in Cartesian space (can cause singularities)
                logger.warning("Using linear movement - may cause singularities")
                success = self.move_linear(target_position, speed, self.default_acceleration)
                
            else:
                # Use the correct ur-rtde inverse kinematics function
                try:
                    new_joints = self.rtde_c.getInverseKinematics(target_position)
                    success = self.move_joints(new_joints, speed)
                except Exception as e:
                    logger.warning(f"IK failed: {e} - falling back to linear movement")
                    success = self.move_linear(target_position, speed, self.default_acceleration)
         
            if success:
                logger.info(f"Moved to position: {target_position}")
                return True
            else:
                logger.error("Failed to move to position")
                return False
            
        except Exception as e:
            logger.error(f"Failed to move to position: {e}")
            return False
    
    def move_joints(self, target_joints: Union[List[float], np.ndarray], 
                   speed: Optional[float] = None) -> bool:
        """
        Move robot joints to specific angles in joint space.
        
        Args:
            target_joints: Target joint angles [j1, j2, j3, j4, j5, j6] in radians
            speed: Joint speed (rad/s), uses default if None
            
        Returns:
            bool: True if movement successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Robot not connected")
            return False
        
        try:
            speed = speed or self.default_joint_speed
            
            # Direct joint movement - no inverse kinematics needed
            success = self.rtde_c.moveJ(target_joints, speed)
            
            if success:
                logger.info(f"Moved joints to: {target_joints}")
                return True
            else:
                logger.error("Failed to move joints")
                return False
            
        except Exception as e:
            logger.error(f"Failed to move joints: {e}")
            return False
    
    def move_relative(self, delta_position: Union[List[float], np.ndarray], 
                     speed: Optional[float] = None, 
                     use_linear: bool = False) -> bool:
        """
        Move robot relative to current position using joint movement (safer).
        
        Args:
            delta_position: Relative movement [dx, dy, dz, drx, dry, drz]
            speed: Movement speed (rad/s for joint movement), uses default if None
            use_linear: If True, use linear movement (can cause singularities)
            
        Returns:
            bool: True if movement successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Robot not connected")
            return False
        
        try:
            current_pos = self.get_position()
            if current_pos is None:
                return False
            
            target_position = current_pos + np.array(delta_position)
            return self.move_to(target_position, speed, use_linear)
            
        except Exception as e:
            logger.error(f"Failed to move relative: {e}")
            return False
    
    def move_servo(self, target_position: Union[List[float], np.ndarray], 
                   speed: Optional[float] = None, 
                   acceleration: Optional[float] = None,
                   dt: float = 0.002) -> bool:
        """
        Move robot to target position using servoL() for smooth servo control.
        This provides continuous servo control with high frequency updates.
        
        Args:
            target_position: Target position [x, y, z, rx, ry, rz]
            speed: Movement speed (m/s), uses default if None
            acceleration: Movement acceleration (m/s²), uses default if None
            dt: Time step for servo control (seconds), default 0.002 (500Hz)
            
        Returns:
            bool: True if movement successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Robot not connected")
            return False
        
        try:
            speed = speed or self.default_speed
            acceleration = acceleration or self.default_acceleration
            
            # Use servoL for smooth servo control
            success = self.rtde_c.servoL(target_position, speed, acceleration, dt, 0.1, 300)
            
            if success:
                logger.info(f"Servo move to position: {target_position}")
                return True
            else:
                logger.error("Failed to move with servo control")
                return False
            
        except Exception as e:
            logger.error(f"Failed to move with servo control: {e}")
            return False
    
    def move_servo_relative(self, delta_position: Union[List[float], np.ndarray], 
                           speed: Optional[float] = None, 
                           acceleration: Optional[float] = None,
                           dt: float = 0.002,
                           lookahead_time: float = 0.01,
                           gain: float = 50) -> bool:
        """
        Move robot relative to current position using servoL() for smooth servo control.
        This provides continuous servo control with high frequency updates.
        
        Args:
            delta_position: Relative movement [dx, dy, dz, drx, dry, drz]
            speed: Movement speed (m/s), uses default if None
            acceleration: Movement acceleration (m/s²), uses default if None
            dt: Time step for servo control (seconds), default 0.002 (500Hz)
            
        Returns:
            bool: True if movement successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Robot not connected")
            return False
        
        try:
            current_pos = self.get_position()
            if current_pos is None:
                return False
            
            # Calculate target position
            target_position = current_pos + np.array(delta_position)
            
            # Validate target position (basic workspace check)
            # if (target_position[2] < 0.1 or target_position[2] > 1.5 or  # Z height limits
            #     abs(target_position[0]) > 1.0 or abs(target_position[1]) > 1.0):  # XY limits
            #     logger.warning(f"Target position {target_position} may be outside workspace")
            #     return False
            
            # Use servoL for smooth servo control
            speed = speed or self.default_speed
            acceleration = acceleration or self.default_acceleration
            
            logger.info(f"Servo move: target={target_position}, speed={speed}, accel={acceleration}")
            success = self.rtde_c.servoL(target_position, speed, acceleration, dt, lookahead_time, gain)
            
            if success:
                logger.info(f"Servo relative move completed: {delta_position}")
                return True
            else:
                logger.error("Failed to move relative with servo control")
                return False
            
        except Exception as e:
            logger.error(f"Failed to move relative with servo control: {e}")
            return False
    
    def servo_stop(self) -> None:
        """Stop servo control and return to normal operation."""
        if self.is_connected():
            try:
                self.rtde_c.servoStop()
                logger.info("Servo control stopped")
            except Exception as e:
                logger.error(f"Error stopping servo control: {e}")

    def move_linear(self, target_position: Union[List[float], np.ndarray], 
                   speed: Optional[float] = None, 
                   acceleration: Optional[float] = None) -> bool:
        """
        Move robot to target position using linear movement (can cause singularities).
        Use with caution - prefer move_to() for safer joint movement.
        
        Args:
            target_position: Target position [x, y, z, rx, ry, rz]
            speed: Movement speed (m/s), uses default if None
            acceleration: Movement acceleration (m/s²), uses default if None
            
        Returns:
            bool: True if movement successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Robot not connected")
            return False
        
        try:
            speed = speed or self.default_speed
            acceleration = acceleration or self.default_acceleration
            
            # Linear movement (can cause singularities)
            success = self.rtde_c.moveL(target_position, speed, acceleration)
            
            if success:
                logger.info(f"Linear move to position: {target_position}")
                return True
            else:
                logger.error("Failed to move linearly to position")
                return False
            
        except Exception as e:
            logger.error(f"Failed to move linearly to position: {e}")
            return False
    
    def emergency_stop(self) -> None:
        """Emergency stop the robot."""
        if self.is_connected():
            try:
                self.rtde_c.stopJ()
                logger.warning("Emergency stop activated")
            except Exception as e:
                logger.error(f"Error during emergency stop: {e}")
    
    def set_speed(self, speed: float) -> None:
        """Set default movement speed."""
        self.default_speed = speed
        logger.info(f"Default speed set to: {speed} m/s")
    
    def set_acceleration(self, acceleration: float) -> None:
        """Set default movement acceleration."""
        self.default_acceleration = acceleration
        logger.info(f"Default acceleration set to: {acceleration} m/s²")
    
    def set_linear_speed(self, velocity: Union[List[float], np.ndarray]) -> bool:
        """
        Set linear speed using speedl() for continuous velocity control.
        
        Args:
            velocity: Linear velocity [vx, vy, vz, wx, wy, wz] in m/s and rad/s
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Robot not connected")
            return False
        
        try:
            # Use speedl() for continuous linear velocity control
            success = self.rtde_c.speedL(velocity)
            
            if success:
                logger.info(f"Set linear speed to: {velocity}")
                return True
            else:
                logger.error("Failed to set linear speed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to set linear speed: {e}")
            return False
    
    def set_joint_speed(self, speed: float) -> None:
        """Set default joint movement speed."""
        self.default_joint_speed = speed
        logger.info(f"Default joint speed set to: {speed} rad/s")
    
    def home_position(self) -> bool:
        """
        Move robot to home position (vertical position).
        
        Returns:
            bool: True if movement successful, False otherwise
        """
        home_joints = [
            0,  # Base: -90 degrees (point forward)
            -math.pi/2,         # Shoulder: 0 degrees (straight up)
            0.0,         # Elbow: 0 degrees (straight)
            -math.pi/2,  # Wrist1: -90 degrees (adjust for vertical)
            0.0,         # Wrist2: 0 degrees (neutral)
            0.0          # Wrist3: 0 degrees (neutral)
        ]
        return self.move_joints(home_joints)
    
    def get_status(self) -> dict:
        """
        Get robot status information.
        
        Returns:
            dict: Status information including connection, position, and joints
        """
        status = {
            "connected": self.is_connected(),
            "position": self.get_position(),
            "joints": self.get_joints(),
            "ip": self.robot_ip
        }
        return status
    
    def wait_for_movement(self, timeout: float = 10.0) -> bool:
        """
        Wait for current movement to complete.
        
        Args:
            timeout: Maximum time to wait (seconds)
            
        Returns:
            bool: True if movement completed, False if timeout
        """
        if not self.is_connected():
            return False
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if robot is still moving using RTDE
                if not self.rtde_r.isProgramRunning():
                    return True
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error waiting for movement: {e}")
                return False
        
        logger.warning("Movement timeout reached")
        return False
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def get_force(self) -> Optional[np.ndarray]:
        """
        Get current TCP force readings.
        
        Returns:
            np.ndarray: Current force [Fx, Fy, Fz, Tx, Ty, Tz] or None if not connected
        """
        if not self.is_connected():
            logger.warning("Robot not connected")
            return None
        
        try:
            force = self.rtde_r.getActualTCPForce()
            return np.array(force)
        except Exception as e:
            logger.error(f"Failed to get force: {e}")
            return None
    
    def get_velocity(self) -> Optional[np.ndarray]:
        """
        Get current TCP velocity.
        
        Returns:
            np.ndarray: Current velocity [vx, vy, vz, wx, wy, wz] or None if not connected
        """
        if not self.is_connected():
            logger.warning("Robot not connected")
            return None
        
        try:
            velocity = self.rtde_r.getActualTCPSpeed()
            return np.array(velocity)
        except Exception as e:
            logger.error(f"Failed to get velocity: {e}")
            return None
    
    def get_joint_velocity(self) -> Optional[np.ndarray]:
        """
        Get current joint velocities.
        
        Returns:
            np.ndarray: Current joint velocities [j1, j2, j3, j4, j5, j6] or None if not connected
        """
        if not self.is_connected():
            logger.warning("Robot not connected")
            return None
        
        try:
            joint_velocity = self.rtde_r.getActualQd()
            return np.array(joint_velocity)
        except Exception as e:
            logger.error(f"Failed to get joint velocity: {e}")
            return None
    
    def get_digital_input(self, pin: int) -> Optional[bool]:
        """
        Get digital input state.
        
        Args:
            pin: Digital input pin number (0-7)
            
        Returns:
            bool: Input state or None if not connected
        """
        if not self.is_connected():
            logger.warning("Robot not connected")
            return None
        
        try:
            digital_inputs = self.rtde_r.getDigitalInputBits()
            return bool(digital_inputs & (1 << pin))
        except Exception as e:
            logger.error(f"Failed to get digital input {pin}: {e}")
            return None
    
    def set_digital_output(self, pin: int, state: bool) -> bool:
        """
        Set digital output state.
        
        Args:
            pin: Digital output pin number (0-7)
            state: Output state (True/False)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Robot not connected")
            return False
        
        try:
            success = self.rtde_c.setDigitalOutput(pin, state)
            if success:
                logger.info(f"Set digital output {pin} to {state}")
            return success
        except Exception as e:
            logger.error(f"Failed to set digital output {pin}: {e}")
            return False
