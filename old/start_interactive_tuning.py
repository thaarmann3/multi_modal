#!/usr/bin/env python3
"""
Start Interactive Control System Tuning
======================================

This script starts the interactive tuning interface for your UR5 robot.
"""

from system_parameter_design import ControlSystemDesigner

def start_interactive_tuning():
    """
    Start the interactive tuning interface.
    """
    print("🤖 UR5 Robot Control System Parameter Tuning")
    print("=" * 50)
    print("This tool will help you fine-tune your robot's control parameters")
    print("using control theory and root locus analysis.")
    print()
    print("Your current parameters from impedence_control_simple.py:")
    print("  k (spring constant) = 5.0")
    print("  b (damping coefficient) = 100.0") 
    print("  α (alpha, potential field strength) = 10.0")
    print()
    print("Starting interactive tuning interface...")
    print()
    
    # Create designer and start interactive mode
    designer = ControlSystemDesigner()
    
    # Create initial system with your current parameters
    initial_system = designer.create_impedance_control_system(
        mass=1.0,
        spring_k=5.0,
        damping_b=100.0,
        alpha=10.0
    )
    
    print(f"✅ Created initial system: {initial_system}")
    print()
    
    # Start interactive tuning
    designer.interactive_tuning()

if __name__ == "__main__":
    start_interactive_tuning()
