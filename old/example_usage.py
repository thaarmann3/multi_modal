#!/usr/bin/env python3
"""
Example usage of the Control System Parameter Design Tool
========================================================

This script demonstrates how to use the control system design tool
for fine-tuning your UR5 robot impedance control parameters.
"""

from system_parameter_design import ControlSystemDesigner
import matplotlib.pyplot as plt

def example_ur5_parameter_tuning():
    """
    Example of how to use the control system designer for UR5 impedance control.
    """
    print("UR5 Impedance Control Parameter Tuning Example")
    print("=" * 50)
    
    # Create the designer
    designer = ControlSystemDesigner()
    
    # Create your current system (based on your impedence_control_simple.py)
    current_system = designer.create_impedance_control_system(
        mass=1.0,        # Approximate robot mass
        spring_k=5.0,    # Your current k value
        damping_b=100.0, # Your current b value  
        alpha=10.0      # Your current alpha value
    )
    
    print(f"Created current system: {current_system}")
    
    # Analyze current system stability
    print("\nAnalyzing current system...")
    stability = designer.analyze_stability()
    print(f"System is stable: {stability['is_stable']}")
    print(f"Stability margin: {stability['stability_margin']:.4f}")
    print(f"Damping ratios: {stability['damping_ratios']}")
    
    # Create some alternative parameter sets for comparison
    print("\nCreating alternative parameter sets...")
    
    # Stiffer system (higher k, higher b)
    stiff_system = designer.create_impedance_control_system(1.0, 15.0, 150.0, 15.0)
    
    # More compliant system (lower k, lower b)
    compliant_system = designer.create_impedance_control_system(1.0, 2.0, 50.0, 5.0)
    
    # Optimized system for good performance
    optimized_system = designer.create_impedance_control_system(1.0, 8.0, 80.0, 12.0)
    
    # Compare all systems
    print("\nComparing systems...")
    systems_to_compare = [current_system, stiff_system, compliant_system, optimized_system]
    comparison_results = designer.compare_systems(systems_to_compare)
    
    # Print comparison results
    print("\nSystem Comparison Results:")
    print("-" * 40)
    for sys_id, results in comparison_results.items():
        if 'error' not in results:
            params = results['parameters']
            print(f"\n{sys_id}:")
            print(f"  Parameters: k={params['spring_k']}, b={params['damping_b']}, α={params['alpha']}")
            print(f"  Overshoot: {results['overshoot']:.2f}%")
            print(f"  Settling Time: {results['settling_time']:.2f}s")
            print(f"  Stability Margin: {results['stability']['stability_margin']:.4f}")
    
    # Try to optimize the current system
    print("\nOptimizing current system for better performance...")
    optimization_result = designer.optimize_parameters(
        target_damping=0.7,      # Good damping ratio
        target_overshoot=5.0,     # Low overshoot
        target_settling_time=2.0  # Fast settling
    )
    
    if optimization_result['success']:
        print("Optimization successful!")
        print("Recommended parameters:")
        for param, value in optimization_result['optimized_parameters'].items():
            print(f"  {param}: {value:.4f}")
    else:
        print("Optimization failed - system may be inherently unstable")
    
    return designer

def interactive_tuning_demo():
    """
    Demonstrate the interactive tuning interface.
    """
    print("\n" + "="*60)
    print("INTERACTIVE TUNING DEMO")
    print("="*60)
    print("This would start the interactive tuning interface.")
    print("You can:")
    print("1. Create new systems with different parameters")
    print("2. Modify existing systems")
    print("3. Analyze stability and performance")
    print("4. Plot root locus diagrams")
    print("5. Optimize parameters automatically")
    print("6. Compare multiple systems")
    print("\nTo start interactive tuning, run:")
    print("python system_parameter_design.py")
    print("Then choose option 7 for interactive mode.")

if __name__ == "__main__":
    # Run the example
    designer = example_ur5_parameter_tuning()
    
    # Show how to start interactive tuning
    interactive_tuning_demo()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Install required packages:")
    print("   pip install -r requirements_control_design.txt")
    print("\n2. Run the main script for interactive tuning:")
    print("   python system_parameter_design.py")
    print("\n3. Use the optimized parameters in your UR5 control script")
    print("   Update impedence_control_simple.py with the recommended values")
