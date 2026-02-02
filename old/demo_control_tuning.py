#!/usr/bin/env python3
"""
Quick Demo of Control System Parameter Tuning
============================================

This script demonstrates the key features of the control system design tool
without requiring interactive input.
"""

from system_parameter_design import ControlSystemDesigner
import matplotlib.pyplot as plt

def demo_control_tuning():
    """
    Demonstrate the control system parameter tuning capabilities.
    """
    print("🎯 Control System Parameter Tuning Demo")
    print("=" * 50)
    
    # Create the designer
    designer = ControlSystemDesigner()
    
    # 1. Create your current system (from impedence_control_simple.py)
    print("\n1. Creating your current system...")
    current_system = designer.create_impedance_control_system(
        mass=1.0,        # Approximate robot mass
        spring_k=5.0,    # Your current k value
        damping_b=100.0, # Your current b value  
        alpha=10.0      # Your current alpha value
    )
    print(f"✅ Created: {current_system}")
    
    # 2. Analyze current system
    print("\n2. Analyzing current system stability...")
    stability = designer.analyze_stability()
    print(f"   Stable: {stability['is_stable']}")
    print(f"   Stability Margin: {stability['stability_margin']:.4f}")
    if stability['damping_ratios']:
        print(f"   Damping Ratio: {stability['damping_ratios'][0]:.4f}")
    
    # 3. Create alternative parameter sets
    print("\n3. Creating alternative parameter sets...")
    
    # Stiffer system
    stiff_system = designer.create_impedance_control_system(1.0, 15.0, 150.0, 15.0)
    print(f"   Stiff system: k=15, b=150, α=15")
    
    # More compliant system  
    compliant_system = designer.create_impedance_control_system(1.0, 2.0, 50.0, 5.0)
    print(f"   Compliant system: k=2, b=50, α=5")
    
    # 4. Compare systems
    print("\n4. Comparing systems...")
    systems_to_compare = [current_system, stiff_system, compliant_system]
    comparison_results = designer.compare_systems(systems_to_compare, plot=False)
    
    print("\n📊 System Comparison Results:")
    print("-" * 50)
    for sys_id, results in comparison_results.items():
        if 'error' not in results:
            params = results['parameters']
            print(f"\n{sys_id}:")
            print(f"   Parameters: k={params['spring_k']}, b={params['damping_b']}, α={params['alpha']}")
            print(f"   Overshoot: {results['overshoot']:.2f}%")
            print(f"   Settling Time: {results['settling_time']:.2f}s")
            print(f"   Stability Margin: {results['stability']['stability_margin']:.4f}")
    
    # 5. Try optimization
    print("\n5. Optimizing parameters...")
    try:
        optimization_result = designer.optimize_parameters(
            target_damping=0.7,      # Good damping ratio
            target_overshoot=5.0,     # Low overshoot
            target_settling_time=2.0  # Fast settling
        )
        
        if optimization_result['success']:
            print("✅ Optimization successful!")
            print("🎯 Recommended parameters:")
            for param, value in optimization_result['optimized_parameters'].items():
                print(f"   {param}: {value:.4f}")
        else:
            print("❌ Optimization failed - trying different approach...")
            
            # Try with different targets
            optimization_result2 = designer.optimize_parameters(
                target_damping=0.5,
                target_overshoot=10.0,
                target_settling_time=3.0
            )
            
            if optimization_result2['success']:
                print("✅ Alternative optimization successful!")
                print("🎯 Recommended parameters:")
                for param, value in optimization_result2['optimized_parameters'].items():
                    print(f"   {param}: {value:.4f}")
            else:
                print("❌ Optimization failed - system may need manual tuning")
                
    except Exception as e:
        print(f"❌ Optimization error: {e}")
    
    # 6. Show how to use the interactive interface
    print("\n" + "="*60)
    print("🚀 INTERACTIVE TUNING INTERFACE")
    print("="*60)
    print("To use the interactive tuning interface:")
    print("1. Run: python3 system_parameter_design.py")
    print("2. Choose from these options:")
    print("   • Option 1: Create new system with custom parameters")
    print("   • Option 2: Modify current system parameters")
    print("   • Option 3: Analyze system stability")
    print("   • Option 4: Plot root locus diagrams")
    print("   • Option 5: Optimize parameters automatically")
    print("   • Option 6: Compare multiple systems")
    print("   • Option 7: Exit")
    
    print("\n💡 TIPS FOR YOUR UR5 ROBOT:")
    print("- Start with your current values: k=5, b=100, α=10")
    print("- Try stiffer parameters for more precise control")
    print("- Try more compliant parameters for smoother interaction")
    print("- Use root locus plots to see how poles move")
    print("- Optimize for your specific performance requirements")
    
    return designer

if __name__ == "__main__":
    designer = demo_control_tuning()
    
    print("\n🎉 Demo complete! You're ready to use the interactive tuning interface.")
