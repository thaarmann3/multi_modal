#!/usr/bin/env python3
"""
Guided Demo of Control System Parameter Tuning
==============================================

This script demonstrates each feature of the control system design tool
step by step, showing you exactly how to use it for your UR5 robot.
"""

from system_parameter_design import ControlSystemDesigner
import matplotlib.pyplot as plt

def guided_demo():
    """
    Step-by-step demonstration of the control system design tool.
    """
    print("🎯 GUIDED DEMO: UR5 Control System Parameter Tuning")
    print("=" * 60)
    
    # Create the designer
    designer = ControlSystemDesigner()
    
    print("\n📋 STEP 1: Create your current system")
    print("-" * 40)
    print("Creating system with your current parameters:")
    print("  Mass: 1.0 kg")
    print("  Spring K: 5.0 N/m")
    print("  Damping B: 100.0 N⋅s/m")
    print("  Alpha: 10.0")
    
    current_system = designer.create_impedance_control_system(
        mass=1.0,
        spring_k=5.0,
        damping_b=100.0,
        alpha=10.0
    )
    print(f"✅ Created: {current_system}")
    
    print("\n📊 STEP 2: Analyze system stability")
    print("-" * 40)
    stability = designer.analyze_stability()
    print(f"System is stable: {stability['is_stable']}")
    print(f"Stability margin: {stability['stability_margin']:.4f}")
    if stability['damping_ratios']:
        print(f"Damping ratio: {stability['damping_ratios'][0]:.4f}")
    print(f"Natural frequencies: {stability['natural_frequencies']}")
    
    print("\n🔧 STEP 3: Create alternative parameter sets")
    print("-" * 40)
    print("Creating different parameter combinations to compare:")
    
    # Stiffer system
    stiff_system = designer.create_impedance_control_system(1.0, 15.0, 150.0, 15.0)
    print(f"  Stiff system: k=15, b=150, α=15")
    
    # More compliant system
    compliant_system = designer.create_impedance_control_system(1.0, 2.0, 50.0, 5.0)
    print(f"  Compliant system: k=2, b=50, α=5")
    
    # Balanced system
    balanced_system = designer.create_impedance_control_system(1.0, 8.0, 80.0, 12.0)
    print(f"  Balanced system: k=8, b=80, α=12")
    
    print("\n📈 STEP 4: Compare systems")
    print("-" * 40)
    systems_to_compare = [current_system, stiff_system, compliant_system, balanced_system]
    comparison_results = designer.compare_systems(systems_to_compare, plot=False)
    
    print("System Performance Comparison:")
    print("-" * 50)
    for sys_id, results in comparison_results.items():
        if 'error' not in results:
            params = results['parameters']
            print(f"\n{sys_id}:")
            print(f"  Parameters: k={params['spring_k']}, b={params['damping_b']}, α={params['alpha']}")
            print(f"  Overshoot: {results['overshoot']:.2f}%")
            print(f"  Settling Time: {results['settling_time']:.2f}s")
            print(f"  Stability Margin: {results['stability']['stability_margin']:.4f}")
    
    print("\n🎯 STEP 5: Optimize parameters")
    print("-" * 40)
    print("Optimizing for good performance:")
    print("  Target damping ratio: 0.7 (good damping)")
    print("  Target overshoot: 5.0% (low overshoot)")
    print("  Target settling time: 2.0s (fast response)")
    
    try:
        optimization_result = designer.optimize_parameters(
            target_damping=0.7,
            target_overshoot=5.0,
            target_settling_time=2.0
        )
        
        if optimization_result['success']:
            print("✅ Optimization successful!")
            print("🎯 Recommended parameters for your UR5:")
            for param, value in optimization_result['optimized_parameters'].items():
                print(f"  {param}: {value:.4f}")
            
            # Show how to use these in your robot code
            print("\n💻 How to use these parameters in your robot code:")
            print("Update your impedence_control_simple.py with:")
            k_opt = optimization_result['optimized_parameters']['spring_k']
            b_opt = optimization_result['optimized_parameters']['damping_b']
            alpha_opt = optimization_result['optimized_parameters']['alpha']
            print(f"  k = {k_opt:.4f}")
            print(f"  b = {b_opt:.4f}")
            print(f"  alpha = {alpha_opt:.4f}")
        else:
            print("❌ Optimization failed - trying alternative approach...")
            
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
                    print(f"  {param}: {value:.4f}")
            else:
                print("❌ Optimization failed - manual tuning recommended")
                
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        print("This is normal - the system may need manual parameter adjustment")
    
    print("\n🔍 STEP 6: Root Locus Analysis")
    print("-" * 40)
    print("Root locus analysis shows how system poles move as parameters change.")
    print("This helps you understand system stability and performance.")
    print("To see root locus plots, you would run:")
    print("  designer.plot_root_locus()")
    print("This shows:")
    print("  • Root locus diagram")
    print("  • Pole-zero map")
    print("  • Bode plot")
    print("  • Step response")
    
    print("\n🚀 HOW TO USE THE INTERACTIVE INTERFACE")
    print("=" * 60)
    print("To use the interactive tuning interface:")
    print("1. Run: python3 system_parameter_design.py")
    print("2. You'll see a menu with these options:")
    print()
    print("   Option 1: Create new system")
    print("   - Enter custom mass, k, b, alpha values")
    print("   - Useful for testing different parameter combinations")
    print()
    print("   Option 2: Modify current system")
    print("   - Change parameters of your current system")
    print("   - Quick way to test parameter variations")
    print()
    print("   Option 3: Analyze stability")
    print("   - Check if your system is stable")
    print("   - See damping ratios and natural frequencies")
    print()
    print("   Option 4: Plot root locus")
    print("   - Visualize how poles move with parameter changes")
    print("   - See Bode plots and step responses")
    print()
    print("   Option 5: Optimize parameters")
    print("   - Automatically find optimal parameters")
    print("   - Set target performance requirements")
    print()
    print("   Option 6: Compare systems")
    print("   - Compare multiple parameter sets")
    print("   - See performance metrics side by side")
    print()
    print("   Option 7: Exit")
    print("   - Exit the interactive interface")
    
    print("\n💡 TIPS FOR YOUR UR5 ROBOT")
    print("=" * 60)
    print("• Start with your current values: k=5, b=100, α=10")
    print("• For stiffer behavior (more precise): increase k and b")
    print("• For more compliant behavior (smoother): decrease k and b")
    print("• Use root locus plots to understand stability")
    print("• Optimize for your specific task requirements")
    print("• Test parameters in simulation before using on robot")
    
    print("\n🎉 Demo complete! You're ready to use the control system design tool.")
    print("Run 'python3 system_parameter_design.py' to start interactive tuning.")

if __name__ == "__main__":
    guided_demo()
