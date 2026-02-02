#!/usr/bin/env python3
"""
Control System Parameter Design Tool
====================================

This script helps fine-tune gain, spring, and damping constants for mechanical systems
using control theory and root locus analysis. Specifically designed for impedance control
systems like UR5 robot control.

Features:
- Root locus analysis for stability assessment
- Transfer function modeling
- Parameter optimization using control theory metrics
- Interactive parameter tuning
- Bode plot analysis
- Step response analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
import control
from control import TransferFunction, bode_plot, step_response, root_locus
import sympy as sp
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class ControlSystemDesigner:
    """
    A comprehensive control system design tool for parameter optimization
    using root locus analysis and control theory.
    """
    
    def __init__(self):
        self.systems = {}
        self.current_system = None
        
    def create_impedance_control_system(self, mass: float = 1.0, 
                                      spring_k: float = 5.0, 
                                      damping_b: float = 100.0,
                                      alpha: float = 10.0) -> str:
        """
        Create an impedance control system model for robot control.
        
        The system models: F = F_potential - F_error = k*(x_actual - x_goal) + b*(x_dot_actual - x_dot_goal)
        
        Parameters:
        -----------
        mass : float
            System mass (kg)
        spring_k : float
            Spring constant (N/m)
        damping_b : float
            Damping coefficient (N⋅s/m)
        alpha : float
            Potential field strength parameter
        """
        
        # Define the system transfer function
        # For impedance control: G(s) = 1/(ms² + bs + k)
        s = control.TransferFunction.s
        G_plant = 1 / (mass * s**2 + damping_b * s + spring_k)
        
        # Potential field controller (simplified as proportional gain)
        K_potential = alpha
        
        # Overall system with potential field
        G_system = K_potential * G_plant
        
        system_id = f"impedance_m{mass}_k{spring_k}_b{damping_b}_a{alpha}"
        self.systems[system_id] = {
            'plant': G_plant,
            'system': G_system,
            'parameters': {
                'mass': mass,
                'spring_k': spring_k,
                'damping_b': damping_b,
                'alpha': alpha
            },
            'type': 'impedance_control'
        }
        
        self.current_system = system_id
        return system_id
    
    def analyze_stability(self, system_id: str = None) -> Dict:
        """
        Analyze system stability using root locus and pole analysis.
        """
        if system_id is None:
            system_id = self.current_system
            
        if system_id not in self.systems:
            raise ValueError(f"System {system_id} not found")
            
        system = self.systems[system_id]
        G_system = system['system']
        
        # Get poles and zeros
        poles = G_system.poles()
        zeros = G_system.zeros()
        
        # Calculate stability metrics
        damping_ratios = []
        natural_frequencies = []
        
        for pole in poles:
            if np.iscomplex(pole):
                wn = abs(pole)
                zeta = -pole.real / wn
                damping_ratios.append(zeta)
                natural_frequencies.append(wn)
        
        # Stability assessment
        is_stable = all(pole.real < 0 for pole in poles)
        margin_stability = min(-pole.real for pole in poles if pole.real < 0) if is_stable else 0
        
        return {
            'poles': poles,
            'zeros': zeros,
            'is_stable': is_stable,
            'damping_ratios': damping_ratios,
            'natural_frequencies': natural_frequencies,
            'stability_margin': margin_stability
        }
    
    def plot_root_locus(self, system_id: str = None, k_range: Tuple[float, float] = (0.1, 1000)):
        """
        Plot root locus for the system with varying gain.
        """
        if system_id is None:
            system_id = self.current_system
            
        if system_id not in self.systems:
            raise ValueError(f"System {system_id} not found")
            
        system = self.systems[system_id]
        G_plant = system['plant']
        
        # Create root locus plot
        plt.figure(figsize=(12, 8))
        
        # Root locus
        plt.subplot(2, 2, 1)
        control.root_locus(G_plant, kvect=np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), 1000))
        plt.title('Root Locus')
        plt.grid(True)
        
        # Pole-zero plot
        plt.subplot(2, 2, 2)
        poles = G_plant.poles()
        zeros = G_plant.zeros()
        
        plt.plot(poles.real, poles.imag, 'x', markersize=10, label='Poles')
        plt.plot(zeros.real, zeros.imag, 'o', markersize=10, label='Zeros')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.title('Pole-Zero Map')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        
        # Bode plot
        plt.subplot(2, 2, 3)
        control.bode_plot(G_plant, dB=True)
        plt.title('Bode Plot')
        
        # Step response
        plt.subplot(2, 2, 4)
        t, y = control.step_response(G_plant)
        plt.plot(t, y)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Step Response')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return self.analyze_stability(system_id)
    
    def optimize_parameters(self, system_id: str = None, 
                          target_damping: float = 0.7,
                          target_overshoot: float = 5.0,
                          target_settling_time: float = 2.0) -> Dict:
        """
        Optimize system parameters using control theory metrics.
        """
        if system_id is None:
            system_id = self.current_system
            
        if system_id not in self.systems:
            raise ValueError(f"System {system_id} not found")
            
        system = self.systems[system_id]
        params = system['parameters']
        
        def objective(x):
            k_new, b_new, alpha_new = x
            
            # Create new system with updated parameters
            mass = params['mass']
            s = control.TransferFunction.s
            G_plant = 1 / (mass * s**2 + b_new * s + k_new)
            G_system = alpha_new * G_plant
            
            # Analyze stability
            stability = self.analyze_stability()
            
            # Calculate performance metrics
            try:
                t, y = control.step_response(G_system, T=np.linspace(0, 10, 1000))
                
                # Calculate overshoot
                final_value = y[-1]
                peak_value = np.max(y)
                overshoot = ((peak_value - final_value) / final_value) * 100
                
                # Calculate settling time (2% criterion)
                settling_idx = np.where(np.abs(y - final_value) <= 0.02 * final_value)[0]
                settling_time = t[settling_idx[0]] if len(settling_idx) > 0 else 10.0
                
                # Calculate damping ratio
                damping_ratio = stability['damping_ratios'][0] if stability['damping_ratios'] else 0
                
                # Objective function (minimize)
                cost = (abs(damping_ratio - target_damping) * 10 +
                       abs(overshoot - target_overshoot) * 0.1 +
                       abs(settling_time - target_settling_time) * 0.1)
                
                return cost
                
            except:
                return 1000  # Large penalty for unstable systems
        
        # Initial guess
        x0 = [params['spring_k'], params['damping_b'], params['alpha']]
        
        # Bounds for parameters
        bounds = [(0.1, 100), (1, 1000), (0.1, 100)]
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            k_opt, b_opt, alpha_opt = result.x
            
            # Create optimized system
            optimized_id = self.create_impedance_control_system(
                mass=params['mass'],
                spring_k=k_opt,
                damping_b=b_opt,
                alpha=alpha_opt
            )
            
            return {
                'success': True,
                'optimized_parameters': {
                    'spring_k': k_opt,
                    'damping_b': b_opt,
                    'alpha': alpha_opt
                },
                'optimized_system_id': optimized_id,
                'cost': result.fun
            }
        else:
            return {
                'success': False,
                'message': 'Optimization failed',
                'cost': result.fun
            }
    
    def compare_systems(self, system_ids: List[str], plot: bool = True) -> Dict:
        """
        Compare multiple systems and their performance.
        """
        results = {}
        
        for sys_id in system_ids:
            if sys_id in self.systems:
                stability = self.analyze_stability(sys_id)
                system = self.systems[sys_id]
                
                # Get step response
                try:
                    t, y = control.step_response(system['system'], T=np.linspace(0, 10, 1000))
                    
                    # Calculate metrics
                    final_value = y[-1]
                    peak_value = np.max(y)
                    overshoot = ((peak_value - final_value) / final_value) * 100
                    
                    settling_idx = np.where(np.abs(y - final_value) <= 0.02 * final_value)[0]
                    settling_time = t[settling_idx[0]] if len(settling_idx) > 0 else 10.0
                    
                    results[sys_id] = {
                        'parameters': system['parameters'],
                        'stability': stability,
                        'overshoot': overshoot,
                        'settling_time': settling_time,
                        'step_response': (t, y)
                    }
                except:
                    results[sys_id] = {
                        'parameters': system['parameters'],
                        'stability': stability,
                        'error': 'Could not compute step response'
                    }
        
        if plot and len(results) > 1:
            plt.figure(figsize=(15, 10))
            
            # Step responses
            plt.subplot(2, 3, 1)
            for sys_id, data in results.items():
                if 'step_response' in data:
                    t, y = data['step_response']
                    plt.plot(t, y, label=f'{sys_id}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Step Responses Comparison')
            plt.legend()
            plt.grid(True)
            
            # Bode plots
            plt.subplot(2, 3, 2)
            for sys_id, data in results.items():
                if sys_id in self.systems:
                    control.bode_plot(self.systems[sys_id]['system'], label=sys_id)
            plt.title('Bode Plot Comparison')
            
            # Parameter comparison
            plt.subplot(2, 3, 3)
            params = [data['parameters'] for data in results.values()]
            param_names = ['spring_k', 'damping_b', 'alpha']
            x = np.arange(len(param_names))
            width = 0.8 / len(params)
            
            for i, (sys_id, param_dict) in enumerate(zip(results.keys(), params)):
                values = [param_dict[name] for name in param_names]
                plt.bar(x + i * width, values, width, label=sys_id)
            
            plt.xlabel('Parameters')
            plt.ylabel('Values')
            plt.title('Parameter Comparison')
            plt.xticks(x + width/2, param_names)
            plt.legend()
            
            # Performance metrics
            plt.subplot(2, 3, 4)
            overshoots = [data.get('overshoot', 0) for data in results.values()]
            plt.bar(results.keys(), overshoots)
            plt.ylabel('Overshoot (%)')
            plt.title('Overshoot Comparison')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 3, 5)
            settling_times = [data.get('settling_time', 0) for data in results.values()]
            plt.bar(results.keys(), settling_times)
            plt.ylabel('Settling Time (s)')
            plt.title('Settling Time Comparison')
            plt.xticks(rotation=45)
            
            # Stability margin
            plt.subplot(2, 3, 6)
            stability_margins = [data['stability'].get('stability_margin', 0) for data in results.values()]
            plt.bar(results.keys(), stability_margins)
            plt.ylabel('Stability Margin')
            plt.title('Stability Margin Comparison')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
        
        return results
    
    def interactive_tuning(self):
        """
        Interactive parameter tuning interface.
        """
        print("=== Interactive Control System Parameter Tuning ===")
        print("Current system parameters:")
        
        if self.current_system:
            params = self.systems[self.current_system]['parameters']
            print(f"  Mass: {params['mass']}")
            print(f"  Spring K: {params['spring_k']}")
            print(f"  Damping B: {params['damping_b']}")
            print(f"  Alpha: {params['alpha']}")
        
        while True:
            print("\nOptions:")
            print("1. Create new system")
            print("2. Modify current system")
            print("3. Analyze stability")
            print("4. Plot root locus")
            print("5. Optimize parameters")
            print("6. Compare systems")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                try:
                    mass = float(input("Enter mass (kg): "))
                    k = float(input("Enter spring constant K: "))
                    b = float(input("Enter damping coefficient B: "))
                    alpha = float(input("Enter alpha (potential field strength): "))
                    
                    sys_id = self.create_impedance_control_system(mass, k, b, alpha)
                    print(f"Created system: {sys_id}")
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
            
            elif choice == '2':
                if not self.current_system:
                    print("No current system. Create one first.")
                    continue
                
                try:
                    k = float(input(f"Enter new spring constant K (current: {self.systems[self.current_system]['parameters']['spring_k']}): "))
                    b = float(input(f"Enter new damping coefficient B (current: {self.systems[self.current_system]['parameters']['damping_b']}): "))
                    alpha = float(input(f"Enter new alpha (current: {self.systems[self.current_system]['parameters']['alpha']}): "))
                    
                    mass = self.systems[self.current_system]['parameters']['mass']
                    sys_id = self.create_impedance_control_system(mass, k, b, alpha)
                    print(f"Updated system: {sys_id}")
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
            
            elif choice == '3':
                if not self.current_system:
                    print("No current system.")
                    continue
                
                stability = self.analyze_stability()
                print(f"\nStability Analysis:")
                print(f"  Stable: {stability['is_stable']}")
                print(f"  Stability Margin: {stability['stability_margin']:.4f}")
                print(f"  Damping Ratios: {stability['damping_ratios']}")
                print(f"  Natural Frequencies: {stability['natural_frequencies']}")
            
            elif choice == '4':
                if not self.current_system:
                    print("No current system.")
                    continue
                
                self.plot_root_locus()
            
            elif choice == '5':
                if not self.current_system:
                    print("No current system.")
                    continue
                
                try:
                    target_damping = float(input("Enter target damping ratio (0.7 recommended): "))
                    target_overshoot = float(input("Enter target overshoot % (5.0 recommended): "))
                    target_settling = float(input("Enter target settling time (2.0 recommended): "))
                    
                    result = self.optimize_parameters(target_damping=target_damping,
                                                   target_overshoot=target_overshoot,
                                                   target_settling_time=target_settling)
                    
                    if result['success']:
                        print(f"\nOptimization successful!")
                        print(f"Optimal parameters:")
                        for param, value in result['optimized_parameters'].items():
                            print(f"  {param}: {value:.4f}")
                    else:
                        print(f"Optimization failed: {result['message']}")
                        
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
            
            elif choice == '6':
                if len(self.systems) < 2:
                    print("Need at least 2 systems to compare.")
                    continue
                
                print("Available systems:")
                for i, sys_id in enumerate(self.systems.keys()):
                    print(f"  {i}: {sys_id}")
                
                try:
                    indices = input("Enter system indices to compare (e.g., 0,1,2): ").split(',')
                    selected_systems = [list(self.systems.keys())[int(i.strip())] for i in indices]
                    self.compare_systems(selected_systems)
                except (ValueError, IndexError):
                    print("Invalid selection.")
            
            elif choice == '7':
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please enter 1-7.")


def main():
    """
    Main function to demonstrate the control system design tool.
    """
    print("Control System Parameter Design Tool")
    print("====================================")
    
    # Create designer instance
    designer = ControlSystemDesigner()
    
    # Create initial system based on your UR5 parameters
    print("\nCreating initial system based on your UR5 parameters...")
    initial_system = designer.create_impedance_control_system(
        mass=1.0,      # Approximate robot mass
        spring_k=5.0,  # Your current k value
        damping_b=100.0,  # Your current b value
        alpha=10.0     # Your current alpha value
    )
    
    print(f"Created system: {initial_system}")
    
    # Analyze initial system
    print("\nAnalyzing initial system...")
    stability = designer.analyze_stability()
    print(f"Initial system stability: {stability['is_stable']}")
    print(f"Stability margin: {stability['stability_margin']:.4f}")
    
    # Create some variations for comparison
    print("\nCreating system variations for comparison...")
    variations = [
        designer.create_impedance_control_system(1.0, 2.0, 50.0, 5.0),   # Lower gains
        designer.create_impedance_control_system(1.0, 10.0, 200.0, 20.0), # Higher gains
        designer.create_impedance_control_system(1.0, 5.0, 100.0, 10.0),  # Original
    ]
    
    # Compare systems
    print("\nComparing systems...")
    comparison_results = designer.compare_systems(variations)
    
    # Start interactive tuning
    print("\nStarting interactive tuning interface...")
    designer.interactive_tuning()


if __name__ == "__main__":
    main()
