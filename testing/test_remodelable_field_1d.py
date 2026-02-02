import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.potential_field_discrete import PotentialFieldDiscrete1DRemodelable

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = np.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = np.exp(z)
    return ez / (1.0 + ez)

class RemodelableSimulator1D:
    def __init__(self, z_bounds=(-2, 2)):
        self.dt = 1/50.0
        
        # Control parameters
        self.damping = 50.0
        self.stiffness = 2.5
        self.force_scale = 15.0

        # Tunneling intent integrator parameters
        self.intent_I = 0.0  # Integrated uphill effort state (accumulates over time)
        self.s0 = 5.0  # Force threshold - only count uphill force above this value
        self.rho = 7.05  # Decay rate - how quickly intent decays when not pushing uphill
        self.I0 = 5.0  # Intent threshold - minimum integrated effort to activate tunneling
        self.beta_I = 1.0  # Sigmoid sharpness - controls transition smoothness from 0 to 1
        self.eps_grad = 1e-6  # Small epsilon to prevent division by zero in gradient normalization
        self.min_grad_norm = 0.25 * self.force_scale  # Minimum gradient magnitude to consider for tunneling
        
        # World setup
        self.z_bounds = z_bounds
        self.pf = PotentialFieldDiscrete1DRemodelable(
            x_bounds=z_bounds, 
            resolution=200
        )
        self.position = 0.0
        self.applied_force = 0.0
        
        # Visualization
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.running = True

    def setup_plot(self):
        self.update_plot()

    def update_plot(self):
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Get potential and gradient values
        potential_values = self.pf.calculate_potential()
        force_values = -self.pf.get_gradient(self.pf.x)  # Force = -gradient
        
        # Plot 1: Potential field with obstacles and bores
        self.ax1.plot(self.pf.x, potential_values, 'b-', linewidth=2, label='Potential', zorder=1)
        
        # Draw obstacles
        for obs in self.pf.obstacles.values():
            obs_potential = self.pf.get_potential(obs['x'])
            self.ax1.plot(obs['x'], obs_potential, 'ro', markersize=10, zorder=10)
            self.ax1.axvline(x=obs['x'], color='r', linestyle='--', alpha=0.3, linewidth=1, zorder=2)
            # Draw obstacle bores
            if obs['x'] in self.pf.obstacle_bores:
                for bore in self.pf.obstacle_bores[obs['x']]:
                    # Draw arrow indicating bore direction
                    arrow_length = bore['width'] * 0.3
                    if bore['direction'] > 0:
                        # Bore on right side
                        self.ax1.annotate('', xy=(obs['x'] + arrow_length, obs_potential * 0.8),
                                         xytext=(obs['x'], obs_potential * 0.8),
                                         arrowprops=dict(arrowstyle='->', color='orange', lw=2, alpha=0.7))
                    else:
                        # Bore on left side
                        self.ax1.annotate('', xy=(obs['x'] - arrow_length, obs_potential * 0.8),
                                         xytext=(obs['x'], obs_potential * 0.8),
                                         arrowprops=dict(arrowstyle='->', color='orange', lw=2, alpha=0.7))
        
        # Draw robot position
        robot_potential = self.pf.get_potential(self.position)
        self.ax1.plot(self.position, robot_potential, 'bo', markersize=14, zorder=11, label='Robot')
        
        # Draw force vector (as vertical line/arrow)
        if abs(self.applied_force) > 0.1:
            force_vis_length = (self.applied_force / self.force_scale) * 0.5
            self.ax1.annotate('', xy=(self.position, robot_potential + force_vis_length),
                             xytext=(self.position, robot_potential),
                             arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.8))
        
        self.ax1.set_xlim(self.z_bounds)
        self.ax1.set_xlabel('Z Position', fontsize=12)
        self.ax1.set_ylabel('Potential', fontsize=12)
        self.ax1.set_title('Remodelable Potential Field (1D) - Left/Right Arrows to Apply Force, R to Reset Bores')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right')
        
        # Info text
        info_text = f"Force scale: {self.force_scale:.1f}\n"
        info_text += f"Position: {self.position:.3f}\n"
        info_text += f"I={self.intent_I:.2f}\n"
        info_text += f"γ={sigmoid(self.beta_I * (self.intent_I - self.I0)):.2f}\n"
        info_text += f"Obstacle bores: {sum(len(b) for b in self.pf.obstacle_bores.values())}"
        
        self.ax1.text(
            0.02, 0.98,
            info_text,
            transform=self.ax1.transAxes,
            va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=1)
        )
        
        # Plot 2: Force field
        self.ax2.plot(self.pf.x, force_values, 'r-', linewidth=2, label='Force (-dV/dz)', zorder=1)
        self.ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5, zorder=0)
        
        # Mark robot position on force plot
        robot_force = -self.pf.get_gradient(self.position)
        self.ax2.plot(self.position, robot_force, 'bo', markersize=10, zorder=10)
        
        # Mark obstacles on force plot
        for obs in self.pf.obstacles.values():
            obs_force = -self.pf.get_gradient(obs['x'])
            self.ax2.plot(obs['x'], obs_force, 'ro', markersize=8, zorder=10)
            self.ax2.axvline(x=obs['x'], color='r', linestyle='--', alpha=0.3, linewidth=1, zorder=2)
        
        self.ax2.set_xlim(self.z_bounds)
        self.ax2.set_xlabel('Z Position', fontsize=12)
        self.ax2.set_ylabel('Force (-dV/dz)', fontsize=12)
        self.ax2.set_title('Force Field')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper right')
        
        plt.tight_layout()

    def on_key_press(self, event):
        if event.key == 'left':
            self.applied_force = -self.force_scale
        elif event.key == 'right':
            self.applied_force = self.force_scale
        elif event.key == 'a':
            self.force_scale = max(0, self.force_scale + 5)
            self.min_grad_norm = 0.25 * self.force_scale  # Update min_grad_norm
            print(f"Force scale: {self.force_scale}")
        elif event.key == 'd':
            self.force_scale = max(0, self.force_scale - 5)
            self.min_grad_norm = 0.25 * self.force_scale  # Update min_grad_norm
            print(f"Force scale: {self.force_scale}")
        elif event.key == 'r':
            self.pf.clear_bores()
            self.intent_I = 0.0
            print("All bores cleared!")

    def on_key_release(self, event):
        if event.key in ['left', 'right']:
            self.applied_force = 0.0

    def on_close(self, event):
        self.running = False

    def update(self):
        if not self.running:
            return False
        
        # Get potential force at current position
        potential_force = -self.pf.get_gradient(self.position)
        
        # External force
        f_ext = self.applied_force
        gradV = -potential_force
        grad_norm = abs(gradV)
        
        # Tunneling logic (1D version)
        if grad_norm < self.min_grad_norm:
            gamma = 0.0
            self.intent_I = max(0.0, self.intent_I - self.dt * self.rho * self.intent_I)
        else:
            # In 1D, gradient direction is just sign
            ghat = 1.0 if gradV >= 0 else -1.0
            f_up = f_ext * ghat  # Uphill force component
            
            if f_up > 0.0:
                u = max(0.0, f_up - self.s0)
                self.intent_I = max(0.0, self.intent_I + self.dt * (u - self.rho * self.intent_I))
            else:
                self.intent_I = max(0.0, self.intent_I - self.dt * self.rho * self.intent_I)
            
            gamma = sigmoid(self.beta_I * (self.intent_I - self.I0))
            
            # Update bore based on tunneling intent
            if gamma > 0.01:
                self.pf.update_bore_from_force(
                    self.position, 
                    f_ext, 
                    gamma
                )
        
        # Update position (simple dynamics)
        pdot = (potential_force + self.stiffness * self.applied_force) / self.damping
        self.position = self.position + pdot * self.dt
        
        # Clamp to bounds
        self.position = np.clip(self.position, self.z_bounds[0], self.z_bounds[1])
        
        self.update_plot()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        return True

    def run(self):
        plt.ion()
        plt.show()
        while self.running:
            if not self.update():
                break
            plt.pause(self.dt)

if __name__ == '__main__':
    sim = RemodelableSimulator1D(z_bounds=(-2, 2))
    
    # Add obstacles
    sim.pf.add_obstacle(0.5, 50, 0.2)
    sim.pf.add_obstacle(-0.5, 100, 0.2)
    sim.pf.add_obstacle(0.0, 25, 0.15)
    
    print("Controls:")
    print("  Left arrow: Apply force left (negative direction)")
    print("  Right arrow: Apply force right (positive direction)")
    print("  A: Increase force scale by 5")
    print("  D: Decrease force scale by 5")
    print("  R: Reset all bores")
    print("\nTry pushing into an obstacle with arrow keys to create a permanent bore!")
    
    sim.setup_plot()
    sim.run()
