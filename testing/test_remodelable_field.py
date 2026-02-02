import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.potential_field_discrete import PotentialFieldDiscreteRemodelable

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = np.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = np.exp(z)
    return ez / (1.0 + ez)

def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12 or n <= max_norm:
        return v
    return v * (max_norm / n)

class RemodelableSimulator:
    def __init__(self, xy_bounds=((-2, 2), (-2, 2))):
        self.dt = 1/50.0
        
        # Control parameters
        self.damping = 50.0
        self.stiffness = 2.5
        self.force_scale = 15.0

        # Tunneling intent integrator parameters
        self.intent_I = 0.0  # Integrated uphill effort state (accumulates over time)
        self.s0 = 5.0  # Force threshold - only count uphill force above this value
        self.rho = 7.05 # Decay rate - how quickly intent decays when not pushing uphill
        self.I0 = 5.0  # Intent threshold - minimum integrated effort to activate tunneling
        self.beta_I = 1.0  # Sigmoid sharpness - controls transition smoothness from 0 to 1
        self.eps_grad = 1e-6  # Small epsilon to prevent division by zero in gradient normalization
        self.use_angle_gate = True  # Require force to be aligned with gradient direction
        self.cos0 = 0.75  # Minimum cosine of angle between force and gradient (when angle gate enabled)
        self.min_grad_norm = 0.25*self.force_scale  # Minimum gradient magnitude to consider for tunneling
        
        
        # World setup
        self.xy_world_bounds = xy_bounds
        self.pf = PotentialFieldDiscreteRemodelable(
            x_bounds=xy_bounds[0], 
            y_bounds=xy_bounds[1], 
            resolution=200
        )
        self.position = np.array([0.0, 0.0])
        self.applied_force = np.array([0.0, 0.0])
        
        # Visualization
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.running = True

    def setup_plot(self):
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        x_bounds, y_bounds = self.xy_world_bounds
        
        # Calculate potential over grid
        Z = self.pf.calculate_potential()
        X, Y = self.pf.X, self.pf.Y
        
        # Normalize for visualization
        Z_min = np.min(Z)
        Z_vis = Z - Z_min
        
        # Contour plot
        self.ax.contourf(X, Y, Z_vis, levels=30, cmap='viridis', alpha=0.6)
        self.ax.contour(X, Y, Z_vis, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # Draw obstacles
        for obs in self.pf.obstacles.values():
            circle = plt.Circle((obs['x'], obs['y']), obs['width'], 
                              color='red', alpha=0.5, fill=False, linewidth=2)
            self.ax.add_patch(circle)
            self.ax.plot(obs['x'], obs['y'], 'ro', markersize=8, zorder=10)
        
        # Draw obstacle bore directions
        for obs_key, bores in self.pf.obstacle_bores.items():
            obs = self.pf.obstacles[obs_key]
            for bore in bores:
                dir_vec = np.array(bore['direction'])
                arrow_length = bore['width'] * 0.6
                self.ax.arrow(obs['x'], obs['y'], 
                             dir_vec[0] * arrow_length, dir_vec[1] * arrow_length,
                             head_width=0.15, head_length=0.1, 
                             fc='orange', ec='orange', alpha=0.7, linewidth=2,
                             zorder=9)
        
        # Draw field bore locations
        for bore in self.pf.field_bores:
            dir_vec = np.array(bore['direction'])
            arrow_length = bore['width'] * 0.6
            self.ax.plot(bore['x'], bore['y'], 'o', color='cyan', markersize=8, zorder=10)
            self.ax.arrow(bore['x'], bore['y'], 
                         dir_vec[0] * arrow_length, dir_vec[1] * arrow_length,
                         head_width=0.12, head_length=0.08, 
                         fc='cyan', ec='cyan', alpha=0.8, linewidth=2,
                         zorder=9)
        
        # Draw robot position
        self.ax.plot(self.position[0], self.position[1], 'bo', markersize=14, zorder=11)
        
        # Draw force vector
        if np.linalg.norm(self.applied_force) > 0.1:
            force_vis = self.applied_force / self.force_scale * 0.3
            self.ax.arrow(self.position[0], self.position[1],
                         force_vis[0], force_vis[1],
                         head_width=0.08, head_length=0.06,
                         fc='blue', ec='blue', alpha=0.8, linewidth=2,
                         zorder=12)
        
        # Info text
        info_text = f"Force scale: {self.force_scale:.1f}\n"
        info_text += f"I={self.intent_I:.2f}\n"
        info_text += f"γ={sigmoid(self.beta_I * (self.intent_I - self.I0)):.2f}\n"
        info_text += f"Obstacle bores: {sum(len(b) for b in self.pf.obstacle_bores.values())}\n"
        info_text += f"Field bores: {self.pf.get_field_bore_count()}"
        
        self.ax.text(
            0.02, 0.98,
            info_text,
            transform=self.ax.transAxes,
            va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=1)
        )
        
        self.ax.set_xlim(x_bounds)
        self.ax.set_ylim(y_bounds)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Remodelable Potential Field - Arrow Keys to Apply Force, R to Reset Bores')
        self.ax.grid(True, alpha=0.3)

    def on_key_press(self, event):
        if event.key == 'left':
            self.applied_force[0] = -self.force_scale
        elif event.key == 'right':
            self.applied_force[0] = self.force_scale
        elif event.key == 'up':
            self.applied_force[1] = self.force_scale
        elif event.key == 'down':
            self.applied_force[1] = -self.force_scale
        elif event.key == 'a':
            self.force_scale = max(0, self.force_scale + 5)
            print(f"Force scale: {self.force_scale}")
        elif event.key == 'd':
            self.force_scale = max(0, self.force_scale - 5)
            print(f"Force scale: {self.force_scale}")
        elif event.key == 'r':
            self.pf.clear_bores(clear_field_bores=True)
            print("All bores cleared!")

    def on_key_release(self, event):
        if event.key in ['left', 'right']:
            self.applied_force[0] = 0.0
        elif event.key in ['up', 'down']:
            self.applied_force[1] = 0.0

    def on_close(self, event):
        self.running = False

    def update(self):
        if not self.running:
            return False
        
        # Get potential force at current position
        potential_force = -self.pf.get_gradient(self.position[0], self.position[1])
        
        # External force
        f_ext = self.applied_force.copy()
        gradV = -potential_force
        grad_norm = float(np.linalg.norm(gradV))
        
        # Tunneling logic
        if grad_norm < self.min_grad_norm:
            gamma = 0.0
            self.intent_I = max(0.0, self.intent_I - self.dt * self.rho * self.intent_I)
        else:
            ghat = gradV / (grad_norm + self.eps_grad)
            f_up = float(np.dot(f_ext, ghat))
            
            if self.use_angle_gate:
                f_norm = float(np.linalg.norm(f_ext))
                if f_norm < 1e-12:
                    angle_ok = False
                else:
                    cosang = f_up / (f_norm + 1e-12)
                    angle_ok = (cosang >= self.cos0)
            else:
                angle_ok = True
            
            if (f_up > 0.0) and angle_ok:
                u = max(0.0, f_up - self.s0)
                self.intent_I = max(0.0, self.intent_I + self.dt * (u - self.rho * self.intent_I))
            else:
                self.intent_I = max(0.0, self.intent_I - self.dt * self.rho * self.intent_I)
            
            gamma = sigmoid(self.beta_I * (self.intent_I - self.I0))
            
            # Update bore based on tunneling intent
            if gamma > 0.01:
                self.pf.update_bore_from_force(
                    self.position[0], 
                    self.position[1], 
                    f_ext, 
                    gamma
                )
        
        # Update position (simple dynamics)
        pdot = (potential_force + self.stiffness * self.applied_force) / self.damping
        self.position = self.position + pdot * self.dt
        
        # Clamp to bounds
        x_bounds, y_bounds = self.xy_world_bounds
        self.position[0] = np.clip(self.position[0], x_bounds[0], x_bounds[1])
        self.position[1] = np.clip(self.position[1], y_bounds[0], y_bounds[1])
        
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
    sim = RemodelableSimulator(xy_bounds=((-2, 2), (-2, 2)))
    
    # Add obstacles
    sim.pf.add_obstacle(0.5, 0.5, 50, 0.2)
    sim.pf.add_obstacle(-0.5, -0.5, 100, 0.2)
    sim.pf.add_obstacle(0.0, 1.0, 25, 0.15)
    
    print("Controls:")
    print("  Arrow keys: Apply force in that direction")
    print("  A: Increase force scale by 5")
    print("  D: Decrease force scale by 5")
    print("  R: Reset all bores")
    print("\nTry pushing into an obstacle with arrow keys to create a permanent bore!")
    
    sim.setup_plot()
    sim.run()
