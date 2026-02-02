import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.potential_field_class import PotentialField, PotentialField1D

ENABLE_XY = True
ENABLE_Z = True

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

def clamp_scalar(v: float, max_val: float) -> float:
    return max(-max_val, min(max_val, v))

class CombinedSimulator:
    def __init__(self, xy_bounds=((-2, 2), (-2, 2)), z_bounds=(-2, 2)):
        self.dt = 1/50.0
        
        if ENABLE_XY:
            self.xy_damping = 50.0
            self.xy_stiffness = 2.5
            self.xy_force_scale = 10.0
            self.xy_intent_I = 0.0
            self.xy_s0 = 5.0
            self.xy_rho = 5.0
            self.xy_I0 = 3.0
            self.xy_beta_I = 1.0
            self.xy_alpha_o = 0.05
            self.xy_vmax_o = 0.20
            self.xy_eps_grad = 1e-6
            self.xy_use_angle_gate = True
            self.xy_cos0 = 0.5
            self.xy_min_grad_norm = 0.02
        
        if ENABLE_Z:
            self.z_damping = 25.0
            self.z_stiffness = 3.0
            self.z_force_scale = 10.0
            self.z_intent_I = 0.0
            self.z_s0 = 5.0
            self.z_rho = 5.0
            self.z_I0 = 1.5
            self.z_beta_I = 1.0
            self.z_alpha_o = 0.075
            self.z_vmax_o = 0.20
            self.z_eps_grad = 1e-6
            self.z_min_grad_norm = 0.02

        if ENABLE_XY:
            self.xy_world_bounds = xy_bounds
            self.pf_xy = PotentialField(x_bounds=xy_bounds[0], y_bounds=xy_bounds[1], resolution=200)
            self.xy_origin = np.array([0.0, 0.0])
            self.xy_position = np.array([0.0, 0.0])
            self.xy_applied_force = np.array([0.0, 0.0])

        if ENABLE_Z:
            self.z_world_bounds = z_bounds
            self.pf_z = PotentialField1D(x_bounds=z_bounds, resolution=1000)
            self.z_origin = 0.0
            self.z_position = 0.0
            self.z_applied_force = 0.0

        self.fig = plt.figure(figsize=(16, 8))
        
        if ENABLE_XY and ENABLE_Z:
            self.ax_xy = self.fig.add_subplot(121)
            self.ax_z = self.fig.add_subplot(122)
        elif ENABLE_XY:
            self.ax_xy = self.fig.add_subplot(111)
            self.ax_z = None
        elif ENABLE_Z:
            self.ax_z = self.fig.add_subplot(111)
            self.ax_xy = None

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.running = True

    def setup_plot(self):
        self.update_plot()

    def update_plot(self):
        if ENABLE_XY and self.ax_xy:
            self.ax_xy.clear()
            x_bounds, y_bounds = self.xy_world_bounds
            x = np.linspace(x_bounds[0], x_bounds[1], 200)
            y = np.linspace(y_bounds[0], y_bounds[1], 200)
            X, Y = np.meshgrid(x, y)

            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    pos_rel = np.array([X[i,j], Y[i,j]]) - self.xy_origin
                    Z[i,j] = self.pf_xy.potential_func(pos_rel[0], pos_rel[1])

            Z_min = np.min(Z)
            Z = Z - Z_min

            self.ax_xy.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
            self.ax_xy.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)

            for obs in self.pf_xy.obstacles.values():
                obs_world = np.array([obs['x'], obs['y']]) + self.xy_origin
                circle = plt.Circle((obs_world[0], obs_world[1]), obs['width'], color='red', alpha=0.5)
                self.ax_xy.add_patch(circle)

            self.ax_xy.plot(self.xy_position[0], self.xy_position[1], 'ro', markersize=12, zorder=10)
            self.ax_xy.plot(self.xy_origin[0], self.xy_origin[1], 'go', markersize=6, zorder=9)

            self.ax_xy.text(
                0.02, 0.98,
                f"I={self.xy_intent_I:.2f}",
                transform=self.ax_xy.transAxes,
                va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

            self.ax_xy.set_xlim(x_bounds)
            self.ax_xy.set_ylim(y_bounds)
            self.ax_xy.set_aspect('equal')
            self.ax_xy.set_xlabel('X')
            self.ax_xy.set_ylabel('Y')
            self.ax_xy.set_title('XY Potential Field - Arrow Keys')
            self.ax_xy.grid(True, alpha=0.3)

        if ENABLE_Z and self.ax_z:
            self.ax_z.clear()
            z_bounds = self.z_world_bounds
            z = np.linspace(z_bounds[0], z_bounds[1], 1000)

            V = np.zeros_like(z)
            for i in range(len(z)):
                pos_rel = z[i] - self.z_origin
                V[i] = self.pf_z.potential_func(pos_rel)

            V_min = np.min(V)
            V = V - V_min

            self.ax_z.plot(z, V, 'b-', linewidth=2, label='Potential')

            for obs in self.pf_z.obstacles.values():
                obs_world = obs['x'] + self.z_origin
                obs_potential_raw = self.pf_z.potential_func(obs['x'])
                obs_potential = obs_potential_raw - V_min
                self.ax_z.plot(obs_world, obs_potential, 'ro', markersize=10)
                self.ax_z.axvline(x=obs_world, color='r', linestyle='--', alpha=0.3, linewidth=1)

            pos_potential_raw = self.pf_z.potential_func(self.z_position - self.z_origin)
            pos_potential = pos_potential_raw - V_min
            self.ax_z.plot(self.z_position, pos_potential, 'ro', markersize=12, zorder=10)
            
            origin_potential_raw = self.pf_z.potential_func(0.0)
            origin_potential = origin_potential_raw - V_min
            self.ax_z.plot(self.z_origin, origin_potential, 'go', markersize=6, zorder=9)

            self.ax_z.text(
                0.02, 0.98,
                f"I={self.z_intent_I:.2f}",
                transform=self.ax_z.transAxes,
                va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

            self.ax_z.set_xlim(z_bounds)
            self.ax_z.set_xlabel('Z')
            self.ax_z.set_ylabel('Potential')
            self.ax_z.set_title('Z Potential Field - W/E Keys')
            self.ax_z.grid(True, alpha=0.3)

        plt.tight_layout()

    def on_key_press(self, event):
        if ENABLE_XY:
            if event.key == 'left':
                self.xy_applied_force[0] = -self.xy_force_scale
            elif event.key == 'right':
                self.xy_applied_force[0] = self.xy_force_scale
            elif event.key == 'up':
                self.xy_applied_force[1] = self.xy_force_scale
            elif event.key == 'down':
                self.xy_applied_force[1] = -self.xy_force_scale
        
        if ENABLE_Z:
            if event.key == 'w':
                self.z_applied_force = self.z_force_scale
            elif event.key == 'e':
                self.z_applied_force = -self.z_force_scale

    def on_key_release(self, event):
        if ENABLE_XY:
            if event.key in ['left', 'right']:
                self.xy_applied_force[0] = 0.0
            elif event.key in ['up', 'down']:
                self.xy_applied_force[1] = 0.0
        
        if ENABLE_Z:
            if event.key in ['w', 'e']:
                self.z_applied_force = 0.0

    def on_close(self, event):
        self.running = False

    def update(self):
        if not self.running:
            return False

        if ENABLE_XY:
            pos_rel = self.xy_position - self.xy_origin
            potential_force = -self.pf_xy.get_gradient(pos_rel[0], pos_rel[1])

            f_ext = self.xy_applied_force.copy()
            gradV = -potential_force
            grad_norm = float(np.linalg.norm(gradV))

            if grad_norm < self.xy_min_grad_norm:
                gamma = 0.0
                self.xy_intent_I = max(0.0, self.xy_intent_I - self.dt * self.xy_rho * self.xy_intent_I)
                o_dot = np.zeros(2)
            else:
                ghat = gradV / (grad_norm + self.xy_eps_grad)
                f_up = float(np.dot(f_ext, ghat))

                if self.xy_use_angle_gate:
                    f_norm = float(np.linalg.norm(f_ext))
                    if f_norm < 1e-12:
                        angle_ok = False
                    else:
                        cosang = f_up / (f_norm + 1e-12)
                        angle_ok = (cosang >= self.xy_cos0)
                else:
                    angle_ok = True

                if (f_up > 0.0) and angle_ok:
                    u = max(0.0, f_up - self.xy_s0)
                    self.xy_intent_I = max(0.0, self.xy_intent_I + self.dt * (u - self.xy_rho * self.xy_intent_I))
                else:
                    self.xy_intent_I = max(0.0, self.xy_intent_I - self.dt * self.xy_rho * self.xy_intent_I)

                gamma = sigmoid(self.xy_beta_I * (self.xy_intent_I - self.xy_I0))
                o_dot = self.xy_alpha_o * gamma * f_up * ghat

            o_dot = clamp_norm(o_dot, self.xy_vmax_o)
            self.xy_origin = self.xy_origin + o_dot * self.dt

            x_bounds, y_bounds = self.xy_world_bounds
            self.xy_origin[0] = np.clip(self.xy_origin[0], x_bounds[0], x_bounds[1])
            self.xy_origin[1] = np.clip(self.xy_origin[1], y_bounds[0], y_bounds[1])

            pdot = (potential_force + self.xy_stiffness * self.xy_applied_force) / self.xy_damping
            self.xy_position = self.xy_position + pdot * self.dt

            self.xy_position[0] = np.clip(self.xy_position[0], x_bounds[0], x_bounds[1])
            self.xy_position[1] = np.clip(self.xy_position[1], y_bounds[0], y_bounds[1])

        if ENABLE_Z:
            pos_rel = self.z_position - self.z_origin
            potential_force = -float(self.pf_z.get_gradient(pos_rel))

            f_ext = self.z_applied_force
            gradV = -potential_force
            grad_norm = abs(gradV)

            if grad_norm < self.z_min_grad_norm:
                gamma = 0.0
                self.z_intent_I = max(0.0, self.z_intent_I - self.dt * self.z_rho * self.z_intent_I)
                o_dot = 0.0
            else:
                ghat = 1.0 if gradV >= 0 else -1.0
                f_up = f_ext * ghat

                if f_up > 0.0:
                    u = max(0.0, f_up - self.z_s0)
                    self.z_intent_I = max(0.0, self.z_intent_I + self.dt * (u - self.z_rho * self.z_intent_I))
                else:
                    self.z_intent_I = max(0.0, self.z_intent_I - self.dt * self.z_rho * self.z_intent_I)

                gamma = sigmoid(self.z_beta_I * (self.z_intent_I - self.z_I0))
                o_dot = self.z_alpha_o * gamma * f_up * ghat

            o_dot = clamp_scalar(o_dot, self.z_vmax_o)
            self.z_origin = self.z_origin + o_dot * self.dt

            z_bounds = self.z_world_bounds
            self.z_origin = np.clip(self.z_origin, z_bounds[0], z_bounds[1])

            pdot = (potential_force + self.z_stiffness * self.z_applied_force) / self.z_damping
            self.z_position = self.z_position + pdot * self.dt

            self.z_position = np.clip(self.z_position, z_bounds[0], z_bounds[1])

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
    if not ENABLE_XY and not ENABLE_Z:
        print("Both simulators disabled!")
        exit()

    sim = CombinedSimulator(xy_bounds=((-2, 2), (-2, 2)), z_bounds=(-2, 2))
    
    if ENABLE_XY:
        sim.pf_xy.add_obstacle(0.5, 0.5, 10, 0.2)
        sim.pf_xy.add_obstacle(-0.5, -0.5, 10, 0.2)
    
    if ENABLE_Z:
        sim.pf_z.add_obstacle(0.5, 20, 0.2)
        sim.pf_z.add_obstacle(-0.5, 10, 0.2)
    
    sim.setup_plot()
    sim.run()
