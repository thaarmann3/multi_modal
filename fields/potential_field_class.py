import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

class PotentialField:
    def __init__(self, x_bounds=(-10, 10), y_bounds=(-10, 10), resolution=100, scaling_factor=0.2):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.resolution = resolution
        self.obstacles = {}
        self.scaling_factor = scaling_factor
        
        x = np.linspace(x_bounds[0], x_bounds[1], resolution)
        y = np.linspace(y_bounds[0], y_bounds[1], resolution)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Symbolic variables
        self.x_sym, self.y_sym = sp.symbols('x y')
        
        # Initialize potential function
        self._update_potential_function()
        
    def _update_potential_function(self):
        # Calculate alpha
        if self.obstacles:
            avg_spike_height = np.mean([obs['height'] for obs in self.obstacles.values()])
            alpha = self.scaling_factor * avg_spike_height
        else:
            alpha = 1.0
        
        # Base paraboloid
        potential = alpha * (self.x_sym**2 + self.y_sym**2)
        
        # Add Gaussian spikes for each obstacle
        for obs in self.obstacles.values():
            spike = obs['height'] * sp.exp(-((self.x_sym - obs['x'])**2 + (self.y_sym - obs['y'])**2) / (2 * obs['width']**2))
            potential += spike
        
        # Lambdify the potential function
        self.potential_func = sp.lambdify([self.x_sym, self.y_sym], potential, 'numpy')
        
        # Calculate gradient symbolically
        grad_x = sp.diff(potential, self.x_sym)
        grad_y = sp.diff(potential, self.y_sym)
        
        # Lambdify single gradient function that returns [grad_x, grad_y]
        self.gradient_func = sp.lambdify([self.x_sym, self.y_sym], [grad_x, grad_y], 'numpy')
    
    def add_obstacle(self, x, y, height, width):
        key = (x, y)
        self.obstacles[key] = {'x': x, 'y': y, 'height': height, 'width': width}
        self._update_potential_function()
        
    def remove_obstacle(self, x, y):
        key = (x, y)
        if key in self.obstacles:
            del self.obstacles[key]
            self._update_potential_function()
            
    def clear_obstacles(self):
        self.obstacles = {}
        self._update_potential_function()
        
    def calculate_potential(self):
        return self.potential_func(self.X, self.Y)
        
    def get_gradient(self, x, y):
        return np.array(self.gradient_func(x, y))
        
    def visualize_3d(self, title="Potential Field"):
        Z = self.calculate_potential()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(self.X, self.Y, Z, cmap='plasma', alpha=0.9, linewidth=0, antialiased=True)
        ax.plot_wireframe(self.X, self.Y, Z, rstride=10, cstride=10, alpha=0.3, color='black', linewidth=0.5)
        
        ax.view_init(elev=60, azim=45)
        plt.colorbar(surf, shrink=0.5, aspect=5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Potential (Z)')
        ax.set_title(title)
        ax.set_box_aspect([1,1,1])
        
        plt.show()
        
    def visualize_contours(self, title="Potential Field Contours"):
        Z = self.calculate_potential()
        
        plt.figure(figsize=(10, 8))
        contour = plt.contour(self.X, self.Y, Z, levels=20)
        plt.clabel(contour, inline=True, fontsize=8)
        plt.colorbar()
        
        for obs in self.obstacles.values():
            plt.plot(obs['x'], obs['y'], 'ro', markersize=10)
            
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.axis('equal')
        plt.grid(True)
        plt.show()


class PotentialField1D:
    """
    1D Potential Field: V(x) where x is position and V is potential.
    Creates a 2D function where Y-axis = potential, X-axis = position.
    """
    def __init__(self, x_bounds=(-10, 10), resolution=1000, scaling_factor=0.2, alpha=1.0):
        """
        Initialize a 1D potential field.
        
        Args:
            x_bounds: Tuple (x_min, x_max) for position range
            resolution: Number of points for evaluation grid
            scaling_factor: Scaling factor for base parabolic potential
        """
        self.x_bounds = x_bounds
        self.resolution = resolution
        self.obstacles = {}  # Key: x position, Value: {'x': x, 'height': h, 'width': w}
        self.scaling_factor = scaling_factor
        self.alpha = alpha
        # Create position grid
        self.x_positions = np.linspace(x_bounds[0], x_bounds[1], resolution)
        
        # Symbolic variable
        self.x_sym = sp.symbols('x')
        
        # Initialize potential function
        self._update_potential_function()
        
    def _update_potential_function(self):
        """Update the symbolic potential function V(x)."""
        # Calculate alpha
        if self.obstacles:
            avg_spike_height = np.mean([obs['height'] for obs in self.obstacles.values()])
            alpha = self.scaling_factor * avg_spike_height
        else:
            alpha = self.alpha
        
        # Base parabola: V(x) = alpha * x^2
        potential = alpha * (self.x_sym**2)
        
        # Add Gaussian spikes for each obstacle
        for obs in self.obstacles.values():
            spike = obs['height'] * sp.exp(-((self.x_sym - obs['x'])**2) / (2 * obs['width']**2))
            potential += spike
        
        # Lambdify the potential function
        self.potential_func = sp.lambdify([self.x_sym], potential, 'numpy')
        
        # Calculate gradient symbolically (force = -dV/dx)
        grad_x = sp.diff(potential, self.x_sym)
        self.gradient_func = sp.lambdify([self.x_sym], grad_x, 'numpy')
    
    def add_obstacle(self, x, height, width):
        """
        Add an obstacle (Gaussian spike) at position x.
        
        Args:
            x: Position of the obstacle
            height: Height (magnitude) of the Gaussian spike
            width: Width (standard deviation) of the Gaussian spike
        """
        key = x
        self.obstacles[key] = {'x': x, 'height': height, 'width': width}
        self._update_potential_function()
        
    def remove_obstacle(self, x):
        """Remove obstacle at position x."""
        key = x
        if key in self.obstacles:
            del self.obstacles[key]
            self._update_potential_function()
            
    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles = {}
        self._update_potential_function()
        
    def get_potential(self, x):
        """
        Get potential value at position x.
        
        Args:
            x: Position (can be scalar or array)
        
        Returns:
            Potential value(s) at position(s) x
        """
        return self.potential_func(x)
    
    def get_gradient(self, x):
        """
        Get gradient (force) at position x. Force = -dV/dx.
        
        Args:
            x: Position (can be scalar or array)
        
        Returns:
            Gradient value(s) at position(s) x (negative gradient = force)
        """
        return np.array(self.gradient_func(x))
    
    def calculate_potential(self):
        """
        Calculate potential over the entire position grid.
        
        Returns:
            Array of potential values
        """
        return self.potential_func(self.x_positions)
        
    def visualize(self, title="1D Potential Field: Potential vs Position", show_obstacles=True):
        """
        Visualize the potential field as a 2D plot: potential (Y-axis) vs position (X-axis).
        
        Args:
            title: Plot title
            show_obstacles: If True, mark obstacle positions on the plot
        """
        potential_values = self.calculate_potential()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.x_positions, potential_values, 'b-', linewidth=2, label='Potential')
        
        # Mark obstacles
        if show_obstacles:
            for obs in self.obstacles.values():
                obs_potential = self.get_potential(obs['x'])
                plt.plot(obs['x'], obs_potential, 'ro', markersize=10, 
                        label='Obstacle' if obs == list(self.obstacles.values())[0] else '')
                plt.axvline(x=obs['x'], color='r', linestyle='--', alpha=0.3, linewidth=1)
        
        plt.xlabel('Position (X)', fontsize=12)
        plt.ylabel('Potential', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def visualize_with_force(self, title="1D Potential Field with Force", show_obstacles=True):
        """
        Visualize both potential and force (negative gradient) on the same plot.
        
        Args:
            title: Plot title
            show_obstacles: If True, mark obstacle positions on the plot
        """
        potential_values = self.calculate_potential()
        force_values = self.get_gradient(self.x_positions)
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot potential on left y-axis
        color1 = 'b'
        ax1.set_xlabel('Position (X)', fontsize=12)
        ax1.set_ylabel('Potential', color=color1, fontsize=12)
        line1 = ax1.plot(self.x_positions, potential_values, color=color1, linewidth=2, label='Potential')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Plot force on right y-axis
        ax2 = ax1.twinx()
        color2 = 'r'
        ax2.set_ylabel('Force (-dV/dx)', color=color2, fontsize=12)
        line2 = ax2.plot(self.x_positions, force_values, color=color2, linewidth=2, linestyle='--', label='Force')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        # Mark obstacles
        if show_obstacles:
            for obs in self.obstacles.values():
                obs_potential = self.get_potential(obs['x'])
                ax1.plot(obs['x'], obs_potential, 'ko', markersize=10, 
                        label='Obstacle' if obs == list(self.obstacles.values())[0] else '')
                ax1.axvline(x=obs['x'], color='k', linestyle='--', alpha=0.3, linewidth=1)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.show()


class PotentialFieldFlat:
    """
    2D Potential Field with flat base (plane with constant gradient 0).
    Spikes merge continuously into the flat plane.
    """
    def __init__(self, x_bounds=(-10, 10), y_bounds=(-10, 10), resolution=100, scaling_factor=0.2):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.resolution = resolution
        self.obstacles = {}
        self.scaling_factor = scaling_factor
        
        x = np.linspace(x_bounds[0], x_bounds[1], resolution)
        y = np.linspace(y_bounds[0], y_bounds[1], resolution)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Symbolic variables
        self.x_sym, self.y_sym = sp.symbols('x y')
        
        # Initialize potential function
        self._update_potential_function()
        
    def _update_potential_function(self):
        # Base plane with constant gradient 0 (flat surface)
        potential = 0
        
        # Add Gaussian spikes for each obstacle
        for obs in self.obstacles.values():
            spike = obs['height'] * sp.exp(-((self.x_sym - obs['x'])**2 + (self.y_sym - obs['y'])**2) / (2 * obs['width']**2))
            potential += spike
        
        # Lambdify the potential function
        self.potential_func = sp.lambdify([self.x_sym, self.y_sym], potential, 'numpy')
        
        # Calculate gradient symbolically
        grad_x = sp.diff(potential, self.x_sym)
        grad_y = sp.diff(potential, self.y_sym)
        
        # Lambdify single gradient function that returns [grad_x, grad_y]
        self.gradient_func = sp.lambdify([self.x_sym, self.y_sym], [grad_x, grad_y], 'numpy')
    
    def add_obstacle(self, x, y, height, width):
        key = (x, y)
        self.obstacles[key] = {'x': x, 'y': y, 'height': height, 'width': width}
        self._update_potential_function()
        
    def remove_obstacle(self, x, y):
        key = (x, y)
        if key in self.obstacles:
            del self.obstacles[key]
            self._update_potential_function()
            
    def clear_obstacles(self):
        self.obstacles = {}
        self._update_potential_function()
        
    def calculate_potential(self):
        return self.potential_func(self.X, self.Y)
        
    def get_gradient(self, x, y):
        return np.array(self.gradient_func(x, y))
        
    def visualize_3d(self, title="Potential Field (Flat Base)"):
        Z = self.calculate_potential()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(self.X, self.Y, Z, cmap='plasma', alpha=0.9, linewidth=0, antialiased=True)
        ax.plot_wireframe(self.X, self.Y, Z, rstride=10, cstride=10, alpha=0.3, color='black', linewidth=0.5)
        
        ax.view_init(elev=60, azim=45)
        plt.colorbar(surf, shrink=0.5, aspect=5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Potential (Z)')
        ax.set_title(title)
        ax.set_box_aspect([1,1,1])
        
        plt.show()
        
    def visualize_contours(self, title="Potential Field Contours (Flat Base)"):
        Z = self.calculate_potential()
        
        plt.figure(figsize=(10, 8))
        contour = plt.contour(self.X, self.Y, Z, levels=20)
        plt.clabel(contour, inline=True, fontsize=8)
        plt.colorbar()
        
        for obs in self.obstacles.values():
            plt.plot(obs['x'], obs['y'], 'ro', markersize=10)
            
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.axis('equal')
        plt.grid(True)
        plt.show()


class PotentialField1DFlat:
    """
    1D Potential Field with flat base (line with constant gradient 0).
    Spikes merge continuously into the flat line.
    """
    def __init__(self, x_bounds=(-10, 10), resolution=1000, scaling_factor=0.2, alpha=1.0):
        """
        Initialize a 1D potential field with flat base.
        
        Args:
            x_bounds: Tuple (x_min, x_max) for position range
            resolution: Number of points for evaluation grid
            scaling_factor: Not used (kept for compatibility)
            alpha: Not used (kept for compatibility)
        """
        self.x_bounds = x_bounds
        self.resolution = resolution
        self.obstacles = {}  # Key: x position, Value: {'x': x, 'height': h, 'width': w}
        self.scaling_factor = scaling_factor
        self.alpha = alpha
        # Create position grid
        self.x_positions = np.linspace(x_bounds[0], x_bounds[1], resolution)
        
        # Symbolic variable
        self.x_sym = sp.symbols('x')
        
        # Initialize potential function
        self._update_potential_function()
        
    def _update_potential_function(self):
        """Update the symbolic potential function V(x)."""
        # Base line with constant gradient 0 (flat line)
        potential = 0
        
        # Add Gaussian spikes for each obstacle
        for obs in self.obstacles.values():
            spike = obs['height'] * sp.exp(-((self.x_sym - obs['x'])**2) / (2 * obs['width']**2))
            potential += spike
        
        # Lambdify the potential function
        self.potential_func = sp.lambdify([self.x_sym], potential, 'numpy')
        
        # Calculate gradient symbolically (force = -dV/dx)
        grad_x = sp.diff(potential, self.x_sym)
        self.gradient_func = sp.lambdify([self.x_sym], grad_x, 'numpy')
    
    def add_obstacle(self, x, height, width):
        """
        Add an obstacle (Gaussian spike) at position x.
        
        Args:
            x: Position of the obstacle
            height: Height (magnitude) of the Gaussian spike
            width: Width (standard deviation) of the Gaussian spike
        """
        key = x
        self.obstacles[key] = {'x': x, 'height': height, 'width': width}
        self._update_potential_function()
        
    def remove_obstacle(self, x):
        """Remove obstacle at position x."""
        key = x
        if key in self.obstacles:
            del self.obstacles[key]
            self._update_potential_function()
            
    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles = {}
        self._update_potential_function()
        
    def get_potential(self, x):
        """
        Get potential value at position x.
        
        Args:
            x: Position (can be scalar or array)
        
        Returns:
            Potential value(s) at position(s) x
        """
        return self.potential_func(x)
    
    def get_gradient(self, x):
        """
        Get gradient (force) at position x. Force = -dV/dx.
        
        Args:
            x: Position (can be scalar or array)
        
        Returns:
            Gradient value(s) at position(s) x (negative gradient = force)
        """
        return np.array(self.gradient_func(x))
    
    def calculate_potential(self):
        """
        Calculate potential over the entire position grid.
        
        Returns:
            Array of potential values
        """
        return self.potential_func(self.x_positions)
        
    def visualize(self, title="1D Potential Field: Potential vs Position (Flat Base)", show_obstacles=True):
        """
        Visualize the potential field as a 2D plot: potential (Y-axis) vs position (X-axis).
        
        Args:
            title: Plot title
            show_obstacles: If True, mark obstacle positions on the plot
        """
        potential_values = self.calculate_potential()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.x_positions, potential_values, 'b-', linewidth=2, label='Potential')
        
        # Mark obstacles
        if show_obstacles:
            for obs in self.obstacles.values():
                obs_potential = self.get_potential(obs['x'])
                plt.plot(obs['x'], obs_potential, 'ro', markersize=10, 
                        label='Obstacle' if obs == list(self.obstacles.values())[0] else '')
                plt.axvline(x=obs['x'], color='r', linestyle='--', alpha=0.3, linewidth=1)
        
        plt.xlabel('Position (X)', fontsize=12)
        plt.ylabel('Potential', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def visualize_with_force(self, title="1D Potential Field with Force (Flat Base)", show_obstacles=True):
        """
        Visualize both potential and force (negative gradient) on the same plot.
        
        Args:
            title: Plot title
            show_obstacles: If True, mark obstacle positions on the plot
        """
        potential_values = self.calculate_potential()
        force_values = self.get_gradient(self.x_positions)
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot potential on left y-axis
        color1 = 'b'
        ax1.set_xlabel('Position (X)', fontsize=12)
        ax1.set_ylabel('Potential', color=color1, fontsize=12)
        line1 = ax1.plot(self.x_positions, potential_values, color=color1, linewidth=2, label='Potential')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Plot force on right y-axis
        ax2 = ax1.twinx()
        color2 = 'r'
        ax2.set_ylabel('Force (-dV/dx)', color=color2, fontsize=12)
        line2 = ax2.plot(self.x_positions, force_values, color=color2, linewidth=2, linestyle='--', label='Force')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        # Mark obstacles
        if show_obstacles:
            for obs in self.obstacles.values():
                obs_potential = self.get_potential(obs['x'])
                ax1.plot(obs['x'], obs_potential, 'ko', markersize=10, 
                        label='Obstacle' if obs == list(self.obstacles.values())[0] else '')
                ax1.axvline(x=obs['x'], color='k', linestyle='--', alpha=0.3, linewidth=1)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.show()
