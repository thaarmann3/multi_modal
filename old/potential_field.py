import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

class PotentialField3D:
    def __init__(self, x_bounds, y_bounds, z_bounds, resolution, alpha, potential_func, beta, grad_func,
                Q: dict, goal_pos: np.ndarray, d_star=None):
        """
        Create a 3D grid for efficient potential field lookup.
        
        Parameters:
        -----------
        x_bounds : tuple (min, max)
            X-axis bounds for the field
        y_bounds : tuple (min, max) 
            Y-axis bounds for the field
        z_bounds : tuple (min, max)
            Z-axis bounds for the field
        resolution : int or tuple (x_res, y_res, z_res)
            Grid resolution
        alpha : float
            Attractive potential strength parameter
        potential_func : callable function
            Function that computes potential at a point: func(point, goal_pos, Q, d_star)
        beta : float
            Repulsive potential strength parameter
        grad_func : callable function
            Function that computes gradient at a point: func(point, goal_pos, Q, d_star)
        Q : dict
            Dictionary containing field parameters and obstacle information
        goal_pos : np.ndarray
            Goal position [x, y, z]
        d_star : float, optional
            Influence distance for obstacles (default: None)
        """
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.goal_pos = goal_pos
        self.d_star = d_star
        
        # Handle resolution
        if isinstance(resolution, int):
            self.x_res = self.y_res = self.z_res = resolution
        else:
            self.x_res, self.y_res, self.z_res = resolution
        
        # Create coordinate arrays
        self.x_coords = np.linspace(x_bounds[0], x_bounds[1], self.x_res)
        self.y_coords = np.linspace(y_bounds[0], y_bounds[1], self.y_res)
        self.z_coords = np.linspace(z_bounds[0], z_bounds[1], self.z_res)
        
        # Create 3D meshgrid
        self.X, self.Y, self.Z = np.meshgrid(
            self.x_coords, self.y_coords, self.z_coords, 
            indexing='ij'
        )
        
        # Pre-compute potential and gradient
        self._compute_field(potential_func, grad_func)
        
        # Create interpolators for smooth lookup
        self._create_interpolators()
    
    def _compute_field(self, potential_func, grad_func):
        """Pre-compute potential and gradient at all grid points."""
        print(f"Computing field for {self.x_res}×{self.y_res}×{self.z_res} grid...")
        
        # Initialize arrays
        self.potential = np.zeros_like(self.X)
        self.gradient = np.zeros((*self.X.shape, 3))  # 3D gradient vector
        
        # Fill arrays
        for i in range(self.x_res):
            for j in range(self.y_res):
                for k in range(self.z_res):
                    point = np.array([self.X[i,j,k], self.Y[i,j,k], self.Z[i,j,k]])
                    self.potential[i,j,k] = potential_func(point, self.goal_pos, self.Q, self.d_star)
                    self.gradient[i,j,k] = grad_func(point, self.goal_pos, self.Q, self.d_star)
    
    def _create_interpolators(self):
        """Create interpolators for smooth lookup between grid points."""
        self.potential_interp = RegularGridInterpolator(
            (self.x_coords, self.y_coords, self.z_coords),
            self.potential,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        self.gradient_interp = RegularGridInterpolator(
            (self.x_coords, self.y_coords, self.z_coords),
            self.gradient,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
    
    def get_potential(self, point):
        """Get potential at a point (with interpolation)."""
        return self.potential_interp(point)
    
    def get_gradient(self, point):
        """Get gradient at a point (with interpolation)."""
        return self.gradient_interp(point)
    
    def get_both(self, point):
        """Get both potential and gradient at a point."""
        return self.get_potential(point), self.get_gradient(point)
    
    def get_exact_grid_point(self, i, j, k):
        """Get exact values at grid indices (fastest lookup)."""
        return {
            'point': [self.X[i,j,k], self.Y[i,j,k], self.Z[i,j,k]],
            'potential': self.potential[i,j,k],
            'gradient': self.gradient[i,j,k]
        }
    
    def find_nearest_grid_index(self, point):
        """Find nearest grid point indices for a given point."""
        x_idx = np.argmin(np.abs(self.x_coords - point[0]))
        y_idx = np.argmin(np.abs(self.y_coords - point[1]))
        z_idx = np.argmin(np.abs(self.z_coords - point[2]))
        return x_idx, y_idx, z_idx
    
    def get_nearest_grid_point(self, point):
        """Get exact values at nearest grid point."""
        i, j, k = self.find_nearest_grid_index(point)
        return self.get_exact_grid_point(i, j, k)
    
    def get_grid_info(self):
        """Get information about the grid."""
        return {
            'shape': self.potential.shape,
            'total_points': self.potential.size,
            'x_range': (self.x_coords[0], self.x_coords[-1]),
            'y_range': (self.y_coords[0], self.y_coords[-1]),
            'z_range': (self.z_coords[0], self.z_coords[-1]),
            'memory_usage_mb': (self.potential.nbytes + self.gradient.nbytes) / 1024**2
        }
    
    def visualize(self, color_by='gradient_magnitude', subsample=1, figsize=(12, 10)):
        """
        Visualize the 3D potential field with color-coded points.
        
        Parameters:
        -----------
        color_by : str
            What to color by: 'gradient_magnitude', 'potential', 'gradient_x', 'gradient_y', 'gradient_z'
        subsample : int
            Subsample factor for visualization (1 = all points, 2 = every other point, etc.)
        figsize : tuple
            Figure size for the plot
        """
        # Subsample the data for visualization
        X_sub = self.X[::subsample, ::subsample, ::subsample]
        Y_sub = self.Y[::subsample, ::subsample, ::subsample]
        Z_sub = self.Z[::subsample, ::subsample, ::subsample]
        potential_sub = self.potential[::subsample, ::subsample, ::subsample]
        gradient_sub = self.gradient[::subsample, ::subsample, ::subsample]
        
        # Determine coloring values
        if color_by == 'gradient_magnitude':
            color_values = np.sqrt(gradient_sub[:,:,:,0]**2 + gradient_sub[:,:,:,1]**2 + gradient_sub[:,:,:,2]**2)
            color_label = 'Gradient Magnitude'
        elif color_by == 'potential':
            color_values = potential_sub
            color_label = 'Potential Value'
        elif color_by == 'gradient_x':
            color_values = gradient_sub[:,:,:,0]
            color_label = 'Gradient X Component'
        elif color_by == 'gradient_y':
            color_values = gradient_sub[:,:,:,1]
            color_label = 'Gradient Y Component'
        elif color_by == 'gradient_z':
            color_values = gradient_sub[:,:,:,2]
            color_label = 'Gradient Z Component'
        else:
            raise ValueError(f"Unknown color_by option: {color_by}")
        
        # Create the plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Flatten arrays for scatter plot
        X_flat = X_sub.flatten()
        Y_flat = Y_sub.flatten()
        Z_flat = Z_sub.flatten()
        color_flat = color_values.flatten()
        
        # Create 3D scatter plot
        scatter = ax.scatter(X_flat, Y_flat, Z_flat, 
                           c=color_flat, 
                           cmap='viridis', 
                           alpha=0.6, 
                           s=20)
        
        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'3D Potential Field Visualization\nColor = {color_label}')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label(color_label)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def visualize_surface_slice(self, slice_axis='z', slice_value=0.0, color_by='gradient_magnitude', figsize=(10, 8)):
        """
        Visualize a 2D slice of the 3D potential field.
        
        Parameters:
        -----------
        slice_axis : str
            Which axis to slice along: 'x', 'y', or 'z'
        slice_value : float
            Value along the slice axis
        color_by : str
            What to color by: 'gradient_magnitude', 'potential', 'gradient_x', 'gradient_y', 'gradient_z'
        figsize : tuple
            Figure size for the plot
        """
        # Find the closest slice index
        if slice_axis == 'x':
            slice_idx = np.argmin(np.abs(self.x_coords - slice_value))
            X_slice = self.Y[slice_idx, :, :]
            Y_slice = self.Z[slice_idx, :, :]
            potential_slice = self.potential[slice_idx, :, :]
            gradient_slice = self.gradient[slice_idx, :, :, :]
        elif slice_axis == 'y':
            slice_idx = np.argmin(np.abs(self.y_coords - slice_value))
            X_slice = self.X[:, slice_idx, :]
            Y_slice = self.Z[:, slice_idx, :]
            potential_slice = self.potential[:, slice_idx, :]
            gradient_slice = self.gradient[:, slice_idx, :, :]
        elif slice_axis == 'z':
            slice_idx = np.argmin(np.abs(self.z_coords - slice_value))
            X_slice = self.X[:, :, slice_idx]
            Y_slice = self.Y[:, :, slice_idx]
            potential_slice = self.potential[:, :, slice_idx]
            gradient_slice = self.gradient[:, :, slice_idx, :]
        else:
            raise ValueError(f"Unknown slice_axis: {slice_axis}")
        
        # Determine coloring values
        if color_by == 'gradient_magnitude':
            color_values = np.sqrt(gradient_slice[:,:,0]**2 + gradient_slice[:,:,1]**2 + gradient_slice[:,:,2]**2)
            color_label = 'Gradient Magnitude'
        elif color_by == 'potential':
            color_values = potential_slice
            color_label = 'Potential Value'
        elif color_by == 'gradient_x':
            color_values = gradient_slice[:,:,0]
            color_label = 'Gradient X Component'
        elif color_by == 'gradient_y':
            color_values = gradient_slice[:,:,1]
            color_label = 'Gradient Y Component'
        elif color_by == 'gradient_z':
            color_values = gradient_slice[:,:,2]
            color_label = 'Gradient Z Component'
        else:
            raise ValueError(f"Unknown color_by option: {color_by}")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create contour plot
        contour = ax.contourf(X_slice, Y_slice, color_values, levels=20, cmap='viridis', alpha=0.8)
        ax.contour(X_slice, Y_slice, color_values, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # Set labels and title
        ax.set_xlabel(f'{slice_axis.upper()} = {slice_value:.3f}')
        ax.set_ylabel('Z Position' if slice_axis != 'z' else 'Y Position')
        ax.set_title(f'2D Slice of Potential Field\n{slice_axis.upper()}={slice_value:.3f}, Color = {color_label}')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(color_label)
        
        plt.tight_layout()
        plt.show()
        
        return fig