import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

class PotentialFieldDiscrete: 
    """
    Discretized implementation of the potential field.
    """
    def __init__(self, x_bounds=(-10, 10), y_bounds=(-10, 10), resolution=100, alpha=1.0):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.resolution = resolution
        self.obstacles = {}
        self.alpha = alpha
        self.x = np.linspace(self.x_bounds[0], self.x_bounds[1], resolution)
        self.y = np.linspace(self.y_bounds[0], self.y_bounds[1], resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Grid spacing for reference
        self.dx = (x_bounds[1] - x_bounds[0]) / (resolution - 1)
        self.dy = (y_bounds[1] - y_bounds[0]) / (resolution - 1)

        
        # Initialize potential array and interpolator
        self._update_potential()
    
    def _update_potential(self):
        """Update the potential array and bicubic interpolator.
        Called only when obstacles are added, removed, or cleared.
        This ensures the potential field and gradient field are updated accordingly.
        """
        # Always use alpha directly for the base paraboloid potential
        alpha = self.alpha
        
        # Recalculate potential array from scratch
        # Base paraboloid: V = alpha * (x^2 + y^2)
        self.potential = alpha * (self.X**2 + self.Y**2)
        
        # Add Gaussian spikes for each obstacle
        for obs in self.obstacles.values():
            spike = obs['height'] * np.exp(-((self.X - obs['x'])**2 + (self.Y - obs['y'])**2) / (2 * obs['width']**2))
            self.potential += spike
        
        # Recreate bicubic interpolator with updated potential
        # Note: RectBivariateSpline expects z[i, j] = value at (x[i], y[j])
        # But meshgrid creates potential[i, j] = value at (x[j], y[i])
        # So we need to transpose the potential array
        self.potential_interpolator = RectBivariateSpline(
            self.x, self.y, self.potential.T, kx=3, ky=3, s=0
        )
    
    def add_obstacle(self, x, y, height, width):
        """Add an obstacle (Gaussian spike) at position (x, y)."""
        key = (x, y)
        self.obstacles[key] = {'x': x, 'y': y, 'height': height, 'width': width}
        self._update_potential()  # Updates potential and gradient fields
    
    def remove_obstacle(self, x, y):
        """Remove obstacle at position (x, y)."""
        key = (x, y)
        if key in self.obstacles:
            del self.obstacles[key]
            self._update_potential()  # Updates potential and gradient fields
    
    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles = {}
        self._update_potential()  # Updates potential and gradient fields
    
    def set_alpha(self, alpha_value):
        """
        Set the alpha value for the base paraboloid potential and update the field.
        
        Args:
            alpha_value: The alpha value to use for the base potential V = alpha * (x^2 + y^2)
        """
        self.alpha = alpha_value
        if not hasattr(self, '_alpha_explicit'):
            self._alpha_explicit = False
        self._alpha_explicit = True  # Mark that alpha was explicitly set
        self._update_potential()  # Recalculate potential with new alpha
    
    def calculate_potential(self):
        """Calculate potential over the entire grid. Returns the stored potential array."""
        return self.potential.copy()
    
    def get_potential(self, x, y):
        """
        Get potential value at position (x, y) using bicubic interpolation.
        
        Args:
            x: X coordinate (scalar or array)
            y: Y coordinate (scalar or array)
        
        Returns:
            Potential value(s) at (x, y)
        """
        # Clamp coordinates to grid bounds
        x = np.clip(x, self.x_bounds[0], self.x_bounds[1])
        y = np.clip(y, self.y_bounds[0], self.y_bounds[1])
        
        # Use bicubic interpolator
        # RectBivariateSpline.__call__(x, y) evaluates at points (x, y)
        result = self.potential_interpolator(x, y, grid=False)
        
        # Handle scalar inputs
        if np.isscalar(x) and np.isscalar(y):
            return float(result)
        return result
    
    def get_gradient(self, x, y):
        """
        Get gradient at position (x, y) by computing derivatives of the bicubic interpolated potential.
        The gradient is computed analytically from the interpolated function.
        
        Args:
            x: X coordinate (scalar or array)
            y: Y coordinate (scalar or array)
        
        Returns:
            Array [grad_x, grad_y] at position (x, y)
        """
        # Clamp coordinates to grid bounds
        x = np.clip(x, self.x_bounds[0], self.x_bounds[1])
        y = np.clip(y, self.y_bounds[0], self.y_bounds[1])
        
        # Compute gradients analytically from the bicubic interpolator
        # __call__(x, y, dx, dy) computes d^(dx+dy) / (dx^dx dy^dy) at (x, y)
        # For gradient: grad_x = dV/dx (dx=1, dy=0), grad_y = dV/dy (dx=0, dy=1)
        grad_x = self.potential_interpolator(x, y, dx=1, dy=0, grid=False)
        grad_y = self.potential_interpolator(x, y, dx=0, dy=1, grid=False)
        
        # Handle scalar inputs
        if np.isscalar(x) and np.isscalar(y):
            return np.array([float(grad_x), float(grad_y)])
        
        return np.array([grad_x, grad_y])
    
    def get_gradient_grid(self):
        """
        Get gradient over the entire grid using bicubic interpolation.
        
        Returns:
            Tuple (grad_x_grid, grad_y_grid) where each is a 2D array matching the grid shape
        """
        try:
            # Use bicubic interpolator to compute gradients on the grid
            grad_x_grid = self.potential_interpolator(self.X, self.Y, dx=1, dy=0, grid=True)
            grad_y_grid = self.potential_interpolator(self.X, self.Y, dx=0, dy=1, grid=True)
            
            # Ensure gradients are valid (no NaN or Inf)
            grad_x_grid = np.nan_to_num(grad_x_grid, nan=0.0, posinf=0.0, neginf=0.0)
            grad_y_grid = np.nan_to_num(grad_y_grid, nan=0.0, posinf=0.0, neginf=0.0)
            
            return grad_x_grid, grad_y_grid
        except (ValueError, AttributeError) as e:
            # If interpolation fails, compute gradients numerically as fallback
            # This can happen if the interpolator is invalid or data is too flat
            # Compute gradients numerically using finite differences
            # Note: np.gradient expects (y, x) ordering for 2D arrays
            grad_y_grid, grad_x_grid = np.gradient(self.potential)
            # Scale by grid spacing
            if len(self.x) > 1:
                grad_x_grid /= (self.x[1] - self.x[0])
            if len(self.y) > 1:
                grad_y_grid /= (self.y[1] - self.y[0])
            return grad_x_grid, grad_y_grid
    
    def visualize_3d(self, title="Potential Field (Discrete)"):
        """Visualize the potential field as a 3D surface."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(self.X, self.Y, self.potential, cmap='plasma', alpha=0.9, linewidth=0, antialiased=True)
        ax.plot_wireframe(self.X, self.Y, self.potential, rstride=10, cstride=10, alpha=0.3, color='black', linewidth=0.5)
        
        ax.view_init(elev=60, azim=45)
        plt.colorbar(surf, shrink=0.5, aspect=5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Potential (Z)')
        ax.set_title(title)
        ax.set_box_aspect([1,1,1])
        
        plt.show()
    
    def visualize_contours(self, title="Potential Field Contours (Discrete)"):
        """Visualize the potential field as contour lines."""
        plt.figure(figsize=(10, 8))
        contour = plt.contour(self.X, self.Y, self.potential, levels=20)
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


class PotentialFieldDiscreteRemodelable(PotentialFieldDiscrete):
    """
    Discretized potential field with permanent directional remodeling.
    When force is applied uphill into an obstacle, creates a permanent "bore"
    that expands the basin on that side only.
    """
    
    def __init__(self, *args, 
                 # Bore size and shape parameters
                 bore_width_default=0.5,  # Default width for field bores (spatial extent)
                 obstacle_bore_width_multiplier=2.0,  # Obstacle bore width = obstacle_width * this multiplier
                 
                 # Bore directional masking parameters
                 side_mask_sharpness=0.3,  # Controls sharpness of directional transition (lower = sharper)
                 distance_falloff_multiplier=1.5,  # Distance falloff = bore_width * this multiplier (larger = wider effect)
                 
                 # Bore strength and reduction parameters
                 obstacle_bore_strength_multiplier=0.5,  # Obstacle bore reduction = strength * height * this multiplier
                 field_bore_strength_multiplier=0.3,  # Field bore reduction = strength * potential * this multiplier
                 field_bore_reduction_curve=1.0,  # Curve parameter: 1.0=linear, >1.0=more sensitive (faster to 0), <1.0=less sensitive (slower to 0)
                 max_reduction_fraction=0.4,  # Maximum fraction of potential that can be reduced (0-1) - legacy, not used for field bores
                 
                 # Obstacle protection parameters
                 peak_protection_radius=0.3,  # Peak protection mask radius = obstacle_width * this multiplier
                 peak_max_reduction_fraction=0.6,  # Maximum reduction at obstacle peak (0-1, lower = more protected)
                 min_obstacle_height_fraction=0.35,  # Minimum fraction of obstacle height that must be preserved (0-1)
                 
                 # Bore merging and accumulation parameters
                 field_bore_merge_distance_multiplier=0.5,  # Merge field bores if within bore_width * this multiplier
                 field_bore_merge_direction_threshold=0.90,  # Merge field bores if direction similarity > this (0-1)
                 obstacle_bore_merge_direction_threshold=0.85,  # Merge obstacle bores if direction similarity > this (0-1)
                 bore_strength_accumulation_rate=0.05,  # How much strength accumulates per update (0-1)
                 
                 # Bore creation detection parameters
                 force_alignment_threshold=0.75,  # Minimum force-gradient alignment to create bore (0-1)
                 obstacle_alignment_threshold=0.4,  # Minimum force-obstacle alignment to create obstacle bore (0-1)
                 obstacle_min_distance_multiplier=0.5,  # Skip obstacle if robot within obstacle_width * this multiplier
                 obstacle_max_distance_multiplier=6.0,  # Create obstacle bore if within obstacle_width * this multiplier
                 
                 **kwargs):
        # Initialize field bores before calling super() to avoid AttributeError
        # Store field bores at any location (not tied to obstacles)
        # Structure: [(bore1, bore2, ...)]
        # Each bore: {'x': x, 'y': y, 'direction': [dx, dy], 'strength': float, 'width': float}
        self.field_bores = []
        
        # Store bore behavior parameters
        self.bore_width_default = bore_width_default
        self.obstacle_bore_width_multiplier = obstacle_bore_width_multiplier
        self.side_mask_sharpness = side_mask_sharpness
        self.distance_falloff_multiplier = distance_falloff_multiplier
        self.obstacle_bore_strength_multiplier = obstacle_bore_strength_multiplier
        self.field_bore_strength_multiplier = field_bore_strength_multiplier
        self.field_bore_reduction_curve = field_bore_reduction_curve
        self.max_reduction_fraction = max_reduction_fraction
        self.peak_protection_radius = peak_protection_radius
        self.peak_max_reduction_fraction = peak_max_reduction_fraction
        self.min_obstacle_height_fraction = min_obstacle_height_fraction
        self.field_bore_merge_distance_multiplier = field_bore_merge_distance_multiplier
        self.field_bore_merge_direction_threshold = field_bore_merge_direction_threshold
        self.obstacle_bore_merge_direction_threshold = obstacle_bore_merge_direction_threshold
        self.bore_strength_accumulation_rate = bore_strength_accumulation_rate
        self.force_alignment_threshold = force_alignment_threshold
        self.obstacle_alignment_threshold = obstacle_alignment_threshold
        self.obstacle_min_distance_multiplier = obstacle_min_distance_multiplier
        self.obstacle_max_distance_multiplier = obstacle_max_distance_multiplier
        
        # Track if alpha was explicitly set BEFORE calling super() (since kwargs might be modified)
        # If alpha was provided in kwargs, mark it as explicit (regardless of value)
        alpha_explicit = 'alpha' in kwargs
        
        super().__init__(*args, **kwargs)
        
        # Set the flag after super() call
        self._alpha_explicit = alpha_explicit
        
        # Store permanent bores for each obstacle
        # Structure: {(obs_x, obs_y): [list of bores]}
        # Each bore: {'direction': [dx, dy], 'strength': float, 'width': float}
        self.obstacle_bores = {}
    
    def _update_potential(self):
        """Update potential with base obstacles and permanent bores."""
        # Always use alpha directly for the base paraboloid potential
        alpha = self.alpha
        
        # Base paraboloid
        self.potential = alpha * (self.X**2 + self.Y**2)
        
        # Add obstacles with their bores
        for obs_key, obs in self.obstacles.items():
            # Base obstacle (symmetric Gaussian)
            spike = obs['height'] * np.exp(-((self.X - obs['x'])**2 + (self.Y - obs['y'])**2) / 
                                            (2 * obs['width']**2))
            self.potential += spike
            
            # Apply directional bores for this obstacle
            if obs_key in self.obstacle_bores:
                for bore in self.obstacle_bores[obs_key]:
                    self._apply_obstacle_bore(obs, bore)
        
        # Apply field bores (at any location)
        if hasattr(self, 'field_bores'):
            for bore in self.field_bores:
                self._apply_field_bore(bore)
        
        # Ensure potential is valid (no NaN or Inf values)
        # Replace any invalid values with a small positive number
        max_valid = np.max(self.potential[np.isfinite(self.potential)]) if np.any(np.isfinite(self.potential)) else 1.0
        self.potential = np.nan_to_num(self.potential, nan=0.0, posinf=max_valid, neginf=0.0)
        # Ensure all values are finite
        if not np.all(np.isfinite(self.potential)):
            # If there are still invalid values, set them to 0
            self.potential[~np.isfinite(self.potential)] = 0.0
        
        # Ensure potential has sufficient variation for interpolation
        # If the potential is too flat, the bicubic interpolator will fail
        potential_range = np.max(self.potential) - np.min(self.potential)
        if potential_range < 1e-6:
            # If potential is too flat, add a tiny amount of variation to prevent interpolation failure
            # This is a numerical stability fix
            tiny_variation = 1e-6 * (self.X**2 + self.Y**2) / (np.max(self.X)**2 + np.max(self.Y)**2)
            self.potential += tiny_variation
        
        # Update interpolator with error handling
        try:
            self.potential_interpolator = RectBivariateSpline(
                self.x, self.y, self.potential.T, kx=3, ky=3, s=0
            )
        except ValueError as e:
            # If bicubic interpolation fails (e.g., data too flat), use linear interpolation
            print(f"Warning: Bicubic interpolation failed, using linear interpolation: {e}")
            # Use linear interpolation (kx=1, ky=1) which is more robust to flat data
            self.potential_interpolator = RectBivariateSpline(
                self.x, self.y, self.potential.T, kx=1, ky=1, s=0
            )
    
    def _apply_obstacle_bore(self, obs, bore):
        """
        Apply a directional bore to reduce potential on one side of obstacle.
        
        Args:
            obs: Obstacle dict with 'x', 'y', 'height', 'width'
            bore: Bore dict with 'direction', 'strength', 'width'
        """
        # Vector from obstacle center to each grid point
        dx = self.X - obs['x']
        dy = self.Y - obs['y']
        dist = np.sqrt(dx**2 + dy**2)
        
        # Normalize direction vector
        dir_vec = np.array(bore['direction'])
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-10:
            return
        dir_vec = dir_vec / dir_norm
        
        # Project grid points onto bore direction
        # Positive projection = on the "bored" side
        projection = dx * dir_vec[0] + dy * dir_vec[1]
        
        # Create directional mask: only affect points on the bored side
        # Use smooth transition to avoid sharp edges
        side_mask = 1.0 / (1.0 + np.exp(-projection / (bore['width'] * self.side_mask_sharpness)))
        
        # Distance-based falloff from obstacle center
        dist_falloff = np.exp(-dist**2 / (2 * (bore['width'] * self.distance_falloff_multiplier)**2))
        
        # Combine: stronger effect closer to obstacle and on the correct side
        bore_mask = side_mask * dist_falloff
        
        # Reduce potential: subtract a function that creates expanded basin
        # Strength determines how much potential is reduced
        bore_reduction = bore['strength'] * obs['height'] * self.obstacle_bore_strength_multiplier * bore_mask
        
        # Only reduce potential in regions where it's elevated (near obstacle)
        # Cap reduction to preserve obstacle structure - make it hard to nullify obstacles
        max_reduction = self.potential * self.max_reduction_fraction
        potential_reduction = np.minimum(bore_reduction, max_reduction)
        
        # Additional protection: ensure obstacle peak always maintains significant height
        # Stronger protection near the obstacle center
        peak_protection_mask = np.exp(-dist**2 / (2 * (obs['width'] * self.peak_protection_radius)**2))
        
        # At the peak, limit reduction to preserve obstacle height
        # Further from peak, allow more reduction
        peak_reduction_limit = obs['height'] * self.peak_max_reduction_fraction * peak_protection_mask
        far_reduction_limit = self.potential * self.max_reduction_fraction * (1 - peak_protection_mask)
        combined_limit = peak_reduction_limit + far_reduction_limit
        
        # Apply the combined limit
        potential_reduction = np.minimum(potential_reduction, combined_limit)
        
        # Final safety check: never reduce potential below a minimum threshold near obstacle center
        min_potential = obs['height'] * self.min_obstacle_height_fraction
        current_potential_after_reduction = self.potential - potential_reduction
        near_peak_mask = dist < obs['width']
        protection_needed = np.maximum(0, min_potential - current_potential_after_reduction) * near_peak_mask
        potential_reduction = potential_reduction - protection_needed
        
        self.potential -= potential_reduction
    
    def _apply_field_bore(self, bore):
        """
        Apply a directional bore at any location in the field.
        When held at a constant location, the bore strength grows, creating a deeper well.
        
        Args:
            bore: Bore dict with 'x', 'y', 'direction', 'strength', 'width'
        """
        # Vector from bore center to each grid point
        dx = self.X - bore['x']
        dy = self.Y - bore['y']
        dist = np.sqrt(dx**2 + dy**2)
        
        # Normalize direction vector
        dir_vec = np.array(bore['direction'])
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-10:
            return
        dir_vec = dir_vec / dir_norm
        
        # Project grid points onto bore direction
        # Positive projection = on the "bored" side
        projection = dx * dir_vec[0] + dy * dir_vec[1]
        
        # Create directional mask: only affect points on the bored side
        # Use smooth transition to avoid sharp edges
        side_mask = 1.0 / (1.0 + np.exp(-projection / (bore['width'] * self.side_mask_sharpness)))
        
        # Distance-based falloff from bore center
        dist_falloff = np.exp(-dist**2 / (2 * (bore['width'] * self.distance_falloff_multiplier)**2))
        
        # Combine: stronger effect closer to bore center and on the correct side
        bore_mask = side_mask * dist_falloff
        
        # Reduce potential: subtract a function that creates expanded basin
        # Strength determines how much potential is reduced
        # As strength grows (when held at constant location), the reduction increases
        # Allow reduction all the way to 0 for continuous improvement
        local_potential = self.potential
        
        # Calculate reduction: scale with strength to allow full reduction to 0
        # Tunable parameters control rate and sensitivity:
        # - field_bore_strength_multiplier: base reduction strength (0-1, higher = stronger base reduction)
        # - field_bore_reduction_curve: controls how reduction scales with strength
        #   * 1.0 = linear scaling
        #   * >1.0 = exponential (more sensitive, faster to 0)
        #   * <1.0 = logarithmic (less sensitive, slower to 0)
        base_mult = self.field_bore_strength_multiplier
        curve = self.field_bore_reduction_curve
        
        # Apply curve to strength: strength^curve
        # When curve=1.0: linear, curve=2.0: quadratic (faster), curve=0.5: square root (slower)
        curved_strength = bore['strength'] ** curve
        
        # Effective factor: interpolates between base_mult and 1.0 based on curved strength
        # This allows smooth transition from base reduction to full reduction (0 potential)
        effective_factor = base_mult + curved_strength * (1.0 - base_mult)
        bore_reduction = curved_strength * local_potential * effective_factor * bore_mask
        
        # Allow reduction all the way to 0 (no cap on max_reduction_fraction)
        # Clamp to prevent negative potential and ensure valid values
        potential_reduction = np.minimum(bore_reduction, local_potential)
        # Ensure reduction is finite and non-negative
        potential_reduction = np.nan_to_num(potential_reduction, nan=0.0, posinf=local_potential, neginf=0.0)
        potential_reduction = np.clip(potential_reduction, 0.0, local_potential)
        
        self.potential -= potential_reduction
        # Ensure potential remains non-negative and finite
        self.potential = np.clip(self.potential, 0.0, None)
        self.potential = np.nan_to_num(self.potential, nan=0.0, posinf=np.max(self.potential[self.potential != np.inf]) if np.any(self.potential != np.inf) else 0.0, neginf=0.0)
    
    def add_bore(self, obstacle_x, obstacle_y, force_direction, bore_strength, bore_width=None):
        """
        Add or update a permanent bore to an obstacle.
        
        Args:
            obstacle_x, obstacle_y: Position of obstacle to bore into
            force_direction: Unit vector indicating direction force was applied FROM
                           (points FROM robot TO obstacle, i.e., uphill direction)
            bore_strength: Strength of bore (0-1, typically from tunneling intent)
            bore_width: Width/spread of bore (defaults to obstacle width * 2)
        """
        obs_key = (obstacle_x, obstacle_y)
        
        if obs_key not in self.obstacles:
            raise ValueError(f"No obstacle at ({obstacle_x}, {obstacle_y})")
        
        obs = self.obstacles[obs_key]
        
        # Normalize force direction
        force_dir = np.array(force_direction)
        force_norm = np.linalg.norm(force_dir)
        if force_norm < 1e-10:
            return
        force_dir = force_dir / force_norm
        
        # Default width
        if bore_width is None:
            bore_width = obs['width'] * self.obstacle_bore_width_multiplier
        
        # Initialize bores list if needed
        if obs_key not in self.obstacle_bores:
            self.obstacle_bores[obs_key] = []
        
        # Check if bore in similar direction already exists
        # Prefer creating new bores over updating existing ones - make directions more stable
        similar_bore_idx = None
        max_similarity = 0.0
        
        # Stricter criteria: only merge if direction is very similar
        merge_direction_threshold = self.obstacle_bore_merge_direction_threshold
        
        for i, existing_bore in enumerate(self.obstacle_bores[obs_key]):
            # Check if directions are similar (dot product > threshold)
            dir_similarity = np.dot(existing_bore['direction'], force_dir)
            if dir_similarity > merge_direction_threshold and dir_similarity > max_similarity:
                max_similarity = dir_similarity
                similar_bore_idx = i
        
        if similar_bore_idx is not None:
            # Update existing bore: accumulate strength (capped at 1.0)
            # But keep direction more stable - only update if extremely similar
            bore = self.obstacle_bores[obs_key][similar_bore_idx]
            bore['strength'] = min(1.0, bore['strength'] + bore_strength * self.bore_strength_accumulation_rate)
            # Only update direction if almost identical (very high similarity)
            if max_similarity > 0.95:
                old_strength = bore['strength'] - bore_strength * self.bore_strength_accumulation_rate
                if old_strength > 0:
                    weight_old = old_strength / bore['strength']
                    weight_new = (bore_strength * self.bore_strength_accumulation_rate) / bore['strength']
                    bore['direction'] = (weight_old * np.array(bore['direction']) + 
                                         weight_new * force_dir).tolist()
                    # Renormalize
                    dir_norm = np.linalg.norm(bore['direction'])
                    if dir_norm > 1e-10:
                        bore['direction'] = (np.array(bore['direction']) / dir_norm).tolist()
        else:
            # Add new bore - prefer creating many bores in different directions
            bore = {
                'direction': force_dir.tolist(),
                'strength': min(1.0, bore_strength),
                'width': bore_width
            }
            self.obstacle_bores[obs_key].append(bore)
        
        # Recompute potential with new bore
        self._update_potential()
    
    def update_bore_from_force(self, robot_x, robot_y, force_vector, tunneling_intent, 
                                min_distance_threshold=0.3, min_gradient_threshold=None):
        """
        Automatically add bores based on applied force at current position.
        Uses inertial effort methodology: bores are created when tunneling intent
        (gamma) is significant, indicating sustained uphill effort.
        
        The tunneling_intent (gamma) parameter already encodes accumulated uphill effort
        via the intent integrator, so no gradient threshold is needed. When gamma is high,
        it means there has been sustained uphill force over time.
        
        Args:
            robot_x, robot_y: Current robot position
            force_vector: External force vector [fx, fy] (uphill = into gradient)
            tunneling_intent: Tunneling intent value (0-1, typically gamma from intent integrator)
            min_distance_threshold: Minimum distance to consider obstacle for boring (legacy, not used for field bores)
            min_gradient_threshold: DEPRECATED - no longer used, kept for backward compatibility
        
        Returns:
            bool: True if updated an existing field bore, False if created a new one or no bore was created
        """
        if tunneling_intent < 0.01:  # No tunneling intent
            return False
        
        # Normalize force direction
        force_norm = np.linalg.norm(force_vector)
        if force_norm < 1e-6:
            return False
        
        force_dir = force_vector / force_norm
        
        # Get gradient at current position to determine uphill direction
        grad = self.get_gradient(robot_x, robot_y)
        grad_norm = np.linalg.norm(grad)
        
        # If gradient is too small, there's no meaningful uphill direction
        # This is just a numerical check, not a threshold for bore creation
        if grad_norm < 1e-6:
            return False
        
        # Normalize gradient direction
        grad_dir = grad / grad_norm
        
        # Check if force is uphill (aligned with gradient direction)
        # Force should be in same direction as gradient (uphill)
        alignment = np.dot(force_dir, grad_dir)
        
        # Only create bore if force is significantly uphill (aligned with gradient)
        # The tunneling_intent (gamma) already encodes sustained effort via intent integrator,
        # so we just need to verify force alignment
        if alignment < self.force_alignment_threshold:  # Not well aligned with gradient
            return False
        
        # Check for nearby obstacles first - create obstacle bores if applicable
        # Find nearest obstacle in force direction
        min_dist = float('inf')
        nearest_obs = None
        
        for obs_key, obs in self.obstacles.items():
            # Vector from robot to obstacle
            to_obstacle = np.array([obs['x'] - robot_x, obs['y'] - robot_y])
            dist = np.linalg.norm(to_obstacle)
            
            # Check if obstacle is too close (skip if robot is inside obstacle)
            # Use obstacle width as threshold instead of fixed distance
            if dist < obs['width'] * self.obstacle_min_distance_multiplier:
                continue
            
            # Check alignment: obstacle should be roughly in force direction
            to_obstacle_norm = to_obstacle / (dist + 1e-10)
            obs_alignment = np.dot(force_dir, to_obstacle_norm)
            
            # Only consider obstacles in the direction force is applied
            if obs_alignment > self.obstacle_alignment_threshold and dist < min_dist:
                min_dist = dist
                nearest_obs = (obs_key, obs, to_obstacle_norm)
        
        # Create obstacle bore if nearby obstacle found
        if nearest_obs is not None:
            obs_key, obs, direction = nearest_obs
            # Create obstacle bore if within reasonable distance
            # Use a distance threshold based on obstacle width
            max_obstacle_distance = obs['width'] * self.obstacle_max_distance_multiplier
            if min_dist < max_obstacle_distance:
                self.add_bore(obs['x'], obs['y'], direction, tunneling_intent)
        
        # Also create field bore at current position
        # tunneling_intent (gamma) already represents accumulated uphill effort
        # Return True if updated existing bore, False if created new one
        return self.add_field_bore(robot_x, robot_y, force_dir, tunneling_intent)
    
    def add_field_bore(self, x, y, force_direction, bore_strength, bore_width=None):
        """
        Add or update a permanent field bore at any location.
        
        Args:
            x, y: Position where bore is created
            force_direction: Unit vector indicating direction force was applied FROM
                           (points FROM robot TO uphill direction)
            bore_strength: Strength of bore (0-1, typically from tunneling intent)
            bore_width: Width/spread of bore (defaults to self.bore_width_default)
        
        Returns:
            bool: True if updated an existing bore, False if created a new one
        """
        # Normalize force direction
        force_dir = np.array(force_direction)
        force_norm = np.linalg.norm(force_dir)
        if force_norm < 1e-10:
            return
        force_dir = force_dir / force_norm
        
        # Default width
        if bore_width is None:
            bore_width = self.bore_width_default
        
        # Check if bore at similar location already exists
        # When held at constant location, merge bores to accumulate strength
        similar_bore_idx = None
        min_dist = float('inf')
        
        # More lenient merging: prioritize location over direction
        # Use a fixed distance threshold (0.1m) for location-based merging
        # This ensures bores accumulate when held at constant location even if direction varies slightly
        location_merge_threshold = 0.1  # Fixed 10cm threshold for location-based merging
        merge_distance_threshold = bore_width * self.field_bore_merge_distance_multiplier
        merge_direction_threshold = self.field_bore_merge_direction_threshold
        
        for i, existing_bore in enumerate(self.field_bores):
            # Check distance from existing bore
            dist = np.sqrt((existing_bore['x'] - x)**2 + (existing_bore['y'] - y)**2)
            
            # Check if directions are similar (dot product > threshold)
            dir_similarity = np.dot(existing_bore['direction'], force_dir)
            
            # Merge if very close (location-based, regardless of direction)
            # OR if close with similar direction
            # This allows strength accumulation when held at constant location
            if dist < location_merge_threshold:
                # Very close - merge regardless of direction (location-based merging)
                if dist < min_dist:
                    min_dist = dist
                    similar_bore_idx = i
            elif dist < merge_distance_threshold and dir_similarity > merge_direction_threshold:
                # Close and direction is similar - also merge
                if dist < min_dist:
                    min_dist = dist
                    similar_bore_idx = i
        
        if similar_bore_idx is not None:
            # Update existing bore: accumulate strength (capped at 1.0)
            # This allows the bore magnitude to grow when held at constant location
            bore = self.field_bores[similar_bore_idx]
            old_strength = bore['strength']
            
            # Accumulate strength more aggressively when updating existing bore
            # Use the full bore_strength (not just accumulation_rate) to grow faster
            strength_increment = bore_strength * self.bore_strength_accumulation_rate
            # Scale increment by tunneling intent to grow faster when intent is high
            strength_increment *= (1.0 + bore_strength)  # Boost when strength is already high
            # Allow strength to exceed 1.0 to enable continuous reduction to 0
            # No upper limit - strength can grow indefinitely for continuous improvement
            bore['strength'] = bore['strength'] + strength_increment
            
            # Keep position fixed - don't update it
            # Update direction by weighted average if directions are reasonably similar
            # This allows the bore to adapt slightly while still accumulating strength
            dir_similarity = np.dot(bore['direction'], force_dir)
            if dir_similarity > 0.5:  # Even more lenient threshold for direction update
                if old_strength > 0:
                    weight_old_dir = old_strength / bore['strength']
                    weight_new_dir = (strength_increment) / bore['strength']
                    bore['direction'] = (weight_old_dir * np.array(bore['direction']) + 
                                         weight_new_dir * force_dir).tolist()
                    # Renormalize
                    dir_norm = np.linalg.norm(bore['direction'])
                    if dir_norm > 1e-10:
                        bore['direction'] = (np.array(bore['direction']) / dir_norm).tolist()
            
            # Recompute potential with updated bore
            self._update_potential()
            return True  # Indicates we updated an existing bore
        else:
            # Add new field bore - prefer creating many smaller bores
            # Strength can exceed 1.0 to allow continuous reduction to 0
            bore = {
                'x': x,
                'y': y,
                'direction': force_dir.tolist(),
                'strength': bore_strength,  # No cap - allow growth beyond 1.0
                'width': bore_width
            }
            self.field_bores.append(bore)
            
            # Recompute potential with new bore
            self._update_potential()
            return False  # Indicates we created a new bore
    
    def get_bore_info(self, obstacle_x, obstacle_y):
        """Get information about bores for a specific obstacle."""
        obs_key = (obstacle_x, obstacle_y)
        if obs_key in self.obstacle_bores:
            return self.obstacle_bores[obs_key].copy()
        return []
    
    def clear_bores(self, obstacle_x=None, obstacle_y=None, clear_field_bores=False):
        """
        Clear bores. 
        
        Args:
            obstacle_x, obstacle_y: If specified, clear only that obstacle's bores
            clear_field_bores: If True, also clear field bores (only when clearing all)
        """
        if obstacle_x is not None and obstacle_y is not None:
            obs_key = (obstacle_x, obstacle_y)
            if obs_key in self.obstacle_bores:
                del self.obstacle_bores[obs_key]
                self._update_potential()
        else:
            self.obstacle_bores = {}
            if clear_field_bores:
                self.field_bores = []
            self._update_potential()
    
    def get_field_bore_count(self):
        """Get the number of field bores."""
        return len(self.field_bores)
    
    def set_alpha(self, alpha_value):
        """
        Set the alpha value for the base paraboloid potential and update the field.
        
        Args:
            alpha_value: The alpha value to use for the base potential V = alpha * (x^2 + y^2)
        """
        self.alpha = alpha_value
        self._alpha_explicit = True  # Mark that alpha was explicitly set
        self._update_potential()  # Recalculate potential with new alpha


class PotentialFieldDiscrete1D:
    """
    1D Discretized implementation of the potential field.
    """
    def __init__(self, x_bounds=(-10, 10), resolution=1000, alpha=1.0):
        """
        Initialize a 1D discretized potential field.
        
        Args:
            x_bounds: Tuple (x_min, x_max) for position range
            resolution: Number of points for evaluation grid
            alpha: Alpha value for base parabolic potential V = alpha * x^2
        """
        self.x_bounds = x_bounds
        self.resolution = resolution
        self.obstacles = {}  # Key: x position, Value: {'x': x, 'height': h, 'width': w}
        self.alpha = alpha
        
        # Create position grid
        self.x = np.linspace(x_bounds[0], x_bounds[1], resolution)
        
        # Grid spacing for reference
        self.dx = (x_bounds[1] - x_bounds[0]) / (resolution - 1)
        
        # Initialize potential array and interpolator
        self._update_potential()
    
    def _update_potential(self):
        """Update the potential array and cubic spline interpolator.
        Called only when obstacles are added, removed, or cleared.
        """
        # Always use alpha directly for the base parabolic potential
        alpha = self.alpha
        
        # Recalculate potential array from scratch
        # Base parabola: V = alpha * x^2
        self.potential = alpha * (self.x**2)
        
        # Add Gaussian spikes for each obstacle
        for obs in self.obstacles.values():
            spike = obs['height'] * np.exp(-((self.x - obs['x'])**2) / (2 * obs['width']**2))
            self.potential += spike
        
        # Recreate cubic spline interpolator with updated potential
        # k=3 for cubic spline (matches bicubic quality in 2D)
        self.potential_interpolator = UnivariateSpline(
            self.x, self.potential, k=3, s=0
        )
    
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
        self._update_potential()  # Updates potential and gradient fields
    
    def remove_obstacle(self, x):
        """Remove obstacle at position x."""
        key = x
        if key in self.obstacles:
            del self.obstacles[key]
            self._update_potential()  # Updates potential and gradient fields
    
    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles = {}
        self._update_potential()  # Updates potential and gradient fields
    
    def calculate_potential(self):
        """Calculate potential over the entire grid. Returns the stored potential array."""
        return self.potential.copy()
    
    def get_potential(self, x):
        """
        Get potential value at position x using cubic spline interpolation.
        
        Args:
            x: Position (scalar or array)
        
        Returns:
            Potential value(s) at position(s) x
        """
        # Clamp coordinates to grid bounds
        x = np.clip(x, self.x_bounds[0], self.x_bounds[1])
        
        # Use cubic spline interpolator
        result = self.potential_interpolator(x)
        
        # Handle scalar inputs
        if np.isscalar(x):
            return float(result)
        return result
    
    def get_gradient(self, x):
        """
        Get gradient at position x by computing derivative of the cubic spline interpolated potential.
        The gradient is computed analytically from the interpolated function.
        
        Args:
            x: Position (scalar or array)
        
        Returns:
            Gradient value(s) at position(s) x
        """
        # Clamp coordinates to grid bounds
        x = np.clip(x, self.x_bounds[0], self.x_bounds[1])
        
        # Compute gradient analytically from the cubic spline interpolator
        # derivative(n=1) computes first derivative dV/dx
        grad = self.potential_interpolator.derivative(n=1)(x)
        
        # Handle scalar inputs
        if np.isscalar(x):
            return float(grad)
        return grad
    
    def visualize(self, title="1D Potential Field (Discrete)", show_obstacles=True):
        """
        Visualize the potential field as a 2D plot: potential (Y-axis) vs position (X-axis).
        
        Args:
            title: Plot title
            show_obstacles: If True, mark obstacle positions on the plot
        """
        potential_values = self.calculate_potential()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.x, potential_values, 'b-', linewidth=2, label='Potential')
        
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
    
    def visualize_with_force(self, title="1D Potential Field with Force (Discrete)", show_obstacles=True):
        """
        Visualize both potential and force (negative gradient) on the same plot.
        
        Args:
            title: Plot title
            show_obstacles: If True, mark obstacle positions on the plot
        """
        potential_values = self.calculate_potential()
        force_values = self.get_gradient(self.x)
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot potential on left y-axis
        color1 = 'b'
        ax1.set_xlabel('Position (X)', fontsize=12)
        ax1.set_ylabel('Potential', color=color1, fontsize=12)
        line1 = ax1.plot(self.x, potential_values, color=color1, linewidth=2, label='Potential')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Plot force on right y-axis
        ax2 = ax1.twinx()
        color2 = 'r'
        ax2.set_ylabel('Force (-dV/dx)', color=color2, fontsize=12)
        line2 = ax2.plot(self.x, force_values, color=color2, linewidth=2, linestyle='--', label='Force')
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


class PotentialFieldDiscrete1DRemodelable(PotentialFieldDiscrete1D):
    """
    1D Discretized potential field with permanent directional remodeling.
    When force is applied uphill into an obstacle, creates a permanent "bore"
    that expands the basin on that side only.
    """
    
    def __init__(self, *args, obstacle_bore_width_multiplier=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Store permanent bores for each obstacle
        # Structure: {obs_x: [list of bores]}
        # Each bore: {'direction': 1 or -1, 'strength': float, 'width': float}
        # direction: 1 = force from left (bore on right side), -1 = force from right (bore on left side)
        self.obstacle_bores = {}
        self.obstacle_bore_width_multiplier = obstacle_bore_width_multiplier
    
    def _update_potential(self):
        """Update potential with base obstacles and permanent bores."""
        # Always use alpha directly for the base parabolic potential
        alpha = self.alpha
        
        # Base parabola
        self.potential = alpha * (self.x**2)
        
        # Add obstacles with their bores
        for obs_key, obs in self.obstacles.items():
            # Base obstacle (symmetric Gaussian)
            spike = obs['height'] * np.exp(-((self.x - obs['x'])**2) / (2 * obs['width']**2))
            self.potential += spike
            
            # Apply directional bores for this obstacle
            if obs_key in self.obstacle_bores:
                for bore in self.obstacle_bores[obs_key]:
                    self._apply_bore(obs, bore)
        
        # Update interpolator
        self.potential_interpolator = UnivariateSpline(
            self.x, self.potential, k=3, s=0
        )
    
    def _apply_bore(self, obs, bore):
        """
        Apply a directional bore to reduce potential on one side of obstacle.
        
        Args:
            obs: Obstacle dict with 'x', 'height', 'width'
            bore: Bore dict with 'direction', 'strength', 'width'
        """
        # Distance from obstacle center
        dx = self.x - obs['x']
        dist = np.abs(dx)
        
        # Direction: 1 = bore on right side (x > obs_x), -1 = bore on left side (x < obs_x)
        direction = bore['direction']
        
        # Create directional mask: only affect points on the bored side
        if direction > 0:
            # Bore on right side (x > obs_x)
            side_mask = 1.0 / (1.0 + np.exp(-dx / (bore['width'] * 0.3)))
        else:
            # Bore on left side (x < obs_x)
            side_mask = 1.0 / (1.0 + np.exp(dx / (bore['width'] * 0.3)))
        
        # Distance-based falloff from obstacle center
        dist_falloff = np.exp(-dist**2 / (2 * (bore['width'] * 1.5)**2))
        
        # Combine: stronger effect closer to obstacle and on the correct side
        bore_mask = side_mask * dist_falloff
        
        # Reduce potential: subtract a function that creates expanded basin
        # Strength determines how much potential is reduced
        bore_reduction = bore['strength'] * obs['height'] * 0.5 * bore_mask
        
        # Only reduce potential in regions where it's elevated (near obstacle)
        # Cap reduction to preserve obstacle structure
        max_reduction = self.potential * 0.6
        potential_reduction = np.minimum(bore_reduction, max_reduction)
        
        self.potential -= potential_reduction
    
    def add_bore(self, obstacle_x, force_direction, bore_strength, bore_width=None):
        """
        Add or update a permanent bore to an obstacle.
        
        Args:
            obstacle_x: Position of obstacle to bore into
            force_direction: Direction force was applied FROM (1 = from left, -1 = from right)
                          In 1D, this is simply the sign: positive = force pushing right, negative = force pushing left
            bore_strength: Strength of bore (0-1, typically from tunneling intent)
            bore_width: Width/spread of bore (defaults to obstacle width * 2)
        """
        if obstacle_x not in self.obstacles:
            raise ValueError(f"No obstacle at {obstacle_x}")
        
        obs = self.obstacles[obstacle_x]
        
        # Normalize direction to -1 or 1
        direction = 1 if force_direction > 0 else -1
        
        # Default width
        if bore_width is None:
            bore_width = obs['width'] * self.obstacle_bore_width_multiplier
        
        # Initialize bores list if needed
        if obstacle_x not in self.obstacle_bores:
            self.obstacle_bores[obstacle_x] = []
        
        # Check if bore in same direction already exists
        # If so, update it; otherwise add new one
        similar_bore_idx = None
        for i, existing_bore in enumerate(self.obstacle_bores[obstacle_x]):
            if existing_bore['direction'] == direction:
                similar_bore_idx = i
                break
        
        if similar_bore_idx is not None:
            # Update existing bore: accumulate strength (capped at 1.0)
            bore = self.obstacle_bores[obstacle_x][similar_bore_idx]
            bore['strength'] = min(1.0, bore['strength'] + bore_strength * 0.05)
        else:
            # Add new bore
            bore = {
                'direction': direction,
                'strength': min(1.0, bore_strength),
                'width': bore_width
            }
            self.obstacle_bores[obstacle_x].append(bore)
        
        # Recompute potential with new bore
        self._update_potential()
    
    def update_bore_from_force(self, robot_x, force_value, tunneling_intent, 
                                min_distance_threshold=0.3):
        """
        Automatically detect nearby obstacles and add bores based on applied force.
        
        Args:
            robot_x: Current robot position
            force_value: External force value (positive = pushing right, negative = pushing left)
            tunneling_intent: Tunneling intent value (0-1, typically gamma)
            min_distance_threshold: Minimum distance to consider obstacle for boring
        """
        if tunneling_intent < 0.01:  # No tunneling intent
            return
        
        if abs(force_value) < 1e-6:
            return
        
        # Direction: 1 if pushing right, -1 if pushing left
        force_direction = 1 if force_value > 0 else -1
        
        # Find nearest obstacle in force direction
        min_dist = float('inf')
        nearest_obs = None
        
        for obs_x, obs in self.obstacles.items():
            # Distance from robot to obstacle
            dist = abs(obs_x - robot_x)
            
            # Check if obstacle is too close
            if dist < min_distance_threshold:
                continue
            
            # Check alignment: obstacle should be in force direction
            if force_direction > 0 and obs_x > robot_x:
                # Pushing right, obstacle is to the right
                if dist < min_dist:
                    min_dist = dist
                    nearest_obs = (obs_x, obs)
            elif force_direction < 0 and obs_x < robot_x:
                # Pushing left, obstacle is to the left
                if dist < min_dist:
                    min_dist = dist
                    nearest_obs = (obs_x, obs)
        
        # Add bore to nearest obstacle
        if nearest_obs is not None:
            obs_x, obs = nearest_obs
            self.add_bore(obs_x, force_direction, tunneling_intent)
    
    def get_bore_info(self, obstacle_x):
        """Get information about bores for a specific obstacle."""
        if obstacle_x in self.obstacle_bores:
            return self.obstacle_bores[obstacle_x].copy()
        return []
    
    def clear_bores(self, obstacle_x=None):
        """Clear bores. If obstacle specified, clear only that obstacle's bores."""
        if obstacle_x is not None:
            if obstacle_x in self.obstacle_bores:
                del self.obstacle_bores[obstacle_x]
                self._update_potential()
        else:
            self.obstacle_bores = {}
            self._update_potential()
