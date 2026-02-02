from typing import Any
import numpy as np


__all__ = ["PotentialField", "Point"]
__version__ = "0.1.0"

class Point:
    def __init__(self, loc: tuple, f: callable):
        self.x = loc[0]
        self.y = loc[1]
        self.z = loc[2]
        self.value = f(loc)

    def __call__(self):
        return self.value
    
    def __str__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z}, value={self.value})"

    def get_value(self):
        return self.value
    
    def get_loc(self):
        return self.x, self.y, self.z


class PotentialField:
    def __init__(self, bounds: tuple[float, float, float], robot_loc: tuple[float, float, float] = (0, 0, 0), point_type=Point):
        """Bounds is a tuple of the form (xbound, ybound, zbound) that represents the volume of the potential field in meters
        The potential field is discretized by a grid in cm
        Unless otherwise stated, the robot is assumed to be at the origin (0, 0, 0).
        """
        self.num_x = int(bounds[0]*100) # turn m to cm
        self.num_y = int(bounds[1]*100)
        self.num_z = int(bounds[2]*100)
        self.robot_loc = robot_loc
        self.point_type = point_type
        
        self.grid = np.empty((self.num_x, self.num_y, self.num_z), dtype=object)

        self.center_x = self.num_x // 2
        self.center_y = self.num_y // 2
        self.center_z = self.num_z // 2

        for i in range(self.num_x):
            for j in range(self.num_y):
                for k in range(self.num_z):
                    world_x = (i - self.center_x) / 100.0  # Convert back to meters
                    world_y = (j - self.center_y) / 100.0
                    world_z = (k - self.center_z) / 100.0
                    self.grid[i, j, k] = self.point_type((world_x, world_y, world_z))

    def __call__(self, loc=None):
        if loc is None:
            return self.get_grid()
        else:
            grid_x = int(loc[0] * 100 + self.center_x)
            grid_y = int(loc[1] * 100 + self.center_y)
            grid_z = int(loc[2] * 100 + self.center_z)
            
            if self.check_bounds(grid_x, grid_y, grid_z):
                return self.grid[grid_x, grid_y, grid_z].get_value()
    
    def __str__(self):
        return f"PotentialField(num_x={self.num_x}, num_y={self.num_y}, num_z={self.num_z})"

    # util functions
    def check_bounds(self, grid_x, grid_y, grid_z):
        if 0 <= grid_x < self.num_x and 0 <= grid_y < self.num_y and 0 <= grid_z < self.num_z:
            return True
        else: 
            raise ValueError(f"Point ({grid_x}, {grid_y}, {grid_z}) is out of bounds")
    
    def add_point(self, loc):
        grid_x = int(loc[0] * 100 + self.center_x)
        grid_y = int(loc[1] * 100 + self.center_y)
        grid_z = int(loc[2] * 100 + self.center_z)

        if self.check_bounds(grid_x, grid_y, grid_z):
            self.grid[grid_x, grid_y, grid_z] = self.point_type((grid_x, grid_y, grid_z))

    def get_point(self, loc):
        grid_x = int(loc[0] * 100 + self.center_x)
        grid_y = int(loc[1] * 100 + self.center_y)
        grid_z = int(loc[2] * 100 + self.center_z)

        if self.check_bounds(grid_x, grid_y, grid_z):
            return self.grid[grid_x, grid_y, grid_z]
        else:
            raise ValueError(f"Point ({loc[0]}, {loc[1]}, {loc[2]}) is out of bounds")

    def get_grid(self):
        return self.grid
    
    def get_grid_size(self):
        return self.num_x, self.num_y, self.num_z