"""
Calculate the distance from a spike center where the potential drops below a given threshold.
Uses the Gaussian spike formula: potential = height * exp(-(distance²) / (2 * width²))
"""

import math

def calculate_distance_for_potential(height, width, target_potential):
    """
    Calculate the distance from spike center where potential drops below target_potential.
    
    Args:
        height: Maximum potential at spike center
        width: Standard deviation (width) of the Gaussian spike
        target_potential: Target potential value to find distance for
    
    Returns:
        Distance in meters where potential equals target_potential
    """
    # Formula: target_potential = height * exp(-(d²) / (2 * width²))
    # Solving for d:
    # target_potential / height = exp(-(d²) / (2 * width²))
    # ln(target_potential / height) = -(d²) / (2 * width²)
    # d² = -ln(target_potential / height) * 2 * width²
    # d = sqrt(-ln(target_potential / height) * 2 * width²)
    
    if target_potential >= height:
        return 0.0  # At or above center potential
    
    ratio = target_potential / height
    if ratio <= 0:
        raise ValueError("Target potential must be positive")
    
    d_squared = -math.log(ratio) * 2 * width**2
    d = math.sqrt(d_squared)
    
    return d

def show_potential_at_distances(height, width, distances):
    """Show potential values at various distances."""
    print(f"\nPotential at various distances (height={height}, width={width}):")
    for dist in distances:
        pot = height * math.exp(-(dist**2) / (2 * width**2))
        print(f"  d={dist:.2f}m: potential={pot:.4f}")

if __name__ == "__main__":
    # Spike parameters
    height = 10
    width = 0.01
    
    # Calculate distances for different potential thresholds
    thresholds = [1.0, 0.01]
    
    print(f"Spike parameters: height={height}, width={width}")
    print("=" * 60)
    
    for threshold in thresholds:
        d = calculate_distance_for_potential(height, width, threshold)
        print(f"\nDistance where potential drops below {threshold}:")
        print(f"  {d:.4f} m ({d*100:.2f} cm)")
        
        # Verify
        potential_at_d = height * math.exp(-(d**2) / (2 * width**2))
        print(f"  Verification - potential at d={d:.4f}m: {potential_at_d:.6f}")
    
    # Show potential at various distances
    show_potential_at_distances(height, width, [0.05, 0.08, 0.09, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18])
