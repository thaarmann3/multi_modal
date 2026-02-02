"""
Optimize grid discretization resolution by comparing error between
discretized and continuous potential field implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.potential_field_discrete import PotentialFieldDiscrete
from fields.potential_field_class import PotentialField

def compute_gradient_error(pf_discrete, pf_original, test_points, resolution):
    """
    Compute gradient error between discrete and original implementations.
    
    Returns:
        mean_error: Mean L2 norm of gradient differences
        max_error: Maximum L2 norm of gradient differences
        computation_time: Time to compute gradients for all test points
    """
    errors = []
    start_time = time.time()
    
    for x, y in test_points:
        grad_discrete = pf_discrete.get_gradient(x, y)
        grad_original = pf_original.get_gradient(x, y)
        error = np.linalg.norm(grad_discrete - grad_original)
        errors.append(error)
    
    computation_time = time.time() - start_time
    errors = np.array(errors)
    
    return {
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'computation_time': computation_time,
        'resolution': resolution
    }

def optimize_resolution(x_bounds=(-5, 5), y_bounds=(-5, 5), 
                       scaling_factor=0.2, n_test_points=100):
    """
    Optimize discretization resolution by testing different resolutions
    and comparing with continuous implementation.
    """
    # Create test points (randomly distributed in the domain)
    np.random.seed(42)
    test_x = np.random.uniform(x_bounds[0] + 0.5, x_bounds[1] - 0.5, n_test_points)
    test_y = np.random.uniform(y_bounds[0] + 0.5, y_bounds[1] - 0.5, n_test_points)
    test_points = list(zip(test_x, test_y))
    
    # Define obstacles for testing
    obstacles = [
        (2.0, 1.0, 5.0, 0.5),
        (-1.5, -2.0, 8.0, 0.3),
        (0.0, 3.0, 6.0, 0.4),
    ]
    
    # Create original (continuous) potential field for reference
    print("Creating reference continuous potential field...")
    pf_original = PotentialField(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        resolution=200,  # High resolution for reference
        scaling_factor=scaling_factor
    )
    for x, y, height, width in obstacles:
        pf_original.add_obstacle(x, y, height, width)
    
    # Test different resolutions
    resolutions = [20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200, 250, 300]
    results = []
    
    print(f"\nTesting {len(resolutions)} different resolutions...")
    print("=" * 80)
    
    for resolution in resolutions:
        print(f"Testing resolution: {resolution}...", end=" ")
        
        try:
            # Create discrete potential field
            pf_discrete = PotentialFieldDiscrete(
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                resolution=resolution,
                scaling_factor=scaling_factor
            )
            
            # Add obstacles
            for x, y, height, width in obstacles:
                pf_discrete.add_obstacle(x, y, height, width)
            
            # Compute error metrics
            result = compute_gradient_error(pf_discrete, pf_original, test_points, resolution)
            results.append(result)
            
            print(f"Mean error: {result['mean_error']:.6f}, "
                  f"Max error: {result['max_error']:.6f}, "
                  f"Time: {result['computation_time']:.4f}s")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    return results, test_points

def plot_optimization_results(results):
    """Plot optimization results."""
    if not results:
        print("No results to plot!")
        return
    
    resolutions = [r['resolution'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    max_errors = [r['max_error'] for r in results]
    computation_times = [r['computation_time'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean error vs resolution
    ax1 = axes[0, 0]
    ax1.plot(resolutions, mean_errors, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Resolution', fontsize=12)
    ax1.set_ylabel('Mean Gradient Error (L2 norm)', fontsize=12)
    ax1.set_title('Mean Gradient Error vs Resolution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Max error vs resolution
    ax2 = axes[0, 1]
    ax2.plot(resolutions, max_errors, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Resolution', fontsize=12)
    ax2.set_ylabel('Max Gradient Error (L2 norm)', fontsize=12)
    ax2.set_title('Max Gradient Error vs Resolution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Plot 3: Computation time vs resolution
    ax3 = axes[1, 0]
    ax3.plot(resolutions, computation_times, 'g-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Resolution', fontsize=12)
    ax3.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax3.set_title('Computation Time vs Resolution', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # Plot 4: Error vs computation time (trade-off)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(computation_times, mean_errors, 
                         c=resolutions, s=100, cmap='viridis', 
                         alpha=0.6, edgecolors='black', linewidth=1)
    ax4.set_xlabel('Computation Time (seconds)', fontsize=12)
    ax4.set_ylabel('Mean Gradient Error (L2 norm)', fontsize=12)
    ax4.set_title('Error vs Computation Time Trade-off', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    plt.colorbar(scatter, ax=ax4, label='Resolution')
    
    plt.tight_layout()
    plt.savefig('discretization_optimization.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'discretization_optimization.png'")
    plt.show()

def find_optimal_resolution(results, error_threshold=None, time_threshold=None):
    """Find optimal resolution based on error and computation time."""
    if not results:
        return None
    
    # Find resolution with best error/time trade-off
    # Score = 1 / (error * time), higher is better
    scores = []
    for r in results:
        # Normalize error and time (use log scale for better comparison)
        error_score = 1.0 / (r['mean_error'] + 1e-10)
        time_score = 1.0 / (r['computation_time'] + 1e-10)
        # Combined score (weighted)
        score = error_score * 0.7 + time_score * 0.3
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_result = results[best_idx]
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Optimal Resolution: {best_result['resolution']}")
    print(f"  Mean Error: {best_result['mean_error']:.6f}")
    print(f"  Max Error: {best_result['max_error']:.6f}")
    print(f"  Median Error: {best_result['median_error']:.6f}")
    print(f"  Std Error: {best_result['std_error']:.6f}")
    print(f"  Computation Time: {best_result['computation_time']:.6f} seconds")
    
    # Also show resolutions that meet thresholds
    if error_threshold:
        print(f"\nResolutions with mean error < {error_threshold}:")
        for r in results:
            if r['mean_error'] < error_threshold:
                print(f"  Resolution {r['resolution']}: error={r['mean_error']:.6f}, time={r['computation_time']:.6f}s")
    
    if time_threshold:
        print(f"\nResolutions with computation time < {time_threshold}s:")
        for r in results:
            if r['computation_time'] < time_threshold:
                print(f"  Resolution {r['resolution']}: error={r['mean_error']:.6f}, time={r['computation_time']:.6f}s")
    
    return best_result

if __name__ == "__main__":
    print("=" * 80)
    print("GRID DISCRETIZATION OPTIMIZATION")
    print("=" * 80)
    print("\nComparing PotentialFieldDiscrete vs PotentialField (continuous)")
    print("to find optimal resolution balancing accuracy and computation time.\n")
    
    # Run optimization
    results, test_points = optimize_resolution(
        x_bounds=(-5, 5),
        y_bounds=(-5, 5),
        scaling_factor=0.2,
        n_test_points=100
    )
    
    if results:
        # Plot results
        plot_optimization_results(results)
        
        # Find optimal resolution
        optimal = find_optimal_resolution(
            results,
            error_threshold=0.1,  # Mean error threshold
            time_threshold=0.1    # Computation time threshold (seconds)
        )
        
        # Print summary table
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Resolution':<12} {'Mean Error':<15} {'Max Error':<15} {'Time (s)':<12}")
        print("-" * 80)
        for r in sorted(results, key=lambda x: x['resolution']):
            print(f"{r['resolution']:<12} {r['mean_error']:<15.6f} {r['max_error']:<15.6f} {r['computation_time']:<12.6f}")
    else:
        print("No results obtained!")
