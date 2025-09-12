#!/usr/bin/env python3
"""
Demo script for Hausdorff dimension calculation.

This script demonstrates the robust box-counting implementation with synthetic data
and shows typical results for different types of signals encountered in climbing force analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from additional_calculations import hausdorff_dimension_boxcount

def demo_hausdorff_calculation():
    """Demonstrate Hausdorff dimension calculation with various signal types."""
    
    print("=== Hausdorff Dimension Calculation Demo ===\n")
    
    # Test cases representing different climbing scenarios
    test_cases = [
        {
            "name": "Smooth Force Application",
            "description": "Clean, controlled force application (low complexity)",
            "generator": lambda t: 50 * np.sin(2 * np.pi * 2 * t) + 20,
            "expected_range": "1.0-1.3"
        },
        {
            "name": "Moderate Complexity",
            "description": "Some variation in force application",
            "generator": lambda t: 50 * np.sin(2 * np.pi * 2 * t) + 20 + 5 * np.random.randn(len(t)),
            "expected_range": "1.2-1.5"
        },
        {
            "name": "High Complexity",
            "description": "Irregular, less controlled movements",
            "generator": lambda t: 50 * np.sin(2 * np.pi * 2 * t) + 20 + 15 * np.random.randn(len(t)),
            "expected_range": "1.4-1.8"
        },
        {
            "name": "Random Walk",
            "description": "Cumulative random movements (fractional Brownian motion proxy)",
            "generator": lambda t: np.cumsum(np.random.randn(len(t))),
            "expected_range": "1.3-1.7"
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"Test {i+1}: {case['name']}")
        print(f"Description: {case['description']}")
        print(f"Expected range: {case['expected_range']}")
        
        # Generate test data
        np.random.seed(42 + i)  # Different seed for each case
        t = np.linspace(0, 2, 500)  # 2 seconds, 250 Hz sampling
        x = case["generator"](t)
        
        # Calculate Hausdorff dimension
        D, debug_info = hausdorff_dimension_boxcount(t, x, return_debug=True)
        
        print(f"Result: D = {D:.3f}")
        print(f"R² = {debug_info['r2']:.3f}")
        print(f"Number of points: {debug_info['n_points']}")
        print(f"Scales used: {len(debug_info['scales'])}")
        print("-" * 50)
        
        results.append({
            "name": case["name"],
            "t": t,
            "x": x,
            "D": D,
            "debug_info": debug_info
        })
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        ax = axes[i]
        
        # Plot time series
        ax.plot(result["t"], result["x"], 'b-', linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Force [%BW]")
        ax.set_title(f"{result['name']}\nD = {result['D']:.3f}, R² = {result['debug_info']['r2']:.3f}")
        ax.grid(True, alpha=0.3)
        
        # Add log-log plot inset
        if result['debug_info']['r2'] > 0.5:  # Only if fit is reasonable
            ax2 = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
            log_eps = result['debug_info']['log_eps']
            log_counts = result['debug_info']['log_counts']
            slope = result['debug_info']['slope']
            r2 = result['debug_info']['r2']
            fit_range = result['debug_info']['fit_range']
            
            ax2.plot(log_eps, log_counts, 'ko', markersize=3, alpha=0.7)
            if fit_range[1] > fit_range[0]:
                ax2.plot(log_eps[fit_range[0]:fit_range[1]], 
                       log_counts[fit_range[0]:fit_range[1]], 
                       'ro', markersize=4)
            
            # Plot fitted line
            x_fit = np.array(log_eps[fit_range[0]:fit_range[1]])
            y_fit = slope * x_fit + result['debug_info']['intercept']
            ax2.plot(x_fit, y_fit, 'r--', linewidth=1.5)
            
            ax2.set_xlabel("log(1/ε)")
            ax2.set_ylabel("log N(ε)")
            ax2.set_title(f"Box-counting\nD={slope:.2f}, R²={r2:.2f}")
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("hausdorff_demo.png", dpi=150, bbox_inches='tight')
    print(f"\nDemo plot saved as 'hausdorff_demo.png'")
    
    # Summary
    print("\n=== Summary ===")
    print("Hausdorff dimension interpretation for climbing force signals:")
    print("- D ≈ 1.0-1.3: Smooth, skillful force application")
    print("- D ≈ 1.3-1.6: Moderate complexity, some variation")
    print("- D ≈ 1.6-1.8: High complexity, less controlled movements")
    print("- D > 1.8: Very irregular patterns (may indicate issues)")
    print("\nNote: Lower dimensions generally indicate more controlled, skillful climbing technique.")

if __name__ == "__main__":
    demo_hausdorff_calculation()
