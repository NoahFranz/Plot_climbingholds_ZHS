"""
Test suite for Hausdorff dimension calculation.

Tests the robust box-counting implementation with synthetic data and edge cases.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from additional_calculations import hausdorff_dimension_boxcount


class TestHausdorffDimension:
    """Test cases for Hausdorff dimension calculation."""
    
    def test_straight_line(self):
        """Test straight line: should give D ≈ 1.00 (±0.03)."""
        t = np.linspace(0, 1, 1000)
        x = 2 * t + 1  # Linear function
        
        D = hausdorff_dimension_boxcount(t, x)
        
        assert not np.isnan(D)
        assert 0.97 <= D <= 1.03, f"Expected D ≈ 1.00, got {D:.3f}"
    
    def test_constant_signal(self):
        """Test constant signal: should return 1.0."""
        t = np.linspace(0, 1, 100)
        x = np.ones(100) * 5.0  # Constant signal
        
        D = hausdorff_dimension_boxcount(t, x)
        
        assert D == 1.0, f"Expected D = 1.0, got {D}"
    
    def test_random_walk(self):
        """Test random walk: should give D ≈ 1.5 (±0.2)."""
        np.random.seed(42)  # For reproducibility
        t = np.linspace(0, 1, 1000)
        # Cumulative sum of N(0,1) - proxy for fractional Brownian motion
        x = np.cumsum(np.random.randn(1000))
        
        D = hausdorff_dimension_boxcount(t, x)
        
        assert not np.isnan(D)
        assert 1.3 <= D <= 1.7, f"Expected D ≈ 1.5, got {D:.3f}"
    
    def test_white_noise_large_n(self):
        """Test white noise with large N: should approach D ≈ 2.0 (±0.5)."""
        np.random.seed(42)
        t = np.linspace(0, 1, 5000)
        x = np.random.randn(5000)  # White noise
        
        D = hausdorff_dimension_boxcount(t, x)
        
        assert not np.isnan(D)
        assert 1.5 <= D <= 2.5, f"Expected D ≈ 2.0, got {D:.3f}"
    
    def test_white_noise_small_n(self):
        """Test white noise with small N: should be looser tolerance."""
        np.random.seed(42)
        t = np.linspace(0, 1, 500)
        x = np.random.randn(500)  # White noise
        
        D = hausdorff_dimension_boxcount(t, x)
        
        assert not np.isnan(D)
        assert 1.5 <= D <= 2.5, f"Expected D ≈ 2.0 (loose), got {D:.3f}"
    
    def test_short_segment(self):
        """Test short segment: should return np.nan."""
        t = np.linspace(0, 1, 30)  # Less than min_points=64
        x = np.random.randn(30)
        
        D = hausdorff_dimension_boxcount(t, x, min_points=64)
        
        assert np.isnan(D), f"Expected np.nan for short segment, got {D}"
    
    def test_length_mismatch(self):
        """Test length mismatch: should return np.nan."""
        t = np.linspace(0, 1, 100)
        x = np.random.randn(99)  # Different length
        
        D = hausdorff_dimension_boxcount(t, x)
        
        assert np.isnan(D), f"Expected np.nan for length mismatch, got {D}"
    
    def test_affine_invariance(self):
        """Test invariance to affine scaling when normalize=True."""
        t = np.linspace(0, 1, 500)
        x = np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(500)
        
        # Original
        D1 = hausdorff_dimension_boxcount(t, x, normalize=True)
        
        # Scaled and shifted
        t_scaled = 10 * t + 5
        x_scaled = 3 * x + 2
        
        D2 = hausdorff_dimension_boxcount(t_scaled, x_scaled, normalize=True)
        
        assert not np.isnan(D1) and not np.isnan(D2)
        assert abs(D1 - D2) < 0.05, f"Affine invariance failed: {D1:.3f} vs {D2:.3f}"
    
    def test_detrend_option(self):
        """Test detrend option."""
        t = np.linspace(0, 1, 500)
        # Signal with linear trend
        x = np.sin(2 * np.pi * 3 * t) + 2 * t + 0.1 * np.random.randn(500)
        
        D_no_detrend = hausdorff_dimension_boxcount(t, x, detrend=False)
        D_detrend = hausdorff_dimension_boxcount(t, x, detrend=True)
        
        assert not np.isnan(D_no_detrend) and not np.isnan(D_detrend)
        # Detrended should be closer to 1.0 for oscillatory signal
        assert abs(D_detrend - 1.0) < abs(D_no_detrend - 1.0)
    
    def test_robust_regression_fallback(self):
        """Test fallback to OLS when robust methods fail."""
        t = np.linspace(0, 1, 200)
        x = np.sin(2 * np.pi * 2 * t) + 0.05 * np.random.randn(200)
        
        # Should work with OLS fallback
        D = hausdorff_dimension_boxcount(t, x, robust_fit="ols")
        
        assert not np.isnan(D)
        assert 0.8 <= D <= 1.5, f"Expected reasonable D, got {D:.3f}"
    
    def test_debug_output(self):
        """Test debug output functionality."""
        t = np.linspace(0, 1, 200)
        x = np.sin(2 * np.pi * 2 * t)
        
        D, debug_info = hausdorff_dimension_boxcount(t, x, return_debug=True)
        
        assert not np.isnan(D)
        assert isinstance(debug_info, dict)
        assert "slope" in debug_info
        assert "r2" in debug_info
        assert "scales" in debug_info
        assert "counts" in debug_info
        assert debug_info["r2"] > 0.8  # Good fit for clean sine wave
    
    def test_scale_mode_options(self):
        """Test different scale generation modes."""
        t = np.linspace(0, 1, 300)
        x = np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(300)
        
        D_auto = hausdorff_dimension_boxcount(t, x, scale_mode="auto")
        D_logspace = hausdorff_dimension_boxcount(t, x, scale_mode="logspace")
        
        assert not np.isnan(D_auto) and not np.isnan(D_logspace)
        # Should be reasonably close
        assert abs(D_auto - D_logspace) < 0.2
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Single point
        D = hausdorff_dimension_boxcount(np.array([0]), np.array([1]))
        assert np.isnan(D)
        
        # Two points
        D = hausdorff_dimension_boxcount(np.array([0, 1]), np.array([1, 2]))
        assert np.isnan(D)  # Should be too few points
        
        # All zeros
        t = np.linspace(0, 1, 100)
        x = np.zeros(100)
        D = hausdorff_dimension_boxcount(t, x)
        assert D == 1.0
    
    def test_performance(self):
        """Test performance for typical interval sizes."""
        import time
        
        t = np.linspace(0, 1, 500)  # Typical interval size
        x = np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(500)
        
        start_time = time.time()
        D = hausdorff_dimension_boxcount(t, x)
        end_time = time.time()
        
        assert not np.isnan(D)
        # Should complete in milliseconds
        assert (end_time - start_time) < 0.1, f"Too slow: {(end_time - start_time)*1000:.1f}ms"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
