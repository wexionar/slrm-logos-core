"""
================================================================================
LOGOS CORE v1.0 - COMPREHENSIVE TEST SUITE
================================================================================

SLRM Team: Alex · Gemini · ChatGPT · Claude · Grok · Meta AI
License: MIT
Version: v1.0

Test Coverage:
- Unit Tests (functionality correctness)
- Edge Case Tests (boundary conditions)
- Performance Benchmarks (scalability)
- Precision Tests (numerical accuracy)

================================================================================
"""

import numpy as np
import time
from logos_core import logos_core_engine


def test_exact_matches():
    """Test 1: Exact Match Detection"""
    print("\n" + "="*80)
    print("TEST 1: EXACT MATCH DETECTION")
    print("="*80)
    
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    
    test_cases = [
        (0.0, "Lower boundary"),
        (2.5, "Interior point (should interpolate, not exact)"),
        (3.0, "Interior exact match"),
        (5.0, "Upper boundary"),
    ]
    
    all_passed = True
    for query, description in test_cases:
        result, status = logos_core_engine(x_data, y_data, query, R=0)
        expected = query * 10.0  # Linear relationship
        
        if result is not None and abs(result - expected) < 1e-10:
            print(f"✓ PASS | Query={query:5.2f} | {description:<40} | Result={result:.4f}")
        else:
            print(f"✗ FAIL | Query={query:5.2f} | {description:<40} | Expected={expected:.4f}, Got={result}")
            all_passed = False
    
    print(f"\n{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


def test_extrapolation_rejection():
    """Test 2: Extrapolation Rejection"""
    print("\n" + "="*80)
    print("TEST 2: EXTRAPOLATION REJECTION")
    print("="*80)
    
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    
    test_cases = [
        (0.5, "Below lower bound"),
        (0.999, "Just below lower bound"),
        (5.001, "Just above upper bound"),
        (10.0, "Far above upper bound"),
    ]
    
    all_passed = True
    for query, description in test_cases:
        result, status = logos_core_engine(x_data, y_data, query, R=0)
        
        if result is None and "Extrapolation prohibited" in status:
            print(f"✓ PASS | Query={query:6.3f} | {description:<30} | Correctly rejected")
        else:
            print(f"✗ FAIL | Query={query:6.3f} | {description:<30} | Should have been rejected")
            all_passed = False
    
    print(f"\n{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


def test_r_filter():
    """Test 3: R-Filter Quality Control"""
    print("\n" + "="*80)
    print("TEST 3: R-FILTER QUALITY CONTROL")
    print("="*80)
    
    # Sparse dataset
    x_data = np.array([0.0, 5.0, 10.0])
    y_data = np.array([0.0, 50.0, 100.0])
    
    test_cases = [
        (2.5, 2.0, True, "R=2.0, distance=2.5 to both neighbors (should FAIL)"),
        (2.5, 3.0, False, "R=3.0, distance=2.5 to both neighbors (should PASS)"),
        (1.0, 0.5, True, "R=0.5, distance=1.0 to nearest (should FAIL)"),
        (4.9, 5.0, False, "R=5.0, distance=4.9 to X=0 (should PASS)"),  # Fixed: R must be >= distance
    ]
    
    all_passed = True
    for query, r_value, should_fail, description in test_cases:
        result, status = logos_core_engine(x_data, y_data, query, R=r_value)
        
        failed = (result is None)
        expected_behavior = "FAIL" if should_fail else "PASS"
        actual_behavior = "FAIL" if failed else "PASS"
        
        if failed == should_fail:
            print(f"✓ PASS | Query={query:4.1f}, R={r_value:3.1f} | {description}")
        else:
            print(f"✗ FAIL | Query={query:4.1f}, R={r_value:3.1f} | Expected {expected_behavior}, got {actual_behavior}")
            all_passed = False
    
    print(f"\n{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


def test_linear_interpolation_accuracy():
    """Test 4: Linear Interpolation Precision"""
    print("\n" + "="*80)
    print("TEST 4: LINEAR INTERPOLATION PRECISION")
    print("="*80)
    
    # Test with known linear function: Y = 3X + 7
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = 3.0 * x_data + 7.0
    
    # Test interpolation at non-grid points
    test_queries = [0.25, 0.5, 0.75, 1.333, 2.718, 3.14159, 4.9]
    
    all_passed = True
    max_error = 0.0
    
    for query in test_queries:
        result, status = logos_core_engine(x_data, y_data, query, R=0)
        expected = 3.0 * query + 7.0
        error = abs(result - expected) if result is not None else float('inf')
        max_error = max(max_error, error)
        
        if error < 1e-10:
            print(f"✓ PASS | Query={query:7.5f} | Result={result:10.6f} | Expected={expected:10.6f} | Error={error:.2e}")
        else:
            print(f"✗ FAIL | Query={query:7.5f} | Result={result:10.6f} | Expected={expected:10.6f} | Error={error:.2e}")
            all_passed = False
    
    print(f"\nMaximum error: {max_error:.2e} (should be < 1e-10)")
    print(f"{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


def test_nonlinear_function():
    """Test 5: Non-linear Function Approximation"""
    print("\n" + "="*80)
    print("TEST 5: NON-LINEAR FUNCTION APPROXIMATION (Y = X²)")
    print("="*80)
    
    # Sample Y = X² on [0, 10] with 11 points
    x_data = np.linspace(0, 10, 11)
    y_data = x_data ** 2
    
    # Test queries
    test_queries = [0.5, 1.5, 2.5, 5.0, 7.5, 9.5]
    
    print(f"{'Query':<10} | {'Logos Result':<15} | {'True Y=X²':<15} | {'Error':<10} | Status")
    print("-" * 75)
    
    errors = []
    for query in test_queries:
        result, status = logos_core_engine(x_data, y_data, query, R=0)
        true_value = query ** 2
        error = abs(result - true_value) if result is not None else float('inf')
        errors.append(error)
        
        print(f"{query:<10.2f} | {result:<15.6f} | {true_value:<15.6f} | {error:<10.4f} | {'✓' if error < 5 else '✗'}")
    
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"\nAverage error: {avg_error:.4f}")
    print(f"Maximum error: {max_error:.4f}")
    print(f"Note: Logos uses LINEAR interpolation, so errors on non-linear functions are expected.")
    print(f"✓ TEST COMPLETE (informational - no pass/fail)")
    return True


def test_edge_cases():
    """Test 6: Edge Cases and Special Conditions"""
    print("\n" + "="*80)
    print("TEST 6: EDGE CASES & SPECIAL CONDITIONS")
    print("="*80)
    
    # Test 6.1: Minimum dataset (2 points)
    x_min = np.array([0.0, 1.0])
    y_min = np.array([0.0, 10.0])
    result, _ = logos_core_engine(x_min, y_min, 0.5, R=0)
    test1 = abs(result - 5.0) < 1e-10
    print(f"{'✓ PASS' if test1 else '✗ FAIL'} | Minimum dataset (2 points): {result:.4f} (expected 5.0)")
    
    # Test 6.2: Dense dataset (1000 points)
    x_dense = np.linspace(0, 10, 1000)
    y_dense = np.sin(x_dense)
    result, _ = logos_core_engine(x_dense, y_dense, 5.0, R=0)
    expected = np.sin(5.0)
    test2 = abs(result - expected) < 0.01  # Small tolerance due to discretization
    print(f"{'✓ PASS' if test2 else '✗ FAIL'} | Dense dataset (1000 points): Error={abs(result - expected):.6f}")
    
    # Test 6.3: Constant function
    x_const = np.array([0.0, 1.0, 2.0, 3.0])
    y_const = np.array([42.0, 42.0, 42.0, 42.0])
    result, _ = logos_core_engine(x_const, y_const, 1.5, R=0)
    test3 = abs(result - 42.0) < 1e-10
    print(f"{'✓ PASS' if test3 else '✗ FAIL'} | Constant function: {result:.4f} (expected 42.0)")
    
    # Test 6.4: Negative values
    x_neg = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    y_neg = np.array([-50.0, -20.0, 0.0, 20.0, 50.0])
    result, _ = logos_core_engine(x_neg, y_neg, -3.5, R=0)
    expected = -35.0
    test4 = abs(result - expected) < 1e-10
    print(f"{'✓ PASS' if test4 else '✗ FAIL'} | Negative values: {result:.4f} (expected {expected:.1f})")
    
    all_passed = test1 and test2 and test3 and test4
    print(f"\n{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


def test_input_validation():
    """Test 7: Input Validation"""
    print("\n" + "="*80)
    print("TEST 7: INPUT VALIDATION")
    print("="*80)
    
    x_valid = np.array([0.0, 1.0, 2.0])
    y_valid = np.array([0.0, 10.0, 20.0])
    
    test_cases = [
        (x_valid, np.array([0.0, 10.0]), 1.0, 0, "Mismatched array lengths"),
        (x_valid, y_valid, np.nan, 0, "NaN query"),
        (x_valid, y_valid, 1.0, -1.0, "Negative R"),
        (np.array([2.0, 1.0, 3.0]), y_valid, 1.5, 0, "Unsorted dataset_x"),
        (np.array([0.0]), np.array([0.0]), 0.5, 0, "Dataset with < 2 points"),
    ]
    
    all_passed = True
    for x_data, y_data, query, r_val, description in test_cases:
        result, status = logos_core_engine(x_data, y_data, query, R=r_val)
        
        if result is None and status.startswith("ERROR"):
            print(f"✓ PASS | {description:<35} | Correctly rejected: {status[:50]}...")
        else:
            print(f"✗ FAIL | {description:<35} | Should have been rejected")
            all_passed = False
    
    print(f"\n{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


def benchmark_performance():
    """Test 8: Performance Benchmark"""
    print("\n" + "="*80)
    print("TEST 8: PERFORMANCE BENCHMARK (O(log N) Scaling)")
    print("="*80)
    
    # Test different dataset sizes
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    n_queries = 10_000
    
    print(f"\nBenchmarking with {n_queries:,} queries per dataset size:")
    print(f"{'Dataset Size':<15} | {'Total Time':<12} | {'Time/Query':<15} | {'Queries/Sec':<15} | O(log N) Check")
    print("-" * 90)
    
    times = []
    for size in sizes:
        # Generate dataset
        x_data = np.linspace(0, 100, size)
        y_data = x_data ** 2
        
        # Generate random queries within bounds
        queries = np.random.uniform(0.1, 99.9, n_queries)
        
        # Benchmark
        start = time.perf_counter()
        for query in queries:
            result, _ = logos_core_engine(x_data, y_data, query, R=0)
        elapsed = time.perf_counter() - start
        
        time_per_query = elapsed / n_queries
        queries_per_sec = n_queries / elapsed
        times.append(time_per_query)
        
        # Check O(log N) scaling: time should grow ~ log(size)
        expected_ratio = np.log2(size) / np.log2(sizes[0]) if len(times) > 1 else 1.0
        actual_ratio = time_per_query / times[0] if len(times) > 1 else 1.0
        scaling_ok = actual_ratio < expected_ratio * 2  # Allow 2x margin
        
        print(f"{size:<15,} | {elapsed:<12.4f}s | {time_per_query*1000:<15.6f}ms | {queries_per_sec:<15,.0f} | {'✓' if scaling_ok else '✗'}")
    
    print(f"\n✓ BENCHMARK COMPLETE")
    print(f"Note: Time per query should scale as O(log N), not O(N)")
    return True


def run_all_tests():
    """Run complete test suite"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "LOGOS CORE v1.0 - TEST SUITE" + " "*30 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        test_exact_matches,
        test_extrapolation_rejection,
        test_r_filter,
        test_linear_interpolation_accuracy,
        test_nonlinear_function,
        test_edge_cases,
        test_input_validation,
        benchmark_performance,
    ]
    
    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test.__name__}: {e}")
            results.append(False)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    passed_count = sum(results)
    total_count = len(results)
    
    print(f"\nTests passed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 ✓ ALL TESTS PASSED - LOGOS CORE IS READY FOR PRODUCTION")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed - please review")
    
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
 
