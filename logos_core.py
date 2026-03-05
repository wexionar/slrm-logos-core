"""
================================================================================
LOGOS CORE v1.0 - 1D Geometric Inference Engine for SLRM
================================================================================

SLRM Team: Alex · Gemini · ChatGPT · Claude · Grok · Meta AI
License: MIT
Version: v1.0

================================================================================
"""

import numpy as np

def logos_core_engine(dataset_x, dataset_y, query_x, R=0):
    """
    LOGOS CORE: 1D Geometric Inference Engine (Glass Box AI)

    Parameters:
    - dataset_x: Ordered array of independent variables (X) - must be strictly increasing
    - dataset_y: Array of dependent variables (Y) - must match length of dataset_x
    - query_x: The value to infer
    - R: Quality Radius (0 = disabled). Maximum allowed distance to support points.
    
    Returns:
    - (result, status_message) tuple where:
        - result: Inferred Y value (or None if error)
        - status_message: Detailed explanation of the inference process
    """

    # --- 0. INPUT VALIDATION ---
    if len(dataset_x) != len(dataset_y):
        return None, "ERROR: dataset_x and dataset_y must have same length"
    
    if len(dataset_x) < 2:
        return None, "ERROR: Dataset must contain at least 2 points"
    
    if not np.all(np.diff(dataset_x) > 0):
        return None, "ERROR: dataset_x must be strictly increasing (sorted)"
    
    if np.isnan(query_x) or np.isinf(query_x):
        return None, "ERROR: query_x must be a finite number"
    
    if R < 0:
        return None, "ERROR: R must be non-negative"

    # --- 1. BOUNDARY VALIDATION (Extrapolation Check) ---
    if query_x < dataset_x[0] or query_x > dataset_x[-1]:
        return None, f"ERROR: Extrapolation prohibited. Query X={query_x:.4f} outside Dataset Universe [{dataset_x[0]:.4f}, {dataset_x[-1]:.4f}]"

    # --- 2. SINGLE SEARCH (O(log N) Efficiency) ---
    # idx_sup is the index of first element > query_x (or len if query >= last element)
    idx_sup = np.searchsorted(dataset_x, query_x, side='right')

    # --- 3. EXACT MATCH CHECK ---
    # Check if query matches the element just before idx_sup
    if idx_sup > 0 and dataset_x[idx_sup - 1] == query_x:
        return dataset_y[idx_sup - 1], f"SUCCESS: Exact match found at X={query_x:.4f}"
    
    # --- 4. BOUNDARY CHECK (After exact match check) ---
    # If idx_sup == 0, query is below dataset (already caught by boundary validation)
    # If idx_sup == len, query is at or above last element
    if idx_sup == len(dataset_x):
        # This should only happen if query_x >= dataset_x[-1]
        # But we already checked query_x <= dataset_x[-1] in boundary validation
        # So this means query_x == dataset_x[-1] exactly
        return dataset_y[-1], f"SUCCESS: Exact match at upper boundary X={dataset_x[-1]:.4f}"

    # If we reach here, the point lies strictly between two indices
    idx_inf = idx_sup - 1
    x_inf, x_sup = dataset_x[idx_inf], dataset_x[idx_sup]
    y_inf, y_sup = dataset_y[idx_inf], dataset_y[idx_sup]

    # --- 5. QUALITY CONTROL (R-Filter) ---
    if R > 0:
        dist_inf = query_x - x_inf
        dist_sup = x_sup - query_x
        if dist_inf > R or dist_sup > R:
            return None, f"ERROR: Insufficient support quality. Distance exceeds R={R:.4f}. " \
                        f"Support points: X_inf={x_inf:.4f} (dist={dist_inf:.4f}), " \
                        f"X_sup={x_sup:.4f} (dist={dist_sup:.4f})"

    # --- 6. DEGENERACY CHECK ---
    dx = x_sup - x_inf
    if dx < 1e-12:
        return None, f"ERROR: Degenerate segment. X_inf={x_inf:.4f} and X_sup={x_sup:.4f} are numerically identical"

    # --- 7. CALCULATION (Weighted Linear Equation) ---
    # Linear interpolation: Y = Y_inf + slope * (X - X_inf)
    slope = (y_sup - y_inf) / dx
    inferred_z = y_inf + slope * (query_x - x_inf)

    return inferred_z, f"SUCCESS: Linear inference performed between X={x_inf:.4f} and X={x_sup:.4f}, slope={slope:.6f}"


# --- COLAB LABORATORY TEST ---
if __name__ == "__main__":
    print("="*85)
    print("LOGOS CORE v1.0 - Test Suite")
    print("="*85)
    
    # 1. Synthetic Dataset (Example: non-uniform ramp)
    x_data = np.array([0.0, 1.0, 2.0, 4.5, 5.0, 8.0, 10.0])
    y_data = np.array([0.0, 10.0, 20.0, 45.0, 50.0, 80.0, 100.0])

    # 2. Define queries to test all edge cases
    test_cases = [
        (5.0, 0.5, "Exact match (interior)"),
        (0.0, 0.5, "Exact match (lower boundary)"),
        (10.0, 0.5, "Exact match (upper boundary)"),
        (11.0, 0.5, "Extrapolation (above)"),
        (-1.0, 0.5, "Extrapolation (below)"),
        (3.0, 0.5, "R-filter violation (dist to 2.0 is 1.0 > 0.5)"),
        (4.7, 1.0, "Successful inference (between 4.5 and 5.0)"),
        (2.5, 0, "No R-filter (R=0)"),
    ]

    print(f"\n{'QUERY':<12} | {'R':<5} | {'RESULT':<15} | {'TEST CASE':<35} | {'STATUS'}")
    print("-" * 120)

    for q_x, q_r, description in test_cases:
        result, status = logos_core_engine(x_data, y_data, q_x, R=q_r)
        res_display = f"{result:.4f}" if result is not None else "None"
        status_short = status.split(':')[0]  # Just show ERROR or SUCCESS
        print(f"{q_x:<12.2f} | {q_r:<5.1f} | {res_display:<15} | {description:<35} | {status_short}")
    
    print("\n" + "="*85)
    print("Test complete. Check results above.")
    print("="*85)
 
