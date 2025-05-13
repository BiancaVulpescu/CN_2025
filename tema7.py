import numpy as np
import matplotlib.pyplot as plt

def polynomial_value(coeffs, x):
    """Evaluate polynomial using Horner's method.
    coeffs should be in form [a0, a1, a2, ..., an] representing a0*x^n + a1*x^(n-1) + ... + an
    """
    result = 0
    for coef in coeffs:
        result = result * x + coef
    return result

def polynomial_derivative(coeffs):
    """Calculate the derivative of a polynomial.
    coeffs should be in form [a0, a1, a2, ..., an]
    """
    n = len(coeffs) - 1
    
    if n == 0:
        return [0]
    
    # For polynomial a0*x^n + a1*x^(n-1) + ... + an
    # Derivative is n*a0*x^(n-1) + (n-1)*a1*x^(n-2) + ... + a_(n-1)
    derivative_coeffs = [(n-i) * coeffs[i] for i in range(n)]
    
    return derivative_coeffs

def polynomial_second_derivative(coeffs):
    """Calculate the second derivative of a polynomial."""
    return polynomial_derivative(polynomial_derivative(coeffs))

def halley_method(coeffs, x0, epsilon=1e-10, k_max=1000):
    """
    Implement Halley's method for finding roots of polynomials.
    
    Parameters:
    - coeffs: List of polynomial coefficients [a0, a1, ..., an] for a0*x^n + a1*x^(n-1) + ... + an
    - x0: Initial starting point
    - epsilon: Precision
    - k_max: Maximum number of iterations
    
    Returns:
    - x: Approximated root
    - iterations: Number of iterations performed
    - converged: Boolean indicating if method converged
    """
    x = x0
    iterations = 0
    
    while iterations < k_max:
        # Calculate polynomial value and its derivatives
        p_x = polynomial_value(coeffs, x)
        dp_x = polynomial_value(polynomial_derivative(coeffs), x)
        d2p_x = polynomial_value(polynomial_second_derivative(coeffs), x)
        
        # Check if we're already at a root (within epsilon)
        if abs(p_x) < epsilon:
            return x, iterations, True
        
        # Calculate denominator for Halley's method
        denominator = 2 * dp_x**2 - p_x * d2p_x
        
        # Check for potential division by zero
        if abs(denominator) < epsilon:
            return x, iterations, False
        
        # Halley's formula
        delta = (2 * p_x * dp_x) / denominator
        x = x - delta
        iterations += 1
        
        # Check for convergence
        if abs(delta) < epsilon:
            return x, iterations, True
    
    # If we reach here, we didn't converge within k_max iterations
    return x, iterations, False

def compute_bounds_R(coeffs):
    """
    Compute the bound R such that all real roots lie in [-R, R].
    Using a simple bound based on coefficient ratios.
    """
    a0 = coeffs[0]  # Leading coefficient
    
    # Compute max absolute value of all other coefficients
    other_coeffs = [abs(coef) for coef in coeffs[1:]]
    if not other_coeffs:
        return 1  # Constant polynomial
        
    max_coef = max(other_coeffs)
    
    # A simple bound: 1 + max(|ai/a0|) for i=1..n
    R = 1 + max_coef / abs(a0)
    return R

def find_all_real_roots(coeffs, epsilon=1e-10, k_max=1000, num_starting_points=20):
    """
    Find all real roots of a polynomial in interval [-R, R] using Halley's method
    with multiple starting points.
    
    Parameters:
    - coeffs: Polynomial coefficients [a0, a1, ..., an]
    - epsilon: Precision
    - k_max: Maximum iterations
    - num_starting_points: Number of initial points to try
    
    Returns:
    - List of distinct real roots found
    """
    # Compute bound R where all real roots lie in [-R, R]
    R = compute_bounds_R(coeffs)
    print(f"Searching for roots in interval [-{R:.6f}, {R:.6f}]")
    
    # Generate starting points
    starting_points = np.linspace(-R, R, num_starting_points)
    
    # Find roots
    all_roots = []  # All roots found, including duplicates
    all_starting_points = []  # Corresponding starting points
    all_iterations = []  # Corresponding iteration counts
    
    for x0 in starting_points:
        root, iterations, converged = halley_method(coeffs, x0, epsilon, k_max)
        
        if converged:
            all_roots.append(root)
            all_starting_points.append(x0)
            all_iterations.append(iterations)
            print(f"Found root {root:.10f} starting from x0 = {x0:.6f} after {iterations} iterations")
    
    # Group roots by their distinct values (considering epsilon)
    distinct_roots = []
    distinct_starting_points = []
    distinct_iterations = []
    
    # For each root found
    for i, root in enumerate(all_roots):
        # Check if this root is already represented in distinct_roots
        found_similar = False
        for j, existing_root in enumerate(distinct_roots):
            if abs(root - existing_root) <= epsilon:
                found_similar = True
                break
        
        # If no similar root was found, add this one as a new distinct root
        if not found_similar:
            distinct_roots.append(root)
            distinct_starting_points.append(all_starting_points[i])
            distinct_iterations.append(all_iterations[i])
    
    print(f"\nFound {len(all_roots)} roots in total, {len(distinct_roots)} of which are distinct")
    
    return distinct_roots, distinct_starting_points, distinct_iterations

def save_roots_to_file(roots, filename='radacini.txt'):
    """
    Save distinct roots to a file.
    
    Each root is written on a separate line with high precision.
    """
    with open(filename, "w") as f:
        for root in roots:
            f.write(f"{root:.10f}\n")
    
    print(f"Cele {len(roots)} rădăcini distincte au fost salvate în {filename}")

def main():
    # Set precision and maximum iterations as per requirements
    p = 10  # Precision parameter
    epsilon = 10**(-p)  # ε = 10^(-p)
    k_max = 1000  # Maximum iterations
    
    print(f"Folosind precizia ε = {epsilon} și numărul maxim de iterații k_max = {k_max}")
    
    # List of test polynomials
    test_polynomials = [
        {"name": "P1(x) = x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)", 
         "coeffs": [1, -6, 11, -6]},
        {"name": "P2(x) = 42x⁴-55x³- 42x² + 49x-6 ", 
         "coeffs": [42, -55, -42, 49, -6]},
        {"name": "P3(x) = 8x⁴-38x³ +49x² -22x +3 ", 
         "coeffs": [8, -38, 49, -22, 3]},
        {"name": "P4(x) = x⁴ - 6x³ + 13x² - 12x + 4", 
         "coeffs": [1, -6, 13, -12, 4]}
    ]
    
    for i, poly in enumerate(test_polynomials, 1):
        print(f"\n\n{'='*50}")
        print(f"Testare polinom {i}: {poly['name']}")
        print(f"{'='*50}")
        
        roots, starting_points, iterations = find_all_real_roots(
            poly['coeffs'], 
            epsilon=epsilon, 
            k_max=k_max, 
            num_starting_points=50  
        )
        
        print(f"\nRădăcini distincte găsite: {len(roots)}")
        for j, root in enumerate(roots):
            print(f"Rădăcina {j+1}: {root:.10f} (găsită pornind de la x0 = {starting_points[j]:.6f} în {iterations[j]} iterații)")
        
        save_roots_to_file(roots, filename=f"radacini_poly_{i}.txt")
        
        # if i == 1:  # testez doar pt primul polinom
        #     specific_x0 = 1.5
        #     print(f"\nTestat metoda lui Halley cu punctul de start specific x0 = {specific_x0}")
        #     root, iterations, converged = halley_method(poly['coeffs'], specific_x0, epsilon, k_max)
        #     print(f"Rădăcină: {root:.10f}, Iterații: {iterations}, Convergență: {converged}")

if __name__ == "__main__":
    main()