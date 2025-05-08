import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

def polynomial_value(coeffs, x):
    """Evaluate polynomial P(x) using Horner's method"""
    result = 0
    for coef in coeffs:
        result = result * x + coef
    return result

def polynomial_derivative(coeffs):
    """Calculate derivative of polynomial P'(x)"""
    n = len(coeffs) - 1
    return [i * coeffs[n-i] for i in range(1, n+1)] + [0]

def polynomial_second_derivative(coeffs):
    """Calculate second derivative of polynomial P''(x)"""
    deriv = polynomial_derivative(coeffs)
    return polynomial_derivative(deriv)

def halley_method(coeffs, x0, epsilon=1e-6, k_max=1000):
    """
    Halley's method to find a root of the polynomial
    
    Parameters:
    coeffs - list of polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
    x0 - initial guess
    epsilon - desired precision
    k_max - maximum number of iterations
    
    Returns:
    x - approximated root
    iterations - number of iterations performed
    convergence - True if method converged, False otherwise
    """
    x = x0
    iterations = 0
    delta = float('inf')  # Initialize delta
    
    # Continue while |Δ| ≥ ε AND k ≤ k_max AND |Δ| ≤ 10^8
    while abs(delta) >= epsilon and iterations < k_max and abs(delta) <= 1e8:
        # Calculate polynomial and its derivatives at x
        p_x = polynomial_value(coeffs, x)
        dp_x = polynomial_value(polynomial_derivative(coeffs), x)
        d2p_x = polynomial_value(polynomial_second_derivative(coeffs), x)
        
        # Calculate A = 2*(P'(x))^2 - P(x)*P''(x)
        A = 2 * dp_x**2 - p_x * d2p_x
        
        # Check for early stopping
        if abs(A) < epsilon:
            return x, iterations, True
            
        # Calculate delta = 2*P(x)*P'(x)/A
        delta = 2 * p_x * dp_x / A
        
        # Update x
        x = x - delta
        iterations += 1
    
    # Check if convergence was achieved
    if abs(delta) < epsilon:
        return x, iterations, True
    else:
        return x, iterations, False

def find_all_real_roots(coeffs, interval=(-10, 10), num_initial_points=20, epsilon=1e-6, k_max=1000):
    """
    Find all real roots of a polynomial in a given interval
    
    Parameters:
    coeffs - list of polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
    interval - tuple (a, b) for search interval [a, b]
    num_initial_points - number of initial guesses to try
    epsilon - desired precision
    k_max - maximum number of iterations
    
    Returns:
    roots - list of distinct real roots
    """
    a, b = interval
    initial_points = np.linspace(a, b, num_initial_points)
    
    # Store found roots and their convergence status
    found_roots = []
    
    for x0 in initial_points:
        root, iterations, converged = halley_method(coeffs, x0)
        
        if converged:
            # Check if this root is already found (within epsilon precision)
            is_new_root = True
            for existing_root in found_roots:
                if abs(root - existing_root) < epsilon:
                    is_new_root = False
                    break
                    
            if is_new_root:
                found_roots.append(root)
    
    return sorted(found_roots)

def compute_bounds_R(coeffs):
    # a_n = coeffs[0]  # Leading coefficient
    a_0 = coeffs[-1]  # Constant term
    
    # Find A = max{|ai| : i = 1..n}
    A = max(abs(coef) for coef in coeffs[1:-1]) if len(coeffs) > 2 else 0
    
    R = (abs(a_0) + A) / abs(a_0)
    return R

if __name__ == "__main__":
    # Exemplu: P(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    coeffs = [1, -6, 11, -6]  # [a_n, a_{n-1}, ..., a_1, a_0]
    
    R = compute_bounds_R(coeffs)
    print(f"Toate radacinile reale se gasesc in intervalul: [-{R}, {R}]")
    
    # Gasim radacinile reale din intervalul [-R, R]
    roots = find_all_real_roots(coeffs, interval=(-R, R), epsilon=1e-6, k_max=1000)
    print(f"Am gasit {len(roots)} radacinile reale: {roots}")
    
    # Test the halley method on a specific initial point
    x0 = 1.5
    root, iterations, converged = halley_method(coeffs, x0)
    print(f"Starting from x0 = {x0}")
    print(f"Approximated root: {root}")
    print(f"Number of iterations: {iterations}")
    print(f"Convergence: {converged}")
    
    # Plot the polynomial and its roots
    x = np.linspace(-R, R, 1000)
    p = np.polynomial.Polynomial(coeffs[::-1])  # Note: numpy uses reversed order
    y = p(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.scatter(roots, [0]*len(roots), color='red', s=50)
    plt.title(f"Polynomial P(x) and its real roots")
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.ylim(-10, 10)  # Adjust as needed
    
    # Show the plot
    plt.tight_layout()
    plt.show()