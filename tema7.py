import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.misc import derivative

class Polynomial:
    def __init__(self, coefficients):
        """
        Initialize a polynomial with coefficients [a₀, a₁, a₂, ..., aₙ]
        where P(x) = a₀xⁿ + a₁xⁿ⁻¹ + ... + aₙ₋₁x + aₙ
        """
        self.coefficients = coefficients
        self.degree = len(coefficients) - 1
    
    def evaluate(self, x):
        """Evaluate polynomial at point x using Horner's method"""
        result = 0
        for coef in self.coefficients:
            result = result * x + coef
        return result
    
    def derivative(self):
        """Calculate the first derivative of the polynomial"""
        derivative_coeffs = []
        for i in range(self.degree):
            derivative_coeffs.append(self.coefficients[i] * (self.degree - i))
        return Polynomial(derivative_coeffs)
    
    def second_derivative(self):
        """Calculate the second derivative of the polynomial"""
        return self.derivative().derivative()

def halley_method(f, df, ddf, x0, epsilon=1e-10, k_max=1000):
    """
    Implementation of Halley's method for finding roots
    
    Parameters:
    - f: Function to find root of
    - df: First derivative of f
    - ddf: Second derivative of f
    - x0: Initial guess
    - epsilon: Precision
    - k_max: Maximum number of iterations
    
    Returns:
    - x: Approximated root
    - k: Number of iterations performed
    - convergence: True if converged, False otherwise
    """
    x = x0
    k = 1
    
    iterations = [(x, f(x))]
    
    while True:
        # Calculate A = 2[f'(x)]² - f(x)f''(x)
        A = 2 * df(x)**2 - f(x) * ddf(x)
        
        # Check if A is too small - algorithm might be unstable
        if abs(A) < epsilon:
            return x, k, True, iterations
        
        # Calculate Δ = 2f(x)f'(x)/A
        delta = 2 * f(x) * df(x) / A
        
        # Update x
        x = x - delta
        
        # Record this iteration
        iterations.append((x, f(x)))
        
        # Increment counter
        k += 1
        
        # Check convergence criteria
        if abs(delta) < epsilon:
            return x, k, True, iterations
        
        # Check maximum iterations or if delta is too large (divergence)
        if k > k_max or abs(delta) > 1e8:
            return x, k, False, iterations

def estimate_root_bounds(poly, padding=1.0):
    """
    Estimate bounds [-R, R] that contain all real roots
    using a simple upper bound formula
    """
    coeffs = poly.coefficients.copy()
    lead_coeff = abs(coeffs[0])
    max_coeff = max(abs(c) for c in coeffs[1:])
    
    # An upper bound for absolute values of all roots
    R = 1 + max_coeff / lead_coeff
    
    # Add some padding
    R = R * padding
    
    return -R, R

def find_all_roots(poly, epsilon=1e-10, k_max=1000, num_start_points=20):
    """
    Find all real roots of a polynomial using Halley's method
    with multiple starting points
    """
    lower_bound, upper_bound = estimate_root_bounds(poly)
    
    # First derivative
    poly_prime = poly.derivative()
    
    # Second derivative
    poly_double_prime = poly.second_derivative()
    
    # Lambda functions for Halley's method
    f = lambda x: poly.evaluate(x)
    df = lambda x: poly_prime.evaluate(x)
    ddf = lambda x: poly_double_prime.evaluate(x)
    
    # Generate starting points
    start_points = np.linspace(lower_bound, upper_bound, num_start_points)
    
    # Find roots from different starting points
    found_roots = []
    convergence_data = []
    
    for x0 in start_points:
        root, iterations, converged, iter_data = halley_method(f, df, ddf, x0, epsilon, k_max)
        
        if converged:
            # Check if this root is already found (within epsilon)
            is_new_root = True
            for existing_root in found_roots:
                if abs(root - existing_root) < epsilon:
                    is_new_root = False
                    break
            
            if is_new_root:
                found_roots.append(root)
                convergence_data.append((x0, root, iterations, iter_data))
    
    # Sort roots
    found_roots.sort()
    
    return found_roots, convergence_data

def plot_polynomial_and_roots(poly, roots, bounds=None, num_points=1000):
    """Plot the polynomial function and mark its roots"""
    if bounds is None:
        bounds = estimate_root_bounds(poly)
    
    x = np.linspace(bounds[0], bounds[1], num_points)
    y = [poly.evaluate(xi) for xi in x]
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label=f"P(x)")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Mark roots
    y_roots = [0] * len(roots)
    plt.scatter(roots, y_roots, color='red', s=50, label='Roots')
    
    plt.grid(True)
    plt.legend()
    plt.title("Polynomial and its roots")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    
    return plt.gcf()

def plot_convergence(convergence_data, bounds=None):
    """Plot convergence paths for each found root"""
    plt.figure(figsize=(12, 6))
    
    for i, (x0, root, iterations, iter_data) in enumerate(convergence_data):
        x_values = [point[0] for point in iter_data]
        y_values = [point[1] for point in iter_data]
        
        plt.plot(x_values, y_values, 'o-', label=f"From x₀={x0:.2f} to root={root:.6f}")
    
    plt.grid(True)
    plt.legend()
    plt.title("Convergence of Halley's Method")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    return plt.gcf()

def test_halley_with_examples():
    """Test Halley's method with the provided examples"""
    examples = [
        # Example 1: P(x) = (x-1)(x-2)(x-3)
        Polynomial([1, -6, 11, -6]),
        
        # Example 2: P(x) = (x-2/3)(x-1/7)(x+1)(x-3/2)
        Polynomial([42, -55, -42, 49, -6]),
        
        # Example 3: P(x) = (x-1)(x-1/2)(x-3)(x-1/4)
        Polynomial([8, -38, 49, -22, 3]),
        
        # Example 4: P(x) = (x-1)²(x-2)²
        Polynomial([1, -6, 13, -12, 4]),
    ]
    
    # Process each example polynomial
    results = []
    
    for i, poly in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Polynomial coefficients: {poly.coefficients}")
        
        # Find all roots
        roots, convergence_data = find_all_roots(poly)
        
        print(f"Found {len(roots)} roots:")
        for j, root in enumerate(roots):
            print(f"  Root {j+1}: x = {root:.10f}, P(x) = {poly.evaluate(root):.10e}")
        
        # Plot polynomial and roots
        fig1 = plot_polynomial_and_roots(poly, roots)
        fig1.savefig(f"example{i+1}_polynomial.png")
        
        # Plot convergence
        fig2 = plot_convergence(convergence_data)
        fig2.savefig(f"example{i+1}_convergence.png")
        
        results.append((roots, convergence_data))
    
    return results

def test_function_example():
    """Test Halley's method with the function example from the assignment"""
    # f(x) = e^x - sin(x)
    f = lambda x: math.exp(x) - math.sin(x)
    df = lambda x: math.exp(x) - math.cos(x)
    ddf = lambda x: math.exp(x) + math.sin(x)
    
    # Starting point from the example
    x0 = -3.0
    
    # Apply Halley's method
    root, iterations, converged, iter_data = halley_method(f, df, ddf, x0)
    
    print("\nFunction example: f(x) = e^x - sin(x)")
    if converged:
        print(f"Found root: x = {root:.10f}")
        print(f"f(x) = {f(root):.10e}")
        print(f"Converged in {iterations} iterations")
    else:
        print("Method did not converge")
    
    # Plot function near the root
    x = np.linspace(-5, 2, 1000)
    y = [f(xi) for xi in x]
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label="f(x) = e^x - sin(x)")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    if converged:
        plt.scatter([root], [0], color='red', s=50, label=f'Root: {root:.6f}')
    
    plt.grid(True)
    plt.legend()
    plt.title("Function and its root")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig("function_example.png")
    
    return root, iterations, converged

def save_results_to_file(polynomial_results, function_result):
    """Save results to a file"""
    with open("halley_method_results.txt", "w") as f:
        f.write("HALLEY'S METHOD RESULTS\n")
        f.write("======================\n\n")
        
        # Polynomial examples
        for i, (roots, _) in enumerate(polynomial_results):
            f.write(f"Example {i+1}:\n")
            f.write(f"Found {len(roots)} roots:\n")
            for j, root in enumerate(roots):
                f.write(f"  Root {j+1}: x = {root:.10f}\n")
            f.write("\n")
        
        # Function example
        root, iterations, converged = function_result
        f.write("Function example: f(x) = e^x - sin(x)\n")
        if converged:
            f.write(f"Found root: x = {root:.10f}\n")
            f.write(f"Converged in {iterations} iterations\n")
        else:
            f.write("Method did not converge\n")

def bonus_implementation():
    """
    Implement N⁴ and N⁵ methods for approximating roots
    as mentioned in the bonus section
    """
    # This would be implemented if needed, based on the article referenced
    pass

def main():
    # Set precision (epsilon) and maximum iterations
    epsilon = 1e-10
    k_max = 1000
    
    print("Testing Halley's method on polynomial examples:")
    polynomial_results = test_halley_with_examples()
    
    print("\nTesting Halley's method on function example:")
    function_result = test_function_example()
    
    # Save results to file
    save_results_to_file(polynomial_results, function_result)
    
    print("\nResults have been saved to 'halley_method_results.txt'")
    print("Plots have been saved as PNG files")

if __name__ == "__main__":
    main()