import numpy as np
import matplotlib.pyplot as plt

def least_squares_polynomial(x_points, y_points, m):
    """
    Calculate polynomial approximation using least squares method
    
    Parameters:
        x_points: list of x coordinates (n+1 points)
        y_points: list of y values at x_points
        m: degree of the polynomial
    
    Returns:
        coefficients of the polynomial (a₀, a₁, ..., aₘ)
    """
    n = len(x_points) - 1
    
    # Build the matrix B
    B = np.zeros((m+1, m+1))
    for i in range(m+1):
        for j in range(m+1):
            B[i, j] = sum(x_points[k] ** (i+j) for k in range(n+1))
    
    # Build the right side vector f
    f = np.zeros(m+1)
    for i in range(m+1):
        f[i] = sum(y_points[k] * (x_points[k] ** i) for k in range(n+1))
    
    # Solve the linear system Ba = f
    a = np.linalg.solve(B, f)
    
    return a[::-1]  # Return in descending order (aₘ, aₘ₋₁, ..., a₁, a₀)

def horner_eval(coeffs, x):
    """
    Evaluate polynomial using Horner's scheme
    
    Parameters:
        coeffs: list of coefficients [aₘ, aₘ₋₁, ..., a₁, a₀]
        x: point where to evaluate the polynomial
    
    Returns:
        value of polynomial at point x
    """
    result = 0
    for coef in coeffs:
        result = result * x + coef
    return result

def polynomial_string(coeffs):
    """Generate a readable string representation of the polynomial"""
    m = len(coeffs) - 1
    terms = []
    
    for i, a in enumerate(coeffs):
        power = m - i
        if a == 0:
            continue
            
        if power == 0:
            terms.append(f"{a:.4f}")
        elif power == 1:
            terms.append(f"{a:.4f}x")
        else:
            terms.append(f"{a:.4f}x^{power}")
            
    return " + ".join(terms).replace("+ -", "- ")

def polynomial_approximation(x_points, y_points, x_eval, m, true_func=None):
    """
    Main function to approximate a function using polynomial least squares
    and evaluate it at a given point
    
    Parameters:
        x_points: list of x coordinates
        y_points: list of y values at x_points
        x_eval: point where to evaluate the approximation
        m: degree of the polynomial
        true_func: actual function (if available)
    """
    # Calculate polynomial coefficients using least squares
    coeffs = least_squares_polynomial(x_points, y_points, m)
    
    # Evaluate polynomial at x_eval using Horner's scheme
    p_eval = horner_eval(coeffs, x_eval)
    
    # Calculate the sum of absolute errors at interpolation points
    errors_sum = sum(abs(horner_eval(coeffs, x) - y) for x, y in zip(x_points, y_points))
    
    # Print results
    print(f"Polynomial degree: {m}")
    print(f"Polynomial: P(x) = {polynomial_string(coeffs)}")
    print(f"P({x_eval}) = {p_eval:.6f}")
    
    if true_func:
        true_value = true_func(x_eval)
        print(f"|P({x_eval}) - f({x_eval})| = |{p_eval:.6f} - {true_value:.6f}| = {abs(p_eval - true_value):.6f}")
    
    print(f"Sum of |P(xi) - yi| = {errors_sum:.6f}")
    
    # Plot the results
    plot_approximation(x_points, y_points, coeffs, true_func)
    
    return coeffs, p_eval, errors_sum

def trigonometric_interpolation(x_points, y_points, x_eval, true_func=None):
    """
    Approximates periodic functions using trigonometric interpolation
    
    Parameters:
        x_points: list of x coordinates (2m+1 points in [0, 2π])
        y_points: list of y values at x_points
        x_eval: point where to evaluate the approximation
        true_func: actual function (if available)
    
    Returns:
        coefficients, approximation at x_eval, error sum
    """
    n = len(x_points) - 1
    if n % 2 != 0:
        raise ValueError("Number of points must be odd (n = 2m)")
    
    m = n // 2  # m in the formula
    
    # Build the T matrix as described in image 3
    T = np.zeros((n+1, n+1))
    
    # Define the basis functions
    def phi_0(x):
        return 1
    
    def phi_k(x, k):
        return np.cos(k * x)
    
    def phi_2m_k(x, k):
        return np.sin(k * x)
    
    # Fill the T matrix
    for i in range(n+1):  # For each row (point)
        # First column is always 1
        T[i, 0] = phi_0(x_points[i])
        
        # Next columns alternate between sin and cos
        for k in range(1, m+1):
            # cos(kx) columns
            T[i, 2*k-1] = phi_k(x_points[i], k)
            # sin(kx) columns
            T[i, 2*k] = phi_2m_k(x_points[i], k)
    
    # Create Y vector with function values
    Y = np.array(y_points)
    
    # Solve the system T * X = Y
    X = np.linalg.solve(T, Y)
    
    # Extract coefficients
    a0 = X[0]
    a_coefs = X[1::2][:m]  # take every other element starting from index 1
    b_coefs = X[2::2][:m]  # take every other element starting from index 2
    
    # Function to evaluate trigonometric approximation at a point
    def eval_trig(x):
        result = a0
        for k in range(1, m+1):
            result += a_coefs[k-1] * np.cos(k * x)
            result += b_coefs[k-1] * np.sin(k * x)
        return result
    
    # Evaluate at the requested point
    t_eval = eval_trig(x_eval)
    
    # Calculate sum of absolute errors
    errors_sum = sum(abs(eval_trig(x) - y) for x, y in zip(x_points, y_points))
    
    # Print results
    print(f"Trigonometric interpolation with m = {m}")
    
    # Format the trigonometric series as a string
    trig_string = f"{a0:.4f}"
    for k in range(1, m+1):
        if a_coefs[k-1] != 0:
            term = f" + {a_coefs[k-1]:.4f}cos({k}x)"
            trig_string += term.replace("+ -", "- ")
        if b_coefs[k-1] != 0:
            term = f" + {b_coefs[k-1]:.4f}sin({k}x)"
            trig_string += term.replace("+ -", "- ")
    
    print(f"T(x) = {trig_string}")
    print(f"T({x_eval}) = {t_eval:.6f}")
    
    if true_func:
        true_value = true_func(x_eval)
        print(f"|T({x_eval}) - f({x_eval})| = |{t_eval:.6f} - {true_value:.6f}| = {abs(t_eval - true_value):.6f}")
    
    print(f"Sum of |T(xi) - yi| = {errors_sum:.6f}")
    
    # Plot the results
    plot_trig_approximation(x_points, y_points, a0, a_coefs, b_coefs, m, true_func)
    
    return (a0, a_coefs, b_coefs), t_eval, errors_sum

def plot_trig_approximation(x_points, y_points, a0, a_coefs, b_coefs, m, true_func=None):
    """Plot the original points, the trigonometric approximation and the true function if provided"""
    plt.figure(figsize=(10, 6))
    
    # Plot the original points
    plt.scatter(x_points, y_points, color='red', label='Data points')
    
    # Create a smooth x range for plotting - over the full period
    x_range = np.linspace(0, 2*np.pi, 1000)
    
    # Function to evaluate trigonometric series
    def eval_trig(x):
        result = a0
        for k in range(1, m+1):
            result += a_coefs[k-1] * np.cos(k * x)
            result += b_coefs[k-1] * np.sin(k * x)
        return result
    
    # Calculate trigonometric values
    t_values = [eval_trig(x) for x in x_range]
    plt.plot(x_range, t_values, 'b-', label='Trigonometric approximation')
    
    # Plot the true function if provided
    if true_func:
        true_values = [true_func(x) for x in x_range]
        plt.plot(x_range, true_values, 'g--', label='True function')
    
    plt.title('Function Approximation using Trigonometric Interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_approximation(x_points, y_points, coeffs, true_func=None):
    """Plot the original points, the approximation polynomial and the true function if provided"""
    plt.figure(figsize=(10, 6))
    
    # Plot the original points
    plt.scatter(x_points, y_points, color='red', label='Data points')
    
    # Create a smooth x range for plotting
    x_range = np.linspace(min(x_points), max(x_points), 1000)
    
    # Calculate polynomial values
    p_values = [horner_eval(coeffs, x) for x in x_range]
    plt.plot(x_range, p_values, 'b-', label='Polynomial approximation')
    
    # Plot the true function if provided
    if true_func:
        true_values = [true_func(x) for x in x_range]
        plt.plot(x_range, true_values, 'g--', label='True function')
    
    plt.title('Function Approximation using Least Squares Polynomial')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example 1: f(x) = x⁴ - 12x³ + 30x² + 12 on [1, 5]
def f1(x):
    return x**4 - 12*x**3 + 30*x**2 + 12

# Generate points for example 1
x0, xn = 1, 5
n = 10  # Number of points - 1
x_points = np.linspace(x0, xn, n+1)
y_points = [f1(x) for x in x_points]

# Point to evaluate
x_eval = 3.0

# Run the approximation with different polynomial degrees
print("Example 1: f(x) = x⁴ - 12x³ + 30x² + 12")
print("="*50)
for m in [2, 3, 4, 5]:
    coeffs, p_eval, errors_sum = polynomial_approximation(x_points, y_points, x_eval, m, f1)
    print("="*50)

# Example 2: Trigonometric functions
# Example 2a: f(x) = sin(x) - cos(x) on [0, 2π]
def f2a(x):
    return np.sin(x) - np.cos(x)

# Example 2b: f(x) = sin(2x) + sin(x) + cos(3x) on [0, 2π]
def f2b(x):
    return np.sin(2*x) + np.sin(x) + np.cos(3*x)

# Example 2c: f(x) = sin²(x) - cos²(x) on [0, 2π]
def f2c(x):
    return np.sin(x)**2 - np.cos(x)**2

# For trigonometric interpolation: use odd number of points (n = 2m)
m_trig = 3  # This will give 2m+1 = 7 points
x_points_trig = np.linspace(0, 2*np.pi, 2*m_trig+1, endpoint=False)  # n = 2m points

# First trigonometric example
print("\nExample 2a: f(x) = sin(x) - cos(x)")
print("="*50)
y_points_trig = [f2a(x) for x in x_points_trig]
x_eval_trig = np.pi/4
trigonometric_interpolation(x_points_trig, y_points_trig, x_eval_trig, f2a)
print("="*50)

# Second trigonometric example
print("\nExample 2b: f(x) = sin(2x) + sin(x) + cos(3x)")
print("="*50)
y_points_trig = [f2b(x) for x in x_points_trig]
trigonometric_interpolation(x_points_trig, y_points_trig, x_eval_trig, f2b)
print("="*50)

# Third trigonometric example
print("\nExample 2c: f(x) = sin²(x) - cos²(x)")
print("="*50)
y_points_trig = [f2c(x) for x in x_points_trig]
trigonometric_interpolation(x_points_trig, y_points_trig, x_eval_trig, f2c)
print("="*50)