import numpy as np
import matplotlib.pyplot as plt

def polynomial_value(coeffs, x):
    result = 0
    for coef in coeffs:
        result = result * x + coef
    return result

def polynomial_derivative(coeffs):
    n = len(coeffs) - 1
    
    if n == 0:
        return [0]
    derivative_coeffs = [(n-i) * coeffs[i] for i in range(n)]
    
    return derivative_coeffs

def polynomial_second_derivative(coeffs):
    return polynomial_derivative(polynomial_derivative(coeffs))

def halley_method(coeffs, x0, epsilon=1e-10, k_max=1000):
    x = x0
    iterations = 0
    
    while iterations < k_max:
        p_x = polynomial_value(coeffs, x)
        dp_x = polynomial_value(polynomial_derivative(coeffs), x)
        d2p_x = polynomial_value(polynomial_second_derivative(coeffs), x)
        
        if abs(p_x) < epsilon:
            return x, iterations, True
        
        A = 2 * dp_x**2 - p_x * d2p_x
        
        if abs(A) < epsilon:
            return x, iterations, False
        
        delta = (2 * p_x * dp_x) /A
        x = x - delta
        iterations += 1
        
        if abs(delta) < epsilon and abs(delta) > 1e8:
            return x, iterations, True
    
    return x, iterations, False

def compute_bounds_R(coeffs):
    a0 = coeffs[0]  
    
    other_coeffs = [abs(coef) for coef in coeffs[1:]]
    if not other_coeffs:
        return 1  
        
    max_coef = max(other_coeffs)
    
    R = (abs(a0) + max_coef )/ abs(a0)
    return R

def find_all_real_roots(coeffs, epsilon=1e-10, k_max=1000, num_starting_points=20):
    R = compute_bounds_R(coeffs)
    print(f"Searching for roots in interval [-{R:.6f}, {R:.6f}]")
    
    starting_points = np.linspace(-R, R, num_starting_points)
    
    all_roots = []  
    all_starting_points = [] 
    all_iterations = []  
    
    for x0 in starting_points:
        root, iterations, converged = halley_method(coeffs, x0, epsilon, k_max)
        
        if converged:
            all_roots.append(root)
            all_starting_points.append(x0)
            all_iterations.append(iterations)
            print(f"Found root {root:.10f} starting from x0 = {x0:.6f} after {iterations} iterations")
    
    distinct_roots = []
    distinct_starting_points = []
    distinct_iterations = []
    
    for i, root in enumerate(all_roots):
        found_similar = False
        for j, existing_root in enumerate(distinct_roots):
            if abs(root - existing_root) <= epsilon:
                found_similar = True
                break
        
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
    p = 10  
    epsilon = 10**(-p)  
    k_max = 1000  
    
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