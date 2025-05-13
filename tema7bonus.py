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

def method_N4(coeffs, x0, epsilon=1e-10, k_max=1000):
    x = x0
    iterations = 0
    
    while iterations < k_max:
        fx = polynomial_value(coeffs, x)
        
        if abs(fx) < epsilon:
            return x, iterations, True
        
        dfx = polynomial_value(polynomial_derivative(coeffs), x)
        
        if abs(dfx) < epsilon:
            return x, iterations, False
        
        y = x - fx / dfx
        fy = polynomial_value(coeffs, y)
        
        denominator = dfx * (fx - fy)
        if abs(denominator) < epsilon:
            return x, iterations, False
        
        x_new = x - (fx**2 + fy**2) / denominator
        
        if abs(x_new - x) < epsilon:
            return x_new, iterations, True
        
        x = x_new
        iterations += 1
    
    return x, iterations, False

def method_N5(coeffs, x0, epsilon=1e-10, k_max=1000):
    x = x0
    iterations = 0
    
    while iterations < k_max:
        fx = polynomial_value(coeffs, x)
        
        if abs(fx) < epsilon:
            return x, iterations, True
        
        dfx = polynomial_value(polynomial_derivative(coeffs), x)
        
        if abs(dfx) < epsilon:
            return x, iterations, False
        
        y = x - fx / dfx
        fy = polynomial_value(coeffs, y)
        
        denominator = dfx * (fx - fy)
        if abs(denominator) < epsilon:
            return x, iterations, False
        
        z = x - (fx**2 + fy**2) / denominator
        fz = polynomial_value(coeffs, z)
        
        if abs(fz) < epsilon:
            return z, iterations, True
        
        x_new = z - fz / dfx
        
        if abs(x_new - x) < epsilon:
            return x_new, iterations, True
        
        x = x_new
        iterations += 1
    
    return x, iterations, False

def compute_bounds_R(coeffs):
    a0 = coeffs[0]  
    
    other_coeffs = [abs(coef) for coef in coeffs[1:]]
    if not other_coeffs:
        return 1  
        
    max_coef = max(other_coeffs)
    
    R = (abs(a0) + max_coef )/ abs(a0)
    return R

def find_all_real_roots(coeffs, method, epsilon=1e-10, k_max=1000, num_starting_points=50):
    
    R = compute_bounds_R(coeffs)
    print(f"Searching for roots in interval [-{R:.6f}, {R:.6f}]")
    
    starting_points = np.linspace(-R, R, num_starting_points)
    
    all_roots = []  
    all_starting_points = [] 
    all_iterations = []  
    
    for x0 in starting_points:
        root, iterations, converged = method(coeffs, x0, epsilon, k_max)
        
        if converged and abs(polynomial_value(coeffs, root)) < epsilon:
            all_roots.append(root)
            all_starting_points.append(x0)
            all_iterations.append(iterations)
    
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
    
    # Sort roots for cleaner output
    indices = np.argsort(distinct_roots)
    distinct_roots = [distinct_roots[i] for i in indices]
    distinct_starting_points = [distinct_starting_points[i] for i in indices]
    distinct_iterations = [distinct_iterations[i] for i in indices]
    
    return distinct_roots, distinct_starting_points, distinct_iterations

def save_roots_to_file(roots, filename):
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
        
        print(f"\nUsing Method N4:")
        roots_n4, starting_points_n4, iterations_n4 = find_all_real_roots(
            poly['coeffs'], 
            method_N4, 
            epsilon=epsilon, 
            k_max=k_max, 
            num_starting_points=50  
        )
        
        print(f"\nRădăcini distincte găsite cu N4: {len(roots_n4)}")
        for j, root in enumerate(roots_n4):
            print(f"Rădăcina {j+1}: {root:.10f} (găsită pornind de la x0 = {starting_points_n4[j]:.6f} în {iterations_n4[j]} iterații)")
        
        save_roots_to_file(roots_n4, filename=f"radacini_n4_poly_{i}.txt")
        
        print(f"\nUsing Method N5:")
        roots_n5, starting_points_n5, iterations_n5 = find_all_real_roots(
            poly['coeffs'], 
            method_N5, 
            epsilon=epsilon, 
            k_max=k_max, 
            num_starting_points=50  
        )
        
        print(f"\nRădăcini distincte găsite cu N5: {len(roots_n5)}")
        for j, root in enumerate(roots_n5):
            print(f"Rădăcina {j+1}: {root:.10f} (găsită pornind de la x0 = {starting_points_n5[j]:.6f} în {iterations_n5[j]} iterații)")
        
        save_roots_to_file(roots_n5, filename=f"radacini_n5_poly_{i}.txt")

if __name__ == "__main__":
    main()