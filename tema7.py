import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

def polynomial_value(coeffs, x):
    result = 0
    for coef in coeffs:
        result = result * x + coef
    return result

def polynomial_derivative(coeffs):
    n = len(coeffs) - 1  
    if n == 0:  
        return [0]
    derivative_coeffs = [i * coeffs[n-i] for i in range(n, 0, -1)]
    return derivative_coeffs

def polynomial_second_derivative(coeffs):
    deriv = polynomial_derivative(coeffs)
    return polynomial_derivative(deriv)

def halley_method(coeffs, x0, epsilon=1e-6, k_max=1000):
    x = x0
    iterations = 0
    delta = None  
    while True:
        p_x = polynomial_value(coeffs, x)
        dp_x = polynomial_value(polynomial_derivative(coeffs), x)
        d2p_x = polynomial_value(polynomial_second_derivative(coeffs), x)

        A = 2 * dp_x**2 - p_x * d2p_x

        if abs(A) < epsilon:
            return x, iterations, False

        delta = 2 * p_x * dp_x / A
        x = x - delta
        iterations += 1

        if abs(delta) < epsilon:
            return x, iterations, True

        if abs(delta) > 1e8 or iterations >= k_max:
            return x, iterations, False
    
def find_all_real_roots(coeffs, interval=(-10, 10), num_initial_points=20, epsilon=1e-6, k_max=1000):

    a, b = interval
    initial_points = np.linspace(a, b, num_initial_points)
    
    found_roots = []
    
    for x0 in initial_points:
        root, iterations, converged = halley_method(coeffs, x0)
        
        if converged:
            is_new_root = True
            for existing_root in found_roots:
                if abs(root - existing_root) < epsilon:
                    is_new_root = False
                    break
                    
            if is_new_root:
                found_roots.append(root)
    
    return sorted(found_roots)

def compute_bounds_R(coeffs):
    a_n = coeffs[0]  # Leading coefficient
    # a_0 = coeffs[-1]  # Constant term
    
    # Find A = max{|ai| : i = 1..n}
    A = max(abs(coef) for coef in coeffs[1:-1]) if len(coeffs) > 2 else 0
    
    R = (abs(a_n) + A) / abs(a_n)
    return R

def save_roots_to_file(roots, filename='radacini.txt'):
    with open(filename, "w") as f:
        for root in roots:
            f.write(f"{root:.10f}\n")  # scrie cu precizie mare
    print(f"Rădăcinile distincte au fost salvate în {filename}")

# def main():
#     # Exemplu: P(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
#     coeffs = [1, -6, 11, -6]  # [a_n, a_{n-1}, ..., a_1, a_0]
    
#     R = compute_bounds_R(coeffs)
#     print(f"Toate radacinile reale se gasesc in intervalul: [-{R}, {R}]")
    
#     # Gasim radacinile reale din intervalul [-R, R]
#     roots = find_all_real_roots(coeffs, interval=(-R, R), epsilon=1e-6, k_max=1000)
#     print(f"Am gasit {len(roots)} radacini reale: {roots}")
#     save_roots_to_file(roots)

#     x0 = 1.5
#     root, iterations, converged = halley_method(coeffs, x0)
#     print(f"Starting from x0 = {x0}")
#     print(f"radacina aproximata este: {root}")
#     print(f"nr de iteratii: {iterations}")
#     print(f"convergenta?: {converged}")
    
    

#     coeffs = [42, -55, -42, 49, -6]
#     coeffs = [8, -38, 49, -22, 3]
#     coeffs = [1, -6, 13, -12, 4]




def main():
    coeffs_list = [
        [1, -6, 11, -6],         # P(x) = (x-1)(x-2)(x-3)
        [42, -55, -42, 49, -6],
        [8, -38, 49, -22, 3],
        [1, -6, 13, -12, 4]
    ]

    for idx, coeffs in enumerate(coeffs_list, start=1):
        print(f"\n=== Polynomial {idx} ===")
        print(f"Coefficients: {coeffs}")

        R = compute_bounds_R(coeffs)
        print(f"Toate radacinile reale se gasesc in intervalul: [-{R}, {R}]")

        roots = find_all_real_roots(coeffs, interval=(-R, R), epsilon=1e-6, k_max=1000)
        print(f"Am gasit {len(roots)} radacini reale: {roots}")
        save_roots_to_file(roots, filename=f"roots_poly_{idx}.txt")

        x0 = 1.5  
        root, iterations, converged = halley_method(coeffs, x0)
        print(f"Halley start x0 = {x0}")
        print(f"radacina aproximata este: {root}")
        print(f"nr de iteratii: {iterations}")
        print(f"convergenta?: {converged}")

        
#         # x = np.linspace(-R, R, 1000)
#         # p = np.polynomial.Polynomial(coeffs[::-1])  
#         # y = p(x)
        
#         # plt.figure(figsize=(10, 6))
#         # plt.plot(x, y)
#         # plt.grid(True)
#         # plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
#         # plt.scatter(roots, [0]*len(roots), color='red', s=50)
#         # plt.title(f"Polynomial P(x) and its real roots")
#         # plt.xlabel('x')
#         # plt.ylabel('P(x)')
#         # plt.ylim(-10, 10)  # Adjust as needed
        
#         # # Show the plot
#         # plt.tight_layout()
#         # plt.show()

if __name__ == "__main__":
    main()