import numpy as np

def polynomial_value(coeffs, x):
    result = 0
    for coef in coeffs:
        result = result * x + coef
    return result

def polynomial_derivative(coeffs):
    n = len(coeffs) - 1
    if n == 0:
        return [0]
    return [i * coeffs[n - i] for i in range(n, 0, -1)]

def method_N4(coeffs, x0, epsilon=1e-6, k_max=1000):
    x = x0
    for _ in range(k_max):
        fx = polynomial_value(coeffs, x)
        dfx = polynomial_value(polynomial_derivative(coeffs), x)

        if abs(dfx) < epsilon:
            return x, _, False

        y = x - fx / dfx
        fy = polynomial_value(coeffs, y)
        denominator = dfx * (fx - fy)

        if abs(denominator) < epsilon:
            return x, _, False

        delta = (fx**2 + fy**2) / denominator
        x_new = x - delta

        if abs(delta) < epsilon:
            return x_new, _, True

        x = x_new

    return x, k_max, False

def method_N5(coeffs, x0, epsilon=1e-6, k_max=1000):
    x = x0
    for _ in range(k_max):
        fx = polynomial_value(coeffs, x)
        dfx = polynomial_value(polynomial_derivative(coeffs), x)

        if abs(dfx) < epsilon:
            return x, _, False

        y = x - fx / dfx
        fy = polynomial_value(coeffs, y)
        denominator = dfx * (fx - fy)

        if abs(denominator) < epsilon:
            return x, _, False

        delta1 = (fx**2 + fy**2) / denominator
        z = x - delta1

        fz = polynomial_value(coeffs, z)
        delta2 = fz / dfx
        x_new = z - delta2

        if abs(x_new - x) < epsilon:
            return x_new, _, True

        x = x_new

    return x, k_max, False

def compute_bounds_R(coeffs):
    a_n = coeffs[0]
    A = max(abs(c) for c in coeffs[1:-1]) if len(coeffs) > 2 else 0
    R = (abs(a_n) + A) / abs(a_n)
    return R

def find_all_real_roots(coeffs, method, interval=(-10, 10), num_initial_points=3000, epsilon=1e-6, k_max=1000):
    a, b = interval
    initial_points = np.linspace(a, b, num_initial_points)
    
    found_roots = []
    
    for x0 in initial_points:
        root, _, converged = method(coeffs, x0, epsilon, k_max)
        
        if converged and abs(polynomial_value(coeffs, root)) < 1e-6:
            if not any(abs(root - existing_root) < 1e-6 for existing_root in found_roots):
                found_roots.append(root)
    
    return sorted(found_roots)

def save_roots(roots, filename):
    with open(filename, "w") as f:
        for r in roots:
            f.write(f"{r:.12f}\n")
    print(f"Rădăcinile distincte au fost salvate în {filename}")

if __name__ == "__main__":
    coeffs = [1, -6, 11, -6]  # Polynomial: (x - 1)(x - 2)(x - 3)
    R = compute_bounds_R(coeffs)
    print(f"Toate rădăcinile reale se află în intervalul: [-{R}, {R}]")

    print("\n Metoda N4:")
    roots_n4 = find_all_real_roots(coeffs, method_N4, interval=(-R, R), num_initial_points=1000)

    print(f"Am găsit {len(roots_n4)} rădăcini reale: {roots_n4}")
    save_roots(roots_n4, "radacini_n4.txt")

    print("\n Metoda N5:")
    roots_n5 = find_all_real_roots(coeffs, method_N5, interval=(-R, R), num_initial_points=1000)

    print(f"Am găsit {len(roots_n5)} rădăcini reale: {roots_n5}")
    save_roots(roots_n5, "radacini_n5.txt")
