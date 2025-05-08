import numpy as np

def polynomial_value(coeffs, x):
    result = 0
    for coef in coeffs:
        result = result * x + coef
    return result

def polynomial_derivative(coeffs):
    n = len(coeffs) - 1
    return [i * coeffs[n - i] for i in range(1, n + 1)]

def method_N4(coeffs, x0, epsilon=1e-6, k_max=1000):
    x = x0
    iterations = 0

    for _ in range(k_max):
        fx = polynomial_value(coeffs, x)
        dfx = polynomial_value(polynomial_derivative(coeffs), x)

        if abs(dfx) < epsilon:
            return x, iterations, False

        y = x - fx / dfx
        fy = polynomial_value(coeffs, y)

        denominator = dfx * (fx - fy)
        if abs(denominator) < epsilon:
            return x, iterations, False

        delta = (fx**2 + fy**2) / denominator
        x_new = x - delta

        if abs(delta) < epsilon:
            return x_new, iterations, True

        x = x_new
        iterations += 1

    return x, iterations, False

def method_N5(coeffs, x0, epsilon=1e-6, k_max=1000):
    x = x0
    iterations = 0

    for _ in range(k_max):
        fx = polynomial_value(coeffs, x)
        dfx = polynomial_value(polynomial_derivative(coeffs), x)

        if abs(dfx) < epsilon:
            return x, iterations, False

        y = x - fx / dfx
        fy = polynomial_value(coeffs, y)

        denominator = dfx * (fx - fy)
        if abs(denominator) < epsilon:
            return x, iterations, False

        delta1 = (fx**2 + fy**2) / denominator
        z = x - delta1

        fz = polynomial_value(coeffs, z)
        delta2 = fz / dfx
        x_new = z - delta2

        if abs(x_new - x) < epsilon:
            return x_new, iterations, True

        x = x_new
        iterations += 1

    return x, iterations, False

def find_all_real_roots(coeffs, method, interval=(-10, 10), num_points=20, epsilon=1e-6, k_max=1000):
    a, b = interval
    points = np.linspace(a, b, num_points)
    roots = []

    for x0 in points:
        root, _, converged = method(coeffs, x0, epsilon, k_max)
        if converged:
            if not any(abs(root - r) < epsilon for r in roots):
                roots.append(root)

    return sorted(roots)

def save_roots(roots, filename):
    with open(filename, "w") as f:
        for r in roots:
            f.write(f"{r:.10f}\n")
    print(f"Roots saved to {filename}")

if __name__ == "__main__":
    coeffs = [1, -6, 11, -6]  # Example: (x - 1)(x - 2)(x - 3)
    interval = (-10, 10)

    print("Finding roots using method N4...")
    roots_n4 = find_all_real_roots(coeffs, method_N4, interval)
    save_roots(roots_n4, "roots_n4.txt")

    print("Finding roots using method N5...")
    roots_n5 = find_all_real_roots(coeffs, method_N5, interval)
    save_roots(roots_n5, "roots_n5.txt")
