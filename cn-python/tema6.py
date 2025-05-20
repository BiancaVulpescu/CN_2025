# Medeleanu Daria 
# Vulpescu Bianca
# 310910401RSL221137
# 310910401RSL221219
# medeleanudaria14@gmail.com
# vulpescubianca@gmail.com
# dariaa8868 
# bianca2544
#procent de rezolvare cu AI: 35% 
#resurse
#https://www.geeksforgeeks.org/horners-method-polynomial-evaluation/
#https://edu.info.uaic.ro/calcul-numeric/CN/lab/6/Tema%206.pdf
import numpy as np
import matplotlib.pyplot as plt

def metoda_celor_mai_mici_patrate(x_points, y_points, m):
    n = len(x_points) - 1
    
    B = np.zeros((m+1, m+1))
    for i in range(m+1):
        for j in range(m+1):
            B[i, j] = sum(x_points[k] ** (i+j) for k in range(n+1))
    
    f = np.zeros(m+1)
    for i in range(m+1):
        f[i] = sum(y_points[k] * (x_points[k] ** i) for k in range(n+1))
    
    a = np.linalg.solve(B, f)
    
    return a[::-1]  #returnez in ordine inversa

def horner_eval(coeffs, x):
    result = 0
    for coef in coeffs:
        result = result * x + coef
    return result

def polynomial_string(coeffs):
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

    coeffs = metoda_celor_mai_mici_patrate(x_points, y_points, m)
    
    p_eval = horner_eval(coeffs, x_eval)
    
    print(f"Polynomial degree: {m}")
    print(f"Polynomial: P(x) = {polynomial_string(coeffs)}")
    print(f"P({x_eval}) = {p_eval:.6f}")
    
    errors_sum = sum(abs(horner_eval(coeffs, x) - y) for x, y in zip(x_points, y_points))
    
    if true_func:
        true_value = true_func(x_eval)
        print(f"|P({x_eval}) - f({x_eval})| = |{p_eval:.6f} - {true_value:.6f}| = {abs(p_eval - true_value):.6f}")
    
    print(f"Sum of |P(xi) - yi| = {errors_sum:.6f}")
    
    plot_approximation(x_points, y_points, coeffs, true_func)
    
    return coeffs, p_eval, errors_sum

def plot_approximation(x_points, y_points, coeffs, true_func=None):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(x_points, y_points, color='red', label='Data points')
    
    x_range = np.linspace(min(x_points), max(x_points), 1000)
    
    p_values = [horner_eval(coeffs, x) for x in x_range]
    plt.plot(x_range, p_values, 'r-', label='Polynomial approximation')
    
    if true_func:
        true_values = [true_func(x) for x in x_range]
        plt.plot(x_range, true_values, 'g--', label='True function')
    
    plt.title('Function Approximation using Least Squares Polynomial')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def trigonometric_interpolation(x_points, y_points, x_eval, true_func=None):

    n = len(x_points) - 1
    if n % 2 != 0:
        raise ValueError("Number of points must be odd (n = 2m)")
    
    m = n // 2 
    
    T = np.zeros((n+1, n+1))
    
    def phi_0(x):
        return 1
    
    def phi_k(x, k):
        return np.cos(k * x)
    
    def phi_2m_k(x, k):
        return np.sin(k * x)
    
    for i in range(n+1):  
        T[i, 0] = phi_0(x_points[i])
        
        for k in range(1, m+1):
            T[i, 2*k-1] = phi_2m_k(x_points[i], k) #sin pt coloanele impare
            T[i, 2*k] = phi_k(x_points[i], k) #cos pt coloanele pare
    
    Y = np.array(y_points)
    
    X = np.linalg.solve(T, Y)

    a0 = X[0]
    a_coefs = X[2::2][:m]  # coeficientii pe linii pare
    b_coefs = X[1::2][:m]  # coeficientii pe linii impare
    print(a_coefs, b_coefs)
    def eval_trig(x):
        result = a0
        for k in range(1, m+1):
            result += a_coefs[k-1] * np.cos(k * x)
            result += b_coefs[k-1] * np.sin(k * x)
        return result
    
    t_eval = eval_trig(x_eval)
    
    print(f"Trigonometric interpolation with m = {m}")
    
    trig_string = f"{a0:.4f}"
    for k in range(1, m+1):
        if a_coefs[k-1] != 0:
            term = f" + {a_coefs[k-1]:.4f}cos({k}x)"
            trig_string += term.replace("+ -", "- ")
        if b_coefs[k-1] != 0:
            term = f" + {b_coefs[k-1]:.4f}sin({k}x)"
            trig_string += term.replace("+ -", "- ")
    
    print(f"T(x) = {trig_string}")
    print(f"T({x_eval}) = {t_eval:.15f}")
    
    if true_func:
        true_value = true_func(x_eval)
        print(f"|T({x_eval}) - f({x_eval})| = |{t_eval:.6f} - {true_value:.6f}| = {abs(t_eval - true_value):.6f}")
    
    plot_trig_approximation(x_points, y_points, a0, a_coefs, b_coefs, m, true_func)
    
    return (a0, a_coefs, b_coefs), t_eval

def plot_trig_approximation(x_points, y_points, a0, a_coefs, b_coefs, m, true_func=None):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(x_points, y_points, color='red', label='Data points')
    
    x_range = np.linspace(0, 2*np.pi, 1000)
    
    def eval_trig(x):
        result = a0
        for k in range(1, m+1):
            result += a_coefs[k-1] * np.cos(k * x)
            result += b_coefs[k-1] * np.sin(k * x)
        return result
    
    t_values = [eval_trig(x) for x in x_range]
    plt.plot(x_range, t_values, 'r-', label='Trigonometric approximation')

    if true_func:
        true_values = [true_func(x) for x in x_range]
        plt.plot(x_range, true_values, 'g--', label='True function')
    
    plt.title('Function Approximation using Trigonometric Interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Example 1: f(x) = x⁴ - 12x³ + 30x² + 12 on [1, 5]
    def f1(x):
        return x**4 - 12*x**3 + 30*x**2 + 12
    x0, xn = 1, 5
    n = 10  
    x_points = np.linspace(x0, xn, n+1)
    y_points = [f1(x) for x in x_points]

    x_eval = 3.0

    print("f(x) = x⁴ - 12x³ + 30x² + 12")
    print("="*50)
    for m in [2, 3, 4, 5]:
        coeffs, p_eval, errors_sum = polynomial_approximation(x_points, y_points, x_eval, m, f1)
        print("="*50)

    # Example 2a: f(x) = sin(x) - cos(x) on [0, 2π]
    def f2a(x):
        return np.sin(x) - np.cos(x)

    # Example 2b: f(x) = sin(2x) + sin(x) + cos(3x) on [0, 2π]
    def f2b(x):
        return np.sin(2*x) + np.sin(x) + np.cos(3*x)

    # Example 2c: f(x) = sin²(x) - cos²(x) on [0, 2π]
    def f2c(x):
        return np.sin(x)**2 - np.cos(x)**2

    m_trig = 3  
    x_points_trig = np.linspace(0, 2*np.pi, 2*m_trig+1, endpoint=False)  # n = 2m points

    print("\nExample 2a: f(x) = sin(x) - cos(x)")
    print("="*50)
    y_points_trig = [f2a(x) for x in x_points_trig]
    x_eval_trig = np.pi/4
    trigonometric_interpolation(x_points_trig, y_points_trig, x_eval_trig, f2a)
    print("="*50)

    print("\nExample 2b: f(x) = sin(2x) + sin(x) + cos(3x)")
    print("="*50)
    y_points_trig = [f2b(x) for x in x_points_trig]
    trigonometric_interpolation(x_points_trig, y_points_trig, x_eval_trig, f2b)
    print("="*50)

    print("\nExample 2c: f(x) = sin²(x) - cos²(x)")
    print("="*50)
    y_points_trig = [f2c(x) for x in x_points_trig]
    trigonometric_interpolation(x_points_trig, y_points_trig, x_eval_trig, f2c)
    print("="*50)
    
if __name__ == "__main__":
    main()