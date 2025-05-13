import numpy as np

def f1(x):
    """F(x1, x2) = x1^2 + x2^2 - 2x1 - 4x2 - 1"""
    return x[0]**2 + x[1]**2 - 2*x[0] - 4*x[1] - 1

def grad_f1(x):
    return np.array([2*x[0] - 2, 2*x[1] - 4])

def f2(x):
    """F(x1, x2) = 3x1^2 - 12x1 + 2x2^2 + 16x2 - 10"""
    return 3*x[0]**2 - 12*x[0] + 2*x[1]**2 + 16*x[1] - 10

def grad_f2(x):
    return np.array([6*x[0] - 12, 4*x[1] + 16])

def f3(x):
    """F(x1, x2) = x1^2 - 4x1x2 + 5x2^2 - 4x2 + 3"""
    return x[0]**2 - 4*x[0]*x[1] + 5*x[1]**2 - 4*x[1] + 3

def grad_f3(x):
    return np.array([2*x[0] - 4*x[1], -4*x[0] + 10*x[1] - 4])

def f4(x):
    """F(x1, x2) = x1^2x2 - 2x1x2^2 + 3x1x2 + 4"""
    return x[0]**2*x[1] - 2*x[0]*x[1]**2 + 3*x[0]*x[1] + 4

def grad_f4(x):
    return np.array([2*x[0]*x[1] - 2*x[1]**2 + 3*x[1], 
                    x[0]**2 - 4*x[0]*x[1] + 3*x[0]])

def approximate_gradient(f, x, h=1e-6):
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x1 = x.copy()
        x2 = x.copy()
        x3 = x.copy()
        x4 = x.copy()
        
        x1[i] += 2*h
        x2[i] += h
        x3[i] -= h
        x4[i] -= 2*h
        
        grad[i] = (-f(x1) + 8*f(x2) - 8*f(x3) + f(x4)) / (12*h)
        
    return grad

def gradient_descent_constant_lr(f, grad_f, initial_x, lr=0.01, max_iter=10000, epsilon=1e-5):
    x = initial_x.copy()
    x_history = [x.copy()]
    
    for k in range(max_iter):
        gradient = grad_f(x)
        
        if np.linalg.norm(gradient) < epsilon:
            break
            
        x = x - lr * gradient
        x_history.append(x.copy())
        
        if k > 0 and np.linalg.norm(x_history[-1] - x_history[-2]) < epsilon:
            break
    
    return np.array(x_history), k

def gradient_descent_backtracking(f, grad_f, initial_x, beta=0.8, max_iter=10000, epsilon=1e-5):
    x = initial_x.copy()
    x_history = [x.copy()]
    
    for k in range(max_iter):
        gradient = grad_f(x)
        
        if np.linalg.norm(gradient) < epsilon:
            break
        
        eta = 1.0
        p = 1
        
        while p < 8 and f(x - eta * gradient) > f(x) - (eta/2) * np.linalg.norm(gradient)**2:
            eta = eta * beta
            p += 1
        
        x = x - eta * gradient
        x_history.append(x.copy())
        
        if k > 0 and np.linalg.norm(x_history[-1] - x_history[-2]) < epsilon:
            break
    
    return np.array(x_history), k

def run_tests():
    results = []
    
    function_settings = [
        {"f": f1, "grad_f": grad_f1, "approx_grad_f": lambda x: approximate_gradient(f1, x), 
         "name": "f1(x1, x2) = x1^2 + x2^2 - 2x1 - 4x2 - 1", "x0": np.array([0.0, 0.0])},
        
        {"f": f2, "grad_f": grad_f2, "approx_grad_f": lambda x: approximate_gradient(f2, x), 
         "name": "f2(x1, x2) = 3x1^2 - 12x1 + 2x2^2 + 16x2 - 10", "x0": np.array([0.0, 0.0])},
        
        {"f": f3, "grad_f": grad_f3, "approx_grad_f": lambda x: approximate_gradient(f3, x), 
         "name": "f3(x1, x2) = x1^2 - 4x1x2 + 5x2^2 - 4x2 + 3", "x0": np.array([0.0, 0.0])},
        
        {"f": f4, "grad_f": grad_f4, "approx_grad_f": lambda x: approximate_gradient(f4, x), 
         "name": "f4(x1, x2) = x1^2x2 - 2x1x2^2 + 3x1x2 + 4", "x0": np.array([1.0, 1.0])}
    ]
    
    for settings in function_settings:
        f = settings["f"]
        grad_f = settings["grad_f"]
        approx_grad_f = settings["approx_grad_f"]
        name = settings["name"]
        x0 = settings["x0"]
        
        print(f"\nTesting {name}")
        print("Initial point:", x0)
        
        x_history_const_lr, iters_const_lr = gradient_descent_constant_lr(
            f, grad_f, x0, lr=0.1, max_iter=1000, epsilon=1e-5
        )
        print("\nConstant LR with analytical gradient:")
        print(f"Minimum found at {x_history_const_lr[-1]} after {iters_const_lr} iterations")
        print(f"Function value: {f(x_history_const_lr[-1])}")
        
        x_history_const_lr_approx, iters_const_lr_approx = gradient_descent_constant_lr(
            f, approx_grad_f, x0, lr=0.1, max_iter=1000, epsilon=1e-5
        )
        print("\nConstant LR with approximate gradient:")
        print(f"Minimum found at {x_history_const_lr_approx[-1]} after {iters_const_lr_approx} iterations")
        print(f"Function value: {f(x_history_const_lr_approx[-1])}")
        
        x_history_bt, iters_bt = gradient_descent_backtracking(
            f, grad_f, x0, beta=0.8, max_iter=1000, epsilon=1e-5
        )
        print("\nBacktracking with analytical gradient:")
        print(f"Minimum found at {x_history_bt[-1]} after {iters_bt} iterations")
        print(f"Function value: {f(x_history_bt[-1])}")
        
        x_history_bt_approx, iters_bt_approx = gradient_descent_backtracking(
            f, approx_grad_f, x0, beta=0.8, max_iter=1000, epsilon=1e-5
        )
        print("\nBacktracking with approximate gradient:")
        print(f"Minimum found at {x_history_bt_approx[-1]} after {iters_bt_approx} iterations")
        print(f"Function value: {f(x_history_bt_approx[-1])}")
        
        results.append({
            "name": name,
            "constant_lr_analytical": {
                "min_point": x_history_const_lr[-1],
                "iterations": iters_const_lr,
                "func_value": f(x_history_const_lr[-1]),
                "path": x_history_const_lr
            },
            "constant_lr_approximate": {
                "min_point": x_history_const_lr_approx[-1],
                "iterations": iters_const_lr_approx,
                "func_value": f(x_history_const_lr_approx[-1]),
                "path": x_history_const_lr_approx
            },
            "backtracking_analytical": {
                "min_point": x_history_bt[-1],
                "iterations": iters_bt,
                "func_value": f(x_history_bt[-1]),
                "path": x_history_bt
            },
            "backtracking_approximate": {
                "min_point": x_history_bt_approx[-1],
                "iterations": iters_bt_approx,
                "func_value": f(x_history_bt_approx[-1]),
                "path": x_history_bt_approx
            }
        })
        
    return results

if __name__ == "__main__":
    results = run_tests()
    