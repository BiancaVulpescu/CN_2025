import numpy as np
X = np.array([
    [1, 0, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 0, 0, 1],
    [0, 1, 0, 0]
])
y = np.array([1, 1, 1, 0, 0, 0, 0, 0])  

# Add bias term to features
X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))  # shape: (8, 5)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# === 10 pt: Use derived log-likelihood and its gradient ===
def log_likelihood_derived(w):
    z = X_with_bias @ w
    return np.sum(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z)))

def grad_log_likelihood_derived(w):
    z = X_with_bias @ w
    return X_with_bias.T @ (y - sigmoid(z))

# === 15 pt: Build log-likelihood and gradient from scratch ===
def log_likelihood_from_scratch(w):
    total = 0
    for i in range(len(y)):
        xi = X_with_bias[i]
        yi = y[i]
        zi = np.dot(w, xi)
        total += yi * np.log(sigmoid(zi)) + (1 - yi) * np.log(1 - sigmoid(zi))
    return total

def grad_log_likelihood_from_scratch(w):
    grad = np.zeros_like(w)
    for i in range(len(y)):
        xi = X_with_bias[i]
        yi = y[i]
        zi = np.dot(w, xi)
        grad += xi * (yi - sigmoid(zi))
    return grad

# === Optimization ===
def gradient_ascent(f, grad_f, w0, lr=0.1, max_iter=1000, epsilon=1e-6):
    w = w0.copy()
    for _ in range(max_iter):
        grad = grad_f(w)
        if np.linalg.norm(grad) < epsilon:
            break
        w += lr * grad  # ascent
    return w

# Initial weights
w0 = np.zeros(X_with_bias.shape[1])

# Run optimization
print("=== BONUS 10 pt ===")
w_derived = gradient_ascent(log_likelihood_derived, grad_log_likelihood_derived, w0)
print("Weights:", w_derived)
print("Log-likelihood:", log_likelihood_derived(w_derived))

print("\n=== BONUS 15 pt ===")
w_scratch = gradient_ascent(log_likelihood_from_scratch, grad_log_likelihood_from_scratch, w0)
print("Weights:", w_scratch)
print("Log-likelihood:", log_likelihood_from_scratch(w_scratch))
