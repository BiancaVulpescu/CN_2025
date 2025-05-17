import numpy as np

X = np.array([
    [1, 0, 0, 0], #A
    [1, 0, 1, 0], #B
    [0, 1, 0, 1], #C
    [0, 0, 0, 1], #D
    [1, 1, 1, 0], #E
    [1, 0, 1, 1], #F
    [1, 0, 0, 1], #G
    [0, 1, 0, 0]  #H
])
y = np.array([1, 1, 1, 0, 0, 0, 0, 0])  

X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))  

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def negative_log_likelihood(w):
    """
    Negative log-likelihood function (for minimization)
    This is the objective function we want to minimize
    """
    z = X_with_bias @ w
    # Using a small epsilon to avoid log(0)
    epsilon = 1e-15
    sigmoid_z = sigmoid(z)
    safe_sigmoid = np.clip(sigmoid_z, epsilon, 1 - epsilon)
    return -np.sum(y * np.log(safe_sigmoid) + (1 - y) * np.log(1 - safe_sigmoid))

def grad_negative_log_likelihood(w):
    """Gradient of the negative log-likelihood function"""
    z = X_with_bias @ w
    return -X_with_bias.T @ (y - sigmoid(z))

def gradient_descent(f, grad_f, w0, lr=0.1, max_iter=10000, epsilon=1e-6):
    """
    Gradient descent algorithm for minimization
    Parameters:
    f: Function to minimize
    grad_f: Gradient of the function
    w0: Initial weights
    lr: Learning rate
    max_iter: Maximum number of iterations
    epsilon: Convergence threshold
    """
    w = w0.copy()
    history = [w.copy()]
    
    for i in range(max_iter):
        grad = grad_f(w)
        if np.linalg.norm(grad) < epsilon:
            break
        w -= lr * grad  # Subtract for minimization
        history.append(w.copy())
        
    return w, history

#Initialize weights
w0 = np.zeros(X_with_bias.shape[1])

# Run gradient descent
final_w, history = gradient_descent(negative_log_likelihood, grad_negative_log_likelihood, w0)

print("Final weights:", final_w)
print("Final negative log-likelihood:", negative_log_likelihood(final_w))

# Compute predictions
def predict(X, w):
    """Make predictions using trained weights"""
    z = X @ w
    return sigmoid(z) >= 0.5

# Add bias term for prediction
X_test_with_bias = X_with_bias

# Make predictions
predictions = predict(X_test_with_bias, final_w)

# Evaluate accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)

# Display the negative log-likelihood values throughout training
neg_log_likelihoods = [negative_log_likelihood(w) for w in history]
print("Negative log-likelihood values:")
for i, nll in enumerate(neg_log_likelihoods[:10]):
    print(f"Iteration {i}: {nll}")
print("...")
print(f"Final (iteration {len(neg_log_likelihoods)-1}): {neg_log_likelihoods[-1]}")

# Verify the log-likelihood calculation matches the formula in the image
# We'll explicitly calculate for each example
def verify_log_likelihood(w):
    """Verify the log-likelihood calculation matches the formula in the image"""
    w0, w1, w2, w3, w4 = w
    
    # Calculate the components according to the formula in the image
    part1 = np.log(sigmoid(w0 + w1))  # For example A
    part2 = np.log(sigmoid(w0 + w1 + w3))  # For example B
    part3 = np.log(sigmoid(w0 + w2 + w4))  # For example C
    part4 = np.log(1 - sigmoid(w0 + w4))  # For example D
    part5 = np.log(1 - sigmoid(w0 + w1 + w2 + w3))  # For example E
    part6 = np.log(1 - sigmoid(w0 + w1 + w3 + w4))  # For example F
    part7 = np.log(1 - sigmoid(w0 + w1 + w4))  # For example G
    part8 = np.log(1 - sigmoid(w0 + w2))  # For example H
    
    # Sum all components
    total = part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8
    
    return total

# Verify the log-likelihood
log_likelihood_value = -negative_log_likelihood(final_w)
verification_value = verify_log_likelihood(final_w)

print(f"\nLog-likelihood calculated using vectorized function: {log_likelihood_value}")
print(f"Log-likelihood verified using formula from image: {verification_value}")
print(f"Difference: {abs(log_likelihood_value - verification_value)}")

# Show predictions for each instance
print("\nPredictions for each instance:")
probs = sigmoid(X_with_bias @ final_w)
for i, (pred, actual, prob) in enumerate("ABCDEFGH"):
    print(f"Instance {chr(65+i)}: Predicted={predictions[i]}, Actual={y[i]}, Probability={probs[i]:.4f}")