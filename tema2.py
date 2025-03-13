import numpy as np

def lu_decomposition_with_custom_diagonals(A, dU, epsilon=1e-10):
    n = A.shape[0]
    success = True
    
    # Check if all elements in dU are non-zero
    if np.any(np.abs(dU) < epsilon):
        return None, None, False
    
    # Make a copy of A to work with
    A_work = A.copy()
    
    # Initialize L and U matrices
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for p in range(n):
        # Set the diagonal element of U
        U[p, p] = dU[p]
        
        # For the first row/column
        if p == 0:
            # Calculate L elements for the first column
            for i in range(n):
                L[i, 0] = A_work[i, 0] / dU[0]
            
            # Calculate U elements for the first row
            for j in range(1, n):
                U[0, j] = A_work[0, j] / L[0, 0]
        else:
            # Calculate the current row of U
            for j in range(p, n):
                sum_val = 0
                for k in range(p):
                    sum_val += L[p, k] * U[k, j]
                if j == p:
                    # For diagonal elements of L
                    L[p, p] = (A_work[p, p] - sum_val) / dU[p]
                else:
                    # For other elements of U
                    U[p, j] = (A_work[p, j] - sum_val) / L[p, p]
            
            # Calculate the current column of L
            for i in range(p+1, n):
                sum_val = 0
                for k in range(p):
                    sum_val += L[i, k] * U[k, p]
                L[i, p] = (A_work[i, p] - sum_val) / dU[p]
        
        # If the calculated L[p, p] is too small, decomposition fails
        if abs(L[p, p]) < epsilon:
            success = False
            break
    
    return L, U, success

def solve_lu_system_custom_diagonals(L, U, b, epsilon=1e-10):
    n = L.shape[0]
    
    # Forward substitution (solving Ly = b)
    y = np.zeros(n)
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i, j] * y[j]
        if abs(L[i, i]) < epsilon:
            print("Cannot perform division: L element too small")
            return None
        y[i] = (b[i] - sum_val) / L[i, i]
    
    # Backward substitution (solving Ux = y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_val = 0
        for j in range(i+1, n):
            sum_val += U[i, j] * x[j]
        if abs(U[i, i]) < epsilon:
            print("Cannot perform division: U element too small")
            return None
        x[i] = (y[i] - sum_val) / U[i, i]
    
    return x

def compute_residual_norm(A, x, b):
    residual = np.dot(A, x) - b
    return np.sqrt(np.sum(residual**2))

def example_custom_diagonals():
    n = 3
    A = np.array([
        [4.0, 0.0, 4.0],
        [1.0, 4.0, 2.0],
        [2.0, 4.0, 6.0]
    ])
    A_original = A.copy()  # Keep original for verification
    
    # Custom diagonal for U
    dU = np.array([2.0, 4.0, 1.0])
    
    b = np.array([6.0, 5.0, 12.0])
    epsilon = 1e-10
    
    # Check if all dU elements are acceptable
    if np.any(np.abs(dU) < epsilon):
        print("Cannot perform LU decomposition: diagonal elements too small")
        return
    
    L, U, success = lu_decomposition_with_custom_diagonals(A, dU, epsilon)
    
    if success:
        print("LU Decomposition successful")
        
        print("L (with custom diagonal):\n", L)
        print("U (with custom diagonal):\n", U)
        print("Verification L*U:\n", np.dot(L, U))
        print("Original A:\n", A_original)
        print("dU after computation (might be adjusted):", dU)
        
        # Compare L*U with original A
        diff_norm = np.linalg.norm(np.dot(L, U) - A_original)
        print("Difference norm ||LU - A||:", diff_norm)
        
        # Solve the system
        x = solve_lu_system_custom_diagonals(L, U, b, epsilon)
        
        if x is not None:
            print("Solution x:", x)
            print("Verification A*x:", np.dot(A_original, x))
            
            # Calculate residual norm
            residual_norm = compute_residual_norm(A_original, x, b)
            print("Residual norm ||Ax - b||₂:", residual_norm)
        else:
            print("Failed to solve the system")
    else:
        print("LU Decomposition failed")

if __name__ == "__main__":
    example_custom_diagonals()