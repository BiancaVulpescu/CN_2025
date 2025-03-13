import numpy as np

def lu_decomposition_with_custom_diagonals(A, dU, epsilon=1e-10):
    
    n = A.shape[0]
    success = True
    
    # Check if all elements in dU are non-zero
    if np.any(np.abs(dU) < epsilon):
        return False
    
    # First column calculation - special case
    for i in range(n):
        A[i, 0] = A[i, 0] / dU[0]
    
    # First row calculation (except diagonal) - special case
    for j in range(1, n):
        A[0, j] = A[0, j] / A[0, 0]   # scoate u12 si u13
    
    # Process the remaining matrix
    for p in range(1, n):
        # Calculate U elements for row p
        for j in range(p, n):
            sum_val = 0
            for k in range(p):
                sum_val += A[p, k] * A[k, j]
            
            if j == p:
                # Calculate diagonal element of L
                A[p, p] = (A[p, p] - sum_val) / dU[p]
            else:
                # Calculate non-diagonal element of U
                A[p, j] = (A[p, j] - sum_val) / A[p, p]
        
        # Calculate L elements for column p
        for i in range(p+1, n):
            sum_val = 0
            for k in range(p):
                sum_val += A[i, k] * A[k, p]
            A[i, p] = (A[i, p] - sum_val) / dU[p]
        
        # If the calculated L[p,p] is too small, decomposition fails
        if abs(A[p, p]) < epsilon:
            success = False
            break
    
    return success

def solve_lu_system_custom_diagonals(A, dU, b, epsilon=1e-10):

    n = A.shape[0]
    
    # Forward substitution (solving Ly = b)
    y = np.zeros(n)
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += A[i, j] * y[j]  # Using A for L values
        y[i] = (b[i] - sum_val) / A[i, i]  # Using A[i,i] for L[i,i]
    
    # Backward substitution (solving Ux = y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_val = 0
        for j in range(i+1, n):
            sum_val += A[i, j] * x[j]  # Using A for U values
        x[i] = (y[i] - sum_val) / dU[i]  # Using dU[i] for U[i,i]
    return x

def compute_residual_norm(A_original, x, b):
    residual = np.dot(A_original, x) - b
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
    
    # Perform LU decomposition (modifies A in place)
    success = lu_decomposition_with_custom_diagonals(A, dU, epsilon)
    
    if success:
       
        print(A)
        print("Diagonal elements of U (dU):", dU)
        
        # Extract L and U for verification
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i > j:  # Below diagonal - L elements
                    L[i, j] = A[i, j]
                elif i == j:  # Diagonal
                    L[i, j] = A[i, j]
                    U[i, j] = dU[i]
                else:  # Above diagonal - U elements
                    U[i, j] = A[i, j]
        
        print("\nExtracted L matrix:\n", L)
        print("Extracted U matrix:\n", U)
        print("Verification L*U:\n", np.dot(L, U))
        print("Original A:\n", A_original)
        
        # Compare L*U with original A
        diff_norm = np.linalg.norm(np.dot(L, U) - A_original)
        print("Difference norm ||LU - A||:", diff_norm)
        
        # Solve the system
        x = solve_lu_system_custom_diagonals(A, dU, b, epsilon)
        
        if x is not None:
            print("\nSolution x:", x)
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