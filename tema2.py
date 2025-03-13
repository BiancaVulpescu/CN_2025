import numpy as np

def lu_decomposition_with_custom_diagonal(A, dU, epsilon=1e-10):
    """
    Compute the LU decomposition of matrix A where the diagonal elements of U
    are specified by the vector dU.
    
    Parameters:
    -----------
    A : numpy.ndarray
        Square matrix to decompose
    dU : numpy.ndarray
        Vector containing the diagonal elements for matrix U
    epsilon : float
        Precision threshold for numerical stability (observation 1)
        
    Returns:
    --------
    A : numpy.ndarray
        The original matrix A is modified to store L and U matrices.
        - Lower triangular part (including diagonal) contains L
        - Upper triangular part (excluding diagonal) contains U
    success : bool
        True if decomposition was successful, False if a pivot is too small
    """
    n = A.shape[0]
    success = True
    
    # Check if all elements in dU are non-zero (greater than epsilon)
    if np.any(np.abs(dU) < epsilon):
        return A, False
    
    # We'll modify A in-place to store both L and U
    # as mentioned in the observation
    
    # Process the matrix row by row
    for p in range(n):
        # Calculate elements of row p of matrix L and U
        for i in range(1, p+1):  # Calculate L elements for row p
            sum_val = 0
            for k in range(1, i):
                sum_val += A[p, k-1] * A[k-1, i-1]
            # Check division by zero (as in observation 1)
            if abs(dU[i-1]) < epsilon:
                print("Cannot perform division: dU element too small")
                success = False
                return A, success
            A[p, i-1] = (A[p, i-1] - sum_val) / dU[i-1]
        
        # Calculate elements of row p of matrix U
        for j in range(p, n):
            sum_val = 0
            for k in range(p):
                sum_val += A[p, k] * A[k, j]
            A[p, j] = A[p, j] - sum_val
        
        # Set diagonal element of L to 1 (as specified in the problem)
        if p < n:
            A[p, p] = 1.0
        
        # Check if the pivoting element is acceptable
        if p < n and abs(A[p, p]) < epsilon:
            success = False
            break
    
    return A, success

def solve_lu_system(A, dU, b, epsilon=1e-10):
    """
    Solve the system Ax = b using the LU decomposition stored in A and dU.
    Following observation 2: Ax = b <=> LUx = b <=> { Ly = b
                                                    { Ux = y
    
    Parameters:
    -----------
    A : numpy.ndarray
        Matrix containing the LU decomposition (L in lower part, U in upper part)
    dU : numpy.ndarray
        Vector containing the diagonal elements of U
    b : numpy.ndarray
        Right-hand side vector
    epsilon : float
        Precision threshold for numerical stability
        
    Returns:
    --------
    x : numpy.ndarray
        Solution vector
    """
    n = A.shape[0]
    
    # Forward substitution (solving Ly = b)
    # According to observation 2
    y = np.zeros(n)
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += A[i, j] * y[j]
        y[i] = b[i] - sum_val  # L has 1s on diagonal
    
    # Backward substitution (solving Ux = y)
    # According to observation 2
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_val = 0
        for j in range(i+1, n):
            sum_val += A[i, j] * x[j]
        # Check division by zero (as in observation 1)
        if abs(dU[i]) < epsilon:
            print("Cannot perform division: dU element too small")
            return None
        x[i] = (y[i] - sum_val) / dU[i]  # U has dU on diagonal
    
    return x

def compute_residual_norm(A, x, b):
    """
    Compute the residual norm ||A*x - b||₂ as mentioned in observation 3.
    
    Parameters:
    -----------
    A : numpy.ndarray
        The original matrix A
    x : numpy.ndarray
        Solution vector
    b : numpy.ndarray
        Right-hand side vector
        
    Returns:
    --------
    float
        The residual norm
    """
    residual = np.dot(A, x) - b
    return np.sqrt(np.sum(residual**2))

# Example usage with proper error handling as per observation 1
def example():
    n = 3
    A = np.array([
        [4.0, 0.0, 4.0],
        [1.0, 4.0, 2.0],
        [2.0, 4.0, 6.0]
    ])
    A_original = A.copy()  # Keep original for verification
    dU = np.array([2.0, 4.0, 1.0])  # Custom diagonal for U
    b = np.array([6.0, 5.0, 12.0])
    epsilon = 1e-10  # As mentioned in observation 1
    
    # Check if all dU elements are acceptable (greater than epsilon)
    if np.any(np.abs(dU) < epsilon):
        print("Cannot perform LU decomposition: dU contains elements too small")
        return
    
    A_lu, success = lu_decomposition_with_custom_diagonal(A, dU, epsilon)
    
    if success:
        print("LU Decomposition successful")
        
        # Extract L and U for verification
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        
        # Fill L (lower triangular with 1s on diagonal)
        for i in range(n):
            L[i, i] = 1.0  # Diagonal is 1
            for j in range(i):
                L[i, j] = A_lu[i, j]
        
        # Fill U (upper triangular with dU on diagonal)
        for i in range(n):
            U[i, i] = dU[i]  # Diagonal from dU
            for j in range(i+1, n):
                U[i, j] = A_lu[i, j]
        
        print("L:\n", L)
        print("U:\n", U)
        print("Verification L*U:\n", np.dot(L, U))
        print("Original A:\n", A_original)
        
        # Solve the system as per observation 2
        x = solve_lu_system(A_lu, dU, b, epsilon)
        
        if x is not None:
            print("Solution x:", x)
            print("Verification A*x:", np.dot(A_original, x))
            
            # Calculate residual norm as per observation 3
            residual_norm = compute_residual_norm(A_original, x, b)
            print("Residual norm ||Ax - b||₂:", residual_norm)
        else:
            print("Failed to solve the system")
    else:
        print("LU Decomposition failed")
example()