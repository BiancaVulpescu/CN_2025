import numpy as np

def allocate_LU_vectors(n):
    size = n * (n + 1) // 2
    L_vec = np.zeros(size)
    U_vec = np.zeros(size)
    return L_vec, U_vec

def index_in_vector(i, j, n):
    return i * (i + 1) // 2 + j if i >= j else j * (j + 1) // 2 + i

def lu_decomposition_optimized(A, dU, epsilon=1e-10):
    n = A.shape[0]
    L_vec, U_vec = allocate_LU_vectors(n)
    success = True
    
    if np.any(np.abs(dU) < epsilon):
        return False, L_vec, U_vec
    
    for i in range(n):
        for j in range(i, n):
            sum_val = sum(L_vec[index_in_vector(i, k, n)] * U_vec[index_in_vector(k, j, n)] for k in range(i))
            if j == i:
                L_vec[index_in_vector(i, i, n)] = (A[i, i] - sum_val) / dU[i]
            else:
                U_vec[index_in_vector(i, j, n)] = (A[i, j] - sum_val) / L_vec[index_in_vector(i, i, n)]
        
        for k in range(i + 1, n):
            sum_val = sum(L_vec[index_in_vector(k, j, n)] * U_vec[index_in_vector(j, i, n)] for j in range(i))
            L_vec[index_in_vector(k, i, n)] = (A[k, i] - sum_val) / dU[i]
        
        if abs(L_vec[index_in_vector(i, i, n)]) < epsilon:
            success = False
            break
    
    return success, L_vec, U_vec

def solve_lu_optimized(L_vec, U_vec, dU, b, n):
    y = np.zeros(n)
    for i in range(n):
        sum_val = sum(L_vec[index_in_vector(i, j, n)] * y[j] for j in range(i))
        y[i] = (b[i] - sum_val) / L_vec[index_in_vector(i, i, n)]
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_val = sum(U_vec[index_in_vector(i, j, n)] * x[j] for j in range(i+1, n))
        x[i] = (y[i] - sum_val) / dU[i]
    
    return x

def reconstruct_A_from_LU(L_vec, U_vec, dU, n):
    A_reconstructed = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                A_reconstructed[i, j] = L_vec[index_in_vector(i, j, n)]
            elif i == j:
                A_reconstructed[i, j] = L_vec[index_in_vector(i, i, n)] * dU[i]
            else:
                A_reconstructed[i, j] = U_vec[index_in_vector(i, j, n)]
    return A_reconstructed

def example_optimized():
    n = 3
    A = np.array([
        [4.0, 0.0, 5.0],
        [1.0, 7.0, 2.0],
        [2.0, 4.0, 6.0]
    ])
    A_original = A.copy()
    dU = np.array([2.0, 4.0, 1.0])
    b = np.array([6.0, 5.0, 12.0])
    epsilon = 1e-10
    
    success, L_vec, U_vec = lu_decomposition_optimized(A, dU, epsilon)
    print("L_vec:", L_vec)
    print("U_vec:", U_vec)
    if success:
        print("LU decomposition successful")
        A_reconstructed = reconstruct_A_from_LU(L_vec, U_vec, dU, n)
        print("Reconstructed A:")
        print(A_reconstructed)
        
        x = solve_lu_optimized(L_vec, U_vec, dU, b, n)
        print("Solution x:", x)
        residual = np.dot(A_original, x) - b
        print("Residual norm:", np.linalg.norm(residual))
    else:
        print("LU decomposition failed")

if __name__ == "__main__":
    example_optimized()
