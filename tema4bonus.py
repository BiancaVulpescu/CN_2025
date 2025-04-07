import numpy as np

def norma_1(A):
    return np.max(np.sum(np.abs(A), axis=0))  # suma maximă a coloanelor

def norma_inf(A):
    return np.max(np.sum(np.abs(A), axis=1))  # suma maximă a rândurilor

# Inițializare V0
def initialize_V0_non_square(A):
    A_T = A.T
    norm1 = norma_1(A)
    norminf = norma_inf(A)
    return A_T / (norm1 * norminf)

def iteratia_schultz_non_square(V, A):
    I = np.eye(A.shape[0])  # Identity matrix of size m x m (matches A @ V)
    C = 2 * I - A @ V
    return V @ C

def aprox_pseudoinverse(A, epsilon=1e-10, k_max=10000):
    V = initialize_V0_non_square(A)
    k = 0

    while True:
        V_prev = V.copy()
        V = iteratia_schultz_non_square(V_prev, A)
        delta_V = np.linalg.norm(V - V_prev)

        k += 1
        if delta_V < epsilon or k > k_max or delta_V > 1e10:
            break

    if delta_V < epsilon:
        return V, k
    else:
        print(f"Divergență după {k} pași")
        return None, k

# Example usage
if __name__ == "__main__":
    A = np.array([[1, 2, 3], [4, 5, 6]])  # Example non-square matrix (2x3)
    pseudoinverse, iterations = aprox_pseudoinverse(A)
    if pseudoinverse is not None:
        print("Pseudoinversa aproximată:")
        print(pseudoinverse)
        print(f"Număr de iterații: {iterations}")