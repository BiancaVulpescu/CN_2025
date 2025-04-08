import numpy as np

def norma_1(A):
    return np.max(np.sum(np.abs(A), axis=0))  # suma maxima a coloanelor

def norma_inf(A):
    return np.max(np.sum(np.abs(A), axis=1))  # suma maxima a randurilor

def initialize_V0_nepatratica(A):
    A_T = A.T
    norm1 = norma_1(A)
    norminf = norma_inf(A)
    return A_T / (norm1 * norminf)

def iteratia_schultz_nepatratica(V, B):
    I = np.eye(B.shape[0]) 
    C = 2 * I + B @ V
    return V @ C

def aprox_inverse(A, epsilon=1e-10, k_max=10000):
    V = initialize_V0_nepatratica(A)
    k = 0
    B = -A

    while True:
        V_prev = V.copy()
        V = iteratia_schultz_nepatratica(V_prev, B)
        delta_V = np.linalg.norm(V - V_prev)

        k += 1
        if delta_V < epsilon or k > k_max or delta_V > 1e10:
            break

    if delta_V < epsilon:
        return V, k
    else:
        print(f"Divergenta dupa {k} pasi")
        return None, k

if __name__ == "__main__":
    A = np.array([[1, 2, 3], [4, 5, 6]]) #mat 2x3
    inv_aprox, iterations = aprox_inverse(A)
    if inv_aprox is not None:
        print("Inversa aproximata:")
        print(inv_aprox)
        print(f"Numar de iteratii: {iterations}")