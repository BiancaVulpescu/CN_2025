import numpy as np

# Norme necesare pentru inițializare
def norma_1(A):
    return np.max(np.sum(np.abs(A), axis=0))

def norma_inf(A):
    return np.max(np.sum(np.abs(A), axis=1))

# Inițializare V0 după metoda standard (folosită de toate metodele)
def initializeaza_V0(A):
    A_T = A.T
    norm1 = norma_1(A)
    norminf = norma_inf(A)
    return A_T / (norm1 * norminf)

def matrice_ai_minus_AV(a, A, V):
    B = -A  # B = -A, calculată o singură dată
    C = B @ V
    C[np.diag_indices_from(C)] += a
    return C

# Formula (1): Schultz
def iteratia_schultz(V, A):
    C = matrice_ai_minus_AV(2, A, V)
    return V @ C

# Formula (2): Li și Li - prima variantă
def iteratia_Li_1(V, A):
    I = np.eye(A.shape[0])
    C = matrice_ai_minus_AV(3, A, V)
    return V @ (3 * I - A @ V @ C)


# Formula (3): Li și Li - a doua variantă
def iteratia_Li_2(V, A):
    I = np.eye(A.shape[0])
    E = I - V @ A
    F = 3 * I - V @ A
    return (I + 0.25 * E @ F @ F) @ V

# Funcție principală de aproximare
def aprox_inverse(A, metoda="schultz", epsilon=1e-10, k_max=10000):
    V = initializeaza_V0(A)
    k = 0

    while True:
        V_prev = V.copy()

        if metoda == "schultz":
            V = iteratia_schultz(V_prev, A)
        elif metoda == "li1":
            V = iteratia_Li_1(V_prev, A)
        elif metoda == "li2":
            V = iteratia_Li_2(V_prev, A)
        else:
            raise ValueError("Metodă necunoscută: folosește 'schultz', 'li1' sau 'li2'")

        delta_V = np.linalg.norm(V - V_prev)
        k += 1

        if delta_V < epsilon or k >= k_max or delta_V > 1e10:
            break

    if delta_V < epsilon:
        return V
    else:
        print(f"Divergență după {k} pași")
        return None

# Exemplu de testare
if __name__ == "__main__":
    n = 3
    A = np.array([[-24, 18, 5], [20, -15, -4], [-5, 4, 1]])
    eps = 1e-10
    kmax = 10000
    inv_exact = np.linalg.inv(A)

    for metoda in ["schultz", "li1", "li2"]:
        print(f"\n=== Metoda: {metoda.upper()} ===")
        inv_aprox = aprox_inverse(A, metoda=metoda)
        if inv_aprox is not None:
            print("Inversă aproximată:")
            print(inv_aprox)
            print("Eroare față de inversa exactă:")
            print(np.linalg.norm(inv_aprox - inv_exact))

