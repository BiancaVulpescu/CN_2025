n = 3
A = [[-24, 18, 5], [20, -15, -4], [-5, 4, 1]]
eps = 1e-10
kmax = 10000
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

# Formula (1) - Schultz
def iteratia_schultz(Vk, A):
    I = np.eye(A.shape[0])
    return Vk @ (2 * I - A @ Vk)

# Formula (2) - Li și Li - prima variantă
def iteratia_Li_1(Vk, A):
    I = np.eye(A.shape[0])
    return Vk @ (3 * I - A @ Vk @ (3 * I - A @ Vk))

# Formula (3) - Li și Li - a doua variantă
def iteratia_Li_2(Vk, A):
    I = np.eye(A.shape[0])
    E = I - Vk @ A
    F = 3 * I - Vk @ A
    return (I + 0.25 * E @ F @ F) @ Vk

# Funcție principală de aproximare
def aprox_inverse(A, metoda="schultz", epsilon=1e-6, k_max=10000):
    V0 = initializeaza_V0(A)
    V1 = V0.copy()
    k = 0

    while True:
        V0 = V1.copy()

        if metoda == "schultz":
            V1 = iteratia_schultz(V0, A)
        elif metoda == "li1":
            V1 = iteratia_Li_1(V0, A)
        elif metoda == "li2":
            V1 = iteratia_Li_2(V0, A)
        else:
            raise ValueError("Metodă necunoscută: folosește 'schultz', 'li1' sau 'li2'")

        delta_V = np.linalg.norm(V1 - V0)
        k += 1

        if delta_V < epsilon or k >= k_max or delta_V > 1e10:
            break

    if delta_V < epsilon:
        return V1
    else:
        print(f"Divergență după {k} pași")
        return None

# Exemplu de testare
if __name__ == "__main__":
    A = np.array([[4.0, 2.0], [3.0, 1.0]])
    inv_exact = np.linalg.inv(A)

    for metoda in ["schultz", "li1", "li2"]:
        print(f"\n=== Metoda: {metoda.upper()} ===")
        inv_aprox = aprox_inverse(A, metoda=metoda)
        if inv_aprox is not None:
            print("Inversă aproximată:")
            print(inv_aprox)
            print("Eroare față de inversa exactă:")
            print(np.linalg.norm(inv_aprox - inv_exact))

