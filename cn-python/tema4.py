import numpy as np

def norma_1(A):
    return np.max(np.sum(np.abs(A), axis=0))  # suma maxima a coloanelor

def norma_inf(A):
    return np.max(np.sum(np.abs(A), axis=1))  # suma maxima a randurilor

def initializeaza_V0(A):
    A_T = A.T
    norm1 = norma_1(A)
    norminf = norma_inf(A)
    return A_T / (norm1 * norminf)

# C = a * I_n - A * V
# B = -A 
#adaug pe diag lui C elem a
def matrice_ai_minus_AV(a, B, V):
    C = B @ V
    C[np.diag_indices_from(C)] += a
    return C

# formula schultz
def metoda_schultz(V, B):
    C = matrice_ai_minus_AV(2, B, V)
    return V @ C

# formula: Li și Li - varianta 1
def metoda_Li_1(V, A, B):
    I = np.eye(A.shape[0])
    C = matrice_ai_minus_AV(3, B, V)
    BVC = B @ V @ C
    np.fill_diagonal(BVC, np.diag(BVC)+3)
    return V @ BVC

# formula: Li și Li - varianta 2
def metoda_Li_2(V, A):
    I = np.eye(A.shape[0])
    E = I - V @ A
    F = 3 * I - V @ A
    C = 0.25 * E @ F @ F
    np.fill_diagonal(C, np.diag(C)+1)
    return C @ V

def aprox_inverse(A, metoda, epsilon=1e-10, k_max=10000):
    V = initializeaza_V0(A)
    B = -A  
    k = 0

    while True:
        V_prev = V.copy()

        if metoda == "schultz":
            V = metoda_schultz(V_prev, B)
        elif metoda == "li1":
            V = metoda_Li_1(V_prev, A, B)
        elif metoda == "li2":
            V = metoda_Li_2(V_prev, A)
        
        delta_V = np.linalg.norm(V - V_prev)
        k += 1

        if delta_V < epsilon or k > k_max or delta_V > 1e10:
            break

    if delta_V < epsilon:
        return V, k
    else:
        print(f"Divergenta dupa {k} pasi")
        return None, k

def genereaza_A(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1 
        if i< n-1:
            A[i, i+1] = 2 
    return A

#construire inversa exacta dupa formula dedusa
def genereaza_inversa_exacta_dupa_form_dedusa(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1  
        for j in range(i + 1, n):
            power = j - i  
            A[i, j] = (-1) ** power * (2 ** power)  # (-1)^pow * 2^pow
    return A


if __name__ == "__main__":
    print("\n=== Punctul 1&2 ===")
    
    # A = np.array([[-24, 18, 5], [20, -15, -4], [-5, 4, 1]])
    np.random.seed(42)
    A = np.random.randint(-10, 10, (3, 3))

    print("Inversa matricei A:")
    print(np.linalg.inv(A))
    for metoda in ["schultz", "li1", "li2"]:
        print(f"\n=== Metoda: {metoda.upper()} ===")
        inv_aprox, k = aprox_inverse(A, metoda=metoda)
        
        print(f"Numar de iteratii: {k}")
        if inv_aprox is not None:
            print("Inversa aproximata:")
            print(inv_aprox)

            identitate = np.eye(A.shape[0])
            eroare_identitate = norma_1(A @ inv_aprox - identitate)
            print(f"|| A * A_aprox - I ||_1 = {eroare_identitate}")

        
    print("\n=== Punctul 3 ===")
    A_3 = genereaza_A(3)
    A_4 = genereaza_A(4)
    A_5 = genereaza_A(5)
    # A_9 = genereaza_A(9)
    # A_100 = genereaza_A(100)
    for A in [A_3, A_4, A_5]:
        inv_exact = genereaza_inversa_exacta_dupa_form_dedusa(A.shape[0])
        print(inv_exact)
        for metoda in ["schultz", "li1", "li2"]:
            print(f"\n=== Metoda: {metoda} ===")
            inv_aprox, k = aprox_inverse(A, metoda=metoda)
            print(f"Numar de iteratii: {k}")
            if inv_aprox is not None:
                print("Inversa aproximata:")
                print(inv_aprox)

                print("Eroare fata de inversa exacta:")
                print(np.linalg.norm(inv_exact - inv_aprox))
