import numpy as np
import random
from numpy.linalg import svd

def citire_matrice_met1(nume_fisier):
    with open(nume_fisier, "r") as f:
        n = int(f.readline().strip())
        
        d = [0] * n
        rare = {i: {} for i in range(n)}
        
        for linie in f:
            linie = linie.strip()
            if not linie:
                continue
            val, i, j = map(str.strip, linie.split(","))
            val, i, j = float(val), int(i), int(j)
            
            if val != 0:
                if i == j:
                    d[i] += val  
                else:
                    rare[i][j] = rare[i].get(j, 0) + val
    
    # Convert dictionaries to lists of tuples (index, value)
    for i in range(n):
        rare[i] = [(j, val) for j, val in rare[i].items()]
    
    return n, d, rare

def genereaza_matrice_rara_simetrica(n, densitate=0.01):
    d = [0] * n 
    rare = {i: [] for i in range(n)}  
    
    nr_elemente_nenule = int((n * n * densitate - n) / 2)
    
    for i in range(n):
        d[i] = random.uniform(1, 100)
    
    pozitii_generate = set()
    while len(pozitii_generate) < nr_elemente_nenule:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        #generam pe partea superioara 
        if i >= j or (i, j) in pozitii_generate:
            continue
        
        pozitii_generate.add((i, j))
        val = random.uniform(1, 100)
        
        rare[i].append((j, val))
        rare[j].append((i, val)) 
    
    return n, d, rare

# def salvare_matrice_rara(nume_fisier, n, d, rare):
#     with open(nume_fisier, "w") as f:
#         f.write(f"{n}\n")
        
#         # Save diagonal elements
#         for i in range(n):
#             if d[i] != 0:
#                 f.write(f"{d[i]}, {i}, {i}\n")
        
#         # Save off-diagonal elements (upper triangle only)
#         for i in range(n):
#             for j, val in sorted(rare[i]):
#                 if i < j:
#                     f.write(f"{val}, {i}, {j}\n")

def verifica_simetrie(n, rare):
    for i in range(n):
        for j, val_ij in rare[i]:
            if j>i:
                found = False
                for k, val_ji in rare[j]:
                    if k == i:
                        if abs(val_ij - val_ji) > 1e-9:
                            return False 
                        found = True
                        break
                if not found:
                    return False  
    return True

def inmultire_matrice_vector(n, d, rare, v):
    rezultat = np.zeros(n)
    
    for i in range(n):
        rezultat[i] += d[i] * v[i]
    
    for i in range(n):
        for j, val in rare[i]:
            rezultat[i] += val * v[j]
    
    return rezultat

def metoda_puterii(n, d, rare, epsilon=1e-9, k_max=1000000):
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    w = inmultire_matrice_vector(n, d, rare, v)
    lambda_k = np.dot(v, w)
    
    k = 0
    while True:
        v = w / np.linalg.norm(w)
        w = inmultire_matrice_vector(n, d, rare, v)
        lambda_k_plus_1 = np.dot(v, w)
        
        if np.linalg.norm(w - lambda_k_plus_1 * v) < epsilon or k >= k_max:
            break
        
        lambda_k = lambda_k_plus_1
        k += 1
    
    return lambda_k_plus_1, v, k

def calculeaza_norma_eroare(n, d, rare, lambda_max, v_max):
    Av = inmultire_matrice_vector(n, d, rare, v_max)
    lambda_v = lambda_max * v_max
    return np.linalg.norm(Av - lambda_v)

def calculeaza_pseudoinversa(A):
    # Calculate SVD decomposition: A = U S V^T
    U, s, Vt = svd(A, full_matrices=True)
    
    # Determine matrix rank (number of positive singular values)
    tol = max(A.shape) * np.finfo(float).eps * np.max(s)
    rang = np.sum(s > tol)
    
    # Calculate S^† matrix
    s_inv = np.zeros((A.shape[1], A.shape[0]))
    for i in range(rang):
        s_inv[i, i] = 1.0 / s[i]
    
    # Calculate pseudoinverse A^† = V S^† U^T
    A_dagger = Vt.T.dot(s_inv).dot(U.T)
    
    return A_dagger, rang, s, U, Vt

def calculeaza_numar_conditionare(s):
    tol = np.finfo(float).eps * np.max(s) * max(s.shape)
    s_pozitive = s[s > tol]
    
    if len(s_pozitive) == 0:
        return float('inf')  # Singular matrix
    
    return np.max(s) / np.min(s_pozitive)

def rezolva_sistem_svd(A, b):
    A_dagger, _, _, _, _ = calculeaza_pseudoinversa(A)
    return A_dagger.dot(b)

def analizeaza_matrice(n, d, rare):
    
    if verifica_simetrie(n, rare):
        print(f"Matricea este simetrică ✓")
    else:
        print(f"Avertisment: Matricea  nu este simetrică!")
    
    lambda_max, v_max, iteratii = metoda_puterii(n, d, rare)
    
    print(f"Metoda puterii:")
    print(f"  - Valoarea proprie de modul maxim: {lambda_max:.10f}")
    print(f"  - Număr de iterații: {iteratii}")
    
    eroare = calculeaza_norma_eroare(n, d, rare, lambda_max, v_max)
    print(f"  - Norma erorii ||Av^max - λ_max * v^max||: {eroare:.10e}")
    
    return lambda_max, v_max, iteratii

def main():
    print("Implementarea metodei puterii pentru matrici rare și simetrice")
    print("------------------------------------------------------------")
    
    n = 600
    _, d, rare = genereaza_matrice_rara_simetrica(n, densitate=0.01)
    analizeaza_matrice( n, d, rare)
    
    nume_fisier = "tema5files/m_rar_sim_2025_256.txt"
    n, d, rare = citire_matrice_met1(nume_fisier)
    analizeaza_matrice( n, d, rare)

    p, n = 800, 500
    print(f"Generare matrice densă de dimensiune {p}x{n}")
    
    A = np.random.rand(p, n) * 100
    b = np.random.rand(p) * 100
    
    # Calculate SVD decomposition and pseudoinverse
    A_dagger, rang, valori_singulare, U, Vt = calculeaza_pseudoinversa(A)
    print(f"Descompunerea SVD calculată")
    
    print(f"\nValorile singulare ale matricei A:")
    print(valori_singulare)
    
    print(f"\nRangul matricei A: {rang}")
    
    numar_conditionare = calculeaza_numar_conditionare(valori_singulare)
    print(f"\nNumărul de condiționare al matricei A: {numar_conditionare:.10e}")
    
    x = rezolva_sistem_svd(A, b)
    print(f"\nSistemul Ax = b rezolvat")
    
    eroare_sistem = np.linalg.norm(b - A.dot(x))
    print(f"Norma erorii ||b - Ax||: {eroare_sistem:.10e}")
    
    print("\nMoore-Penrose pseudoinversa A† = V S† U^T:")
    print(f"Dimensiune: {A_dagger.shape}")
    
    print("\nDESCOMPUNEREA SVD A FOST REALIZATĂ CU SUCCES!")

if __name__ == "__main__":
    main()