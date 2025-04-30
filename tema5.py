import os
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

def inmultire_matrice_vector(p, n, d, rare, v):
    rezultat = np.zeros(p)
    
    for i in range(min(p, n)):
        rezultat[i] += d[i] * v[i]
    
    for i in range(p):
        for j, val in rare[i]:
            if j < n:  
                rezultat[i] += val * v[j]
    
    return rezultat

import numpy as np

def metoda_puterii(n, d, rare, epsilon=1e-9, k_max=1000000):
    x = np.random.rand(n)
    while np.linalg.norm(x) == 0:
        x = np.random.rand(n)
    v = x / np.linalg.norm(x)

    w = inmultire_matrice_vector(n, n, d, rare, v)
    lambda_k = np.dot(w, v)
    k = 0

    while np.linalg.norm(w - lambda_k * v) > n * epsilon and k <= k_max:
        v = w / np.linalg.norm(w)
        w = inmultire_matrice_vector(n, n, d, rare, v)
        lambda_k = np.dot(w, v)
        k += 1

    return lambda_k, v, k


def calculeaza_norma_eroare(n, d, rare, lambda_max, v_max):
    Av = inmultire_matrice_vector(n, n, d, rare, v_max)
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
        print(f"Matricea este simetrica ")
    else:
        print(f"Avertisment: Matricea  nu este simetrica!")
    
    lambda_max, v_max, iteratii = metoda_puterii(n, d, rare)
    
    print(f"Metoda puterii:")
    print(f"  - Valoarea proprie de modul maxim: {lambda_max:.10f}")
    print(f"  - Numar de iteratii: {iteratii}")
    
    eroare = calculeaza_norma_eroare(n, d, rare, lambda_max, v_max)
    print(f"  - Norma erorii ||Av^max - λ_max * v^max||: {eroare:.10e}")
    
    return lambda_max, v_max, iteratii

def main():
    
    # Cazul 1: p = n > 500 - matrice patratica, rara si simetrica
    n = 600
    _, d, rare = genereaza_matrice_rara_simetrica(n, densitate=0.01)
    analizeaza_matrice( n, d, rare)
    
    # Cazul 2: Citire matrice din fisier
    file_names = [f for f in os.listdir("tema5files") if os.path.isfile(os.path.join("tema5files", f))]
    for f in file_names:
        nume_fisier = f"tema5files/{f}"
        n, d, rare = citire_matrice_met1(nume_fisier)
        analizeaza_matrice( n, d, rare)

    #Cazul 3 p > n - matrice clasica, nerara
    p, n = 800, 500
    
    A = np.random.rand(p, n) * 100
    b = np.random.rand(p) * 100
    
    # Calculate SVD decomposition and pseudoinverse
    A_dagger, rang, valori_singulare, U, Vt = calculeaza_pseudoinversa(A)
    print(f"Descompunerea SVD calculata")
    
    print(f"\nValorile singulare ale matricei A:")
    print(valori_singulare)
    
    print(f"\nRangul matricei A: {rang}")
    
    numar_conditionare = calculeaza_numar_conditionare(valori_singulare)
    print(f"\nNumarul de conditionare al matricei A: {numar_conditionare:.10e}")
    
    x = rezolva_sistem_svd(A, b)
    print(f"\nSistemul Ax = b rezolvat")
    
    eroare_sistem = np.linalg.norm(b - A.dot(x))
    print(f"Norma erorii ||b - Ax||: {eroare_sistem:.10e}")
    
    print("\nMoore-Penrose pseudoinversa A† = V S† U^T:")
    print(f"Dimensiune: {A_dagger.shape}")
    
    print("\nDESCOMPUNEREA SVD A FOST REALIZATA CU SUCCES!")

if __name__ == "__main__":
    main()