import numpy as np
import random
import time

def citire_matrice_met1(nume_fisier):
    with open(nume_fisier, "r") as f:
        n = int(f.readline().strip())
        
        d = [0] * n
        rare = {i: {} for i in range(n)}
        
        # Citim fiecare element nenul
        for linie in f:
            linie = linie.strip()
            if not linie:
                continue
            val, i, j = map(str.strip, linie.split(","))
            val, i, j = float(val), int(i), int(j)  # Convertim la tipurile corecte
            
            if val != 0:
                if i == j:
                    d[i] += val  
                else:
                    if j in rare[i]:
                        rare[i][j] += val  
                    else:
                        rare[i][j] = val
    
    # Convertim dictionarele in liste de tupluri (indice, valoare)
    for i in range(n):
        rare[i] = [(j, val) for j, val in rare[i].items()]
    
    return n, d, rare

def genereaza_matrice_rara_simetrica(n, densitate=0.01):
    """
    Generează o matrice rară și simetrică de dimensiune n x n
    cu densitate aproximativă specificată (procent din elemente nenule)
    și doar elemente pozitive.
    """
    d = [0] * n  # Diagonala principală
    rare = {i: [] for i in range(n)}  # Elementele nenule din afara diagonalei
    
    # Calculăm numărul aproximativ de elemente nenule în afara diagonalei principale
    # Pentru o matrice simetrică, trebuie să generăm doar jumătate din elemente
    nr_elemente_nenule = int((n * n * densitate - n) / 2)
    
    # Generăm elementele diagonalei principale (pozitive)
    for i in range(n):
        d[i] = random.uniform(1, 100)  # Valori pozitive între 1 și 100
    
    # Generăm elementele din afara diagonalei
    pozitii_generate = set()
    while len(pozitii_generate) < nr_elemente_nenule:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        
        # Asigurăm-ne că i < j și că poziția nu a fost deja generată
        if i >= j or (i, j) in pozitii_generate:
            continue
        
        pozitii_generate.add((i, j))
        val = random.uniform(1, 100)  # Valori pozitive între 1 și 100
        
        # Adăugăm elementul în structura de date
        rare[i].append((j, val))
        # Pentru ca matricea să fie simetrică, adăugăm și elementul transpus
        rare[j].append((i, val))
    
    return n, d, rare

def salvare_matrice_rara(nume_fisier, n, d, rare):
    """Salvează matricea rară în format text"""
    with open(nume_fisier, "w") as f:
        # Prima linie conține dimensiunea matricei
        f.write(f"{n}\n")
        
        # Salvăm elementele diagonale
        for i in range(n):
            if d[i] != 0:
                f.write(f"{d[i]}, {i}, {i}\n")
        
        # Salvăm elementele din afara diagonalei (doar jumătatea superioară pentru simetrică)
        for i in range(n):
            for j, val in sorted(rare[i]):
                if i < j:  # Salvăm doar elementele din triunghiul superior
                    f.write(f"{val}, {i}, {j}\n")

def verifica_simetrie(n, d, rare):
    """Verifică dacă matricea este simetrică (A = A^T)"""
    # Pentru o matrice simetrică, fiecare element (i,j) trebuie să aibă 
    # un corespondent (j,i) cu aceeași valoare
    
    # Convertim lista de tupluri într-un dicționar pentru verificare rapidă
    matrice_dict = {}
    for i in range(n):
        for j, val in rare[i]:
            matrice_dict[(i, j)] = val
    
    # Verificăm că pentru fiecare element (i,j), elementul (j,i) există și are aceeași valoare
    for (i, j), val in matrice_dict.items():
        if i != j:  # Nu verificăm diagonala
            if (j, i) not in matrice_dict or abs(matrice_dict[(j, i)] - val) > 1e-9:
                return False
    
    return True

def inmultire_matrice_vector(n, d, rare, v):
    """
    Înmulțește matricea rară A cu vectorul v: rezultat = A * v
    """
    rezultat = np.zeros(n)
    
    # Contribuția diagonalei principale
    for i in range(n):
        rezultat[i] += d[i] * v[i]
    
    # Contribuția elementelor din afara diagonalei
    for i in range(n):
        for j, val in rare[i]:
            rezultat[i] += val * v[j]
    
    return rezultat

def metoda_puterii(n, d, rare, epsilon=1e-9, k_max=1000000):
    # Se alege vectorul inițial v^(0) aleator, de normă euclidiană 1
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)  # Normalizare
    
    # Calculăm w = Av
    w = inmultire_matrice_vector(n, d, rare, v)
    
    # Calculăm coeficientul Rayleigh λ = (v, w)
    lambda_k = np.dot(v, w)
    
    k = 0
    while True:
        # Actualizăm v^(k+1) = w / ||w||
        v = w / np.linalg.norm(w)
        
        # Calculăm w = Av
        w = inmultire_matrice_vector(n, d, rare, v)
        
        # Calculăm noul coeficient Rayleigh λ_(k+1) = (v, w)
        lambda_k_plus_1 = np.dot(v, w)
        
        # Verificăm criteriul de oprire: ||w - λ_k * v|| < ε sau k > k_max
        if np.linalg.norm(w - lambda_k_plus_1 * v) < epsilon or k >= k_max:
            break
        
        lambda_k = lambda_k_plus_1
        k += 1
    
    return lambda_k_plus_1, v, k

def calculeaza_norma_eroare(n, d, rare, lambda_max, v_max):
    """
    Calculează norma ||Av^max - λ_max * v^max||
    """
    Av = inmultire_matrice_vector(n, d, rare, v_max)
    lambda_v = lambda_max * v_max
    return np.linalg.norm(Av - lambda_v)

def main():
    print("Implementarea metodei puterii pentru matrici rare și simetrice")
    print("------------------------------------------------------------")
    
    # 1. Generare matrice rară și simetrică pentru n > 500
    n = 600  # Exemplu pentru n > 500
    n, d, rare = genereaza_matrice_rara_simetrica(n, densitate=0.01)
    print(f"Matrice generată de dimensiune {n}x{n} ")
    
    # Verificăm dacă matricea generată este simetrică
    if verifica_simetrie(n, d, rare):
        print("Matricea generată este simetrică ✓")
    else:
        print("Avertisment: Matricea generată nu este simetrică!")
    
    lambda_max_gen, v_max_gen, iteratii_gen = metoda_puterii(n, d, rare)
    
    print(f"Metoda puterii pentru matricea generată:")
    print(f"  - Valoarea proprie de modul maxim: {lambda_max_gen:.10f}")
    print(f"  - Număr de iterații: {iteratii_gen}")
    
    # Calculăm norma erorii pentru matricea generată
    eroare_gen = calculeaza_norma_eroare(n, d, rare, lambda_max_gen, v_max_gen)
    print(f"  - Norma erorii ||Av^max - λ_max * v^max||: {eroare_gen:.10e}")
    
    print("\n------------------------------------------------------------\n")
    
    nume_fisier = "/tema5files/m_rar_sim_2025_256.txt"  # Actualizați cu calea corectă
    n_citit, d_citit, rare_citit = citire_matrice_met1(nume_fisier)
    print(f"Matrice citită din fișier de dimensiune {n_citit}x{n_citit}")
    
    # Verificăm dacă matricea citită este simetrică
    if verifica_simetrie(n_citit, d_citit, rare_citit):
        print("Matricea citită este simetrică ✓")
    else:
        print("Avertisment: Matricea citită nu este simetrică!")
    
    lambda_max_citit, v_max_citit, iteratii_citit = metoda_puterii(n_citit, d_citit, rare_citit)
    
    print(f"Metoda puterii pentru matricea citită:")
    print(f"  - Valoarea proprie de modul maxim: {lambda_max_citit:.10f}")
    print(f"  - Număr de iterații: {iteratii_citit}")
    
    # Calculăm norma erorii pentru matricea citită
    eroare_citit = calculeaza_norma_eroare(n_citit, d_citit, rare_citit, lambda_max_citit, v_max_citit)
    print(f"  - Norma erorii ||Av^max - λ_max * v^max||: {eroare_citit:.10e}")


if __name__ == "__main__":
    main()