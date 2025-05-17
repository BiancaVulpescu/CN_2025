import numpy as np
def citire_matrice_met1(nume_fisier):
    with open(nume_fisier, "r") as f:
        # Citim dimensiunea matricei
        n = int(f.readline().strip())
        
        # Inițializăm vectorul diagonal și dicționarul pentru elementele nenule
        d = [0] * n
        rare = {i: {} for i in range(n)}

        # Citim fiecare element nenul
        for linie in f:
            linie = linie.strip()
            if not linie:  # Ignorăm liniile goale
                continue
            val, i, j = map(str.strip, linie.split(","))
            val, i, j = float(val), int(i), int(j)  # Convertim la tipurile corecte

            if i == j:
                d[i] += val  # Element diagonal
            else:
                if j in rare[i]:
                    rare[i][j] += val  # Adunăm valorile pentru aceiași indici
                else:
                    rare[i][j] = val  # Adăugăm o nouă valoare
    # Convertim dicționarele rare[i] în liste de tupluri pentru consistență
    for i in range(n):
        rare[i] = [(val, j) for j, val in rare[i].items()]

    return n, d, rare

def citire_matrice_met2(file_path):
    valori = []  # Store non-zero values
    ind_col = []  # Store column indices
    inceput_linii = [0]  # Start positions for each row (0-based indexing)
    
    with open(file_path, 'r') as file:
        n = int(file.readline().strip())  # Read matrix dimension
        element_count = 0
        row_data = {}  # Temporary dictionary to store values for each row

        for line in file:
            val, i, j = map(float, line.strip().split(','))
            i, j = int(i), int(j)
            
            if i not in row_data:
                row_data[i] = {}
            
            # Sum values for duplicate indices
            if j in row_data[i]:
                row_data[i][j] += val
            else:
                row_data[i][j] = val
        print(row_data)
        # Process the row data into compressed row format
        for i in range(n):
            if i in row_data:
                for j, val in sorted(row_data[i].items()):
                    valori.append(val)
                    ind_col.append(j)
                    element_count += 1
                inceput_linii.append(element_count)
            
    
    return n, valori, ind_col, inceput_linii

def gauss_seidel_met_1(n, d, rare, b, eps=1e-10, max_iter=10000):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Inițializare cu valori specificate
    # x = np.zeros(n)  # Inițializare cu zero
    
    k = 0  # Contor pentru numărul de iterații
    
    while k < max_iter:
        x_old = x.copy()
        
        for i in range(n):
            suma = 0.0
            diag = d[i]
            for val, j in rare[i]:
                if j != i:
                    suma += val * x[j]  # Elemente non-diagonale
            if diag == 0:
                raise ValueError("Element diagonal zero. Metoda Gauss-Seidel nu poate fi aplicată.")
            x[i] = (b[i] - suma) / diag  # Actualizare Gauss-Seidel
        
        print(f"Iteratia {k}, {x}")
        k += 1
        
        # Verificare criteriu de oprire ||x - x_old|| < eps
        if np.linalg.norm(x - x_old, ord=np.inf) < eps:
            break
    
    return x, k

def gauss_seidel_met_2(n, valori, ind_col, inceput_linii, b, eps=1e-10, max_iter=10000):
    # x = np.zeros(n)  # Inițializare cu zero
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Inițializare cu valori specificate
    k = 0  # Contor pentru numărul de iterații
    
    while k < max_iter:
        x_old = x.copy()
        
        for i in range(n):
            suma = 0.0
            diag = 0.0
            
            for idx in range(inceput_linii[i], inceput_linii[i+1]):
                j = ind_col[idx]
                val = valori[idx]
                
                if j == i:
                    diag = val  # Element diagonal
                else:
                    suma += val * x[j]  # Elemente non-diagonale
            
            if diag == 0:
                raise ValueError("Element diagonal zero. Metoda Gauss-Seidel nu poate fi aplicată.")
            
            x[i] = (b[i] - suma) / diag  # Actualizare Gauss-Seidel
        k += 1
        
        # Verificare criteriu de oprire ||x - x_old|| < eps
        if np.linalg.norm(x - x_old, ord=np.inf) < eps:
            break
    
    return x, k

def verif_diag_elem_nenule(d, n):
    if len(d)!= n:
        return False
    return True

#Variabile generale
eps = 1e-15
b = [6.0, 7.0, 8.0, 9.0, 1.0]  # Vector b

# Metoda 1:
n, d, rare = citire_matrice_met1("tema3files/a_test.txt")
print("Dimensiune:", n)
print("Diagonală:", d)
print("Elemente nenule non-diagonale:", rare)
if (verif_diag_elem_nenule(d, n) == True):
    sol, nr_iter_dict = gauss_seidel_met_1(n, d, rare, b, eps)
    #Afișare pentru verificare
    print("Soluția Metoda 1 cu Dicționar:", sol)
    print("Număr de iterații Metoda 1 cu Dicționar:", nr_iter_dict)

    # Metoda 2:
    n, valori, ind_col, inceput_linii = citire_matrice_met2("tema3files/a_test.txt")

    sol_crs, iters_crs = gauss_seidel_met_2(n, valori, ind_col, inceput_linii, b, eps)
    print("Soluția CRS:", sol_crs)
    print("Număr de iterații CRS:", iters_crs)

    print("Vector valori:", valori)
    print("Vector ind_col:", ind_col)
    print("Vector inceput_linii:", inceput_linii)
else:
    print("Metoda Gauss-Seidel nu poate fi aplicată deoarece elementele diagonale sunt nule.")