#pentru metoda 2 de calcul : 
#https://edu.info.uaic.ro/calcul-numeric/CN/lab/3/smr.pdf
#procent de rezolvare cu AI 20%
import numpy as np
def citire_vector_rar (nume_fisier):
    vector_rar = {}
    with open(nume_fisier, "r") as f:
        lungime = int(f.readline().strip())
        for index, linie in enumerate(f):
            linie = linie.strip()
            if linie:
                valoare = float(linie)
                vector_rar[index] = valoare
    return vector_rar, lungime 

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
            if(val!=0):
                if i == j:
                    d[i] += val  
                else:
                    if j in rare[i]:
                        rare[i][j] += val  
                    else:
                        rare[i][j] = val  
    for i in range(n):
        rare[i] = [(val, j) for j, val in rare[i].items()]

    return n, d, rare

def citire_matrice_met2(file_path):
    valori = []  
    ind_col = []  
    inceput_linii = [0]  
    
    with open(file_path, 'r') as file:
        n = int(file.readline().strip())  
        element_count = 0
        row_data = {} 

        for line in file:
            line = line.strip()
            if not line:
                continue
            val, i, j = map(float, line.split(','))
            i, j = int(i), int(j)
            
            if i not in row_data:
                row_data[i] = {}
            
            if j in row_data[i]:
                row_data[i][j] += val
            else:
                row_data[i][j] = val
        for i in range(n):
            if i in row_data:
                for j, val in sorted(row_data[i].items()):
                    valori.append(val)
                    ind_col.append(j)
                    element_count += 1
                inceput_linii.append(element_count)
            
    
    return n, valori, ind_col, inceput_linii

def gauss_seidel_met_1(n, d, rare, b, eps=1e-10, max_iter=10000):
    x = np.zeros(n)  
    k = 0  
    
    while k < max_iter:
        x_old = x.copy()
        
        for i in range(n):
            suma = 0.0
            diag = d[i]
            for val, j in rare[i]:
                if j != i:
                    suma += val * x[j]  
            if diag == 0:
                raise ValueError("Element diagonal zero. Metoda Gauss-Seidel nu poate fi aplicată.")
            x[i] = (b[i] - suma) / diag  # Actualizare Gauss-Seidel
            
        
        k += 1
        
        # Verificare criteriu de oprire ||x - x_old|| < eps
        if np.linalg.norm(x - x_old, ord=np.inf) < eps:
            break
        if np.linalg.norm(x - x_old, ord=np.inf) > 100000000000000000000:
            print("Sistemul nu converge.")
            break
    if(k == max_iter):
        print("Numar maxim de iteratii atins.")
    return x, k

def gauss_seidel_met_2(n, valori, ind_col, inceput_linii, b, eps=1e-10, max_iter=10000):
    x = np.zeros(n)  
    k = 0  
    
    while k < max_iter:
        x_old = x.copy()
        
        for i in range(n):
            suma = 0.0
            diag = 0.0
            
            for idx in range(inceput_linii[i], inceput_linii[i+1]):
                j = ind_col[idx]
                val = valori[idx]
                
                if j == i:
                    diag = val 
                else:
                    suma += val * x[j] 
            
            if diag == 0:
                raise ValueError("Element diagonal zero. Metoda Gauss-Seidel nu poate fi aplicată.")
            
            x[i] = (b[i] - suma) / diag  # Actualizare Gauss-Seidel
        k += 1
        
        # Verificare criteriu de oprire ||x - x_old|| < eps
        if np.linalg.norm(x - x_old, ord=np.inf) < eps:
            break
        if np.linalg.norm(x - x_old, ord=np.inf) > 100000000000000000000:
            print("Sistemul nu converge.")
            break
    if(k == max_iter):
        print("Numar maxim de iteratii atins.")
    return x, k

def verif_diag_elem_nenule(d, n):
    if len(d)!= n:
        return False
    return True

def calc_norma_1(n, d, rare, x, b):
    # Calculam A * x
    Ax = np.zeros(n)
    for i in range(n):
        suma = d[i] * x[i]  
        for val, j in rare[i]:
            suma += val * x[j] 
        Ax[i] = suma
    
    # Calculam reziduurile A * x - b
    reziduuri = np.zeros(n)
    for i in range(n):
        reziduuri[i] = Ax[i] - b[i]
    
    # Norma infinit
    norma_inf = np.linalg.norm(reziduuri, ord=np.inf)
    return norma_inf

def calc_norma_2(n, valori, ind_col, inceput_linii, x, b):
    # Calculam A * x
    Ax = np.zeros(n)
    for i in range(n):
        suma = 0.0
        for idx in range(inceput_linii[i], inceput_linii[i+1]):
            j = ind_col[idx]
            suma += valori[idx] * x[j]
        Ax[i] = suma
    
    # Calculam reziduurile A * x - b
    reziduuri = np.zeros(n)
    for i in range(n):
        reziduuri[i] = Ax[i] - b[i] 
    
    # Norma infinit
    norma_inf = np.linalg.norm(reziduuri, ord=np.inf)
    return norma_inf

#Variabile generale
eps = 1e-15
for i in range(1,6):
    print("Iteratia", i)
    a_path = f"tema3files/a_{i}.txt"
    b_path = f"tema3files/b_{i}.txt"
    b, lung_b = citire_vector_rar(b_path)
    print("Metoda1:")
    # Metoda 1:
    n, d, rare = citire_matrice_met1(a_path)
    if (verif_diag_elem_nenule(d, n) == True):

        # print("Dimensiune:", n)
        # print("Diagonală:", d)
        # print("Elemente nenule non-diagonale:", rare)
        sol, nr_iter_dict = gauss_seidel_met_1(n, d, rare, b, eps)
        if lung_b != n:
            raise ValueError("Dimensiunea vectorului b nu corespunde cu dimensiunea matricei A.")
        #Afisare pentru verificare
        print("Soluția Metoda 1 cu Dicționar:", sol)
        print("Număr de iterații Metoda 1 cu Dicționar:", nr_iter_dict)
        # Calcul norma infinit pentru metoda 1
        norma_met_1 = calc_norma_1(n, d, rare, sol, b)
        print("Norma infinit pentru metoda 1:", norma_met_1)


        print("Metoda2:")
        # Metoda 2:
        n, valori, ind_col, inceput_linii = citire_matrice_met2(a_path)
        if lung_b != n:
            raise ValueError("Dimensiunea vectorului b nu corespunde cu dimensiunea matricei A.")  
        sol_crs, iters_crs = gauss_seidel_met_2(n, valori, ind_col, inceput_linii, b, eps)
        print("Soluția Metoda 2 cu CRS:", sol_crs)
        print("Număr de iterații Metoda 2 cu CRS:", iters_crs)
        #Calcul norma infinit pentru metoda 2
        norma_met2 = calc_norma_2(n, valori, ind_col, inceput_linii, sol_crs, b)
        print("Norma infinit pentru metoda 2:", norma_met2)
        # print("Vector valori:", valori)
        # print("Vector ind_col:", ind_col)
        # print("Vector inceput_linii:", inceput_linii)
    else: 
        print("Metoda Gauss-Seidel nu poate fi aplicată deoarece elementele diagonale sunt nule.")

