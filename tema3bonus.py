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
eps= 1e-10
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
                    if abs(val) >= eps:  
                        valori.append(val)
                        ind_col.append(j)
                        element_count += 1
                inceput_linii.append(element_count)
            
    
    return n, valori, ind_col, inceput_linii

def suma_matrici_met1(n_a, d_a, rare_a, n_b, d_b, rare_b):
    n_sum = max(n_a, n_b)
    d_sum = [0] * n_sum
    rare_sum = {i: {} for i in range(n_sum)}

    for i in range(n_a):
        d_sum[i] += d_a[i]
    for i in range(n_b):
        d_sum[i] += d_b[i]

    for i in range(n_a):
        for val, j in rare_a[i]:
            if j in rare_sum[i]:
                rare_sum[i][j] += val
            else:
                rare_sum[i][j] = val
    for i in range(n_b):
        for val, j in rare_b[i]:
            if j in rare_sum[i]:
                rare_sum[i][j] += val
            else:
                rare_sum[i][j] = val

    for i in range(n_sum):
        rare_sum[i] = [(val, j) for j, val in rare_sum[i].items()]

    return n_sum, d_sum, rare_sum
eps= 1e-10
def suma_matrici_met2(n_a, valori_a, ind_col_a, inceput_linii_a, n_b, valori_b, ind_col_b, inceput_linii_b):
    n_sum = max(n_a, n_b)
    valori_sum = []
    ind_col_sum = []
    inceput_linii_sum = [0]
    for i in range(n_sum):
        row_sum = {}

        # Adăugăm elementele din prima matrice
        if i < n_a:
            for idx in range(inceput_linii_a[i], inceput_linii_a[i+1]):
                j = ind_col_a[idx]
                val = valori_a[idx]
                row_sum[j] = val

        # Adăugăm elementele din a doua matrice
        if i < n_b:
            for idx in range(inceput_linii_b[i], inceput_linii_b[i+1]):
                j = ind_col_b[idx]
                val = valori_b[idx]
                if j in row_sum:
                    row_sum[j] += val
                else:
                    row_sum[j] = val

        for j, val in sorted(row_sum.items()):
             if abs(val) >= eps:
                valori_sum.append(val)
                ind_col_sum.append(j)
        inceput_linii_sum.append(len(valori_sum))

    return n_sum, valori_sum, ind_col_sum, inceput_linii_sum

def verifica_suma_met1(n_sum, d_sum, rare_sum, d_aplusb, rare_aplusb, eps=1e-10):
    # Verificăm diagonalele
    for i in range(n_sum):
        if abs(d_sum[i] - d_aplusb[i]) >= eps:
            return False

    # Verificăm elementele non-diagonale
    for i in range(n_sum):
        rare1 = {j: val for val, j in rare_sum[i]}
        rare2 = {j: val for val, j in rare_aplusb[i]}
        if rare1.keys() != rare2.keys():
            return False
        for j in rare1:
            if abs(rare1[j] - rare2[j]) >= eps:
                return False

    return True

def verif_diag_elem_nenule(d, n):
    if len(d)!= n:
        return False
    return True
def verifica_suma_met2(valori_sum, ind_col_sum, inceput_linii_sum, 
                        valori_aplusb, ind_col_aplusb, inceput_linii_aplusb, eps=1e-10):

     if inceput_linii_sum != inceput_linii_aplusb:
         return False
     if ind_col_sum != ind_col_aplusb:
         return False
     if len(valori_sum) != len(valori_aplusb):
         return False
 
     # Verificăm valorile nenule cu toleranța eps
     for v1, v2 in zip(valori_sum, valori_aplusb):
         if abs(v1 - v2) >= eps:
             return False
 
     return True

eps = 1e-10
a_path = f"tema3files/a.txt"
b_path = f"tema3files/b.txt"
aplusb_path = f"tema3files/aplusb.txt"
print("Metoda1:")
# Metoda 1:
n_a, d_a, rare_a = citire_matrice_met1(a_path)
n_b, d_b, rare_b = citire_matrice_met1(b_path)
n_aplusb, d_aplusb, rare_aplusb = citire_matrice_met1(aplusb_path)

if (n_aplusb != max(n_a,n_b)):
    raise ValueError("Dimensiunea matricei A+B nu corespunde cu dimensiunea matricii A + B.")

if (verif_diag_elem_nenule(d_a, n_a) == True and verif_diag_elem_nenule(d_b, n_b) == True):

    # Metoda 1 Suma dintre cele 2 matrici a si b
    n_sum, d_sum, rare_sum = suma_matrici_met1(n_a, d_a, rare_a, n_b, d_b, rare_b)
    if (verifica_suma_met1(n_sum, d_sum, rare_sum, d_aplusb, rare_aplusb, eps)== True):
        print("Metoda 1: suma matricilor A si B este corecta.")
    else: 
        print("Metoda 1: suma matricilor A si B este incorecta.")

    print("Metoda2:")
    # Metoda 2:
    
    n_a, valori_a, ind_col_a, inceput_linii_a = citire_matrice_met2(a_path)
    n_b, valori_b, ind_col_b, inceput_linii_b = citire_matrice_met2(b_path)
    n_aplusb, valori_aplusb, ind_col_aplusb, inceput_linii_aplusb = citire_matrice_met2(aplusb_path)
    
    n_sum, valori_sum, ind_col_sum, inceput_linii_sum = suma_matrici_met2( n_a, valori_a, ind_col_a, inceput_linii_a, n_b, valori_b, ind_col_b, inceput_linii_b)
    
    if (verifica_suma_met2(valori_sum, ind_col_sum, inceput_linii_sum, valori_aplusb, ind_col_aplusb, inceput_linii_aplusb, eps) == True):
        print("Suma este corectă (metoda 2).")
    else:
        print("Suma este incorectă (metoda 2).")

else: 
    print("Metoda Gauss-Seidel nu poate fi aplicată deoarece elementele diagonale sunt nule.")
