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
eps= 1e-10
def citire_matrice_met2(file_path):
    valori = []  # Store non-zero values
    ind_col = []  # Store column indices
    inceput_linii = [0]  # Start positions for each row (0-based indexing)
    
    with open(file_path, 'r') as file:
        n = int(file.readline().strip())  # Read matrix dimension
        element_count = 0
        row_data = {}  # Temporary dictionary to store values for each row

        for line in file:
            line = line.strip()
            if not line:
                continue
            val, i, j = map(float, line.split(','))
            i, j = int(i), int(j)
            
            if i not in row_data:
                row_data[i] = {}
            
            # Sum values for duplicate indices
            if j in row_data[i]:
                row_data[i][j] += val
            else:
                row_data[i][j] = val
        # Process the row data into compressed row format
        for i in range(n):
            if i in row_data:
                for j, val in sorted(row_data[i].items()):
                    valori.append(val)
                    ind_col.append(j)
                    element_count += 1
                inceput_linii.append(element_count)
            
    
    return n, valori, ind_col, inceput_linii

def suma_matrici_met1(n_a, d_a, rare_a, n_b, d_b, rare_b):
    n_sum = max(n_a, n_b)
    d_sum = [0] * n_sum
    rare_sum = {i: {} for i in range(n_sum)}

    # Adăugăm elementele diagonale
    for i in range(n_a):
        d_sum[i] += d_a[i]
    for i in range(n_b):
        d_sum[i] += d_b[i]

    # Adăugăm elementele non-diagonale
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

    # Convertim dicționarele rare_sum[i] în liste de tupluri
    for i in range(n_sum):
        rare_sum[i] = [(val, j) for j, val in rare_sum[i].items()]

    return n_sum, d_sum, rare_sum

def suma_matrici_met2(n_a, valori_a, ind_col_a, inceput_linii_a, n_b, valori_b, ind_col_b, inceput_linii_b):
    n_sum = max(n_a, n_b)
    valori_sum = []
    ind_col_sum = []
    inceput_linii_sum = [0]
    # reconstruim dictionarul cu valorile sumate
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

        # Adăugăm elementele în format CRS
        for j, val in sorted(row_sum.items()):
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
def verifica_suma_met2(n_sum, valori_sum, ind_col_sum, inceput_linii_sum, 
                       valori_aplusb, ind_col_aplusb, inceput_linii_aplusb, eps=1e-10):
    def crs_to_dict(valori, ind_col, inceput_linii):
        coord_map = {}
        for i in range(len(inceput_linii) - 1):
            start = inceput_linii[i]
            end = inceput_linii[i + 1]
            for idx in range(start, end):
                j = ind_col[idx]
                coord_map[(i, j)] = coord_map.get((i, j), 0) + valori[idx]
        return coord_map

    sum_dict = crs_to_dict(valori_sum, ind_col_sum, inceput_linii_sum)
    expected_dict = crs_to_dict(valori_aplusb, ind_col_aplusb, inceput_linii_aplusb)

    # Check for missing or extra coordinates
    if set(sum_dict.keys()) != set(expected_dict.keys()):
        print("Coordonate lipsă sau în plus:")
        print("Doar în sumă:", set(sum_dict.keys()) - set(expected_dict.keys()))
        print("Doar în aplusb:", set(expected_dict.keys()) - set(sum_dict.keys()))
        return False

    # Check for mismatched values
    for coord in sum_dict:
        diff = abs(sum_dict[coord] - expected_dict[coord])
        if diff >= eps:
            print(f"Valoare diferită la {coord}: sum={sum_dict[coord]}, aplusb={expected_dict[coord]}, diff={diff}")
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
    
    if (verifica_suma_met2(n_sum, valori_sum, ind_col_sum, inceput_linii_sum, valori_aplusb, ind_col_aplusb, inceput_linii_aplusb, eps) == True):
        print("Suma este corectă (metoda 2).")
    else:
        print("Suma este incorectă (metoda 2).")

else: 
    print("Metoda Gauss-Seidel nu poate fi aplicată deoarece elementele diagonale sunt nule.")
