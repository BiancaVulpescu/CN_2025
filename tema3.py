import numpy as np
def citire_matrice_rara(nume_fisier):
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

def verif_diag_elem_nenule(d, n):
    if len(d)!= n:
        return False
    return True

def read_sparse_matrix(file_path):
    """Reads a sparse matrix from a file and stores it in compressed row format."""
    valori = []  # Store non-zero values
    ind_col = []  # Store column indices
    inceput_linii = [0]  # Start positions for each row (0-based indexing)
    
    with open(file_path, 'r') as file:
        n = int(file.readline().strip())  # Read matrix dimension
        current_row = 0
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
            inceput_linii.append(element_count)
            if i in row_data:
                for j, val in sorted(row_data[i].items()):
                    valori.append(val)
                    ind_col.append(j)
                    element_count += 1
            
    
    return n, valori, ind_col, inceput_linii

# Exemplu de utilizare:
eps = 1e-15
n, d, rare = citire_matrice_rara("tema3files/a_test.txt")
# verif_diag_elem_nenule(d, n)

# # Afișare pentru verificare
# print("Dimensiune:", n)
# print("Diagonală:", d)
# print("Elemente nenule non-diagonale:", rare)

n, valori, ind_col, inceput_linii = read_sparse_matrix("tema3files/a_test.txt")
    
print("Vector valori:", valori)
print("Vector ind_col:", ind_col)
print("Vector inceput_linii:", inceput_linii)

