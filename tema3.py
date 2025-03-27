def citire_matrice_rara(nume_fisier):
    with open(nume_fisier, "r") as f:
        # Citim dimensiunea matricei
        n = int(f.readline().strip())
        
        # Inițializăm vectorul diagonal și dicționarul pentru elementele nenule
        d = [0] * n
        rare = {i: [] for i in range(n)}

        # Citim fiecare element nenul
        for linie in f:
            linie = linie.strip()
            if not linie:  # Ignorăm liniile goale
                continue
            val, i, j = map(str.strip, linie.split(","))
            val, i, j = float(val), int(i), int(j)  # Convertim la tipurile corecte

            if i == j:
                d[i] = val  # Element diagonal
            else:
                rare[i].append((val, j))  # Element nenul non-diagonal

    return n, d, rare

# Exemplu de utilizare:
n, d, rare = citire_matrice_rara("tema3files/a_test.txt")

# Afișare pentru verificare
print("Dimensiune:", n)
print("Diagonală:", d)
print("Elemente nenule non-diagonale:", rare)
