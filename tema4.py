import numpy as np
import matplotlib.pyplot as plt

# Funcția pentru a genera matricea A de dimensiune n
def genereaza_A(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1  # Diagonala principală
        if i < n - 1:
            A[i, i+1] = 2  # Prima diagonală superioară
    return A

# Norme necesare pentru inițializare
def norma_1(A):
    return np.max(np.sum(np.abs(A), axis=0))  # suma maximă a coloanelor

def norma_inf(A):
    return np.max(np.sum(np.abs(A), axis=1))  # suma maximă a rândurilor

# Inițializare V0
def initializeaza_V0(A):
    A_T = A.T
    norm1 = norma_1(A)
    norminf = norma_inf(A)
    return A_T / (norm1 * norminf)

# C = a * I_n - A * V, dar scris eficient ca B = -A, C = B @ V, apoi adăugăm a pe diagonală
def matrice_ai_minus_AV(a, B, V):
    C = B @ V
    C[np.diag_indices_from(C)] += a
    return C

# Formula (1): Schultz
def iteratia_schultz(V, B):
    C = matrice_ai_minus_AV(2, B, V)
    return V @ C

# Formula (2): Li și Li - varianta 1
def iteratia_Li_1(V, A, B):
    I = np.eye(A.shape[0])
    C = matrice_ai_minus_AV(3, B, V)
    return V @ (3 * I - A @ V @ C)

# Formula (3): Li și Li - varianta 2 (nu folosește B)
def iteratia_Li_2(V, A):
    I = np.eye(A.shape[0])
    E = I - V @ A
    F = 3 * I - V @ A
    return (I + 0.25 * E @ F @ F) @ V

# Funcția principală
def aprox_inverse(A, metoda="schultz", epsilon=1e-10, k_max=10000):
    V = initializeaza_V0(A)
    B = -A  # ✅ calculăm o singură dată în tot programul
    k = 0

    while True:
        V_prev = V.copy()

        if metoda == "schultz":
            V = iteratia_schultz(V_prev, B)
        elif metoda == "li1":
            V = iteratia_Li_1(V_prev, A, B)
        elif metoda == "li2":
            V = iteratia_Li_2(V_prev, A)
        else:
            raise ValueError("Metodă necunoscută: folosește 'schultz', 'li1' sau 'li2'")

        delta_V = np.linalg.norm(V - V_prev)
        k += 1

        if delta_V < epsilon or k > k_max or delta_V > 1e10:
            break

    if delta_V < epsilon:
        return V, k
    else:
        print(f"Divergență după {k} pași")
        return None, k
    
# Funcție pentru a investiga forma inversei pentru diferite valori ale lui n
def investiga_inversa():
    # Vom analiza matricea pentru diferite valori ale lui n
    for n in range(2, 7):
        print(f"\n========== Analiza pentru n = {n} ==========")
        A = genereaza_A(n)
        print("Matricea A:")
        print(A)
        
        # Calculăm inversa exactă
        inv_exact = np.linalg.inv(A)
        print("\nInversa exactă:")
        print(inv_exact)
        
        # Observăm tiparul din inversa exactă
        print("\nObservații:")
        for i in range(n):
            for j in range(n):
                if abs(inv_exact[i, j]) > 1e-10:  # Ignorăm valorile foarte apropiate de zero
                    print(f"A_inv[{i}, {j}] = {inv_exact[i, j]}")

def forma_generala_inversa(n):
    A_inv = np.zeros((n, n))
    
    # Completăm matricea conform formulei deduse inductiv
    for i in range(n):
        for j in range(i, n):
            if i == j:
                A_inv[i, j] = 1  # Diagonala principală
            else:
                # Calculăm valoarea conform formulei deduse: (-2)^(j-i)
                A_inv[i, j] = (-2)**(j-i)
    
    return A_inv

def compara_inverse(n, metoda="schultz"):
    A = genereaza_A(n)
    
    # Inversa exactă calculată cu numpy
    inv_exact_np = np.linalg.inv(A)
    
    # Inversa calculată cu formula dedusă
    inv_exact_formula = forma_generala_inversa(n)
    
    # Inversa aproximată cu algoritm iterativ
    inv_aprox, k = aprox_inverse(A, metoda=metoda)
    
    # Calculăm și afișăm normele diferențelor
    norma_formula_np = np.linalg.norm(inv_exact_formula - inv_exact_np)
    print(f"||A^(-1)_formula - A^(-1)_numpy|| = {norma_formula_np:.4e}")
    
    if inv_aprox is not None:
        norma_aprox_exact = np.linalg.norm(inv_aprox - inv_exact_formula)
        print(f"||A^(-1)_exact - A^(-1)_aprox|| = {norma_aprox_exact:.4e}")
        print(f"Metoda {metoda} a converged în {k} iterații")
    else:
        print(f"Metoda {metoda} nu a converged.")
    
    return inv_exact_formula, inv_aprox

# Funcție pentru a verifica dacă inversa calculată cu formula este corectă
def verifica_inversa(n):
    A = genereaza_A(n)
    A_inv = forma_generala_inversa(n)
    
    # Verificăm dacă A * A_inv = I
    produs = A @ A_inv
    identitate = np.eye(n)
    eroare = np.linalg.norm(produs - identitate)
    
    print(f"Verificare pentru n = {n}:")
    print(f"||A * A_inv - I|| = {eroare:.4e}")
    
    return eroare < 1e-10

# Funcția pentru testare și vizualizare
def analiza_completa():
    # Investigăm inversa pentru diferite valori de n pentru a deduce forma generală
    investiga_inversa()
    
    print("\n\n========== VERIFICARE FORMULA ==========")
    # Verificăm formula pentru diferite valori de n
    for n in range(2, 10):
        verifica_inversa(n)
    
    print("\n\n========== COMPARARE METODE ==========")
    # Comparăm metodele pentru n = 10
    n = 10
    print(f"\nComparație pentru n = {n}:")
    inv_exact, _ = compara_inverse(n, "schultz")
    print("\nSchultz:")
    _, inv_schultz = compara_inverse(n, "schultz")
    print("\nLi varianta 1:")
    _, inv_li1 = compara_inverse(n, "li1")
    print("\nLi varianta 2:")
    _, inv_li2 = compara_inverse(n, "li2")
    
    # Vizualizăm inversa matricei pentru n = 10
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(inv_exact, cmap='viridis')
    plt.colorbar()
    plt.title('Inversa exactă (formula)')
    
    if inv_schultz is not None:
        plt.subplot(2, 2, 2)
        plt.imshow(inv_schultz, cmap='viridis')
        plt.colorbar()
        plt.title('Inversa aproximată (Schultz)')
    
    if inv_li1 is not None:
        plt.subplot(2, 2, 3)
        plt.imshow(inv_li1, cmap='viridis')
        plt.colorbar()
        plt.title('Inversa aproximată (Li1)')
    
    if inv_li2 is not None:
        plt.subplot(2, 2, 4)
        plt.imshow(inv_li2, cmap='viridis')
        plt.colorbar()
        plt.title('Inversa aproximată (Li2)')
    
    plt.tight_layout()
    plt.savefig('inverse_comparison.png')
    plt.close()
    
    print("\nVizualizare salvată în fișierul 'inverse_comparison.png'")

if __name__ == "__main__":
    analiza_completa()