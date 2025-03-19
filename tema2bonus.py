import numpy as np

def index_in_vector(i, j):
    return i * (i + 1) // 2 + j if i >= j else j * (j + 1) // 2 + i

def descompunere_LU_optimizata(A, dU, epsilon=1e-15):
    n = A.shape[0]
    #alocarea vectorilor L si U
    size = n * (n + 1) // 2
    L_vec = np.zeros(size)
    U_vec = np.zeros(size)
    
    success = True
    
    if np.any(np.abs(dU) < epsilon):
        return False, L_vec, U_vec
    
    for i in range(n):

        #calcul pentru U (coltul dreapta sus)
        for j in range(i, n):
            
            sum_val = sum(L_vec[index_in_vector(i, k)] * U_vec[index_in_vector(k, j)] for k in range(i))
            
            #elementul de pe diagonala principala
            if j == i:
                L_vec[index_in_vector(i, i)] = (A[i, i] - sum_val) / dU[i]
                U_vec[index_in_vector(i, i)] = dU[i]
            else:
                U_vec[index_in_vector(i, j)] = (A[i, j] - sum_val) / L_vec[index_in_vector(i, i)]
        
        #calcul pentru L (coltul stanga jos)
        for k in range(i + 1, n):
            sum_val = sum(L_vec[index_in_vector(k, j)] * U_vec[index_in_vector(j, i)] for j in range(i))
            L_vec[index_in_vector(k, i)] = (A[k, i] - sum_val) / dU[i]
        
        if abs(L_vec[index_in_vector(i, i)]) < epsilon:
            success = False
            break
    
    return success, L_vec, U_vec

def rezolvare_x_optimizata(L_vec, U_vec, dU, b, n):
    # substitutia directa (Ly = b)
    y = np.zeros(n)
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val = sum(L_vec[index_in_vector(i, j)] * y[j] )
        y[i] = (b[i] - sum_val) / L_vec[index_in_vector(i, i)]
    
    # substitutia inversa (Ux = y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_val = 0
        for j in range(i+1, n):
            sum_val = sum(U_vec[index_in_vector(i, j)] * x[j] )
        x[i] = (y[i] - sum_val) / dU[i]
    
    return x

def reconstruieste_A_din_LU(L_vec, U_vec, dU, n):
    A_reconstruit = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sum_val = 0
            for k in range(n):
                L_val = L_vec[index_in_vector(i, k)] if i >= k else 0.0
                U_val = U_vec[index_in_vector(k, j)] if k <= j else 0.0
                if k == j:
                    U_val = dU[k]
                sum_val += L_val * U_val
            A_reconstruit[i, j] = sum_val
    return A_reconstruit

def example_optimized():
    print('Pentru random scrie \'random\', iar pentru exemplu mic scrie \'mic\'')
    tip = input()
    if(tip == 'random'):
        print('Introduceti dimensiunea matricei:')
        n = int(input())
        print('Introduceti epsilon:')
        epsilon = float(input())

        A = np.random.rand(n, n) * 10  
        dU = np.random.rand(n) * 10 + epsilon  
        b = np.random.rand(n) * 10  
    else: 
        if(tip == 'mic'):
            n = 3
            A = np.array([
                [4.0, 0.0, 5.0],
                [1.0, 7.0, 2.0],
                [2.0, 4.0, 6.0]
            ])            
            dU = np.array([2.0, 4.0, 1.0])
            b = np.array([6.0, 5.0, 12.0])
            epsilon = 1e-15
    
    success, L_vec, U_vec = descompunere_LU_optimizata(A, dU, epsilon)
    print("L_vec:", L_vec)
    print("U_vec:", U_vec)
    if success:
        print("Descompunerea LU a fost facuta cu succes!")
        A_reconstruit = reconstruieste_A_din_LU(L_vec, U_vec, dU, n)
        print("A reconstruit:\n", A_reconstruit)
        
        x = rezolvare_x_optimizata(L_vec, U_vec, dU, b, n)
        print("Solutia x:", x)
        residual = np.subtract(np.dot(A, x), b)
        print("Norma reziduala:", np.linalg.norm(residual))
    else:
        print("Descompunerea LU a dat fail!")

if __name__ == "__main__":
    example_optimized()
