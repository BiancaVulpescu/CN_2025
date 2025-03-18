import numpy as np

def descompunerea_LU(A, dU, epsilon=1e-15):
    
    n = A.shape[0]
    success = True
    
    # verificare suplimentara pentru a evita impartirea la 0
    if np.any(np.abs(dU) < epsilon):
        return False
    
    # calculul primei coloane din L
    for i in range(n):
        A[i, 0] = A[i, 0] / dU[0]
    
    # calculul primei linii din U fara diagonala
    for j in range(1, n):
        if np.abs(A[0, 0]) <  epsilon:
            return False
        else:
            A[0, j] = A[0, j] / A[0, 0]   # scoate u01 si u02
    
    for p in range(1, n): # merge pe toate liniile
        #pune pe partea lui U pe linii
        for j in range(p, n): 
            sum_val = 0
            for k in range(p): 
                sum_val += A[p, k] * A[k, j] #calculeaza sumele de simetrice pana la diagonala
            
            if j == p:
                # calculeaza elementul de pe diagonala de pe L
                A[p, p] = (A[p, p] - sum_val) / dU[p]
            else:
                # calculeaza elementele de pe linia p de pe U
                if(abs(A[p, p]) < epsilon):
                    return False
                else:
                    A[p, j] = (A[p, j] - sum_val) / A[p, p]
        
        # calculeaza elementele din L pe coloana p pana la diagonala
        for i in range(p+1, n):
            sum_val = 0
            for k in range(p):
                sum_val += A[i, k] * A[k, p]
            A[i, p] = (A[i, p] - sum_val) / dU[p]
        
        # test final de verificare a elementului de pe diagonala impartirea la 0
        if abs(A[p, p]) < epsilon:
            success = False
            break
    
    return success

def rezolvare_sistem(A, dU, b, epsilon=1e-15):

    n = A.shape[0]
    
    # substitutia directa (Ly = b)
    y = np.zeros(n)
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += A[i, j] * y[j] 
        y[i] = (b[i] - sum_val) / A[i, i]  
    
    # substitutia inversa (Ux = y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_val = 0
        for j in range(i+1, n):
            sum_val += A[i, j] * x[j]  
        x[i] = (y[i] - sum_val) / dU[i]  
    return x

def solve_lu_system_library(A, b):
    x = np.linalg.solve(A, b)
    return x

def compute_residual_norm(residual):
    return np.sqrt(np.sum(residual**2))

def compute_determinant(A, dU):
    n = A.shape[0]
    detA = 1
    for i in range(n):
        detA *= A[i][i]
    for i in dU:
        detA *= i
    return detA

def example_custom_diagonals():
    print('Pentru random scrie \'random\', iar pentru exemplu mic scrie \'mic\'')
    tip = input()
    if(tip == 'random'):
        print('Introduceti dimensiunea matricei:')
        n = int(input())
        print('Introduceti epsilon:')
        epsilon = float(input())
        A = np.random.rand(n, n) * 10  
        A_original = A.copy()  
        print("Matricea A initiala este:\n", A)
        
        dU = np.random.rand(n) * 10 + epsilon  
        print("Diagonala U este: ", dU)
        
        b = np.random.rand(n) * 10  
        print("Vectorul b este: ", b)
    
    else: 
        if(tip == 'mic'):
            n = 3
            A = np.array([
                [4.0, 0.0, 4.0],
                [1.0, 4.0, 2.0],
                [2.0, 4.0, 6.0]
            ])
            A_original = A.copy()  
            print("Matricea A initiala este:\n", A)
            dU = np.array([2.0, 4.0, 1.0])
            print("Diagonala U este: ", dU)
            b = np.array([6.0, 5.0, 12.0])
            print("Vectorul b este: ", b)
            epsilon = 1e-15
    if np.any(np.abs(dU) < epsilon):
        print("Nu se poate face despompunerea LU cu valori mai mici decat epsilon intrucat se va ajunge la impartirea la 0")
        return
    
    success = descompunerea_LU(A, dU, epsilon)
    
    if success:
       
        print("Descompunerea A = LU este:\n", A)

        detA = compute_determinant(A, dU)
        print("Determinant of A:", detA)

        # extragem matricele L si U pentru verificare
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i > j:  
                    L[i, j] = A[i, j]
                elif i == j: 
                    L[i, j] = A[i, j]
                    U[i, j] = dU[i]
                else:  
                    U[i, j] = A[i, j]
        
        print("Matricea L:\n", L)
        print("Matricea U:\n", U)  
        print("Verificam L*U:\n", np.dot(L, U))
        print("Matricea A initiala:\n", A_original)        
        
        xLU = rezolvare_sistem(A, dU, b, epsilon)
        
        if xLU is not None:
            print("\nSolutia xLU este:", xLU)
            
            residual = np.subtract(np.dot(A_original, xLU), b)
            residual_norm = compute_residual_norm(residual)
            print("Norma reziduala ||Ax - b||2:", residual_norm)


            x_lib = solve_lu_system_library(A_original, b)
            print('Solutia folosind biblioteca numpy: ', x_lib)
            
            residual = np.subtract(xLU,x_lib)
            residual_norm = compute_residual_norm(residual)
            print("Norma reziduala ||xLU - x_lib||2:", residual_norm)


            A_inv = np.linalg.inv(A_original)
            print("Inversa lui A este:\n", A_inv)

            residual = np.subtract(xLU, np.dot(A_inv, b))
            residual_norm = compute_residual_norm(residual)
            print("Norma reziduala ||xLU - A_inv*b||2:", residual_norm)

        else:
            print("Rezolvarea sistemului a dat fail")
        
    else:
        print("Descompunerea LU a dat fail")

if __name__ == "__main__":
    example_custom_diagonals()