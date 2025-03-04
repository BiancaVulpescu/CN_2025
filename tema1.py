#ex1
def find_machine_epsilon():
    m = 1
    u = 10 ** -m

    while (1.0 + u) != 1.0:  
        m += 1
        u = 10 ** -m
    print(f"Precizia masina este: {u}, iar m este {m}")

    if (1.0 + 10 ** -(m+1) == 1.0):
        print("Verificare trecuta cu succes")
    else: 
        print("Verificare esuata")

find_machine_epsilon()
