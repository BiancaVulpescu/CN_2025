#ex1
import random
import struct


u = 0
def setU(value):
    global u
    u = value
def getU():
    return u

def find_machine_epsilon():
    m = 1
    u = 10 ** -m

    while (1.0 + u) != 1.0:  
        m += 1
        u = 10 ** -m

    m -= 1
    u = 10 ** -(m)
    print(f"Precizia masina este: {u}, iar m este {m}")

    if (1.0 + 10 ** -(m) != 1.0):
        print("Verificare trecuta cu succes")
    else: 
        print("Verificare esuata")
    setU(u)

find_machine_epsilon()

def addition_nonassociativity():
    u = getU()
    x = 1.0
    y = u / 10
    z = u / 10
    if (x+y)+z!=x+(y+z):
        print("Adunarea nu este asociativa:)")
    else: 
        print("Adunarea este asociativa:()")
addition_nonassociativity()


def random_numbers():
    return random.uniform(0, 1),random.uniform(0, 1), random.uniform(0, 1)
def good_example():
    #contains representable numbers
    return 0.5, 0.1, 0.2
def bad_example():
    #contains unrepresentable numbers
    return 0.1, 0.2, 0.3
def float2bin(f):
    ''' Convert float to 64-bit binary string.

    Attributes:
        :f: Float number to transform.
    '''
    # [d] = struct.unpack(">Q", struct.pack(">d", f))
    # return f'{d:064b}'
    bits = struct.unpack('!Q', struct.pack('!d', f))[0]
    return f'{bits:064b}'
    
def multiplication_nonassociativity():
    
    x, y, z = good_example()
    print(float2bin(x))
    print(float2bin(y))
    a, b, c = bad_example()

    print(f"x = {x}, y = {y}, z= {z}" )
    if (x*y)*z!=x*(y*z):
        print("Inmultirea nu este asociativa:)")
    else: 
        print("Inmultirea este asociativa:()")
    print(f"a = {a}, b = {b}, c= {c}" )
    if (a*b)*c!=a*(b*c):
        print("Inmultirea nu este asociativa:)")
    else: 
        print("Inmultirea este asociativa:()")

multiplication_nonassociativity()

