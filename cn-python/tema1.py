from functools import reduce
import math
import random
import struct
import time


def run_all():
    u = 0
    def setU(value):
        global u
        u = value
    def getU():
        return u

    def find_machine_epsilon():
        m = 0
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
    print("ex1")
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
        print(f"(x+y)+z={(x+y)+z}")
        print(f"x+(y+z)={x+(y+z)}")
    print("ex2a")
    addition_nonassociativity()

    def float2bin(f):
        #pack the float f as a double precision binary big endian (most significant bit first) string
        packed = struct.pack('!d', f)
        #unpack the double precision binary string as an unsigned long long
        #[0] the first and only element of the tuple returned by struct.unpack
        unpacked = struct.unpack('!Q', packed)[0]
        return f'{unpacked:064b}'

    def random_numbers():
        d = random.uniform(0, 1)
        e = random.uniform(0, 1)
        f = random.uniform(0, 1)
        print("d= ", float2bin(d))
        print("e= ", float2bin(e))
        print("f= ", float2bin(f))
        return d, e, f
    def good_example():
        #contains one representable number => asociative
        print('0.5 = ', float2bin(0.5))
        print('0.1 = ', float2bin(0.1))
        print('0.2 = ', float2bin(0.2))
        return 0.5, 0.1, 0.2
    def bad_example():
        #contains unrepresentable numbers => non-asociative
        print('0.1 = ', float2bin(0.1))
        print('0.2 = ', float2bin(0.2))
        print('0.3 = ', float2bin(0.3))
        return 0.1, 0.2, 0.3

        
    def multiplication_nonassociativity():
        
        x, y, z = good_example()
        a, b, c = bad_example()

        print(f"x = {x}, y = {y}, z= {z}" )
        print(f"(x*y)*z={(x*y)*z}")
        print(f"x*(y*z)={x*(y*z)}")
        if (x*y)*z!=x*(y*z):
            print("Inmultirea nu este asociativa:)")
        else: 
            print("Inmultirea este asociativa:()")
        print(f"a = {a}, b = {b}, c= {c}" )
        print(f"(a*b)*c={(a*b)*c}")
        print(f"a*(b*c)={a*(b*c)}")
        if (a*b)*c!=a*(b*c):
            print("Inmultirea nu este asociativa:)")
        else: 
            print("Inmultirea este asociativa:()")
    print("ex2b")
    multiplication_nonassociativity()

    def horner_evaluate(coeff, value):
        if len(coeff) == 1:
            return coeff[0]
        return value*horner_evaluate(coeff[1:], value)+coeff[0]

    def sinus():

        # def P1(x, c1, c2) : return x*(1 + x**2*(c1 + x**2*c2))
        # def P2(x, c1, c2, c3) : return x*(1 + x**2*(-c1 + x**2*(c2 - x**2*c3)))
        # def P3(x, c1, c2, c3, c4) : return x*(1 + x**2*(-c1 + x**2*(c2 + x**2*(-c3 + x**2*c4))))
        # def P4(x, c3, c4) : return x*(1 + x**2*(-0.166 + x**2*(0.00833 + x**2*(-c3 + x**2*c4))))
        # def P5(x, c3, c4) : return x*(1 + x**2*(-0.1666 + x**2*(0.008333 + x**2*(-c3 + x**2*c4))))
        # def P6(x, c3, c4) : return x*(1 + x**2*(-0.16666 + x**2*(0.0083333 + x**2*(-c3 + x**2*c4))))
        # def P7(x, c1, c2, c3, c4, c5) : return x*(1 + x**2*(-c1 + x**2*(c2 + x**2*(-c3 + x**2*(c4 + x**2*(-c5))))))
        # def P8(x, c1, c2, c3, c4, c5, c6) : return x*(1 + x**2*(-c1 + x**2*(c2 + x**2*(-c3 + x**2*(c4 + x**2*(-c5+ x**2*c6))))))

        c1 = 0.16666666666666666666666666666667
        c2 = 0.00833333333333333333333333333333
        c3 = 1.984126984126984126984126984127e-4
        c4 = 2.7557319223985890652557319223986e-6
        c5 = 2.5052108385441718775052108385442e-8
        c6 = 1.6059043836821614599392377170155e-10
        timep1 = 0
        timep2 = 0
        timep3 = 0
        timep4 = 0
        timep5 = 0
        timep6 = 0
        timep7 = 0
        timep8 = 0

        best_polynomials = [0] * 9
        for _ in range(10000):
            nr = random.uniform(-math.pi/2, math.pi/2)
            val_sin_exact = math.sin(nr)
            
            # p1 = P1(nr, c1, c2)
            # p2 = P2(nr, c1, c2, c3)
            # p3 = P3(nr, c1, c2, c3, c4)
            # p4 = P4(nr, c3, c4)
            # p5 = P5(nr, c3, c4)
            # p6 = P6(nr, c3, c4)
            # p7 = P7(nr, c1, c2, c3, c4, c5)
            # p8 = P8(nr, c1, c2, c3, c4, c5, c6)

            errors = []
            start_time = time.time_ns()
            p1 = horner_evaluate([1, 0, -c1, 0, c2], nr)
            errorp1 = abs(p1 - val_sin_exact)
            timep1 += time.time_ns()- start_time
            errors.append((1, errorp1))
            
            start_time = time.time_ns()
            p2 = horner_evaluate([1, 0, -c1, 0, c2, 0, -c3], nr)
            errorp2 = abs(p2 - val_sin_exact)
            timep2 += time.time_ns()- start_time
            errors.append((2, errorp2))

            start_time = time.time_ns()
            p3 = horner_evaluate([1, 0, -c1, 0, c2, 0, -c3, 0, c4], nr)
            errorp3 = abs(p3 - val_sin_exact)
            timep3 += time.time_ns()- start_time
            errors.append((3, errorp3))

            start_time = time.time_ns()
            p4 = horner_evaluate([1, 0, -0.166, 0, 0.00833, 0, -c3, 0, c4], nr)
            errorp4 = abs(p4 - val_sin_exact)
            timep4 += time.time_ns()- start_time
            errors.append((4, errorp4))

            start_time = time.time_ns()
            p5 = horner_evaluate([1, 0, -0.1666, 0, 0.008333, 0, -c3, 0, c4], nr)
            errorp5 = abs(p5 - val_sin_exact)
            timep5 += time.time_ns()- start_time
            errors.append((5, errorp5))
            
            start_time = time.time_ns()
            p6 = horner_evaluate([1, 0, -0.16666, 0, 0.0083333, 0, -c3, 0, c4], nr)
            errorp6 = abs(p6 - val_sin_exact)
            timep6 += time.time_ns()- start_time
            errors.append((6, errorp6))
            
            start_time = time.time_ns()
            p7 = horner_evaluate([1, 0, -c1, 0, c2, 0, -c3, 0, c4, 0, -c5], nr)
            errorp7 = abs(p7 - val_sin_exact)
            timep7 += time.time_ns()- start_time
            errors.append((7, errorp7))
            
            start_time = time.time_ns()
            p8 = horner_evaluate([1, 0, -c1, 0, c2, 0, -c3, 0, c4, 0, -c5, 0, c6], nr)
            errorp8 = abs(p8 - val_sin_exact)
            timep8 += time.time_ns()- start_time
            errors.append((8, errorp8))

            errors.sort(key = lambda x : x[1])
            
            for (idx, err) in errors[:3]:
                best_polynomials[idx] += 1

        print(best_polynomials)

        hierarchy = sorted(range(1,9), key=lambda k: best_polynomials[k], reverse=True)
        print(hierarchy[:3])

        best_time_polynomials = [(1,timep1), (2,timep2), (3,timep3), (4,timep4), (5,timep5), (6,timep6), (7,timep7), (8,timep8)]
        best_time_polynomials.sort(key = lambda x : x[1])
        print(best_time_polynomials)

    print("ex3")
    sinus()