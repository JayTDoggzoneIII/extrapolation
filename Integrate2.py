from math import pi, sin, cos, exp, prod, fabs, log, acos
from random import uniform
from sys import stdin, stdout, setrecursionlimit
from gc import disable

gets = input
puts = print
input = stdin.readline
print = stdout.write


N = 7
a, b, alpha, beta = 2.1, 3.3, 0.4, 0
EPS = 1e-6

def Gauss(A:list, B:list):
    n = len(A)
    a = [[A[i][j] for j in range(n)] for i in range(n)]
    [*b] = B
    det = 1
    
    for i in range(n-1):
        k = i
        for j in range(i+1, n): 
            if (fabs(a[j][i]) > fabs(a[k][i])): k = j
        if (k != i):
            a[i], a[k] = a[k], a[i]
            b[i], b[k] = b[k], b[i]
            det = -det
 
        for j in range(i+1, n):
            t = a[j][i]/a[i][i]
            for k in range(i+1, n): a[j][k] -= t*a[i][k]
            b[j] -= t*b[i]
 
    for i in reversed(range(n)):
        for j in range(i + 1, n):
            t = a[i][j]
            b[i] -= t*b[j]
        t = 1/a[i][i]
        det *= a[i][i]
        b[i] *= t
    return det, b


def f(x:float) -> float:
    return 4.5*cos(7*x)*exp(-2*x/3) + 1.4*sin(1.5*x)*exp(-x/3) + 3;

def p(x:float) -> float:
    return 1/(x-a)**alpha

def F(x:float) -> float:
    return f(x)*p(x);

def Tartaglia_Kardano(pol:list) -> list:
    b,c,d = pol
    
    q = (2*b*b*b/27 - b*c/3 + d)/2
    p = (3*c - b*b)/9
    r = fabs(p)**0.5
    phi = acos(q/r/r/r)
    
    return -2*r*cos(phi/3) - b/3, 2*r*cos(pi/3 - phi/3) - b/3, 2*r*cos(pi/3 + phi/3) - b/3
    
def Durand_Kerner(expr:list) -> list:
    n = 3
    params = [uniform(0,10) for _ in range(n)]   
    pol = lambda x: sum(x**(n-i) * expr[i] for i in range(n+1))
    while (any(abs(pol(params[k])) > 1e-12 for k in range(n))):
        for i in range(n):
            params[i] = params[i] - pol(params[i])/prod((params[i] - params[j]) for j in range(i))/prod((params[i] - params[j]) for j in range(i+1,n))
    return params

def composite_Newton_Cotes_quadrature_rule(f, down:float, up:float, z:list):
    n = 3
    
    k = len(z)-1
    ans = 0
    mu = [[0]*n for i in range(k+1)]
    A = [[0]*n for i in range(k+1)]
    for i in range(1,k+1):
        mu[i][0] = ((z[i] - down)**(1 - alpha) - (z[i-1] - down)**(1 - alpha))/(1 - alpha)
        mu[i][1] = ((z[i] - down)**(2 - alpha) - (z[i-1] - down)**(2 - alpha))/(2 - alpha) + down*mu[i][0]
        mu[i][2] = ((z[i] - down)**(3 - alpha) - (z[i-1] - down)**(3 - alpha))/(3 - alpha) + 2*down*mu[i][1] - down*down*mu[i][0]
    
        z_m = (z[i-1]+z[i])/2
        A[i][0] = (mu[i][2] - mu[i][1]*(z_m + z[i]) + mu[i][0]*z_m*z[i])/(z_m - z[i-1])/(z[i] - z[i-1])
        A[i][1] = -(mu[i][2] - mu[i][1]*(z[i-1] + z[i]) + mu[i][0]*z[i-1]*z[i])/(z_m - z[i-1])/(z[i] - z_m)
        A[i][2] = (mu[i][2] - mu[i][1]*(z_m + z[i-1]) + mu[i][0]*z_m*z[i-1])/(z[i] - z_m)/(z[i] - z[i-1])
        
        ans += A[i][0]*f(z[i-1]) + A[i][1]*f(z_m) + A[i][2]*f(z[i])
    
    return ans;

def composite_Gaussian_quadrature_rule(f, down:float, up:float, z:list):
    n = 3
    
    k = len(z)-1
    
    ans = 0
    mu = [[0]*(2*n) for i in range(k+1)]
    for i in range(1,k+1):
        mu[i][0] = ((z[i] - down)**(1 - alpha) - (z[i-1] - down)**(1 - alpha))/(1 - alpha)
        
        mu[i][1] = ((z[i] - down)**(2 - alpha) - (z[i-1] - down)**(2 - alpha))/(2 - alpha) + down*mu[i][0]
        
        mu[i][2] = ((z[i] - down)**(3 - alpha) - (z[i-1] - down)**(3 - alpha))/(3 - alpha) + 2*down*mu[i][1] - down*down*mu[i][0]
        
        mu[i][3] = ((z[i] - down)**(4 - alpha) - (z[i-1] - down)**(4 - alpha))/(4 - alpha) + 3*down*mu[i][2] - 3*down*down*mu[i][1] + down*down*down*mu[i][0]
        
        mu[i][4] = ((z[i] - down)**(5 - alpha) - (z[i-1] - down)**(5 - alpha))/(5 - alpha) + 4*down*mu[i][3] - 6*down*down*mu[i][2] + 4*down*down*down*mu[i][1] - down**4*mu[i][0]
        
        mu[i][5] = ((z[i] - down)**(6 - alpha) - (z[i-1] - down)**(6 - alpha))/(6 - alpha) + 5*down*mu[i][4] - 10*down*down*mu[i][3] + 10*down*down*down*mu[i][2] - 5*down**4*mu[i][1] + down**5*mu[i][0]
        
        A = [[0]*n for i in range(n)]
        B = [0]*n
        for s in range(n):
            for j in range(n):
                A[s][j] = mu[i][j+s]
            B[s] = -mu[i][n+s]
        det, a_s = Gauss(A,B)
        x_s = Tartaglia_Kardano(reversed(a_s))
        for s in range(n):
            for j in range(n):
                A[s][j] = x_s[j]**s
            B[s] = mu[i][s]
        det, A_s = Gauss(A,B)
        ans += sum(A_s[i]*f(x_s[i]) for i in range(n))
    return ans;

  
def Richardson_extrapolation(I, h:list = [0.6, 0.3, 0.15], down:float = a, up:float = b) -> float:
    z = [[down], [down], [down]]
    
    while (1.2/len(z[0]) >= h[0]):
        z[0].append(z[0][-1]+h[0])
    
    while (1.2/len(z[1]) >= h[1]):
        z[1].append(z[1][-1]+h[1])
        
    while (1.2/len(z[2]) >= h[2]):
        z[2].append(z[2][-1]+h[2])
        
    I_s = [I(f,a,b,z[0]), I(f,a,b,z[1]), I(f,a,b,z[2])]
    
    A = (I_s[2] - I_s[1])/(I_s[1] - I_s[0])
    m = -log(fabs(A))/log(2)
    
    H = [[-1, h[0]**m, h[0]**(m+1)], [-1, h[1]**m, h[1]**(m+1)], [-1, h[2]**m, h[2]**(m+1)]]
    S = [-I_s[0], -I_s[1], -I_s[2]]
    det, JC = Gauss(H,S)
    
    i = 2
    while (fabs(JC[0] - I_s[i-1]) >= EPS):
        i += 1
        h.append(h[i-1]/2)
        
        z.append([down])
        
        while (1.2/len(z[i]) >= h[i]):
            z[i].append(z[i][-1]+h[i])
            
        I_s.append(I(f,a,b,z[i]))
        
        A = (I_s[i] - I_s[i-1])/(I_s[i-1] - I_s[i-2])
        m = -log(fabs(A))/log(2)                    
        puts(m)
        H = [[0]*(i+1) for _ in range(i+1)]
        for k in range(i+1):
            for j in range(i+1):
                H[k][j] = -1 if (not j) else h[k]**(m+j)
        S.append(-I_s[i])
        det, JC = Gauss(H,S)
    
    return I_s[i]+sum(JC[j+1]*h[i]**(m+j) for j in range(i)), h[i], i
     
def main() -> int:
    disable()
    global i1,i2,i3
    F_ans = 4.461512705324278
    
    print("2.1\n")
    
    AnsNK, hNK, i1 = Richardson_extrapolation(composite_Newton_Cotes_quadrature_rule)
    print("  СКФ Ньютона-Котеса с экстраполяцией Ричардсона:\n    S + R = %.14f\n"%(AnsNK))
    print("  Длина шага разбиения интервала:\n    h = %.7f\n\n"%(hNK))
    
    print("2.2\n")
    
    AnsG, hG,i2  = Richardson_extrapolation(composite_Gaussian_quadrature_rule)
    print("  СКФ Гаусса с экстраполяцией Ричардсона:\n    S + R = %.14f\n"%(AnsG))
    print("  Длина шага разбиения интервала:\n    h = %.7f\n\n"%(hG))
    
    print("2.3\n")
    print("  Работаем с СКФ Гаусса\n")
    z = [[],[],[]]
    z[0] = [2.1, 3.3]
    z[1] = [2.1, 2.7, 3.3]
    z[2] = [2.1, 2.4, 2.7, 3, 3.3]
    h = [1.2, 0.6, 0.3]
    I1,I2,I3 = composite_Gaussian_quadrature_rule(f,a,b,z[0]), composite_Gaussian_quadrature_rule(f,a,b,z[1]), composite_Gaussian_quadrature_rule(f,a,b,z[2])
    I_s = [I1, I2, I3]
    A = (I3 - I2)/(I2 - I1)
    m = -log(fabs(A))/log(2)
    Rh2 = (I2 - I1)/(2**m - 1)
    
    h_opt = 0.6*(EPS/fabs(Rh2))**(1/m)
    n = round((b-a)/h_opt)
    h_opt = (b-a)/n
    print("  Используя оценку по Рунге и Эйткену, получаем:\n  Разбиваем Интревал на %d частей\n  h_opt = %.7f\n\n"%(n,h_opt))
    H = [[-1, h[0]**m, h[0]**(m+1)], [-1, h[1]**m, h[1]**(m+1)], [-1, h[2]**m, h[2]**(m+1)]]
    S = [-I_s[0], -I_s[1], -I_s[2]]
    det, JC = Gauss(H,S)
    
    i = 2
    i += 1
    h.append(h_opt)
        
    z4 = [a]
        
    while (1.2/len(z4) >= h_opt):
        z4.append(z4[-1]+h_opt)
            
    I_s.append(composite_Gaussian_quadrature_rule(f,a,b,z4))
        
    A = (I_s[i] - I_s[i-1])/(I_s[i-1] - I_s[i-2])
    m = -log(fabs(A))/log(2)                    
    puts(m)
    H = [[0]*(i+1) for _ in range(i+1)]
    for k in range(i+1):
        for j in range(i+1):
            H[k][j] = -1 if (not j) else h[k]**(m+j)
    S.append(-I_s[i])
    det, JC = Gauss(H,S)    
    
    AnsOpt, h_opt, i3 = I_s[i]+sum(JC[j+1]*h_opt**(m+j) for j in range(i)), h_opt, i
    
    print("  СКФ Гаусса с экстраполяцией Ричардсона и выбранным оптимальным шагом по Рунге и Эйткену:\n    S + R = %.14f\n"%(AnsOpt))
    print("  Длина шага разбиения интервала:\n    h = %.7f\n\n"%(h_opt))    
    
    return 0;


if (__name__ == "__main__"): 
    main()    
