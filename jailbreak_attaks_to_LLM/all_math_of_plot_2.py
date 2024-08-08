import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
import math

#linear aprox of y = kx + b
def lin_reg(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b

# my MNK func of y = kx + b
def MNK(x:list, y:list):
    x_avg = mean(x)
    print(fr'$x_avg = {x_avg}$')
    x_in_sq_avg = mean([el ** 2 for el in x])
    print(fr'avg(x^2) = {x_in_sq_avg}')
    y_avg = mean(y)
    print(fr'$x_avg = {y_avg}$')
    y_in_sq_avg = mean([el ** 2 for el in y])
    print(fr'avg(x^2) = {y_in_sq_avg}')
    xy_avg = mean([x_el * y_el for x_el, y_el in zip(x, y)])
    print(rf'$avg(xy) = {xy_avg}$')
    k = (xy_avg - x_avg * y_avg) / (x_in_sq_avg - x_avg ** 2)
    print(rf'k = {k}')
    k_sigma = (1 / len(x)) * math.sqrt((y_in_sq_avg - y_avg ** 2) / (x_in_sq_avg - x_avg ** 2) - k ** 2)
    print(rf'$\sigma_k = {k_sigma}$')
    b = y_avg - k * x_avg
    print(rf"b = {b}")
    b_sigma = k_sigma * math.sqrt(x_in_sq_avg - x_avg ** 2)
    print(rf'$\sigma_b = {b_sigma}$')
    return(k, k_sigma, b, b_sigma)


stop = int(input())

nu = [1.7070, 2.0866, 4.2673, 2.511, 3.1819, 3.8760]

#data of plot #1

X1 = np.array([1/i for i in nu])
Y1 = np.array([183, 142, 69, 120, 94, 75])
mnk_1 = MNK(X1, Y1)






k1, b1 = lin_reg(X1, Y1)




#making plot #1
plt.scatter(X1, Y1, marker='o')
plt.plot(X1, k1 * X1 + b1)



plt.title(r'$\Lambda(\frac{1}{\nu})$ linear aproximation')
plt.legend([r'$\nu_1 = 1,4570$, Meg', rf'lenear of $\nu_1$: $k = {str(mnk_1[0])[:stop]} \pm {str(mnk_1[1])[:stop - 2]}$, $b = {str(mnk_1[2])[:stop]} \pm {str(mnk_1[3])[:stop - 2]}$'])
plt.xlabel(r'$\frac{1}{\nu}, s \cdot 10^{-6}$')
plt.ylabel(r'$\Lambda, m \cdot 10^{-6}$')
plt.show()
