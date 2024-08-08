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

#data of plot #1

X1 = np.array([-2, -1, 0, 1, 2])
Y1 = np.array([-344, -172, 0, 196, 384])
mnk_1 = MNK(X1, Y1)

#data of plot #2
X2 = np.array([-1, 0, 1])
Y2 = np.array([-260, 0, 272])
mnk_2 = MNK(X2, Y2)


#data of plot
X3 = np.array([-1, 0, 1])
Y3 = np.array([-540, 0, 584])
mnk_3 = MNK(X3, Y3)


k1, b1 = lin_reg(X1, Y1)
k2, b2 = lin_reg(X2, Y2)
k3, b3 = lin_reg(X3, Y3)



#making plot #1
plt.scatter(X1, Y1, marker='o')
plt.plot(X1, k1 * X1 + b1)


#making plot #2
plt.scatter(X2, Y2, marker='^')
plt.plot(X2, k2 * X2 + b2)

#making plot #3
plt.scatter(X3, Y3)
plt.plot(X3, k3 * X3 + b3)


plt.title(r'$Y(n)$ linear aproximation')
plt.legend([r'$\nu_1 = 1,4570$, Meg', rf'lenear of $\nu_1$: $k = {str(mnk_1[0])[:stop]} \pm {str(mnk_1[1])[:stop - 2]}$, $b = {str(mnk_1[2])[:stop]} \pm {str(mnk_1[3])[:stop - 2]}$',
           r'$\nu_2 = 2,1515$, Meg', rf'lenear of $\nu_2$: $k = {str(mnk_2[0])[:stop]} \pm {str(mnk_2[1])[:stop - 2]}$, $b = {str(mnk_2[2])[:stop]} \pm {str(mnk_2[3])[:stop - 2]}$',
           r'$\nu_3 = 4,3971$, Meg', rf'lenear of $\nu_3$: $k = {str(mnk_3[0])[:stop]} \pm {str(mnk_3[1])[:stop - 2]}$, $b = {str(mnk_3[2])[:stop - 1]} \pm {str(mnk_3[3])[:stop - 2]}$'])
plt.xlabel(r'n')
plt.ylabel(r'$Y(n)$, mkm')
plt.show()


