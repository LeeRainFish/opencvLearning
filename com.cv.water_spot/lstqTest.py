import numpy as  np
import scipy.linalg as la
import matplotlib.pyplot as plt
m = 100
x = np.linspace(-1, 1, m)
xi = x + np.random.normal(0, 0.05, 100)
yi = 1 + 2 * xi + np.random.normal(0, 0.05, 100)
print( xi,"#xi")
print (yi,"#yi")
A = np.vstack([xi**0, xi**1])
print(A,"#A")

# 变成1行多列
# >>> d = a.reshape((1,-1))
#
# 变成多行1列：
# >>> d = a.reshape((-1,1))