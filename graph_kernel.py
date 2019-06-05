import numpy as np
from matplotlib import pyplot as plt

# print('Hello')
# RBFカーネル
def gauss_kernel(a, b):
    theta_1 = 1.0
    theta_2 = 1.0
    return theta_1 * np.exp(-(np.abs(a - b)**2)/theta_2)

a = np.linspace(1, 4, 16)
K_mat = np.empty((len(a), len(a)), dtype=float)
for i in range(len(a)):
    for j in range(len(a)):
        K_mat[i,j] = gauss_kernel(a[i], a[j])

# print(K_mat)
zero_vec = np.zeros(len(a), dtype=float)
y = np.random.multivariate_normal(zero_vec, K_mat)
# print(y)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(a,y)
plt.show()
