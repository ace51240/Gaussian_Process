import numpy as np
from matplotlib import pyplot as plt

def gauss_kernel(a, b):
    theta_1 = 1.0
    theta_2 = 1.0
    return theta_1 * np.exp(-(np.abs(a - b)**2)/theta_2)

def gpr(x_test, x_train, y_train):
    N = len(x_train)
    K = np.empty((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            K[i][j] = gauss_kernel(x_train[i], x_train[j])
    K_inv = np.linalg.inv(K) # O(N^3)でコスト高いが改善策があるらしい
    yy = K_inv @ y_train
    # print(np.shape(yy))

    M = len(x_test)
    k = np.empty(N, dtype=float)
    mu = np.empty(M, dtype=float)
    var = np.empty(M, dtype=float)
    for i in range(M):
        for j in range(N):
            k[j] = gauss_kernel(x_train[j], x_test[i])
        s = gauss_kernel(x_test[i], x_test[i])
        mu[i] = k.T @ yy
        var[i] = s - k @ K_inv @ k.T
    return mu, var

def make_data(N):
    N = 5
    x_train = np.linspace(1, 4, N)
    y_train = np.random.rand(N) * 5.0
    return x_train, y_train

def make_data2(N):
	x_train = np.linspace(1, 4, N)
	zero_vec = np.zeros_like(x_train, dtype=float)
	K_mat = np.empty((N, N), dtype=float)
	for i in range(N):
		for j in range(N):
			K_mat[i,j] = gauss_kernel(x_train[i], x_train[j])
	y_train = np.random.multivariate_normal(zero_vec, K_mat)
	return x_train, y_train
    
def show_graph(x_train, y_train, x_test, y_mu, y_var):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_test, y_mu, marker='o', label='test')
    ax.scatter(x_train, y_train, marker='x', label='train')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    N = 5
    M = 100
    x_train, y_train = make_data(N)
    # x_train, y_train = make_data2(N)

    # x_test = np.random.rand(M) * 3 + 1
    x_test = np.linspace(1, 4, M)

    mu, var = gpr(x_test, x_train, y_train)
    # print(x_test)
    # print(np.shape(mu))
    # print(np.shape(var))
    show_graph(x_train, y_train, x_test, mu, var)