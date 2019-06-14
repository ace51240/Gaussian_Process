import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from sklearn.utils.extmath import cartesian

def kgauss(X, tau, sigma, eta):
    N = len(X)
    X = X.reshape((1,N))
    z = np.sum(X**2, axis=0).reshape(1,N)
    K = np.tile(z.T, (1,N)) + np.tile(z,(N,1)) - 2 * (X.T @ X)
    K = tau * np.exp(-1 * K / sigma) + eta * np.eye(N)
    return K

def gauss_kernel(a, b, tau, sigma):
    return tau * np.exp(-(np.abs(a - b)**2)/sigma)

def gauss_kernel_3D(x_vector : np.ndarray, y_vector : np.ndarray, tau, sigma, eta):
    sub_vec = x_vector - y_vector
    norm = np.linalg.norm(sub_vec)**2
    delta = 0
    if np.sum(x_vector == y_vector) == len(x_vector) :
        delta = 1
    return tau * np.exp(-1 * norm / sigma) + eta * delta

def gpr(x_test, x_train, y_train, tau, sigma, eta):
    N = len(x_train)
    K = kgauss(x_train, tau, sigma, eta)
    K_inv = np.linalg.inv(K) # O(N^3)でコスト高いが改善策があるらしい
    yy = K_inv @ y_train

    M = len(x_test)
    k = np.empty(N, dtype=float)
    mu = np.empty(M, dtype=float)
    var = np.empty(M, dtype=float)
    for i in range(M):
        for j in range(N):
            k[j] = gauss_kernel(x_train[j], x_test[i], tau, sigma)
        s = gauss_kernel(x_test[i], x_test[i], tau, sigma)
        mu[i] = k.T @ yy
        var[i] = s - k @ K_inv @ k.T
    return mu, var


def gpr_3D(x_test, x_train, y_train, tau, sigma, eta):
    N = x_train.shape[1]
    M = x_test.shape[1]
    K = np.empty((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            K[i, j] = gauss_kernel_3D(x_train[:,i], x_train[:,j], tau, sigma, eta)
    # K = kgauss(x_train, tau, sigma, eta)
    K_inv = np.linalg.inv(K) # O(N^3)でコスト高いが改善策があるらしい
    yy = K_inv @ y_train
    # print('Hello')
    mu = np.empty(M, dtype=float)
    var = np.empty(M, dtype=float)
    for i in range(M):
        k = np.empty(N, dtype=float)
        for j in range(N):
            k[j] = gauss_kernel_3D(x_train[:,j], x_test[:,i], tau, sigma, 0)
        s = gauss_kernel_3D(x_test[:,i], x_test[:,i], tau, sigma, eta)
        mu[i] = k.T @ yy
        var[i] = s - k @ K_inv @ k.T
    return mu, var

# 乱数
def make_data(N):
    x_train = np.linspace(1, 4, N)
    y_train = np.random.rand(N) * 5.0
    return x_train, y_train

# RBFカーネルを用いたなめらかな分布
def make_data2(N, tau, sigma):
	x_train = np.linspace(1, 4, N)
	zero_vec = np.zeros_like(x_train, dtype=float)
	K_mat = np.empty((N, N), dtype=float)
	for i in range(N):
		for j in range(N):
			K_mat[i,j] = gauss_kernel(x_train[i], x_train[j], tau, sigma)
	y_train = np.random.multivariate_normal(zero_vec, K_mat)
	return x_train, y_train

def make_data_3D():
    N = 3
    x1 = np.linspace(0, 5, N)
    x2 = np.linspace(0, 5, N)
    X1, X2 = np.meshgrid(x1, x2)
    xx = np.c_[np.ravel(X1), np.ravel(X2)]
    print(xx.T)
    z = np.random.rand(xx.shape[0]) * 4 + 1
    return xx.T, z

# 図を作成
def show_graph(x_train, y_train, x_test, y_mu, y_var):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_test, y_mu, marker='o', label='test')
    ax.scatter(x_train, y_train, marker='x', label='train')
    plt.legend()
    plt.show()

# 図を作成
def show_graph_3D(x_train, y_train):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_train[0,:], x_train[1,:], y_train)
    ax.set_title("Scatter Plot")
    plt.show()

if __name__ == "__main__":
    # N = 5 # 訓練データ数
    x_train, y_train = make_data_3D()
    show_graph_3D(x_train, y_train)
    M = 10 # テストデータ数
    x_test = np.linspace(0, 5, M)
    X1, X2 = np.meshgrid(x_test, x_test)
    x_test = np.c_[np.ravel(X1), np.ravel(X2)].T
    # ハイパーパラメータ
    tau = 1.0
    sigma = 0.1
    eta = 0.1
    mu, var = gpr_3D(x_test, x_train, y_train, tau, sigma, eta)
    print('Hello')
    print(np.shape(mu))
    print(np.shape(var))
    show_graph_3D(x_test, var)