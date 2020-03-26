import numpy as np
from matplotlib import pyplot as plt

# 共分散行列の効率的な計算
def kgauss(X, tau, sigma, eta):
    N = len(X)
    X = X.reshape((1,N))
    z = np.sum(X**2, axis=0).reshape(1,N)
    K = np.tile(z.T, (1,N)) + np.tile(z,(N,1)) - 2 * (X.T @ X)
    K = tau * np.exp(-1 * K / sigma) + eta * np.eye(N)
    return K

def gauss_kernel(a, b, tau, sigma):
    return tau * np.exp(-((a - b)**2)/sigma)

def gpr(xtest, xtrain, ytrain, tau, sigma, eta):
    N = len(xtrain)
    K = kgauss(xtrain, tau, sigma, eta)
    K_inv = np.linalg.inv(K) # O(N^3)でコスト高いが改善策があるらしい
    yy = K_inv @ ytrain

    M = len(xtest)
    k = np.empty(N)
    mu = np.empty(M)
    var = np.empty(M)
    for i in range(M):
        for j in range(N):
            k[j] = gauss_kernel(xtrain[j], xtest[i], tau, sigma)
        s = gauss_kernel(xtest[i], xtest[i], tau, sigma)
        mu[i] = k.T @ yy
        var[i] = s - k @ K_inv @ k.T
    return mu, var

# 図を作成
def show_graph(xtrain, ytrain, xtest, ymu, yvar):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xtest, ymu, marker='o', label='test')
    ax.scatter(xtrain, ytrain, marker='x', label='train')
    plt.legend()
    plt.show()

# RBFカーネルを用いたなめらかな分布
def make_data(xtrain, tau, sigma, eta):
    N = len(xtrain)
    zero_vec = np.zeros_like(xtrain)
    K_mat = kgauss(xtrain, tau, sigma, eta)
    ytrain = np.random.multivariate_normal(zero_vec, K_mat)
    return ytrain

if __name__ == "__main__":
    xtrain = np.linspace(1, 4, 5)
    # ハイパーパラメータ
    tau = 1.0
    sigma = 0.4
    eta = 0.1
    ytrain = make_data(xtrain, tau, sigma, eta)
    xtest = np.linspace(1, 4, 100)
    mu, var = gpr(xtest, xtrain, ytrain, tau, sigma, eta)
    show_graph(xtrain, ytrain, xtest, mu, var)