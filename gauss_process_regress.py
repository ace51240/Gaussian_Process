import numpy as np
from matplotlib import pyplot as plt
# import scipy as sp
from scipy import optimize


# 各要素が二乗誤差の行列の作成
def make_squ_err_mat(X):
    N = len(X)
    X = X.reshape((1,N))
    z = np.sum(X**2, axis=0).reshape(1,N)
    return np.tile(z.T, (1,N)) + np.tile(z,(N,1)) - 2 * (X.T @ X) # 和の二乗の展開式

# 共分散行列の効率的な計算
def kgauss(X, tau, sigma, eta):
    return np.exp(tau) * np.exp(make_squ_err_mat(X) / -np.exp(sigma)) + np.exp(eta) * np.eye(len(X))

def dif_tau(X, tau, sig, eta):
    return kgauss(X, tau, sig, eta) - np.exp(eta) * np.eye(len(X))

def dif_sigma(X, tau, sig, eta):
    return (kgauss(X, tau, sig, eta) - np.exp(eta) * np.eye(len(X))) * np.exp(-sig) * make_squ_err_mat(X)

def dif_eta(X, eta):
    return np.exp(eta) * np.eye(len(X))

def func_L(params, xtrain, ytrain):
    tau = params[0]
    sigma = params[1]
    eta = params[2]
    K = kgauss(xtrain, tau, sigma, eta)
    K_inv = np.linalg.inv(K) # O(N^3)でコスト高いが改善策があるらしい
    ky_theta = K_inv @ ytrain.T
    k_tau = dif_tau(xtrain, tau, sigma, eta)
    k_sig = dif_sigma(xtrain, tau, sigma, eta)
    k_eta = dif_eta(xtrain, eta)
    L_out = -np.trace(K_inv @ k_tau) + ky_theta.T @ k_tau @ ky_theta - np.trace(K_inv @ k_sig) + ky_theta.T @ k_sig @ ky_theta - np.trace(K_inv @ k_eta) + ky_theta.T @ k_eta @ ky_theta
    return L_out

def est_param(xtrain, ytrain, tau, sigma, eta, solver = 'CG'):
    print('solver : {}'.format(solver))
    param_result = optimize.minimize(func_L, np.array([tau, sigma, eta]), args=(xtrain, ytrain), method=solver)
    print('estimation : {}'.format(param_result.success))
    return param_result.x

def gauss_kernel(a, b, tau, sigma):
    return np.exp(tau) * np.exp(-((a - b)**2)/np.exp(sigma))

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


# RBFカーネルを用いたなめらかな分布
def make_data(xtrain, tau, sigma, eta):
    N = len(xtrain)
    zero_vec = np.zeros_like(xtrain)
    K_mat = kgauss(xtrain, tau, sigma, eta)
    np.random.seed(4)
    ytrain = np.random.multivariate_normal(zero_vec, K_mat) # 正規分布に従いランダム生成
    return ytrain

if __name__ == "__main__":
    tau = np.log(1)
    sigma = np.log(1)
    eta = np.log(0.1)
    xtrain = np.linspace(1, 4, 4)
    ytrain = make_data(xtrain, tau, sigma, eta)
    solvers = ['CG', 'BFGS', 'L-BFGS-B']
    # solvers = ['BFGS']
    mu_list = []
    var_list = []
    for solver in solvers:
        # ハイパーパラメータ(初期値)
        [tau_out, sigma_out, eta_out] = est_param(xtrain, ytrain, tau, sigma, eta, solver)
        print('tau : {}, sigma : {}, eta :{}'.format(tau_out, sigma_out, eta_out))
        xtest = np.linspace(1, 4, 50)
        mu, var = gpr(xtest, xtrain, ytrain, tau_out, sigma_out, eta_out)
        mu_list.append(mu)
        var_list.append(var)
    for i in range(len(solvers)):
        plt.scatter(xtest, mu_list[i], marker='o', label=solvers[i])
    plt.scatter(xtrain, ytrain, marker='x', label='train')
    plt.legend()
    plt.show()

