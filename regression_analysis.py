import gauss_process_regress as gpr
import stock_getter as sg
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # code = int(input())
    code = 3694 # 株式会社OPTiM

    stock_values, stock_dates = sg.get_stock_info(code)
    print(stock_dates)
    dates_md, dates_dt = sg.shape_dates(stock_dates)
    N = len(dates_dt)

    ytrain = stock_values
    ytrain_ave = np.average(ytrain)
    ytrain = ytrain - ytrain_ave

    xtrain = np.zeros(N)
    for i in range(1,N):
        xtrain[i] = xtrain[i-1] + abs(dates_dt[i]-dates_dt[i-1]).days

    xtest = np.linspace(0, 40, 101) # 適当な範囲

    # パラメータ初期値
    tau = np.log(1)
    sigma = np.log(1)
    eta = np.log(0.1)

    # 非線形最適化関数
    # CG : 共役勾配法
    # BFGS : BFGS法
    # L-BFGS-B : 範囲制約付き目盛り制限BFGS法
    # 最適化関数とパラメータ初期値によっては逆行列が計算できない場合がある
    solvers = ['CG', 'BFGS', 'L-BFGS-B']
    mu_list = []

    for solver in solvers:
        [tau_out, sigma_out, eta_out] = gpr.est_param(xtrain, ytrain, tau, sigma, eta, solver)
        print('tau : {}, sigma : {}, eta :{}'.format(tau_out, sigma_out, eta_out))
        mu, var = gpr.gpr(xtest, xtrain, ytrain, tau_out, sigma_out, eta_out)
        mu_list.append(mu)

    for i in range(len(solvers)):
        plt.scatter(xtest, mu_list[i]+ytrain_ave, marker='o', label=solvers[i])
    plt.scatter(xtrain, ytrain+ytrain_ave, marker='x', label='train')
    plt.xticks(xtrain, dates_md, rotation=90)
    plt.title('Ticker_Symbol : {}'.format(code))
    plt.legend()
    plt.grid()
    plt.show()