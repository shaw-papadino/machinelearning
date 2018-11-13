import numpy as np
import pandas as pd
from scipy.optimize import minimize

def logistic2(x0, x1, w):
    """ロジスティック回帰モデル"""
    """シグモイド関数の中に直線の式"""
    y = 1 / (1 + np.exp(-(w[0] * x0 + w[1] * x1 + w[2])))
    return y

def cee_logistic2(w, x, t):
    """平均交差エントロピー誤差関数"""
    X_n = x.shape[0]
    y = logistic2(x[:,0], x[:,1], w)

    cee = 0

    for n in range(len(y)):
        # log0が入ると-infになってしまうのでlog(1+x)をしている
        # 正規化もそうだがlog0が入っていることがwarningの元凶だと考える
        cee = cee - (t[n, 0] * np.log1p(y[n]) + (1 - t[n, 0]) * np.log1p(1 - y[n]))
    cee = cee / X_n
    return cee

def dcee_logistic2(w, x, t):
    """平均交差エントロピー誤差をパラメータで偏微分する関数"""
    X_n = x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    dcee = np.zeros(3)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n, 0]) * x[n, 0]
        dcee[1] = dcee[1] + (y[n] - t[n, 0]) * x[n, 1]
        dcee[2] = dcee[2] + (y[n] - t[n, 0])
    dcee = dcee / X_n
    return dcee

def show3d_logistic2(ax, w):
    """モデルを表示する関数"""
    xn = 50
    # np.linspace(a,b,c)
    # aからbまでc分割したスカラーを返す
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    # リストの要素から格子点を生成する
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic2(xx0, xx1, w)
    ax.plot_surface(xx0, xx1, y, color="blue", edgecolor="gray", rstride=5,cstride=5, alpha=0.3)

def show_data2_3d(ax, x, t):
    """データを表示する関数"""
    c = [[0.5, 0.5, 0.5], [1, 1, 1]]
    for i in range(2):
        ax.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], 1-i, marker="o", color=c[i], linestyle="none", markersize=5, alpha=0.8, markeredgecolor="red")
    ax.view_init(elev=25, azim=-30)

def fit_logistic2(w_init, x, t):
    """勾配法"""
    res1 = minimize(cee_logistic2, w_init, args=(x,t), jac=dcee_logistic2, method="CG")
    return res1.x

def min_max(x, axis=None):
    """行列を正規化する関数"""
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def covariance_and_inverse(a, b):
    """a,bの異なる変数の共分散をして、逆行列を返す関数"""
    co = np.cov(a,b)
    return np.linalg.inv(co)

def mah(x, g):
    """マハラノビス距離を返す関数"""
    return np.dot(np.dot(np.transpose(x), g), x)

def csv_read(csv_f):
    # データ読み込み
    df = pd.read_csv(csv_f,header=None)
    N = len(df)
    T = np.zeros((N,2))
    X = np.zeros((N,2))
    # for i in N:
    for k in range(2):
       # G&B
        X[:,k] = df[k]

    T[:,0] = df[2]
    T[:, 1] = [1 if i == 0 else 0 for i in T[:,0]]
    X = min_max(X)
    return X, T, N
