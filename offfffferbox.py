import numpy as np
import pandas as pd

"""自分が登録しているオファーボックスへのオファー数を予測するモデルの作成"""
"""D次元線形回帰モデルなので、変数は柔軟に増減できます"""


def fit_plane(x1,x2,t):
    """D次元線形回帰モデル"""
    X = np.zeros((len(x1),3))
    t = np.zeros(len(t))
    for i in range(len(x1)):
        # x
        X[i,0] = x1[i]
        # y
        X[i,1] = x2[i]
        # b
        X[i,2] =1
        # z
        t[i] = t[i]

    X0_min = np.min(X[:,0])
    X0_max = np.max(X[:,0])

    X1_min = np.min(X[:,1])
    X1_max = np.max(X[:,1])

    W = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,t))
    return W

def prediction(W, x1, x2):
    """予測"""
    y = W[0] * x1 + W[1] * x2 + W[2]
    return y

if __name__ == "__main__":

    df = pd.read_csv("offerbox.csv")
    s = df["検索"][0:25]
    p = df["プロフィール"][0:25]
    o = df["オファー"][0:25]

    # Wには平均二乗誤差を最小にする係数が入る
    W = fit_plane(s, p, o)
    # x1,x2に新しい値を入れると、予測した値を返す
    # y = prediction(W,x1,x2)
    print("W:{0}".format(W))
