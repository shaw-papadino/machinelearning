import numpy as np
import pandas as pd
import sys
import os.path
from study_machine_learning_calc import logistic2, cee_logistic2,dcee_logistic2, fit_logistic2, min_max, csv_read

def logistic(*csv_f):
    """シグモイド関数を用いたロジスティック回帰モデル"""
    # 初期パラメータ
    W_init = [-1, 0, 0]
    # X0 = 2つの変数のデータが入る2次元配列, T = X0のクラスが入る2次元配列 , N0 = 要素数
    X0, T, N0 = csv_read(csv_f[0])
    # W0 = w0,w1,w2 の最適なパラメータが入るリスト
    W0 = fit_logistic2(W_init, X0, T)
    # Y = XがT=1の時の確率が入るリスト
    Y = logistic2(X0[:,0],X0[:,1],W0)

    # テストデータが指定されている場合に行う
    if len(csv_f) > 1:
        # X1 = 2つの変数のデータが入る2次元配列, T = X0のクラスが入る2次元配列 , N1 = 要素数
        X1, T, N1 = csv_read(csv_f[1])
        # Y = XがT=1の時の確率が入るリスト
        # Wのパラメータは学習データのWの値を使用
        Y = logistic2(X1[:,0],X1[:,1],W0)
    else:
        pass

    # クラスを格納するリスト生成
    TT = np.zeros((N0))
    for i, y in enumerate(Y):
        # 0 <= y <= 1
        if y >= 0.5:
            TT[i] = 1
        else:
            TT[i] = 0

    print("W0 = {0},W1 = {1},W2 = {2}".format(round(W0[0],3),round(W0[1],3),round(W0[2],3)))
    print("正解率は{0}%です".format(np.sum(T[:,0] == TT)))

if __name__ == "__main__":
    args = sys.argv
    # 学習データとテストデータが引数で設定されている場合
    if len(args) == 3:
        if os.path.splitext(args[1])[1] == ".csv" or os.path.splitext(args[2])[1] == ".csv":
            logistic(args[1], args[2])
        else:
            print("csvファイルでお願い申す。")
    # 学習データのみの場合
    elif len(args) == 2:
        if os.path.splitext(args[1])[1] == ".csv":
            logistic(args[1])
        else:
            print("csvファイルでお願い申す。")

    elif len(args) == 1:
        print("学習させたいデータのファイルを引数に設定してくだせえ。")

    else:
        print("学習させたいデータのファイルは１つにしてくだせえ。")
