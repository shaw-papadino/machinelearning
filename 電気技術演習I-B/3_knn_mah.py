import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path
from study_machine_learning_calc import min_max, covariance_and_inverse, mah, csv_read

def knn_mah(*csv_f):
    """マハラノビス距離を用いたk近傍法"""

    # X0 = 2つの変数のデータが入る2次元配列, T0 = X0のクラスが入る2次元配列 , N0 = 要素数
    X0, T0, N0 = csv_read(csv_f[0])

    # テストデータが指定されている場合に行う
    if len(csv_f) > 1:
        X1, T1, N1 = csv_read(csv_f[1])
    else:
        X1, T1, N1 = X0, T0, N0

    # 引数の逆行列が入る
    g = covariance_and_inverse(X0[:,0],X0[:,1])
    knn_mah_list = np.zeros((N0,N0))
    knn_mah_result = []
    TT = np.zeros((N0))
    #最適なKを見つけるため
    for k in range(3,16):
        for i, t in enumerate(X0):
            for j, r in enumerate(X1):
                knn_mah_list[i,j] = mah(r - t, g)
            # 値が小さい順にk個のインデックスを取得
            # それぞれのインデックスのクラスを取得し、合計をとる
            k_sum = sum(T1[np.argpartition(knn_mah_list[i,:], k)[:k],0])
            # 多数決をし、クラスを決定する
            if k_sum / k >= 0.5 :
                TT[i] = 1
            else:
                TT[i] = 0

        knn_mah_result.append(np.sum(T0[:,0] == TT))

    """結果表示部"""
    print("正解率は{0}%です。".format(max(knn_mah_result)))
    print("最も正解率の高いkは{0}です。".format(knn_mah_result.index(max(knn_mah_result))+3))
    print(knn_mah_result)

    x = [3,4,5,6,7,8,9,10,11,12,13,14,15]
    plt.plot(x,knn_mah_result)
    plt.xlabel('k')
    plt.ylabel('accuracy rate')
    plt.show()

if __name__ == "__main__":
    args = sys.argv
    # 学習データとテストデータが引数で設定されている場合
    if len(args) == 3:
        if os.path.splitext(args[1])[1] == ".csv" or os.path.splitext(args[2])[1] == ".csv":
            knn_mah(args[1], args[2])
        else:
            print("csvファイルでお願い申す。")
    # 学習データのみの場合
    elif len(args) == 2:
        if os.path.splitext(args[1])[1] == ".csv":
            knn_mah(args[1])
        else:
            print("csvファイルでお願い申す。")

    elif len(args) == 1:
        print("学習させたいデータのファイルを引数に設定してくだせえ。")

    else:
        print("学習させたいデータのファイルは１つにしてくだせえ。")
