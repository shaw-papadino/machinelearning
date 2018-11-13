
import numpy as np
import sys
import os.path

def linear_regression(ply_f):
    """D次元線形回帰モデル"""
    f = open(ply_f)
    lines = f.readlines()
    lines = lines[11:]
    llines = []
    for ll in lines:

        if ll.find("100000") == -1:
            llines.append(ll[:-1])

    X = np.zeros((len(llines),3))
    t = np.zeros(len(llines))
    for i, l in enumerate(llines):
        l = l.split()
    #     print(type(l[0]))
        # x
        X[i,0] = l[0]
        # y
        X[i,1] = l[1]
        # b
        X[i,2] =1
        # z
        t[i] = l[2]

    W = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,t))

    print("W0 = {0},W1 = {1},W2 = {2}".format(round(W[0],3),round(W[1],3),round(W[2],3)))

if __name__ == "__main__":
    args = sys.argv
    if len(args) == 2:
        if os.path.splitext(args[1])[1] == ".ply":
            linear_regression(args[1])
        else:
            print("plyファイルでお願い申す。")

    elif len(args) == 1:
        print("学習させたいデータのファイルを引数に設定してくだせえ。")

    else:
        print("学習させたいデータのファイルは１つにしてくだせえ。")
