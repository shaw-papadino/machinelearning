1. data.ply を読み込み、近似平面を算出するプログラム

  ○ 1_d_linear_regression.py
    ・data.ply を引数としてプログラムを実行してください。
      → python 1_d_linear_regression.py data.ply
    ・出力はパラメータW0,W1,W2です。

2. data.csv を読み込み、教科書の2次元入力 2クラス分類の方法で識別し、正解率を算出するプログラム

  ○ 2_logistic.py
    ・学習データ(data.csv)を引数としてプログラムを実行してください。
      → python 2_logistic.py data.csv
    ・もし、学習データ(data.csv)とテストデータ(test.csv)を引数とすれば、２つのデータの類似度が算出されます。
      → python 2_logistic.py data.csv test.csv
    ・出力はパラメータW0,W1,W2です。

3. data.csv を読み込み、マハラノビス距離を使ったkNN法で識別し、正解率を算出するプログラム

  ○ 3_knn_mah.py
  ・学習データ(data.csv)を引数としてプログラムを実行してください。
    → python 3_knn_mah.py data.csv
  ・もし、学習データ(data.csv)とテストデータ(test.csv)を引数とすれば、２つのデータの類似度が算出されます。
    → python 3_knn_mah.py data.csv test.csv
  ・出力は、最も正解率の高いk、その正解率、kを３から１５まで変化させた際の正解率の値とグラフです。

補足: study_machine_learning_calc.py は、今回の課題で必要とする関数をまとめたファイルです。