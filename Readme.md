# 株価をガウス過程回帰で分析するプログラム
 
現在の近日から過去約40日前までの株価を訓練データとして取得し，ガウス過程による回帰で分析を行うプログラムをPythonで実装しました．

## ファイル
- gauss_process_regress.py
  - ガウス過程による分析を行う．
  - このファイル単体で実行可能．
  - パラメータ推定に用いる非線形最適化関数は以下の手法を用いている．
  最適化関数とパラメータ初期値によっては逆行列が計算できずエラーが生じる可能性がある．
    - CG : 共役勾配法
    - BFGS : BFGS法
    - L-BFGS-B : 範囲制約付き目盛り制限BFGS法
- stock_getter.py
  - 日経経済新聞の株価情報をWebスクレイピングで取得する関数get_stock_infoが含まれている．
  - 任意の証券コードを引数にすることで，対象の企業の株価情報を取得する．
- regression_analysis.py
  - 上記2ファイルのパッケージを用いて株価の株価分析を行う．
  - 実行すると，株式会社OPTiMの株価情報を取得し，ガウス過程で分析を行う．
  - 最後に株価情報をプロットしたグラフを出力する．
- screenshot.png
  - 実行結果のグラフのスクリーンショットを表す．
    - Ticker_Symbol : 証券コード
    - 縦軸 : 株価
    - 横軸 : 時系列
    - CG, BFGS, L-BFGS-B : パラメータ推定に用いた非線形最適関数
    - train : stock_getterで取得した訓練データ

# 必要なコンパイラやパッケージ

## コンパイラ
- Python 3.7.4 64bitで実行を確認済み

## パッケージ
- gauss_process_regress.py
  - numpy
  - matplotlib
  - scipy
- stock_getter.py
  - numpy
  - pandas
  - datetime
  - re
- regression_analysis.py
  - gauss_process_regress
  - stock_getter
  - numpy
  - matplotlib

上記のパッケージはPythonのAnaconda等の開発環境をインストールする際にデフォルトで含まれている場合が多い．

以下のコマンドでインストールし，パスを通せば実行できるはず
```bash
pip install package_name
```
 
# 実行コマンド
 
以下のコマンドで実行できるはず

```bash
git clone https://github.com/ace51240/Gaussian_Process.git 
cd Gaussian_Process
git checkout for_StockAnalysis
python regression_analysis.py
```

 