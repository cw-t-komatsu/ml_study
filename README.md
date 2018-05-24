# scikit-learnの学習メモ

	$ python -V
	Python 3.6.4 :: Anaconda, Inc.
	$ pip --version
	pip 10.0.1
	$ pip list | grep jupyter
	jupyter 1.0.0
	$ pip list | grep scikit-learn
	scikit-learn 0.19.1
	$ pip list | grep hyper
	hyperopt 0.1
	$ pip list | grep mlxtend
	mlxtend 0.12.0

このプログラムは突然消すかプライベートにするかも  
また、色々なサイトを摘み食いしているので間違っている情報や訳があると思う  

scikit-learnを使って、同じデータ(iris)に対して  
分析アルゴリズムを使って分類可視化をして比較する。  

基本的にはscikit-learnの公式ドキュメントのサンプルを動かしつつ、動作になれていく  
[http://scikit-learn example](http://scikit-learn.org/stable/auto_examples/index.html)

## 作って動作確認したもののメモ

作って確認したもの

| 機械学習        | 項目                           | 関数名                                        |
| :-------------- | :------------                  | :------------                                 |
| 分類            | SVM                            | SVC                                           |
| 分類            | SVM                            | LinearSVC                                     |
| 分類            | 決定木                         | DecisionTreeClassifier                        |
| 分類            | ランダムフォレスト             | RandomForestClassifier                        |
| 分類            | 確率的勾配降下法による線形分類 | SGDClassifier                                 |
| 分類            | ガウス過程分類                 | GaussianProcessClassifier                     |
| 分類            | アンサンブル学習               | RandomForest,DecisionTree,ExtraTrees,AdaBoost |
| 分類            | アンサンブル学習               | RandomForest,DecisionTree,ExtraTrees,AdaBoost |
| クラスタリング  | K-means                        | MiniBatchKMeans                               |
| クラスタリング  | K-means                        | KMeans                                        |
| クラスタリング  | K-means                        | Birch                                         |
| クラスタリング  | MeanShift                      | MeanShift                                     |
| クラスタリング  | ガウス混合分布                 | GMM                                           |
| クラスタリング  | 変分ガウス混合分布             | VBGMM                                         |
| 回帰            | ロジスティック回帰             | LogisticRegression                            |
| 回帰            | 確率的勾配降下法による回帰     | SGDRegressor                                  |
| 回帰            | Lasso回帰                      | Lasso                                         |
| 回帰            | ElasticNet                     | ElasticNet                                    |
| 回帰            | Ridge回帰                      | Ridge                                         |
| 回帰            | SVM                            | SVR(linear)                                   |
| 回帰            | SVM                            | SVR(rbf)                                      |
| 回帰            | アンサンブル学習               |                                               |
| 次元削減        | 主成分分析                     | PCA                                           |
| 次元削減        | カーネル主成分分析             | KernelPCA                                     |
| 次元削減        | 多次元尺度構成法               | MDS                                           |
| 次元削減        | 多様体学習                     | Isomap                                        |
| 次元削減        | 多様体学習                     | locally_linear_embedding                      |
| 次元削減        | 多様体学習                     | SpectralEmbedding                             |
| 次元削減        | t-SNE                          | TSNE                                          |


未作成

| 機械学習        | 項目                                  | 関数名                     |
| :-------------- | :------------                         | :------------              |
| 分類            | パーセプトロン                        |                            |
| 分類            | ニューラルネットワーク                |                            |
| 分類            | GBDT(Gradient Boosting Decision Tree) | GradientBoostingClassifier |
| クラスタリング  | k近傍法                               | KNeighborsClassifier       |


流行りのGBDTとXGBoostとかその辺触れるようにしときたい

###### XGBoost
* Gradient BoostingとRandom Forestsを組み合わせたアンサンブル学習である
* [XGBoostの主な特徴と理論の概要](https://qiita.com/yh0sh/items/1df89b12a8dcd15bd5aa)

## 構築メモ
* anacondaにインストールする時はコマンドが違うはず
* 決定木の可視化(jupyter)が上手くいかなかったため、以下のライブラリを追加でインストールした  
	* brew install graphviz
	* pip install pydotplus
	* pip install pydot
	* pip install pydot-ng
* 可視化を楽にするため、プラグインを追加
	* pip install mlxtend
	* pip install seaborn
* パラメータチューニング用にライブラリを追加
	* pip install hyperopt <-(インストール失敗する、Python3で動かなかった)
	* pip install --upgrade git+git://github.com/hyperopt/hyperopt.git

## コーディング規約
* 理解が進んできたら読むことを推奨
* 公式ドキュメントに規約が書いてあるので、簡易にまとめる
* [scikit-learn Code Review Guidelines](http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines)
* [変数やメソッドの命名規則](http://kaisk.hatenadiary.com/entry/2014/12/15/215502)

## サンプルプログラムの動作確認と学習をしていった順番（独学なのでこの順序がいいかわからない）
1. 機械学習モデル各種の動作確認と可視化（iris）(2次元,3次元,4次元)
2. 教師データの準備と精度評価（クロスバリデーション）
	* [scikit-learn を用いた交差検証（Cross-validation）とハイパーパラメータのチューニング（grid search）](https://qiita.com/tomov3/items/039d4271ed30490edf7b)
	* 評価値の指標についてはサイトを参照 （[機械学習で使う指標まとめ](http://www.procrasist.com/entry/9-metrics)）
	* [Document scoring parameter](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
3. ハイパーパラメータのチューニング（グリッドサーチ、ランダムサーチ、SMBO(hyperopt)）
4. 異なるデータで上記の1-3を動作確認(MNIST)


## メモ
* 多次元の可視化にはt-SNEが人気
* 終わり）1.GBDTの動作確認する
* サンプルプログラムの命名規則整理する
* サンプルデータセットの中身を確認する
* 2.他のデータセットに対しても、動作確認できるようにプログラムを整理する
* 3.比較プログラムに、ハイパーパラメータ調整を加えたものを作る
* 4.分類、回帰、クラスタリング、次元削減の比較プログラムを整理する
* 回帰の可視化が微妙だから、mlxtendの仕様確認する


## 公式のサンプル分類
| 英文                                  | 直訳                                  |
| :------------------------------------ | :------------------------------------ |
| General examples                      | 一般的な例                            |
| Examples based on real world datasets | 実世界のデータセットに基づく例        |
| Biclustering                          | ヒートマップを作るときのサンプル      |
| Calibration                           | キャリブレーション                    |
| Classification                        | 分類                                  |
| Clustering                            | クラスタリング                        |
| Covariance estimation                 | 共分散推定                            |
| Cross decomposition                   | クロス分解                            |
| Dataset example                       | データセットの例                      |
| Decomposition                         | 分解                                  |
| Ensemble methods                      | アンサンブルメソッド                  |
| Tutorial exercises                    | チュートリアル演習                    |
| Feature Selection                     | 機能の選択                            |
| Gaussian Process for Machine Learning | 機械学習のためのガウスプロセス        |
| Generalized Linear Models             | 一般化された線形モデル                |
| Manifold learning                     | 多様体学習                            |
| Gaussian Mixture Models               | ガウス混合モデル                      |
| Model Selection                       | モデル選択                            |
| Multioutput methods                   | マルチ出力メソッド                    |
| Nearest Neighbors                     | 近傍法                                |
| Neural Networks                       | ニューラルネットワーク                |
| Preprocessing                         | 前処理用のライブラリ                  |
| Semi Supervised Classification        | 半教師あり学習                        |
| Support Vector Mac                    | サポートベクターマシン                |
| Working with text documents           | テキスト文書                          |
| Decision Trees                        | 分類木                                |
|                                       |                                       |

## scikit-learnでできることの一覧
[APIリファレンス](http://scikit-learn.org/stable/modules/classes.html)

| 分類                   | 関数                         |
| :-                      | :-                            |
| データセット           | sklearn.datasets             |
| 交差検証やチューニング | sklearn.model_selection      |
| モデル評価             | sklearn.metrics              |
| SVM                    | sklearn.svm                  |
| 決定木                 | sklearn.tree                 |
| ナイーブベイズ         | sklearn.naive_bayes          |
| 近傍法                 | sklearn.neighbors            |
| アンサンブル学習       | sklearn.ensemble             |
| ガウス過程             | sklearn.gaussian_process     |
| 線形モデル             | sklearn.linear_model         |
| カーネル近似           | sklearn.kernel_approximation |
| カーネルRidge回帰      | sklearn.kernel_ridge         |
| クラスタリング         | sklearn.cluster              |
| 混合ガウスモデル(GMM)  | sklearn.mixture              |
| 次元削減               | sklearn.decomposition        |
| 次元削減（多様体）     | sklearn.manifold             |
| ニューラルネットワーク | sklearn.neural_network       |
| 半教師つき学習         | sklearn.semi_supervised      |


* 使ったことがない・使い道を知らない関数

| 分類                                 | 関数                          |
| :-                                    | :-                             |
| ユーティリティ                       | sklearn.base                  |
| 共分散推定                           | sklearn.covariance            |
| 交差次元削減？                       | sklearn.cross_decomposition   |
| 線形判別分析？                       | sklearn.discriminant_analysis |
| Dummy estimators                     |                               |
| エラー処理と警告                     | sklearn.exceptions            |
| 生データや画像から特徴抽出をする機能 | sklearn.feature_extraction    |
| Feature Selection                    |                               |
| isotonic回帰                         | sklearn.isotonic              |
| 多クラス分類器                       | sklearn.multiclass            |
| 多クラス回帰                         | sklearn.multioutput           |
| パイプラインを利用したユーティリティ | sklearn.pipeline              |
| 正規化、2値化などの                  | sklearn.preprocessing         |
| Random projection                    |                               |
| ユーティリティ                       | sklearn.utils                 |
|                                      |                               |

0.20で削除予定  

| 分類                 | 関数                     |
| :                    | :                        |
| クロスバリデーション | sklearn.cross_validation |
|                      |                          |

## scikit-leanrのサンプルデータセットのメモ
* iris dataset
	* classification向き
	* 言わずと知れたアイリスさんちのデータ
	* 4カラム、150レコード、3ラベル
* wine dataset
	* classification向き
	* ワインの中に含まれる化学物質の量などを説明変数として、ワインの等級（3クラス）を予測
	* 13カラム、178レコード、ラベル
	* [説明変数の参考](https://qiita.com/Dixhom/items/7c33a1dc85144e1da822)
* digits dataset
	* classification向き
	* 0 ～ 9 の 10 文字の手書きの数字を 64 (8×8) 個の画素に分解したもの。
	* MNISTみたいなもの
	* 64カラム、1797レコード、10ラベル
* breast cancer wisconsin dataset
	* classification向き
	* ウィスコンシンの乳がんのデータ
	* 検査値32ラベル、診断569ケース、診断結果2ラベル(良性腫瘍 or 悪性腫瘍)
* boston house-prices dataset
	* regression向き
	* 米国ボストン市郊外における地域別の住宅価格のデータセット
	* 14カラム、506レコード
* diabetes dataset 
	* regression向き
	* 糖尿病患者 442 人の検査数値と 1 年後の疾患進行状況（正規化済み）
	* 10カラム、442レコード
* linnerud dataset
	* multivariate regression向き
	* 成人男性に対してフィットネスクラブで測定した 3 つの生理学的特徴と 3 つの運動能力の関係。
	* 説明変数3,目的変数3,レコード数20


[まとめてくれているサイトを参考に](https://pythondatascience.plavox.info/scikit-learn/scikit-learn%E3%81%AB%E4%BB%98%E5%B1%9E%E3%81%97%E3%81%A6%E3%81%84%E3%82%8B%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88)




