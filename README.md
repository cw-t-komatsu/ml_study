# ml_study

	$ python -V
	Python 3.6.4 :: Anaconda, Inc.
	$ pip --version
	pip 10.0.1
	$ pip list | grep jupyter
	jupyter 1.0.0
	$ pip list | grep scikit-learn
	scikit-learn 0.19.1


このプログラムは突然消すかも  

scikit-learnを使って、同じデータ(iris)に対して  
分析アルゴリズムを使って分類可視化をして比較する。  

基本的にはscikit-learnの公式ドキュメントのサンプルを動かしつつ、動作になれていく  
[http://scikit-learn example](http://scikit-learn.org/stable/auto_examples/index.html)

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

## 作って動作確認したもののメモ

作ったもの

| 機械学習        | 項目                           | 関数名                                        |
| :-------------- | :------------                  | :------------                                 |
| 分類            | 決定木                         | DecisionTreeClassifier                        |
| 分類            | ランダムフォレスト             | RandomForestClassifier                        |
| 分類            | 確率的勾配降下法による線形分類 | SGDClassifier                                 |
| 分類            | SVM                            | SVC, LinearSVC                                |
| 分類            | ガウス過程分類                 | GaussianProcessClassifier                     |
| 分類            | アンサンブル学習               | RandomForest,DecisionTree,ExtraTrees,AdaBoost |
| クラスタリング  | K-means                        | KMeans                                        |
| 回帰            | ロジスティック回帰             | LogisticRegression                            |
| 次元削減        | 主成分分析                     | PCA                                           |

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
決定木の可視化(jupyter)が上手くいかなかったため、以下のライブラリを追加でインストールした
brew install graphviz
pip install pydotplus
pip install pydot
pip install pydot-ng

## コーディング規約
* 理解が進んできたら読むことを推奨
* 公式ドキュメントに規約が書いてあるので、簡易にまとめる
* [scikit-learn Code Review Guidelines](http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines)
* [変数やメソッドの命名規則](http://kaisk.hatenadiary.com/entry/2014/12/15/215502)

## メモ
* データ数が増えてきたらクロスバリデーションで精度判定をすべき
* 参考
* [scikit-learn を用いた交差検証（Cross-validation）とハイパーパラメータのチューニング（grid search）](https://qiita.com/tomov3/items/039d4271ed30490edf7b)



