# Googleフォーム入力用：Titanic生存予測

## valid Accuracy

```text
0.8731343283582089
```

四捨五入して入力する場合：

```text
0.873134
```

## 実装のポイント

以下をそのままフォームに貼り付けてよい。

```text
本課題では，まずtrain_local.csv，valid.csv，eval.csvの行数，列名，欠損値，データ型を確認した。EDAでは，Sex，Pclass，Embarked，Cabin，NameなどとSurvivedの関係に注目した。特に，女性の生存率が男性より高く，1等客室の乗客の生存率が高いことを確認した。

欠損値については，数値列は中央値，カテゴリ列は最頻値で補完した。Cabinは欠損が非常に多いため，客室番号をそのまま使うのではなく，HasCabinとDeckという特徴量に変換した。カテゴリ変数であるSex，Embarked，Title，DeckはOne-Hot Encodingによって数値化した。

特徴量エンジニアリングとして，FamilySize，IsAlone，Title，HasCabin，Deckを作成した。FamilySizeとIsAloneは家族構成，Titleは氏名に含まれる敬称，HasCabinとDeckは客室情報を表す。これらはEDAから得られた仮説にもとづいて作成した。

モデルは，Logistic Regression，Random Forest，Gradient Boostingを比較した。ベースラインではPclass，Sex，Age，Fare，Embarkedを使用し，その後，新しい特徴量を追加してスコアを比較した。valid.csvに対する最良のAccuracyは0.873134であり，特徴量追加によってベースラインより改善した。
```

## Notebookファイル

アップロードするファイル：

```text
notebooks/titanic_eda_feature_engineering_solution.ipynb
```

## eval予測CSV

アップロードするファイル：

```text
outputs/submission_eval.csv
```

## レジュメ

アップロードするファイル：

```text
reports/titanic_resume_ja.md
```

## GitHub URLを書く欄がある場合

```text
https://github.com/23610252hoang/titanic-survival-eda-ml
```

## 注意

名前，表示名，学術的誠実性の同意は，自分で確認して入力すること。
