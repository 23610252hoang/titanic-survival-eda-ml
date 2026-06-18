# 提出フォーム用回答: Titanic生存予測

## Valid Accuracy

```text
0.873134
```

## 実装内容の要約

本課題では、Titanic乗客データを用いて `Survived` を予測するモデルを作成しました。最初に `train_local`、`valid`、`eval` の行数、列数、欠損値、データ型を確認し、EDAによって `Sex`、`Pclass`、`Age`、`Cabin`、`Name` などと生存率の関係を調べました。

欠損値については、数値列を中央値、カテゴリ列を最頻値で補完しました。`Cabin` は欠損が多いため、そのまま使わず、Cabin情報の有無を表す `HasCabin` と、Cabinの先頭文字から作る `Deck` に変換しました。カテゴリ変数はOne-Hot Encodingで数値化し、未知カテゴリが出てもエラーにならないよう `handle_unknown="ignore"` を指定しました。

特徴量エンジニアリングとして、`FamilySize`、`IsAlone`、`Title`、`HasCabin`、`Deck` を作成しました。これらはEDAから得た「性別、客室等級、家族構成、敬称、Cabin情報が生存率に関係する可能性がある」という仮説にもとづいています。

モデルは Logistic Regression、Random Forest、Gradient Boosting を比較しました。ベースラインでは `Pclass`、`Sex`、`Age`、`Fare`、`Embarked` を使用し、その後に新しい特徴量を追加してスコアを比較しました。最終的に `feature_logreg` と `feature_rf` が `valid.csv` に対して Accuracy `0.873134` となり、ベースラインより改善しました。

## Notebookファイル

```text
notebooks/titanic_eda_feature_engineering_solution.ipynb
```

## eval予測CSV

```text
outputs/submission_eval.csv
```

## レポート

```text
reports/titanic_resume_ja.md
```

## GitHub URL

```text
https://github.com/23610252hoang/titanic-survival-eda-ml
```
