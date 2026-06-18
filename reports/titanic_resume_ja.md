# レジュメ: Titanic生存予測

## 1. 目的

Titanicの乗客データを用いて、生存有無 `Survived` を予測するモデルを作成しました。目的はAccuracyを高めることだけではなく、データ確認、EDA、仮説構築、特徴量エンジニアリング、モデル比較、検証、提出用CSV作成までの一連の分析プロセスを実装することです。

## 2. データ確認

使用したデータは以下です。

- `train_local (1).csv`: 学習データ。`Survived` を含みます。
- `valid (1).csv`: 検証データ。`Survived` を含みます。
- `eval (1).csv`: 予測対象データ。`Survived` は含みません。
- `sample_submission.csv`: 提出形式のサンプルです。

主な列は `PassengerId`、`Survived`、`Pclass`、`Name`、`Sex`、`Age`、`SibSp`、`Parch`、`Ticket`、`Fare`、`Cabin`、`Embarked` です。

欠損値が目立つ列として、`Age`、`Cabin`、`Embarked` がありました。特に `Cabin` は欠損が多いため、そのまま使うのではなく、Cabin情報の有無やDeck情報として特徴量化しました。

## 3. EDAで注目した点

- `Sex`: 女性の生存率が男性より高く、生存予測に強く関係すると考えました。
- `Pclass`: 1等客室の乗客は2等・3等より生存率が高い傾向がありました。
- `Embarked`: 乗船港によって生存率に差が見られました。
- `Cabin`: 欠損が多いものの、Cabin情報が存在するかどうか自体に意味がある可能性があります。
- `Name`: 敬称から性別、年齢、社会的立場を抽出できると考えました。

## 4. 欠損値処理

欠損値処理はscikit-learnの `Pipeline` 内で行いました。

- 数値列: 中央値で補完
- カテゴリ列: 最頻値で補完
- 未知カテゴリ: `OneHotEncoder(handle_unknown="ignore")` で対応

中央値を使った理由は、外れ値の影響を平均値より受けにくいためです。カテゴリ列は、単純で安定した方法として最頻値を採用しました。

## 5. 特徴量エンジニアリング

EDAから得た仮説をもとに、以下の特徴量を作成しました。

- `FamilySize`: `SibSp + Parch + 1`
- `IsAlone`: 1人で乗船しているかどうか
- `Title`: `Name` から抽出した敬称
- `HasCabin`: Cabin情報があるかどうか
- `Deck`: Cabinの先頭文字

これらの特徴量は、家族構成、社会的立場、客室情報が生存率に関係する可能性があるという仮説に基づいています。

## 6. モデル比較

比較したモデルは以下です。

- Logistic Regression
- Random Forest
- Gradient Boosting

検証データに対する結果:

| 実験名 | Valid Accuracy |
| --- | ---: |
| baseline_logreg | 0.805970 |
| baseline_rf | 0.820896 |
| feature_gb | 0.850746 |
| feature_logreg | 0.873134 |
| feature_rf | 0.873134 |

特徴量を追加した `feature_logreg` と `feature_rf` が最も高いAccuracyとなり、EDAにもとづく特徴量追加が有効であることを確認できました。

## 7. 学んだこと

- EDAによって重要そうな変数を見つけることができました。
- 欠損値処理とカテゴリ変数変換を `Pipeline` に組み込むことで、学習・検証・予測に同じ前処理を適用できました。
- ベースラインを作成してから特徴量を追加することで、改善効果を比較しやすくなりました。
- 分析結果をビジネス視点で説明する重要性を学びました。

## 8. 改善点

- Cross Validationによる安定性確認
- ハイパーパラメータ調整
- `AgeGroup`、`FareGroup`、`TicketGroup` の追加
- 特徴量重要度の可視化
- XGBoostやLightGBMなどの追加モデル検証

## 9. 提出ファイル

`eval.csv` に対する予測結果として `outputs/submission_eval.csv` を作成しました。このファイルは以下の2列を持ちます。

- `PassengerId`
- `Survived`
