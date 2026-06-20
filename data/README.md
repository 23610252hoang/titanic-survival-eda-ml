# データフォルダ

公開リポジトリには、実在人物や授業データを含まない完全な合成サンプルを収録しています。

```text
data/sample_titanic.csv
```

このファイルは列構成、欠損値、前処理コードの確認用です。件数が少ないため、README記載のモデルAccuracyを再現する用途には使用しません。

元実験を再現する場合は、利用許可のあるCSVを次の場所へ配置します。

```text
data/raw/train_local (1).csv
data/raw/valid (1).csv
data/raw/eval (1).csv
data/raw/sample_submission.csv
```

元データは授業・課題用のファイルであるため、Git管理対象外にしています。
