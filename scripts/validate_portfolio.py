"""公開ポートフォリオ用サンプルデータとNotebookの構造を検証します。"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "sample_titanic.csv"
NOTEBOOK_PATH = ROOT / "notebooks" / "titanic_eda_feature_engineering_solution.ipynb"
REQUIRED_COLUMNS = {
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
}


def main() -> None:
    with SAMPLE_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) < 10:
        raise ValueError("サンプルデータは10行以上必要です。")
    if set(rows[0]) != REQUIRED_COLUMNS:
        raise ValueError("サンプルデータの列構成が想定と異なります。")
    if len({row["PassengerId"] for row in rows}) != len(rows):
        raise ValueError("PassengerIdが重複しています。")
    if {row["Survived"] for row in rows} - {"0", "1"}:
        raise ValueError("Survivedは0または1である必要があります。")

    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    if not isinstance(notebook.get("cells"), list) or not notebook["cells"]:
        raise ValueError("Notebookにセルがありません。")

    print(f"検証成功: sample={len(rows)}行, notebook={len(notebook['cells'])}セル")


if __name__ == "__main__":
    main()
