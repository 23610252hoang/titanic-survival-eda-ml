# Resume: Titanic Survival Prediction

## 1. Mục tiêu

Dự án này dự đoán cột `Survived` của hành khách Titanic. Mục tiêu không chỉ là tăng Accuracy, mà là thực hiện đúng quy trình phân tích dữ liệu: kiểm tra dữ liệu gốc, EDA, đặt giả thuyết, xử lý dữ liệu, tạo đặc trưng mới, huấn luyện mô hình và đánh giá trên `valid.csv`.

## 2. Kết quả kiểm tra dữ liệu gốc

Các file được sử dụng:

- `train_local (1).csv`: 623 dòng, 12 cột, có `Survived`.
- `valid (1).csv`: 134 dòng, 12 cột, có `Survived`.
- `eval (1).csv`: 134 dòng, 11 cột, không có `Survived`.
- `sample_submission.csv`: 134 dòng, 2 cột, dùng làm mẫu định dạng nộp.

Các cột chính gồm `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

Trong `train_local`, các cột thiếu dữ liệu là:

- `Age`: thiếu 119 giá trị.
- `Cabin`: thiếu 487 giá trị.
- `Embarked`: thiếu 2 giá trị.

## 3. EDA và các điểm chú ý

Tỉ lệ sống sót trong tập train khoảng 38.4%, còn không sống sót khoảng 61.6%.

Một số quan sát quan trọng:

- Theo `Sex`: nữ có tỉ lệ sống sót khoảng 75.1%, nam khoảng 18.7%. Đây là biến rất mạnh.
- Theo `Pclass`: hạng 1 có tỉ lệ sống sót khoảng 63.9%, hạng 2 khoảng 45.4%, hạng 3 khoảng 24.9%.
- Theo `Embarked`: hành khách lên ở cảng C có tỉ lệ sống sót cao hơn S và Q trong dữ liệu này.
- `Cabin` thiếu nhiều, nhưng việc có thông tin cabin hay không có thể phản ánh hạng vé hoặc vị trí trên tàu.
- `Name` chứa danh xưng như `Mr`, `Mrs`, `Miss`, `Master`, có thể đại diện cho giới tính, độ tuổi và địa vị xã hội.

## 4. Xử lý thiếu dữ liệu

Tôi xử lý missing values trong `Pipeline`:

- Các cột số như `Age`, `Fare` được điền bằng median.
- Các cột category như `Embarked`, `Sex`, `Title`, `Deck` được điền bằng mode.
- `Cabin` không được dùng trực tiếp vì thiếu quá nhiều. Thay vào đó tạo `HasCabin` và `Deck`.

Lý do chọn median cho cột số là median ít bị ảnh hưởng bởi outlier hơn mean. Với category, mode là cách đơn giản và ổn định khi số lượng missing nhỏ.

## 5. Chuyển đổi biến category

Các biến category được mã hóa bằng One-Hot Encoding:

- `Sex`
- `Embarked`
- `Title`
- `Deck`

Tôi dùng `handle_unknown="ignore"` để mô hình không lỗi nếu `valid.csv` hoặc `eval.csv` có category chưa xuất hiện trong train.

## 6. Feature engineering

Các feature mới được tạo:

- `FamilySize = SibSp + Parch + 1`: tổng số người trong gia đình, bao gồm bản thân hành khách.
- `IsAlone`: bằng 1 nếu `FamilySize == 1`, thể hiện hành khách đi một mình.
- `Title`: trích danh xưng từ `Name`, ví dụ `Mr`, `Mrs`, `Miss`, `Master`.
- `HasCabin`: bằng 1 nếu có thông tin cabin, bằng 0 nếu thiếu.
- `Deck`: chữ cái đầu của `Cabin`, đại diện cho khu vực cabin.

Các feature này được tạo vì EDA cho thấy giới tính, hạng vé, gia đình và thông tin cabin đều có khả năng liên quan đến xác suất sống sót.

## 7. So sánh baseline và mô hình cải thiện

Baseline sử dụng các cột:

- `Pclass`
- `Sex`
- `Age`
- `Fare`
- `Embarked`

Kết quả trên `valid.csv`:

| Experiment | Valid Accuracy |
| --- | ---: |
| baseline_logreg | 0.805970 |
| baseline_rf | 0.820896 |
| feature_gb | 0.850746 |
| feature_logreg | 0.873134 |
| feature_rf | 0.873134 |

Mô hình tốt nhất đạt Accuracy khoảng `0.873134` trên `valid.csv`. So với baseline Logistic Regression, Accuracy tăng từ khoảng `0.806` lên `0.873`, tức là feature engineering đã giúp cải thiện rõ rệt.

## 8. Điểm làm tốt và chưa tốt

Điểm làm tốt:

- Tạo được các feature dễ giải thích và có liên hệ với bối cảnh Titanic.
- Áp dụng cùng một preprocessing cho train, valid và eval.
- Dùng `Pipeline` để giảm lỗi xử lý thiếu dữ liệu và encoding không nhất quán.
- Có so sánh baseline với mô hình sau khi thêm feature.

Điểm chưa tốt:

- Chưa tuning hyperparameter sâu.
- Chưa kiểm tra cross-validation.
- `AgeGroup` và `FareGroup` chưa được đưa vào mô hình cuối cùng.
- Chưa thử các mô hình mạnh hơn như XGBoost hoặc LightGBM.

## 9. Hướng cải thiện tiếp theo

Các hướng có thể thử thêm:

- Dùng cross-validation để đánh giá ổn định hơn.
- Tuning `RandomForestClassifier` hoặc `GradientBoostingClassifier`.
- Thử chia nhóm `Age` và `Fare` thành các khoảng.
- Khai thác thêm `Ticket`, ví dụ nhóm vé giống nhau.
- So sánh feature importance để giải thích mô hình tốt hơn.

## 10. File đầu ra

File dự đoán cho `eval.csv` là:

`submission_eval.csv`

File này có đúng 2 cột:

- `PassengerId`
- `Survived`

Số dòng bằng với `eval.csv`: 134 dòng.
