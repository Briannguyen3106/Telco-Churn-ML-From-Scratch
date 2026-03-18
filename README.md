# 🚀 Telco Customer Churn Prediction: Machine Learning From Scratch

## 📌 Giới thiệu Dự án
Dự án này tập trung giải quyết bài toán dự đoán tỷ lệ khách hàng rời mạng (Customer Churn) của một công ty viễn thông. Điểm đặc biệt của dự án không nằm ở việc gọi các thư viện có sẵn, mà là **xây dựng toàn bộ các thuật toán Machine Learning cốt lõi từ con số 0 (from scratch)** chỉ sử dụng thư viện Toán học `NumPy`. 

Mục tiêu là thấu hiểu sâu sắc bản chất toán học, đạo hàm, và cấu trúc dữ liệu đằng sau các mô hình học máy kinh điển.

## 🧠 Các Thuật toán đã tự xây dựng (Implemented from Scratch)
Toàn bộ các mô hình dưới đây được thiết kế theo chuẩn hướng đối tượng (OOP), bao gồm các thuật toán tối ưu (Gradient Descent, Exhaustive Search) và các kỹ thuật chống Overfitting:

1. **Logistic Regression (Lasso/L1 Regularization):** Tối ưu hóa bằng Gradient Descent.
2. **K-Nearest Neighbors (KNN):** Tích hợp tính toán khoảng cách Mahalanobis và trọng số Uniform.
3. **Decision Tree (CART/ID3/C4.5):** Thuật toán đệ quy vét cạn (Exhaustive Search) đo lường bằng Gini/Entropy/Gain Ratio.
4. **Random Forest:** Kỹ thuật Ensemble Learning với Bootstrap Sampling và Feature Randomness.
5. **Linear Support Vector Machine (SVM):** Tối ưu hóa Soft-Margin bằng Hinge Loss và Stochastic Gradient Descent.
6. **Mixed Naive Bayes:** Kiến trúc "lõi kép" kết hợp Gaussian Naive Bayes (cho biến liên tục) và Bernoulli Naive Bayes có Laplace Smoothing (cho biến nhị phân).

## 📊 Bảng So sánh Hiệu năng Thực tế
Dữ liệu đã qua bước tiền xử lý, mã hóa One-Hot, và cân bằng dán nhãn bằng kỹ thuật **SMOTE**. Dưới đây là kết quả đánh giá trên tập Test:

| Mô hình | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression Lasso** | 0.7321 | 0.4975 | 0.7834 | 0.6085 |
| **KNN (k=14, p=mahalanobis)** | 0.6851 | 0.4475 | 0.7861 | 0.5703 |
| **Random Forest (criterion=gini)** | **0.7612** | **0.5352** | 0.7727 | **0.6324** |
| **Decision Tree (max_depth=5)** | 0.7385 | 0.5054 | 0.7567 | 0.6060 |
| **Linear SVM (lr=0.0001, lambda=0.01)** | 0.6859 | 0.4517 | **0.8503** | 0.5900 |
| **Mixed Naive Bayes (alpha=1.0)** | 0.7036 | 0.4662 | 0.7941 | 0.5875 |

## 💡 Key Insights (Phân tích Chuyên sâu)
* **Random Forest là mô hình xuất sắc nhất:** Đạt F1-Score (0.6324) và Độ chính xác cao nhất. Điều này chứng minh mối quan hệ giữa các đặc trưng của khách hàng viễn thông mang tính **phi tuyến tính** cao, và kỹ thuật Ensemble Learning đã khắc phục hoàn hảo điểm yếu của cây quyết định đơn lẻ.
* **SVM là "Vua bắt cá" (Highest Recall):** Linear SVM đạt Recall ấn tượng (0.8503), bắt được hầu hết các khách hàng có ý định rời mạng. Tuy nhiên, do bản chất vẽ siêu mặt phẳng tuyến tính trên tập dữ liệu đã SMOTE, mô hình chấp nhận hy sinh Precision để đổi lấy Recall ("Thà bắt nhầm còn hơn bỏ sót").
* **Sức mạnh của Mixed Naive Bayes:** Việc thiết kế một hệ thống đánh giá hỗn hợp (Gaussian cho cước phí/thời gian + Bernoulli cho các biến One-Hot) giúp mô hình hoạt động cực kỳ ổn định với chi phí tính toán (Computational Cost) gần như bằng không so với việc lặp Gradient Descent.

## 🛠️ Tech Stack
* **Ngôn ngữ:** Python
* **Lõi Toán học & Thuật toán:** `NumPy`
* **Xử lý Dữ liệu:** `Pandas`
* **Trực quan hóa:** `Matplotlib`, `Seaborn`

---
*Tác giả: Nguyễn Danh Bảo*