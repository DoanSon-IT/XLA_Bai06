# iris_classifier.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu Iris từ file CSV
data = pd.read_csv("data/Iris.csv")
X = data.drop(columns=['Species'])  # Các đặc trưng của Iris
y = data['Species']  # Nhãn của lớp

# Tách dữ liệu huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bộ phân loại CART (Gini Index)
cart_model = DecisionTreeClassifier(criterion="gini", random_state=42)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)

# In kết quả của CART
print("CART Model (Gini Index)")
print(f"Accuracy: {accuracy_score(y_test, y_pred_cart) * 100:.2f}%")
print(classification_report(y_test, y_pred_cart))

# Bộ phân loại ID3 (Information Gain)
id3_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
id3_model.fit(X_train, y_train)
y_pred_id3 = id3_model.predict(X_test)

# In kết quả của ID3
print("\nID3 Model (Information Gain)")
print(f"Accuracy: {accuracy_score(y_test, y_pred_id3) * 100:.2f}%")
print(classification_report(y_test, y_pred_id3))

# Bộ phân loại SVM
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# In kết quả của SVM
print("\nSVM Model (Linear Kernel)")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm) * 100:.2f}%")
print(classification_report(y_test, y_pred_svm))
