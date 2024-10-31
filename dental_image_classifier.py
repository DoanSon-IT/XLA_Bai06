# dental_classifier.py
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Hàm đọc ảnh và đặc trưng từ file XML
def load_data(image_folder, xml_folder):
    X = []
    y = []
    for xml_file in os.listdir(xml_folder):
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()
        
        # Lấy nhãn từ XML
        label = root.find('object').find('name').text
        y.append(label)
        
        # Đọc ảnh tương ứng
        image_file = xml_file.replace('.xml', '.jpg')
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        # Resize ảnh và chuyển sang vector đặc trưng
        image_resized = cv2.resize(image, (512, 512))
        features = image_resized.flatten()
        X.append(features)
    
    return np.array(X), np.array(y)

# Đường dẫn tới thư mục ảnh và XML
image_folder = "data/dental_images/class1"
xml_folder = "data/dental_images/class2"

# Tải dữ liệu
X, y = load_data(image_folder, xml_folder)

# Tách dữ liệu huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
