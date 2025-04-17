import streamlit as st
import pandas as pd

st.title("ข้อมูลการทำนายดอกเบี้ยสำหรับฝากเงินในธนาคาร")
st.header("ข้อมูลการทำนายดอกเบี้ยสำหรับฝากเงินในธนาคาร")

st.image('./img/kairung.jpg')
st.subheader("Kairung Hengpraprohm")

dt=pd.read_csv('./data/cirrhosis.csv')
st.header("ข้อมูลดอกไม้")
st.write(dt.head(10))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# โหลดไฟล์
df = pd.read_csv('your_file.csv')  # เปลี่ยนเป็นชื่อไฟล์คุณ
X = df.drop('interest_rate', axis=1)
y = df['interest_rate']

# ถ้ามี categorical column
X = pd.get_dummies(X)

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สเกลข้อมูล
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ทดสอบหลายค่า K
k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())  # แปลงค่าความผิดพลาดให้เป็นบวก

# หาค่า K ที่ให้ MSE ต่ำสุด
best_k = k_range[np.argmin(cv_scores)]
print(f'Best K: {best_k}')
print(f'Lowest MSE: {min(cv_scores)}')

# วาดกราฟ
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, marker='o')
plt.xlabel('K')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('K vs. MSE (Lower is Better)')
plt.grid(True)
plt.show()
