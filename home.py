from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="การพยากรณ์โรคตับ", layout="wide")

st.title("📌 การพยากรณ์โรคตับ")
st.header("👨🏽‍⚕️ ด้วยเทคนิคเหมืองแร่ข้อมูล (KNN Classifier)")

# ✅ โหลดภาพหลัก
if os.path.exists('Liver_disease01.jpg'):
    st.image('Liver_disease01.jpg')
else:
    st.warning("⚠️ ไม่พบภาพ Liver_disease01.jpg")

# ✅ โหลดข้อมูล
dt = pd.read_csv('./cirrhosis.csv')
st.subheader("🧬 แสดงตัวอย่างข้อมูลโรคตับ")
st.write(dt.head())

# ✅ สถิติเพศ
st.subheader("📊 สถิติจำนวนเพศ")
sex_counts = dt['Sex'].value_counts()
sex_df = pd.DataFrame(sex_counts).rename(columns={'Sex': 'จำนวน'})
st.bar_chart(sex_df)

# ✅ เฉลี่ยอายุแยกตามเพศ
st.subheader("📈 ค่าเฉลี่ยอายุแยกตามเพศ")
avg_age = dt.groupby('Sex')['Age'].mean()
st.bar_chart(avg_age)

# ✅ ฟอร์มรับค่าจากผู้ใช้ (ต้องรู้ว่ามีฟีเจอร์อะไรบ้างใน X)
st.subheader("🔮 ทำนายโรคตับจากข้อมูลที่คุณป้อน")

# ตัวอย่างใช้ฟีเจอร์ 5 ตัว
A1 = st.number_input("กรอกอายุ (Age)", min_value=1.0, max_value=100.0, value=50.0)
A2 = st.selectbox("เลือกเพศ", options=[0, 1], format_func=lambda x: "หญิง" if x == 0 else "ชาย")
A3 = st.number_input("Bilirubin", value=1.0)
A4 = st.number_input("Albumin", value=3.0)
A5 = st.number_input("INR", value=1.0)

# 📌 เตรียมข้อมูลและทำนายเมื่อกดปุ่ม
if st.button("✅ ทำนายผล"):
    try:
        X = dt[['Age', 'Sex', 'Bilirubin', 'Albumin', 'INR']]
        y = dt['Stage']  # เปลี่ยนเป็น target column ที่คุณใช้จริง

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        x_input = np.array([[A1, A2, A3, A4, A5]])
        prediction = model.predict(x_input)

        st.success(f"ผลการทำนาย: โรคตับระยะที่ {prediction[0]}")
        
        if int(prediction[0]) >= 3:
            st.image("./img/H2.jpg", caption="พบความเสี่ยงสูง")
        else:
            st.image("./img/H3.jpg", caption="ความเสี่ยงต่ำ")
    
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการทำนาย: {e}")
