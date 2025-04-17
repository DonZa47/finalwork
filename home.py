import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ระบบพยากรณ์โรคตับ", layout="wide")

st.title("📌 ระบบพยากรณ์โรคตับด้วย KNN")
st.header("👨🏽‍⚕️ วิเคราะห์ข้อมูลโรคตับ (Cirrhosis)")

# ✅ โหลดภาพหลัก
if os.path.exists('Liver_disease01.jpg'):
    st.image('Liver_disease01.jpg', caption="ข้อมูลโรคตับ")
else:
    st.warning("⚠️ ไม่พบภาพ Liver_disease01.jpg")

# ✅ โหลดข้อมูล
csv_path = 'cirrhosis.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    st.subheader("🧬 ข้อมูลตัวอย่าง")
    st.write(df.head(10))

    # ✅ ตรวจสอบค่าที่หายไป
    df = df.dropna()
    
    # ✅ แปลง Sex เป็นตัวเลขหากจำเป็น
    if df['Sex'].dtype == 'object':
        df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    
    # ✅ สถิติเพศ
    st.subheader("📊 สถิติเพศ")
    sex_counts = df['Sex'].value_counts()
    st.bar_chart(sex_counts)

    # ✅ อายุเฉลี่ยตามเพศ
    st.subheader("📈 อายุเฉลี่ยตามเพศ")
    avg_age = df.groupby('Sex')['Age'].mean()
    st.bar_chart(avg_age)

    # ✅ ฟอร์มกรอกข้อมูลทำนาย
    st.subheader("🔮 ป้อนข้อมูลเพื่อตรวจความเสี่ยงโรคตับ")

    # ✅ ตัวอย่างใช้ฟีเจอร์ 5 ตัว
    A1 = st.number_input("อายุ (Age)", 1, 100, 45)
    A2 = st.selectbox("เพศ", options=[0, 1], format_func=lambda x: "หญิง" if x == 0 else "ชาย")
    A3 = st.number_input("Bilirubin", 0.0, 30.0, 1.2)
    A4 = st.number_input("Albumin", 0.0, 10.0, 3.5)
    A5 = st.number_input("INR", 0.0, 10.0, 1.0)

    if st.button("✅ ทำนายผล"):
        try:
            X = df[['Age', 'Sex', 'Bilirubin', 'Albumin', 'INR']]
            y = df['Stage']  # ใช้ 'Stage' เป็น target

            # ✅ เตรียมโมเดล
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(X_scaled, y)

            # ✅ เตรียมข้อมูลผู้ใช้
            user_input = np.array([[A1, A2, A3, A4, A5]])
            user_input_scaled = scaler.transform(user_input)

            prediction = model.predict(user_input_scaled)
            st.success(f"🎯 ผลการทำนาย: อยู่ในระยะที่ {prediction[0]}")

            # ✅ แสดงภาพตามผลลัพธ์
            if int(prediction[0]) >= 3:
                if os.path.exists('./img/H2.jpg'):
                    st.image('./img/H2.jpg', caption="⚠️ ความเสี่ยงสูง")
                else:
                    st.warning("ไม่พบภาพ H2.jpg")
            else:
                if os.path.exists('./img/H3.jpg'):
                    st.image('./img/H3.jpg', caption="✅ ความเสี่ยงต่ำ")
                else:
                    st.warning("ไม่พบภาพ H3.jpg")

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {e}")
else:
    st.error("❌ ไม่พบไฟล์ข้อมูล cirrhosis.csv กรุณาวางไว้ในโฟลเดอร์เดียวกับไฟล์นี้")

