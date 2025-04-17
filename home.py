import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# โหลดข้อมูล
@st.cache_data
def load_data():
    data = pd.read_csv("liver_patient_data.csv")
    return data

data = load_data()
st.title("แอปทำนายโรคตับด้วย KNN")

# แสดงข้อมูล
if st.checkbox("แสดงข้อมูล"):
    st.write(data.head())

# เตรียมข้อมูล
data = data.dropna()
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

X = data.drop(['Dataset'], axis=1)
y = data['Dataset']  # 1 = ป่วย, 2 = ไม่ป่วย

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# โมเดล KNN
k = st.slider("เลือกจำนวน K", 1, 15, 5)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# ประเมินผล
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"ความแม่นยำของโมเดล: {acc:.2f}")

# แบบฟอร์มสำหรับป้อนข้อมูลใหม่
st.header("ทำนายจากข้อมูลผู้ป่วยรายใหม่")
age = st.number_input("อายุ", min_value=1, max_value=120, value=45)
gender = st.selectbox("เพศ", ["ชาย", "หญิง"])
tb = st.number_input("Total Bilirubin", value=1.0)
db = st.number_input("Direct Bilirubin", value=0.5)
alp = st.number_input("Alkaline Phosphotase", value=200)
sgpt = st.number_input("Alamine Aminotransferase", value=30)
sgot = st.number_input("Aspartate Aminotransferase", value=30)
tp = st.number_input("Total Protiens", value=6.5)
alb = st.number_input("Albumin", value=3.0)
ag_ratio = st.number_input("Albumin and Globulin Ratio", value=1.0)

input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [0 if gender == "ชาย" else 1],
    'Total_Bilirubin': [tb],
    'Direct_Bilirubin': [db],
    'Alkaline_Phosphotase': [alp],
    'Alamine_Aminotransferase': [sgpt],
    'Aspartate_Aminotransferase': [sgot],
    'Total_Protiens': [tp],
    'Albumin': [alb],
    'Albumin_and_Globulin_Ratio': [ag_ratio]
})

# ทำนายผล
if st.button("ทำนายผล"):
    result = knn.predict(input_data)
    if result[0] == 1:
        st.error("ผลการทำนาย: มีแนวโน้มเป็นโรคตับ")
    else:
        st.success("ผลการทำนาย: ไม่เป็นโรคตับ")