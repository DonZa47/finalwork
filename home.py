from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("การพยากรณ์โรคตับต่างๆ")
st.header("👨🏽‍⚕️👨🏽‍⚕️ด้วยเทคนิคเหมืองแร่ข้อมูล👨🏽‍⚕️👨🏽‍⚕️")

st.image('./img/Liver disease01.jpg')

c1, c2, c3 = st.columns(3)
with c1:
    st.image('./img/Liver disease02.jpg')
with c2:
    st.image('./img/Liver disease03.jpg')
with c3:
    st.image('./img/Liver disease04.jpg')

dt = pd.read_csv('./data/cirrhosis.csv')

st.header("ข้อมูลโรคตับ")
st.write(dt.head(5))

count_male = dt.groupby('Sex').size()[1]
count_female = dt.groupby('Sex').size()[0]
dx = [count_male, count_female]
dx2 = pd.DataFrame(dx, index=["Male", "Female"])
st.bar_chart(dx2)

st.subheader("สถิติข้อมูลโรคตับ")
st.write(dt.describe())
st.write("สถิติจำนวนเพศหญิง=0 เพศชาย=1")
st.write(dt.groupby('Sex')['Sex'].count())
count_male = dt.groupby('Sex').size()[0]
dx = [count_male, count_female]
dx2 = pd.DataFrame(dx, index=["Male", "Female"])
st.bar_chart(dx2)

st.subheader("ข้อมูลแยกตามเพศ")
count_male = dt.groupby('Sex').size()[1]
count_female = dt.groupby('Sex').size()[0]
dx = [count_male, count_female]
dx2 = pd.DataFrame(dx, index=["male", "Female"])
st.bar_chart(dx2)

st.subheader("ข้อมูลค่าเฉลี่ยอายุแยกตามเพศ")
average_male_age = dt[dt['Sex'] == 1]['Age'].mean()
average_female_age = dt[dt['Sex'] == 0]['Age'].mean()
dxavg = [average_male_age, average_female_age]
dxavg2 = pd.DataFrame(dxavg, index=["male", "Female"])
st.bar_chart(dxavg2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

# รับค่าจากผู้ใช้
inputs = []
for i in range(1, 19):
    val = st.number_input(f"กรุณาเลือกข้อมูล{i}")
    inputs.append(val)

if st.button("ทำนายผล"):
    dt = pd.read_csv("./data/cirrhosis.csv")
    dt = dt.dropna()

    for col in dt.columns:
        if dt[col].dtype == 'object':
            dt[col] = pd.factorize(dt[col])[0]

    # ลบคอลัมน์ ID หากมี
    if 'ID' in dt.columns:
        X = dt.drop(['Stage', 'ID'], axis=1)
    else:
        X = dt.drop('Stage', axis=1)

    y = dt['Stage']

    st.write("คอลัมน์ที่ใช้ในการทำนาย:", X.columns.tolist())
    if X.shape[1] != 18:
        st.error(f"จำนวนคุณลักษณะไม่ตรงกัน: ต้องการ 18 แต่มี {X.shape[1]}")
    else:
        Knn_model = KNeighborsClassifier(n_neighbors=3)
        Knn_model.fit(X, y)

        x_input = np.array([inputs])
        out = Knn_model.predict(x_input)

        st.success(f"ผลการทำนาย Stage: {out[0]}")

        # แสดงผลตาม Stage ที่ทำนายได้
        if out[0] == 1:
            st.image("./img/Liver disease01.jpg", caption="Stage 1: ระยะเริ่มต้น")
            st.write("ผลการทำนายคือ Stage 1 - ระยะเริ่มต้นของโรค")
        elif out[0] == 2:
            st.image("./img/Liver disease02.jpg", caption="Stage 2: ระยะปานกลาง")
            st.write("ผลการทำนายคือ Stage 2 - เริ่มมีความรุนแรง ต้องเฝ้าระวัง")
        elif out[0] == 3:
            st.image("./img/Liver disease03.jpg", caption="Stage 3: ระยะรุนแรง")
            st.write("ผลการทำนายคือ Stage 3 - โรครุนแรง ควรพบแพทย์ทันที")
        elif out[0] == 4:
            st.image("./img/Liver disease04.jpg", caption="Stage 4: ระยะวิกฤต")
            st.write("ผลการทำนายคือ Stage 4 - ระยะอันตราย ต้องได้รับการรักษาโดยเร็ว")
        else:
            st.warning("ไม่สามารถจำแนก Stage ได้")
else:
    st.write("ไม่ทำนาย")
