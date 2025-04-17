from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("การพยากรณ์โรคหัวใจล้มเหลว")
st.header("👨🏽‍⚕️👨🏽‍⚕️ด้วยเทคนิคเหมืองแร่ข้อมูล👨🏽‍⚕️👨🏽‍⚕️")

st.image('./img/H1.jpg')

c1,c2,c3=st.columns(3)
with c1:
    st.write('./img/H1.jpg')
with c2:
    st.write('./img/H2.jpg')
with c3:
    st.write("")

dt= pd.read_csv('./data/heart2.csv')

st.header("ข้อมูลโรคหัวใจ")
st.write(dt.head(10))

count_male = dt.groupby('Sex').size()[1]
count_female = dt.groupby('Sex').size()[0]
dx = [count_male, count_female]
dx2 = pd.DataFrame(dx, index=["Male", "Female"])
st.bar_chart(dx2)

st.subheader("สถิติข้อมูลโรคหัวใจ")
st.write(dt.describe())
st.write("สถิติจำนวนเพศหญิง=0 เพสชาย=1")
st.write(dt.groupby('Sex')['Sex'].count())
count_male = dt.groupby('Sex').size()[0]
dx = [count_male, count_female]
dx2 =pd.DataFrame(dx, index=["Male","Female"])
st.bar_chart(dx2)

st.subheader("ข้อมูลแยกตามเพศ")
count_male = dt.groupby('Sex').size()[1]
count_female = dt.groupby('Sex').size()[0]
dx = [count_male, count_female]
dx2 = pd.DataFrame(dx, index=["male", "Female"])
st.bar_chart(dx2)

st.subheader("ข้อมุลค่าเฉลี่ยอายุแยกตามเพศ")
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

A1 = st.number_input("กรุณาเลือกข้อมูลอายุ")
A2 = st.number_input("กรุณาเลือกข้อมุลเพศชาย=1 เพศหญิง=0")
A3 = st.number_input("กรุณาเลือกข้อมูล3")
A4 = st.number_input("กรุณาเลือกข้อมูล4")
A5 = st.number_input("กรุณาเลือกข้อมูล5")
A6 = st.number_input("กรุณาเลือกข้อมูล6")
A7 = st.number_input("กรุณาเลือกข้อมูล7")
A8 = st.number_input("กรุณาเลือกข้อมูล8")
A9 = st.number_input("กรุณาเลือกข้อมูล9")
A10 = st.number_input("กรุณาเลือกข้อมูล10")
A11 = st.number_input("กรุณาเลือกข้อมูล11")

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length")
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width")

if st.button("ทำนายผล"):
    #st.write("ทำนาย")
   dt = pd.read_csv("./data/heart2.csv") 
   X = dt.drop('HeartDisease', axis=1)
   y = dt.HeartDisease   

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  
    
   x_input = np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == '1':
    st.image("./img/H2.jpg")
   else:
    st.image("./img/H3.jpg")
else:
    st.write("ไม่ทำนาย")