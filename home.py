import streamlit as st
import pandas as pd

st.title("🐷🐷🐷Website Developing using Python🐷🐷")
st.header("🍖🍖Website Developing using Python🍖🍖")

st.image('./img/kairung.jpg')
st.subheader("Kairung Hengpraprohm")

dt=pd.read_csv('./data/iris-3.csv')
st.header("ข้อมูลดอกไม้")
st.write(dt.head(10))

st.subheader("สถิติข้อมูลดอกไม้ Iris")
st.write('ผลรวม')
cl1,cl2,cl3,cl4=st.columns(4)
cl1.write(dt['sepallength'].sum())
cl2.write(dt['sepalwidth'].sum())
cl3.write(dt['petallength'].sum())
cl4.write(dt['petalwidth'].sum())

st.write("กราฟแท่ง")
a=dt['sepallength'].sum()
b=dt['sepalwidth'].sum()
c=dt['petallength'].sum()
d=dt['petalwidth'].sum()
dx=[a,b,c,d]
cx=pd.DataFrame(dx,index=["sepallength", "sepalwidth", "petallength","petalwidth"])
st.bar_chart(cx)

st.write('ค่าเฉลี่ย')
cl11,cl12,cl13,cl14=st.columns(4)
cl11.write(dt['sepallength'].mean())
cl12.write(dt['sepalwidth'].mean())
cl13.write(dt['petallength'].mean())
cl14.write(dt['petalwidth'].mean())

st.write("Area_Chart")
a=dt['sepallength'].mean()
b=dt['sepalwidth'].mean()
c=dt['petallength'].mean()
d=dt['petalwidth'].mean()
dxt=[a,b,c,d]
cxx=pd.DataFrame(dxt,index=["sepallength", "sepalwidth", "petallength","petalwidth"])
st.area_chart(cxx)


st.write('ค่ามากที่สุด')
cl21,cl22,cl23,cl24=st.columns(4)
cl21.write(dt['sepallength'].max())
cl22.write(dt['sepalwidth'].max())
cl23.write(dt['petallength'].max())
cl24.write(dt['petalwidth'].max())

import numpy as np
import matplotlib.pyplot as plt
labels = ['Men', 'Women','','']
sizes = [35,25,15,25]
explode = (0, 0.1,0,0) 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
st.pyplot(fig1)


st.write('ค่าน้อยที่สุด')
cl31,cl32,cl33,cl34=st.columns(4)
cl31.write(dt['sepallength'].min())
cl32.write(dt['sepalwidth'].min())
cl33.write(dt['petallength'].min())
cl34.write(dt['petalwidth'].min())

st.write("Line_Chart")
cc=[3,8,1,10]
st.line_chart(cc)fjghf
